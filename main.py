#END-TO-END PIPELINE:
#  1. LOAD AND MERGE CBIS-DDSM CSVS                         (data_pipeline)
#  2. STRATIFIED TRAIN/VAL/TEST SPLIT                       (data_pipeline)
#  3. HANDLE CLASS IMBALANCE                                (config.imbalance_strategy)
#  4. COMPARE CNN ARCHITECTURES (FROZEN TRANSFER LEARNING)  (model_comparison)
#  5. COMPARE CLASS-IMBALANCE STRATEGIES ON THE WINNER      (model_comparison)
#  6. HYPERPARAMETER-TUNE THE WINNING ARCHITECTURE          (hyperparameter_tuning)
#  7. TRAIN FROZEN, THEN FINE-TUNE                          (fine_tune)
#  8. EVALUATE ON THE HELD-OUT TEST SET                     (evaluate)
#  9. EXPLAIN PREDICTIONS WITH GRAD-CAM + DEEPLIFT          (explainability)

#EVERY FIGURE/TABLE IS SAVED TO cfg.model_outputs_dir 

import os
from dataclasses import replace

import numpy as np
import pandas as pd
import tensorflow as tf

from config import Config
from data_pipeline import (
    build_dataset_df,
    load_raw_csvs,
    make_tf_dataset,
    oversample_minority_class,
    split_dataset,
)
from evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_metric_bar,
    plot_roc_curve,
    plot_threshold_curve,
    plot_training_curves,
    select_threshold,
    sweep_thresholds,
)
from explainability import compare_gradcam_and_deeplift, plot_explainability_grid
from fine_tune import fine_tune_model
from hyperparameter_tuning import (
    best_trial,
    make_keras_train_fn,
    run_hyperparameter_search,
    trials_to_dataframe,
)
from model_comparison import best_backbone, compare_architectures, compare_imbalance_strategies
from train import train_frozen_model


#SELECT ONE TEST EXAMPLE PER PREDICTION OUTCOME: CORRECTLY CLASSIFIED BENIGN, CORRECTLY CLASSIFIED MALIGNANT, FALSE POSITIVE, FALSE NEGATIVE,
#(RATHER THAN AN ARBITRARY SAMPLE OF TEST IMAGES). A CATEGORY IS OMITTED IF NO SUCH EXAMPLE EXISTS IN THE TEST SET.
def select_explainability_examples(test_df, true_labels, pred_labels):
    wanted = {
        "correct_benign": lambda t, p: t == 0 and p == 0,
        "correct_malignant": lambda t, p: t == 1 and p == 1,
        "false_positive": lambda t, p: t == 0 and p == 1,
        "false_negative": lambda t, p: t == 1 and p == 0,
    }
    label_text = {
        "correct_benign": "Correctly classified\nbenign",
        "correct_malignant": "Correctly classified\nmalignant",
        "false_positive": "False positive",
        "false_negative": "False negative",
    }
    found = {}
    for i, (t, p) in enumerate(zip(true_labels, pred_labels)):
        for key, cond in wanted.items():
            if key not in found and cond(t, p):
                found[key] = i
    return [
        (label_text[key], test_df["real_path"].values[idx])
        for key, idx in found.items()
    ]


def main(cfg: Config = Config()):
    #seed every controlled sources of randomness for reproducible results.
    #tf.random.set_seed also stabilises keras data augmentation, but
    #parallel tf.data operations(mapping (num_parallel_class=AUTOTUNE)) may still cause minor nondeterminism.
    #so repeated runs may produce close but not identical results
    import random

    import tensorflow as tf

    random.seed(cfg.random_state)
    np.random.seed(cfg.random_state)
    tf.random.set_seed(cfg.random_state)

    #GPU CHECK
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"Training on GPU: {[g.name for g in gpus]}")
        #enable GPU memory growth to avoid pre-allocating all VRAM
        #this reduces out-of-memory errors when multiple models run in 1 session
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            #memory growth to be set before GPU use. if initialised, ignore error as it only affects memory allocation strategy, not correctness
            print(f"Could not enable GPU memory growth ({exc}); continuing with default allocation.")
    else:
        print("No GPU visible to TensorFlow - training will run on CPU")

    os.makedirs(cfg.model_outputs_dir, exist_ok=True)

    def artifact(name: str) -> str:
        return os.path.join(cfg.model_outputs_dir, name)

    #1 and 2. LOAD CBIS-DDSM DATASET + TRAIN/VALIDATION/TEST SPLIT
    mass_df, dicom_df = load_raw_csvs(cfg.mass_csv_path, cfg.dicom_csv_path)
    dataset_df = build_dataset_df(mass_df, dicom_df, cfg.jpeg_prefix_old, cfg.jpeg_prefix_new)
    train_df, val_df, test_df = split_dataset(
        dataset_df,
        holdout_size=cfg.val_test_split,
        test_size_of_holdout=cfg.test_split_of_holdout,
        random_state=cfg.random_state,
    )
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    #3. CLASS IMBALANCE (OVERSAMPLING HAPPENS AT THE DATAFRAME LEVEL. class_weight / focal_loss ARE APPLIED LATER, DURING TRAINING)
    if cfg.imbalance_strategy == "oversample":
        train_df = oversample_minority_class(train_df, random_state=cfg.random_state)
        print(f"train after oversampling: {len(train_df)}")

    #create tf.data datasets
    train_ds = make_tf_dataset(train_df, cfg.img_size, cfg.batch_size, shuffle=True)
    val_ds = make_tf_dataset(val_df, cfg.img_size, cfg.batch_size)
    test_ds = make_tf_dataset(test_df, cfg.img_size, cfg.batch_size)

    #4. COMPARE ARCHITECTURES
    comparison = compare_architectures(
        train_ds, val_ds, test_ds, train_df["label"].values, test_df["label"].values, cfg,
        val_labels=val_df["label"].values,
    )
    print(comparison["table"])
    comparison["table"].to_csv(artifact("table_5_1_architecture_comparison.csv"), index=False)
    comparison["computational_table"].to_csv(artifact("table_5_8_computational_comparison.csv"), index=False)

    ax = plot_metric_bar(comparison["table"], metric="roc_auc", title="ROC-AUC comparison of candidate CNN architectures")
    ax.figure.savefig(artifact("figure_5_1_roc_auc_comparison.png"), dpi=150, bbox_inches="tight")

    winner = best_backbone(comparison)
    print(f"Best backbone by validation ROC-AUC: {winner}")

    #5. CLASS-IMBALANCE STRATEGY COMPARISON ON THE WINNING ARCHITECTURE
    imbalance_comparison = compare_imbalance_strategies(
        winner, train_df, val_ds, test_ds, test_df["label"].values, cfg,
        val_labels=val_df["label"].values,
    )
    print(imbalance_comparison["table"])
    imbalance_comparison["table"].to_csv(artifact("table_5_2_class_imbalance.csv"), index=False)

    #6. HYPERPARAMETER SEARCH ON THE WINNING ARCHITECTURE
    train_fn = make_keras_train_fn(winner, train_df, val_df, cfg)
    hp_results = run_hyperparameter_search(cfg, train_fn)
    top_trial = best_trial(hp_results)
    print(f"Best hyperparameters: {top_trial.params} (val_auc={top_trial.val_auc:.4f})")
    trials_to_dataframe(hp_results).to_csv(artifact("table_5_3_hyperparameter_search.csv"), index=False)

    #rebuild the final model with the winning hyperparameters
    final_train_ds = make_tf_dataset(
        train_df, cfg.img_size, top_trial.params["batch_size"], shuffle=True
    )
    final_val_ds = make_tf_dataset(val_df, cfg.img_size, top_trial.params["batch_size"])
    final_test_ds = make_tf_dataset(test_df, cfg.img_size, top_trial.params["batch_size"])

    #config for the final model: tuned hyperparameters, everything else unchanged
    tuned_cfg = replace(
        cfg,
        learning_rate=top_trial.params["learning_rate"],
        dropout=top_trial.params["dropout"],
        dense_units=top_trial.params["dense_units"],
    )
    #training model (frozen stage). goes through train_frozen_model so the configured imbalance strategy
    #(class weights / focal loss) and the early-stopping / LR-reduction callbacks are applied
    model, history = train_frozen_model(
        winner, final_train_ds, final_val_ds, train_df["label"].values, tuned_cfg
    )

    #evaluate the frozen model before fine-tuning, to report the before/after comparison
    frozen_pred_probs = model.predict(final_test_ds).flatten()
    frozen_metrics = compute_metrics(test_df["label"].values, frozen_pred_probs)

    #7. FINE-TUNE
    model, ft_history = fine_tune_model(
        model, final_train_ds, final_val_ds, tuned_cfg, train_labels=train_df["label"].values
    )

    #checkpoint right after fine-tuning, before eval/explainability. since training is most expensive, saving here prevents training time costs
    #if a crash was to happen in a later step. Reloads with explain_only() function below.
    checkpoint_path = artifact("final_model_finetuned.keras")
    model.save(checkpoint_path)
    print(f"Saved fine-tuned model checkpoint to {checkpoint_path}")

    #7b. CALIBRATE THE DECISION THRESHOLD ON THE VALIDATION SET (never on test - see evaluate.sweep_thresholds)
    val_probs = model.predict(final_val_ds).flatten()
    threshold_sweep = sweep_thresholds(val_df["label"].values, val_probs)
    threshold_sweep.to_csv(artifact("table_threshold_sweep.csv"), index=False)

    calibrated_threshold = select_threshold(threshold_sweep, target_recall=cfg.target_malignant_recall)
    print(f"Calibrated decision threshold (validation, target malignant recall={cfg.target_malignant_recall}): {calibrated_threshold:.2f}")

    thresh_ax = plot_threshold_curve(threshold_sweep, chosen_threshold=calibrated_threshold)
    thresh_ax.figure.savefig(artifact("figure_5_5c_threshold_calibration.png"), dpi=150, bbox_inches="tight")

    #8. EVALUATE ON TEST SET
    # both at the default 0.5 threshold (for comparability with earlier results) and at the threshold calibrated above
    pred_probs = model.predict(final_test_ds).flatten()
    metrics = compute_metrics(test_df["label"].values, pred_probs)
    calibrated_metrics = compute_metrics(test_df["label"].values, pred_probs, threshold=calibrated_threshold)
    print("Default threshold (0.50):")
    print(metrics["classification_report"])
    print("Confusion matrix:\n", metrics["confusion_matrix"])
    print(f"Calibrated threshold ({calibrated_threshold:.2f}):")
    print(calibrated_metrics["classification_report"])
    print("Confusion matrix:\n", calibrated_metrics["confusion_matrix"])
    print("ROC-AUC (threshold-independent):", metrics["roc_auc"])

    #effect of fine-tuning (frozen vs fine-tuned)
    def _row(stage, m):
        report = m["classification_report"]
        return {
            "stage": stage,
            "accuracy": report["accuracy"],
            "precision_malignant": report.get("1", {}).get("precision", float("nan")),
            "recall_malignant": report.get("1", {}).get("recall", float("nan")),
            "f1_malignant": report.get("1", {}).get("f1-score", float("nan")),
            "roc_auc": m["roc_auc"],
        }

    finetune_table = pd.DataFrame([_row("Frozen backbone", frozen_metrics), _row("Fine-tuned", metrics)])
    finetune_table.to_csv(artifact("table_5_4_finetuning_effect.csv"), index=False)

    #final model performance, default threshold vs calibrated threshold
    pd.DataFrame([
        _row("Final model (threshold=0.50)", metrics),
        _row(f"Final model (calibrated threshold={calibrated_threshold:.2f})", calibrated_metrics),
    ]).to_csv(artifact("table_5_5_final_performance.csv"), index=False)

    #confusion matrices at both thresholds
    cm_ax = plot_confusion_matrix(metrics["confusion_matrix"], title="Confusion matrix - final model (threshold=0.50)")
    cm_ax.figure.savefig(artifact("figure_5_5b_confusion_matrix.png"), dpi=150, bbox_inches="tight")

    cm_cal_ax = plot_confusion_matrix(
        calibrated_metrics["confusion_matrix"],
        title=f"Confusion matrix - final model (calibrated threshold={calibrated_threshold:.2f})",
    )
    cm_cal_ax.figure.savefig(artifact("figure_5_5b_confusion_matrix_calibrated.png"), dpi=150, bbox_inches="tight")

    roc_ax = plot_roc_curve(metrics["roc_fpr"], metrics["roc_tpr"], title="ROC curve - final model")
    roc_ax.figure.savefig(artifact("figure_5_5_roc_curve.png"), dpi=150, bbox_inches="tight")

    #training/validation accuracy + loss curves (frozen stage)
    acc_ax = plot_training_curves(history, "accuracy")
    acc_ax.figure.savefig(artifact("figure_5_6_accuracy_curve.png"), dpi=150, bbox_inches="tight")

    loss_ax = plot_training_curves(history, "loss")
    loss_ax.figure.savefig(artifact("figure_5_6_loss_curve.png"), dpi=150, bbox_inches="tight")

    #explainability (one example per outcome), selected using the calibrated threshold
    pred_labels = (pred_probs > calibrated_threshold).astype(int)
    examples = select_explainability_examples(test_df, test_df["label"].values, pred_labels)
    background_paths = train_df["real_path"].values[: cfg.deeplift_background_samples]

    import tensorflow as tf

    def _load(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [cfg.img_size, cfg.img_size])
        return tf.cast(img, tf.float32).numpy()

    background = np.stack([_load(p) for p in background_paths])

    row_labels, images, cams, deeplift_maps = [], [], [], []
    deeplift_method = "DeepLIFT"
    for label, path in examples:
        image = _load(path)
        cam, deeplift_map, deeplift_method = compare_gradcam_and_deeplift(
            model, image, background, return_method=True
        )
        row_labels.append(label)
        images.append(image)
        cams.append(cam)
        deeplift_maps.append(deeplift_map)

    if images:
        plot_explainability_grid(
            images, cams, deeplift_maps, row_labels,
            save_path=artifact("figure_5_7_explainability_comparison.png"),
            deeplift_label=deeplift_method,
        )
        print(f"Saved explainability grid with rows: {row_labels} (DeepLIFT column: {deeplift_method})")
    else:
        print("No explainability examples found (unexpected - check test set predictions).")

    return {
        "model": model,
        "comparison": comparison,
        "imbalance_comparison": imbalance_comparison,
        "hp_results": hp_results,
        "frozen_metrics": frozen_metrics,
        "metrics": metrics,
        "calibrated_threshold": calibrated_threshold,
        "calibrated_metrics": calibrated_metrics,
        "threshold_sweep": threshold_sweep,
    }


#RE-RUN EVALUATION + EXPLAINABILITY FROM A SAVED CHECKPOINT(AFTER FINE-TUNE), SKIPPING ALL TRAINING. 
#USED TO ITERATE ON STEP 8/9 WITHOUT COSTING THE FULL TRAINING RUN AGAIN
#USAGE: python main.py --explain-only model_outputs/final_model_finetuned.keras
def explain_only(cfg: Config, checkpoint_path: str):
    import tensorflow as tf

    model = tf.keras.models.load_model(checkpoint_path)

    #rebuilds the data split, (steps 1-3, no training)
    mass_df, dicom_df = load_raw_csvs(cfg.mass_csv_path, cfg.dicom_csv_path)
    dataset_df = build_dataset_df(mass_df, dicom_df, cfg.jpeg_prefix_old, cfg.jpeg_prefix_new)
    train_df, val_df, test_df = split_dataset(
        dataset_df,
        holdout_size=cfg.val_test_split,
        test_size_of_holdout=cfg.test_split_of_holdout,
        random_state=cfg.random_state,
    )
    val_ds = make_tf_dataset(val_df, cfg.img_size, cfg.batch_size)
    test_ds = make_tf_dataset(test_df, cfg.img_size, cfg.batch_size)

    def artifact(name: str) -> str:
        return os.path.join(cfg.model_outputs_dir, name)

    #7b. CALIBRATE THE DECISION THRESHOLD ON THE VALIDATION SET (same as in main())
    val_probs = model.predict(val_ds).flatten()
    threshold_sweep = sweep_thresholds(val_df["label"].values, val_probs)
    calibrated_threshold = select_threshold(threshold_sweep, target_recall=cfg.target_malignant_recall)
    print(f"Calibrated decision threshold (validation, target malignant recall={cfg.target_malignant_recall}): {calibrated_threshold:.2f}")

    #8. EVALUATE ON TEST SET
    pred_probs = model.predict(test_ds).flatten()
    metrics = compute_metrics(test_df["label"].values, pred_probs)
    calibrated_metrics = compute_metrics(test_df["label"].values, pred_probs, threshold=calibrated_threshold)
    print("Default threshold (0.50):")
    print(metrics["classification_report"])
    print(f"Calibrated threshold ({calibrated_threshold:.2f}):")
    print(calibrated_metrics["classification_report"])
    print("ROC-AUC (threshold-independent):", metrics["roc_auc"])

    cm_ax = plot_confusion_matrix(metrics["confusion_matrix"], title="Confusion matrix - final model (threshold=0.50)")
    cm_ax.figure.savefig(artifact("figure_5_5_confusion_matrix.png"), dpi=150, bbox_inches="tight")
    cm_cal_ax = plot_confusion_matrix(
        calibrated_metrics["confusion_matrix"],
        title=f"Confusion matrix - final model (calibrated threshold={calibrated_threshold:.2f})",
    )
    cm_cal_ax.figure.savefig(artifact("figure_5_5b_confusion_matrix_calibrated.png"), dpi=150, bbox_inches="tight")
    roc_ax = plot_roc_curve(metrics["roc_fpr"], metrics["roc_tpr"], title="ROC curve - final model")
    roc_ax.figure.savefig(artifact("figure_5_5_roc_curve.png"), dpi=150, bbox_inches="tight")

    #9. EXPLAINABILITY (ONE EXAMPLE PER OUTCOME), selected at the calibrated threshold
    pred_labels = (pred_probs > calibrated_threshold).astype(int)
    examples = select_explainability_examples(test_df, test_df["label"].values, pred_labels)
    background_paths = train_df["real_path"].values[: cfg.deeplift_background_samples]

    import tensorflow as tf

    def _load(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [cfg.img_size, cfg.img_size])
        return tf.cast(img, tf.float32).numpy()

    background = np.stack([_load(p) for p in background_paths])

    row_labels, images, cams, deeplift_maps = [], [], [], []
    deeplift_method = "DeepLIFT"
    for label, path in examples:
        image = _load(path)
        cam, deeplift_map, deeplift_method = compare_gradcam_and_deeplift(
            model, image, background, return_method=True
        )
        row_labels.append(label)
        images.append(image)
        cams.append(cam)
        deeplift_maps.append(deeplift_map)

    if images:
        plot_explainability_grid(
            images, cams, deeplift_maps, row_labels,
            save_path=artifact("figure_5_7_explainability_comparison.png"),
            deeplift_label=deeplift_method,
        )

    return {"model": model, "metrics": metrics, "calibrated_threshold": calibrated_threshold, "calibrated_metrics": calibrated_metrics}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2 and sys.argv[1] == "--explain-only":
        explain_only(Config(), sys.argv[2])
    else:
        main()