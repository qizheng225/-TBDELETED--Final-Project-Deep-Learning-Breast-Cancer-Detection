#TRAIN AND EVALUATE EVERY BACKBONE IN cfg.backbones UNDER AN IDENTICAL PROTOCOL (SAME DATA, SAME IMBALANCE HANDLING, SAME EPOCHS) 
# AND COLLECT THE RESULTS INTO ONE COMPARISON TABLE

from dataclasses import replace
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import os
import tempfile

from config import Config
from data_pipeline import make_tf_dataset, oversample_minority_class
from evaluate import compute_metrics
from train import labels_from_dataset, train_frozen_model


def compute_computational_stats(model, test_ds, n_inference_batches: int = 5) -> Dict:
    #trainable/total parameter counts, on-disk model size, and average per image inference time
    trainable_params = int(sum(int(np.prod(w.shape)) for w in model.trainable_weights))
    total_params = int(model.count_params())

    #model size on disk. save to a temp file, measure and clean up
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.keras")
        model.save(path)
        model_size_mb = os.path.getsize(path) / (1024 * 1024)

    #average inference time per image over a few test batches
    per_image_times = []
    for x, _ in test_ds.take(n_inference_batches):
        start = time.perf_counter()
        model.predict(x, verbose=0)
        elapsed = time.perf_counter() - start
        per_image_times.append(elapsed / x.shape[0])
    avg_inference_ms = float(np.mean(per_image_times) * 1000) if per_image_times else float("nan")

    return {
        "trainable_params": trainable_params,
        "total_params": total_params,
        "model_size_mb": model_size_mb,
        "avg_inference_ms_per_image": avg_inference_ms,
    }


def compare_architectures(
    train_ds,
    val_ds,
    test_ds,
    train_labels,
    test_labels,
    cfg: Config,
    backbones: List[str] = None,
    measure_computational: bool = True,
    val_labels=None,
) -> Dict:
    #train each backbone (frozen feature extraction stage) and evaluate it on the validation and test sets
    #model selection only uses validation ROC-AUC, so the test set never influences which backbone wins
    #val_labels is derived from val_ds if not given (val_ds must not be shuffled, so label order matches model.predict)
    backbones = backbones or cfg.backbones
    if val_labels is None:
        val_labels = labels_from_dataset(val_ds)

    models, histories, metrics_by_backbone, val_metrics_by_backbone = {}, {}, {}, {}
    rows = []

    for backbone_name in backbones:
        #training model (timed for the computational comparison)
        start = time.perf_counter()
        model, history = train_frozen_model(
            backbone_name, train_ds, val_ds, train_labels, cfg
        )
        train_seconds = time.perf_counter() - start

        #validation predictions/metrics
        val_probs = model.predict(val_ds).flatten()
        val_metrics = compute_metrics(val_labels, val_probs)
        val_report = val_metrics["classification_report"]

        #test predictions/metrics
        pred_probs = model.predict(test_ds).flatten()
        metrics = compute_metrics(test_labels, pred_probs)

        models[backbone_name] = model
        histories[backbone_name] = history
        metrics_by_backbone[backbone_name] = metrics
        val_metrics_by_backbone[backbone_name] = val_metrics

        report = metrics["classification_report"]
        row = {
            "backbone": backbone_name,
            "accuracy": report["accuracy"],
            "precision_malignant": report.get("1", {}).get("precision", float("nan")),
            "recall_malignant": report.get("1", {}).get("recall", float("nan")),
            "f1_malignant": report.get("1", {}).get("f1-score", float("nan")),
            "roc_auc": metrics["roc_auc"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_recall_malignant": val_report.get("1", {}).get("recall", float("nan")),
            "train_seconds": train_seconds,
        }
        if measure_computational:
            row.update(compute_computational_stats(model, test_ds))
        rows.append(row)

    #ranked by VALIDATION roc-auc (selection criterion). "roc_auc" below is the TEST roc-auc, reported for information
    full_table = pd.DataFrame(rows).sort_values("val_roc_auc", ascending=False).reset_index(drop=True)

    classification_cols = [
        "backbone", "val_roc_auc", "val_recall_malignant",
        "accuracy", "precision_malignant", "recall_malignant", "f1_malignant", "roc_auc",
    ]
    computational_cols = ["backbone", "train_seconds"]
    if measure_computational:
        computational_cols += [
            "trainable_params", "total_params", "model_size_mb", "avg_inference_ms_per_image",
        ]

    return {
        # Dictionary with
        # "table": pandas DataFrame COMPARING ACCURACY / PRECISION / RECALL / F1 / AUC
        # "computational_table": TRAINING TIME / PARAMS / SIZE / INFERENCE TIME PER BACKBONE
        # "models": {backbone_name: trained keras.Model}
        # "histories": {backbone_name: History}
        # "metrics": {backbone_name: metrics dict from evaluate.compute_metrics}
        "table": full_table[classification_cols],
        "computational_table": full_table[computational_cols],
        "models": models,
        "histories": histories,
        "metrics": metrics_by_backbone,
        "val_metrics": val_metrics_by_backbone,
    }


def best_backbone(comparison_result: Dict) -> str:
    #picks the backbone with the highest validation ROC AUC
    table = comparison_result["table"]
    if "val_roc_auc" in table.columns and table["val_roc_auc"].notna().any():
        return table.loc[table["val_roc_auc"].idxmax(), "backbone"] #falls back to the top row if there is no validation column
    return table.iloc[0]["backbone"]


#compares the 3 class imbalance strategies on a fixed backbone
def compare_imbalance_strategies(
    backbone_name: str,
    train_df,
    val_ds,
    test_ds,
    test_labels,
    cfg: Config,
    strategies: List[str] = ("class_weight", "oversample", "focal_loss"),
    val_labels=None,
) -> Dict:
    #trains the same backbone once per strategy, using a copy of cfg with only imbalance_strategy changed
    #rest of protocol (epochs, learning rate, etc) stays identical across comparison
    #strategies are ranked on validation metrics
    if val_labels is None:
        val_labels = labels_from_dataset(val_ds)
    models = {}
    rows = []

    for strategy in strategies:
        strategy_cfg = replace(cfg, imbalance_strategy=strategy)

        #oversampling happens at dataframe level, before tf.data
        #dataset is built, classweight/focal_loss are applied during model.fit
        if strategy == "oversample":
            strat_train_df = oversample_minority_class(train_df, random_state=cfg.random_state)
        else:
            strat_train_df = train_df

        strat_train_ds = make_tf_dataset(strat_train_df, cfg.img_size, cfg.batch_size, shuffle=True)

        model, _ = train_frozen_model(
            backbone_name, strat_train_ds, val_ds, strat_train_df["label"].values, strategy_cfg
        )
        val_probs = model.predict(val_ds).flatten()
        val_metrics = compute_metrics(val_labels, val_probs)
        val_report = val_metrics["classification_report"]

        pred_probs = model.predict(test_ds).flatten()
        metrics = compute_metrics(test_labels, pred_probs)
        models[strategy] = model

        report = metrics["classification_report"]
        rows.append(
            {
                "strategy": strategy,
                "accuracy": report["accuracy"],
                "precision_malignant": report.get("1", {}).get("precision", float("nan")),
                "recall_malignant": report.get("1", {}).get("recall", float("nan")),
                "f1_malignant": report.get("1", {}).get("f1-score", float("nan")),
                "roc_auc": metrics["roc_auc"],
                "val_roc_auc": val_metrics["roc_auc"],
                "val_recall_malignant": val_report.get("1", {}).get("recall", float("nan")),
            }
        )

    table = pd.DataFrame(rows).sort_values("val_roc_auc", ascending=False).reset_index(drop=True)
    return {"table": table, "models": models}


def best_imbalance_strategy(result: Dict, metric: str = "val_roc_auc") -> str:
    #picks a strategy using a VALIDATION metric. state the chosen criterion in the report
    #(e.g. pass metric="val_recall_malignant" if recall is prioritised over discrimination)
    table = result["table"]
    return table.loc[table[metric].idxmax(), "strategy"]