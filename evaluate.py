#METRIC COMPUTATIONS, AS SMALL PURE FUNCTIONS SO EACH CAN BE UNIT TESTED WITH SYNTHETIC ARRAYS INSTEAD OF A TRAINED MODEL

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


#GENERATING PREDICTIONS
def predictions_to_labels(pred_probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(pred_probs) > threshold).astype(int)


#CLASSIFICATION METRICS
def compute_metrics(y_true: np.ndarray, pred_probs: np.ndarray, threshold: float = 0.5) -> Dict:
    #single entry point returning every metric used in the report
    y_true = np.asarray(y_true)
    pred_probs = np.asarray(pred_probs).flatten()
    pred_labels = predictions_to_labels(pred_probs, threshold)

    report = classification_report(y_true, pred_labels, output_dict=True, zero_division=0)

    #Confusion matrix
    cm = confusion_matrix(y_true, pred_labels)

    #ROC-AUC (requiers both class to be present)
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, pred_probs)
        fpr, tpr, _ = roc_curve(y_true, pred_probs)
    else:
        auc, fpr, tpr = float("nan"), np.array([]), np.array([])

    return {
        "classification_report": report,
        "confusion_matrix": cm,
        "roc_auc": auc,
        "roc_fpr": fpr,
        "roc_tpr": tpr,
        "pred_labels": pred_labels,
    }


#DECISION-THRESHOLD CALIBRATION
#selects the threshold on validation data, then apply it once to the test set
def sweep_thresholds(y_true: np.ndarray, pred_probs: np.ndarray, thresholds: Optional[np.ndarray] = None) -> pd.DataFrame:
    #sweep candidate thresholds on validation predictions
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)

    rows = []
    for t in thresholds:
        metrics = compute_metrics(y_true, pred_probs, threshold=float(t))
        report = metrics["classification_report"]
        rows.append({
            "threshold": float(t),
            "precision_malignant": report.get("1", {}).get("precision", float("nan")),
            "recall_malignant": report.get("1", {}).get("recall", float("nan")),
            "f1_malignant": report.get("1", {}).get("f1-score", float("nan")),
            "accuracy": report["accuracy"],
        })
    return pd.DataFrame(rows)


def select_threshold(
    sweep_table: pd.DataFrame,
    target_recall: Optional[float] = None,
    beta: Optional[float] = None,
) -> float:
   # uses a selection rule of either recall target or F-beta maximisation
    if (target_recall is None) == (beta is None):
        raise ValueError("pass exactly one of target_recall or beta")

    table = sweep_table.sort_values("threshold").reset_index(drop=True)

    if target_recall is not None:
        eligible = table[table["recall_malignant"] >= target_recall]
        if eligible.empty:
            raise ValueError(
                f"No threshold in the sweep reaches malignant recall >= {target_recall} "
                f"(max recall in sweep was {table['recall_malignant'].max():.3f}). "
                "Lower target_recall, or the model itself may need improving."
            )
        #the eligible thresholds are the ones meeting the recall floor
        #the highest of those gives the best precision without breaching the floor
        return float(eligible["threshold"].max())

    precision = table["precision_malignant"]
    recall = table["recall_malignant"]
    b2 = beta ** 2
    denom = (b2 * precision) + recall
    fbeta = np.where(denom > 0, (1 + b2) * precision * recall / denom, 0.0)
    return float(table.loc[np.argmax(fbeta), "threshold"])


def plot_threshold_curve(sweep_table: pd.DataFrame, chosen_threshold: Optional[float] = None, ax=None):
    #precision/recall (malignant class) against decision threshold, with the calibrated threshold marked
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    table = sweep_table.sort_values("threshold")
    ax.plot(table["threshold"], table["precision_malignant"], label="Precision (malignant)")
    ax.plot(table["threshold"], table["recall_malignant"], label="Recall (malignant)")
    ax.axvline(0.5, color="gray", linestyle=":", label="Default (0.5)")
    if chosen_threshold is not None:
        ax.axvline(chosen_threshold, color="black", linestyle="--", label=f"Calibrated ({chosen_threshold:.2f})")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Precision/recall vs. decision threshold (validation set)")
    ax.legend()
    return ax


#ROC CURVE
def plot_roc_curve(fpr, tpr, title: str = "ROC Curve", ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    return ax


#ACCURACY / LOSS CURVES
def plot_training_curves(history, metric: str = "accuracy", ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    train_vals = history.history[metric]
    val_vals = history.history[f"val_{metric}"]
    n_epochs = len(train_vals)
    epochs = range(1, n_epochs + 1) #start at epoch 1 instead of 0

    ax.plot(epochs, train_vals)
    ax.plot(epochs, val_vals)
    ax.set_title(f"Model {metric.capitalize()}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.capitalize())
    ax.legend(["Train", "Validation"])
    ax.set_xlim(1, n_epochs) #to show last epoch tick
    ax.set_xticks(range(1, n_epochs + 1)) #increments of 1

    return ax


#CONFUSION MATRIX
def plot_confusion_matrix(cm, class_names=("Benign", "Malignant"), title="Confusion Matrix", ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    #annotates each cell with its count
    thresh = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    return ax


#BAR CHART COMPARING A METRIC ACROSS MODELS/BACKBONES
def plot_metric_bar(table, metric: str = "roc_auc", label_col: str = "backbone", title=None, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(table[label_col], table[metric])
    ax.set_ylabel(metric.replace("_", " ").upper())
    ax.set_ylim(0, 1)
    ax.set_title(title or f"{metric.replace('_', ' ').upper()} comparison")
    for i, v in enumerate(table[metric]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
    return ax