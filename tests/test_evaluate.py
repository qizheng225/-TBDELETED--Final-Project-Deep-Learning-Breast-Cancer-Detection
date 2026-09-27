import numpy as np
import pytest

from evaluate import compute_metrics, predictions_to_labels, select_threshold, sweep_thresholds


def test_predictions_to_labels_thresholds_correctly():
    probs = np.array([0.1, 0.49, 0.5, 0.51, 0.9])
    labels = predictions_to_labels(probs, threshold=0.5)
    assert list(labels) == [0, 0, 0, 1, 1]


def test_compute_metrics_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    pred_probs = np.array([0.01, 0.02, 0.98, 0.99])
    metrics = compute_metrics(y_true, pred_probs)

    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"].tolist() == [[2, 0], [0, 2]]
    assert metrics["classification_report"]["accuracy"] == 1.0


def test_compute_metrics_worst_case_predictions():
    y_true = np.array([0, 0, 1, 1])
    pred_probs = np.array([0.99, 0.98, 0.02, 0.01])  # everything flipped
    metrics = compute_metrics(y_true, pred_probs)

    assert metrics["roc_auc"] == 0.0
    assert metrics["classification_report"]["accuracy"] == 0.0


def test_compute_metrics_handles_single_class_gracefully():
    y_true = np.array([1, 1, 1])
    pred_probs = np.array([0.6, 0.7, 0.8])
    metrics = compute_metrics(y_true, pred_probs)
    assert np.isnan(metrics["roc_auc"])  # AUC undefined with a single class present


def test_compute_metrics_confusion_matrix_shape_matches_binary_case():
    y_true = np.array([0, 1, 0, 1, 0])
    pred_probs = np.array([0.2, 0.8, 0.3, 0.4, 0.9])
    metrics = compute_metrics(y_true, pred_probs)
    assert metrics["confusion_matrix"].shape == (2, 2)

def test_sweep_thresholds_recall_decreases_as_threshold_rises():
    # a threshold separable mock case, recall can only fall (or stay flat) as the threshold rises
    y_true = np.array([0, 0, 1, 1, 1])
    pred_probs = np.array([0.2, 0.4, 0.3, 0.6, 0.9])
    table = sweep_thresholds(y_true, pred_probs, thresholds=np.array([0.1, 0.35, 0.5, 0.7, 0.95]))
    recalls = table.sort_values("threshold")["recall_malignant"].tolist()
    assert all(a >= b for a, b in zip(recalls, recalls[1:]))


def test_sweep_thresholds_returns_one_row_per_threshold():
    y_true = np.array([0, 1, 0, 1])
    pred_probs = np.array([0.1, 0.9, 0.2, 0.8])
    thresholds = np.array([0.3, 0.5, 0.7])
    table = sweep_thresholds(y_true, pred_probs, thresholds=thresholds)
    assert len(table) == len(thresholds)
    assert set(table["threshold"]) == set(thresholds.tolist())


def test_select_threshold_by_target_recall_meets_the_floor():
    y_true = np.array([0, 0, 0, 1, 1, 1, 1])
    pred_probs = np.array([0.1, 0.3, 0.45, 0.4, 0.6, 0.8, 0.95])
    table = sweep_thresholds(y_true, pred_probs)
    chosen = select_threshold(table, target_recall=0.9)

    metrics_at_chosen = compute_metrics(y_true, pred_probs, threshold=chosen)
    recall = metrics_at_chosen["classification_report"]["1"]["recall"]
    assert recall >= 0.9


def test_select_threshold_by_target_recall_picks_highest_eligible_threshold():
    # among thresholds that all satisfy the recall floor, the highest (best precision) one should win
    y_true = np.array([0, 0, 1, 1])
    pred_probs = np.array([0.1, 0.2, 0.8, 0.9])
    table = sweep_thresholds(y_true, pred_probs, thresholds=np.array([0.15, 0.5, 0.75]))
    # recall_malignant is 1.0 at every one of these thresholds for this mock separable case
    assert (table["recall_malignant"] == 1.0).all()
    chosen = select_threshold(table, target_recall=0.9)
    assert chosen == 0.75


def test_select_threshold_raises_when_target_recall_unreachable():
    y_true = np.array([0, 0, 1, 1])
    pred_probs = np.array([0.1, 0.2, 0.3, 0.4])  # never confidently predicts malignant
    table = sweep_thresholds(y_true, pred_probs, thresholds=np.array([0.6, 0.8]))
    with pytest.raises(ValueError):
        select_threshold(table, target_recall=0.95)


def test_select_threshold_requires_exactly_one_selection_rule():
    y_true = np.array([0, 1])
    pred_probs = np.array([0.2, 0.8])
    table = sweep_thresholds(y_true, pred_probs, thresholds=np.array([0.5]))
    with pytest.raises(ValueError):
        select_threshold(table)  # neither given
    with pytest.raises(ValueError):
        select_threshold(table, target_recall=0.9, beta=2.0)  # both given


def test_select_threshold_by_fbeta_favours_recall_as_beta_grows():
    y_true = np.array([0, 0, 0, 1, 1])
    pred_probs = np.array([0.2, 0.3, 0.55, 0.5, 0.6])
    table = sweep_thresholds(y_true, pred_probs)

    low_beta_threshold = select_threshold(table, beta=0.5)
    high_beta_threshold = select_threshold(table, beta=4.0)

    #a beta that weights recall heavily should never pick a higher (stricter) threshold than one that weights precision heavily
    assert high_beta_threshold <= low_beta_threshold