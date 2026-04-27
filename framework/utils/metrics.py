"""
Evaluation metrics for anomaly detection.
"""

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)


def compute_anomaly_score(
    anomaly_maps: np.ndarray, reduction: str = "mean"
) -> np.ndarray:
    """
    Aggregate pixel-level anomaly maps into per-image anomaly scores.

    Args:
        anomaly_maps: Array of shape (N, H, W) or (N, C, H, W).
        reduction: Aggregation strategy — ``'mean'``, ``'max'``, or ``'percentile95'``.

    Returns:
        1-D array of shape (N,) with per-image anomaly scores.
    """
    if anomaly_maps.ndim == 4:
        anomaly_maps = anomaly_maps.mean(axis=1)  # average over channels

    if reduction == "mean":
        return anomaly_maps.reshape(len(anomaly_maps), -1).mean(axis=1)
    elif reduction == "max":
        return anomaly_maps.reshape(len(anomaly_maps), -1).max(axis=1)
    elif reduction == "percentile95":
        return np.percentile(
            anomaly_maps.reshape(len(anomaly_maps), -1), 95, axis=1
        )
    else:
        raise ValueError(f"Unknown reduction '{reduction}'")


def evaluate_anomaly_detection(
    scores: np.ndarray, labels: np.ndarray
) -> Dict[str, float]:
    """
    Compute standard anomaly detection metrics.

    Args:
        scores: Per-image anomaly scores (higher → more anomalous).
        labels: Ground-truth binary labels (0 = normal, 1 = anomaly).

    Returns:
        Dictionary with keys: ``auroc``, ``auprc``, ``fpr_at_95tpr``.
    """
    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)

    fprs, tprs, _ = roc_curve(labels, scores)
    idx = np.searchsorted(tprs, 0.95)
    fpr_at_95tpr = float(fprs[min(idx, len(fprs) - 1)])

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "fpr_at_95tpr": fpr_at_95tpr,
    }


def optimal_threshold(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    """
    Find the threshold that maximises the F1-score on the ROC curve.

    Returns:
        Tuple of (threshold, precision, recall).
    """
    precisions, recalls, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = int(np.argmax(f1_scores[:-1]))
    return float(thresholds[best_idx]), float(precisions[best_idx]), float(recalls[best_idx])
