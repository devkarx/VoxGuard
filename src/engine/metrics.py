"""
Evaluation metrics for deepfake / anti-spoofing detection.

Implements Equal Error Rate (EER), accuracy, and minimum Detection Cost
Function (minDCF) — the standard metrics used across ASVspoof challenges.
"""

import numpy as np
from sklearn.metrics import roc_curve


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> tuple:
    """
    Compute the Equal Error Rate.

    EER is the operating point where the false acceptance rate equals the
    false rejection rate on the ROC curve.

    Args:
        labels: Ground truth binary labels (0 = genuine, 1 = spoof).
        scores: Predicted scores (higher = more likely spoof).

    Returns:
        Tuple of (eer, threshold) where threshold is the decision boundary
        that achieves the EER.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

    # Find the point where FPR and FNR cross
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float(np.mean([fpr[idx], fnr[idx]]))
    threshold = float(thresholds[idx])
    return eer, threshold


def compute_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    """
    Compute simple binary classification accuracy.

    Args:
        labels:      Ground truth binary labels.
        predictions: Predicted binary labels (after thresholding).

    Returns:
        Accuracy as a float between 0 and 1.
    """
    correct = np.sum(labels == predictions)
    return float(correct / len(labels))


def compute_min_dcf(
    labels: np.ndarray,
    scores: np.ndarray,
    p_target: float = 0.05,
    c_miss: float = 1.0,
    c_fa: float = 10.0,
) -> tuple:
    """
    Compute the minimum normalized Detection Cost Function.

    This is the primary metric used in NIST SRE and ASVspoof evaluations.
    It weights missed detections and false alarms differently based on
    the assumed prior probability of encountering a spoof.

    Args:
        labels:   Ground truth binary labels (0 = genuine, 1 = spoof).
        scores:   Predicted scores (higher = more likely spoof).
        p_target: Prior probability of a target (spoof) trial.
        c_miss:   Cost of a missed detection.
        c_fa:     Cost of a false alarm.

    Returns:
        Tuple of (min_dcf, threshold).
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

    # Normalized DCF at every threshold
    dcf = c_miss * fnr * p_target + c_fa * fpr * (1.0 - p_target)
    default_dcf = min(c_miss * p_target, c_fa * (1.0 - p_target))
    normalized_dcf = dcf / default_dcf

    idx = np.argmin(normalized_dcf)
    return float(normalized_dcf[idx]), float(thresholds[idx])
