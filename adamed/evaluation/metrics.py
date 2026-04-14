"""
Evaluation metrics for domain adaptation experiments.
"""

import numpy as np
from typing import Dict, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute standard classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for AUC)

    Returns:
        Dictionary with accuracy, precision, recall, F1, and optionally AUC
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["auc"] = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    return metrics


def compute_domain_adaptation_metrics(
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    source_preds: np.ndarray,
    target_preds: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute domain adaptation quality metrics.

    Includes:
    - Source accuracy (baseline: how well does the model work on source?)
    - Proxy A-distance (measures domain divergence in feature space)
    - Feature distribution statistics

    Args:
        source_features: Features extracted from source domain
        target_features: Features extracted from target domain
        source_labels: True source labels
        target_labels: True target labels (may be dummy)
        source_preds: Model predictions on source
        target_preds: Model predictions on target (optional)

    Returns:
        Dictionary of adaptation metrics
    """
    metrics = {}

    # Source accuracy (sanity check)
    metrics["source_accuracy"] = float(accuracy_score(source_labels, source_preds))

    # Proxy A-distance
    metrics["proxy_a_distance"] = compute_a_distance(source_features, target_features)

    # Feature space statistics
    # Mean and std of features per domain — large differences indicate poor alignment
    source_mean = np.mean(source_features, axis=0)
    target_mean = np.mean(target_features, axis=0)
    mean_distance = np.linalg.norm(source_mean - target_mean)
    metrics["feature_mean_distance"] = float(mean_distance)

    # Variance ratio (should be ~1 if well-aligned)
    source_var = np.var(source_features, axis=0).mean()
    target_var = np.var(target_features, axis=0).mean()
    metrics["feature_variance_ratio"] = float(target_var / (source_var + 1e-8))

    return metrics


def compute_a_distance(
    source_features: np.ndarray,
    target_features: np.ndarray,
    n_estimators: int = 10,
) -> float:
    """
    Compute proxy A-distance between source and target features.

    The A-distance measures how distinguishable two domains are.
    It's computed as: d_A = 2(1 - 2 * error) where error is the
    test error of a linear classifier trained to distinguish domains.

    Interpretation:
    - d_A ≈ 0: domains are indistinguishable (good adaptation)
    - d_A ≈ 2: domains are perfectly separable (no adaptation)

    For our zero-shot experiments, d_A stays near 2 throughout training,
    confirming that domain alignment fails.

    Reference: Ben-David et al., "A theory of learning from different domains" (2010)

    Args:
        source_features: Source domain features, shape (n_source, dim)
        target_features: Target domain features, shape (n_target, dim)
        n_estimators: Number of random train/test splits for stability

    Returns:
        Proxy A-distance value
    """
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import cross_val_score

    # Create domain classification dataset
    X = np.concatenate([source_features, target_features], axis=0)
    y = np.concatenate([
        np.zeros(len(source_features)),
        np.ones(len(target_features)),
    ])

    # Shuffle
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    # Train linear classifier with cross-validation
    clf = SGDClassifier(loss="hinge", max_iter=1000, random_state=42)
    scores = cross_val_score(clf, X, y, cv=min(5, len(X) // 2), scoring="accuracy")

    # A-distance formula
    error = 1 - scores.mean()
    a_distance = 2 * (1 - 2 * error)

    return float(np.clip(a_distance, 0, 2))
