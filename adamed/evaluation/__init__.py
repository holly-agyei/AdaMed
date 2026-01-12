"""
Evaluation and visualization tools for AdaMed experiments.

Modules:
    metrics: Classification and domain adaptation metrics
    visualization: Plotting functions for training curves, feature spaces, etc.
"""

from .metrics import (
    compute_classification_metrics,
    compute_domain_adaptation_metrics,
    compute_a_distance,
)
from .visualization import (
    plot_training_curves,
    plot_feature_space,
    plot_gradient_analysis,
    plot_domain_confusion,
)

__all__ = [
    "compute_classification_metrics",
    "compute_domain_adaptation_metrics",
    "compute_a_distance",
    "plot_training_curves",
    "plot_feature_space",
    "plot_gradient_analysis",
    "plot_domain_confusion",
]
