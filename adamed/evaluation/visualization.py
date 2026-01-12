"""
Visualization functions for AdaMed experiments.

Generates publication-quality figures for:
- Training curves (loss, accuracy, gradient norms)
- Feature space visualization (t-SNE)
- Domain confusion analysis
- Ablation studies
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import seaborn as sns
from typing import Dict, Optional, List, Tuple
from sklearn.manifold import TSNE
import os


# Consistent style for all figures
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {"source": "#2196F3", "target": "#FF5722", "good": "#4CAF50", "poor": "#F44336"}
FIGSIZE = (12, 4)


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Plot training loss curves and accuracy.

    Creates a 3-panel figure:
    1. Label loss over epochs
    2. Domain loss over epochs
    3. Label and domain accuracy over epochs

    The key insight visible in these plots: as domain loss decreases,
    label loss INCREASES — the adversarial training destroys task-relevant features.

    Args:
        history: Training history dictionary from DANNTrainer
        save_path: Path to save figure (optional)
        show: Whether to display figure interactively

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = range(len(history["label_loss"]))

    # Panel 1: Label loss
    axes[0].plot(epochs, history["label_loss"], color=COLORS["source"], linewidth=2, label="Label Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Task Loss (Source Domain)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Domain loss
    axes[1].plot(epochs, history["domain_loss"], color=COLORS["target"], linewidth=2, label="Domain Loss")
    if "alpha" in history:
        ax_alpha = axes[1].twinx()
        ax_alpha.plot(epochs, history["alpha"], color="gray", linewidth=1, linestyle="--", alpha=0.5, label="α")
        ax_alpha.set_ylabel("α (reversal strength)", color="gray")
        ax_alpha.legend(loc="upper left")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Domain Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Accuracies
    axes[2].plot(epochs, history["label_acc"], color=COLORS["good"], linewidth=2, label="Label Acc")
    axes[2].plot(epochs, history["domain_acc"], color=COLORS["poor"], linewidth=2, label="Domain Acc")
    axes[2].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_title("Classification Accuracy")
    axes[2].legend()
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_accuracy_curve(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot label accuracy curve with confidence band."""
    fig, ax = plt.subplots(figsize=(8, 5))

    epochs = range(len(history["label_acc"]))
    ax.plot(epochs, history["label_acc"], color=COLORS["source"], linewidth=2, label="Label Accuracy")

    # Add rolling average for smoothing
    if len(history["label_acc"]) > 5:
        window = 5
        smoothed = np.convolve(history["label_acc"], np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(history["label_acc"])), smoothed,
                color=COLORS["source"], linewidth=3, alpha=0.5, label="Smoothed (5-epoch)")

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random Baseline")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Label Prediction Accuracy During Training", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_feature_space(
    features: np.ndarray,
    domains: np.ndarray,
    labels: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "Feature Space (t-SNE)",
    perplexity: int = 30,
) -> plt.Figure:
    """
    t-SNE visualization of learned feature space.

    Color-codes points by domain (and optionally by label).
    In successful domain adaptation, source and target clusters should overlap.
    In our experiments, they remain clearly separated.

    Args:
        features: Feature vectors, shape (n_samples, feature_dim)
        domains: Domain labels, shape (n_samples,)
        labels: Task labels for marker differentiation (optional)
        save_path: Path to save figure
        title: Figure title
        perplexity: t-SNE perplexity parameter

    Returns:
        matplotlib Figure object
    """
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    features_2d = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot by domain
    source_mask = domains == 0
    target_mask = domains == 1

    ax.scatter(
        features_2d[source_mask, 0], features_2d[source_mask, 1],
        c=COLORS["source"], alpha=0.5, s=20, label="Source (Spanish CGM)",
        edgecolors="none",
    )
    ax.scatter(
        features_2d[target_mask, 0], features_2d[target_mask, 1],
        c=COLORS["target"], alpha=0.5, s=20, label="Target (Ghanaian Proxy)",
        edgecolors="none",
    )

    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10, markerscale=3)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_gradient_analysis(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot gradient norm evolution during training.

    This is the key diagnostic for understanding DANN failure:
    - Feature extractor gradients should be balanced between task and domain
    - If domain gradients dominate, adversarial signal overwhelms task learning
    - If feature gradients vanish, the model has collapsed

    Args:
        history: Training history with gradient norm fields
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    epochs = range(len(history.get("grad_norm_features", [])))

    if "grad_norm_features" in history and history["grad_norm_features"]:
        ax.plot(epochs, history["grad_norm_features"],
                color="#2196F3", linewidth=2, label="Feature Extractor")
    if "grad_norm_labels" in history and history["grad_norm_labels"]:
        ax.plot(epochs, history["grad_norm_labels"],
                color="#4CAF50", linewidth=2, label="Label Predictor")
    if "grad_norm_domain" in history and history["grad_norm_domain"]:
        ax.plot(epochs, history["grad_norm_domain"],
                color="#FF5722", linewidth=2, label="Domain Classifier")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Gradient L2 Norm", fontsize=12)
    ax.set_title("Gradient Flow Analysis", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Annotate the problematic region
    if "grad_norm_domain" in history and len(history["grad_norm_domain"]) > 10:
        max_epoch = np.argmax(history["grad_norm_domain"])
        ax.annotate(
            "Domain gradients dominate\n→ task signal destroyed",
            xy=(max_epoch, history["grad_norm_domain"][max_epoch]),
            xytext=(max_epoch + 5, history["grad_norm_domain"][max_epoch] * 2),
            arrowprops=dict(arrowstyle="->", color="red"),
            fontsize=9, color="red",
        )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_domain_confusion(
    domain_true: np.ndarray,
    domain_pred: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot confusion matrix for domain classifier.

    In successful adaptation: domain classifier should be at ~50% accuracy
    (can't distinguish domains). In our case: near 100% (domains easily separable).

    Args:
        domain_true: True domain labels
        domain_pred: Predicted domain labels
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    from sklearn.metrics import confusion_matrix as cm_func

    cm = cm_func(domain_true, domain_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Source", "Target"],
        yticklabels=["Source", "Target"],
        ax=ax,
    )
    ax.set_xlabel("Predicted Domain", fontsize=12)
    ax.set_ylabel("True Domain", fontsize=12)
    ax.set_title("Domain Classification Confusion Matrix", fontsize=14)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_data_distributions(
    source_data: np.ndarray,
    target_data: np.ndarray,
    feature_names: List[str],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot feature distributions for source vs target domains.

    Shows density plots for each feature, highlighting distribution shift.

    Args:
        source_data: Source domain data, shape (n_source, time_steps, n_features)
        target_data: Target domain data, shape (n_target, time_steps, n_features)
        feature_names: Names for each feature
        save_path: Path to save figure

    Returns:
        matplotlib Figure object
    """
    n_features = len(feature_names)
    fig, axes = plt.subplots(1, n_features, figsize=(4 * n_features, 4))

    if n_features == 1:
        axes = [axes]

    for f, (ax, name) in enumerate(zip(axes, feature_names)):
        # Flatten across patients and time steps
        source_vals = source_data[:, :, f].flatten()
        target_vals = target_data[:, :, f].flatten()

        # Remove NaNs
        source_vals = source_vals[~np.isnan(source_vals)]
        target_vals = target_vals[~np.isnan(target_vals)]

        ax.hist(source_vals, bins=50, alpha=0.5, density=True,
                color=COLORS["source"], label="Source")
        ax.hist(target_vals, bins=50, alpha=0.5, density=True,
                color=COLORS["target"], label="Target")
        ax.set_title(name, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.suptitle("Feature Distributions: Source vs Target", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
