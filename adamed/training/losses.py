"""
Loss functions for AdaMed domain adaptation experiments.

The fundamental challenge: standard DANN loss assumes both source AND target
samples flow through the domain classifier. In zero-shot (no target data),
the domain loss provides no adaptation signal — just source memorization.

This file contains:
1. Standard DANN loss (for baseline comparison)
2. Proposed prior-informed loss (experimental, documented failure)
3. Alpha scheduling utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple


def compute_alpha(epoch: int, max_epochs: int = 100) -> float:
    """
    Compute gradient reversal strength using Ganin schedule.

    The schedule smoothly increases alpha from 0 to 1:
        alpha = 2 / (1 + exp(-10 * p)) - 1, where p = epoch / max_epochs

    Why not constant alpha?
    - Starting with alpha=0 lets the feature extractor learn task-relevant features first
    - Gradually increasing forces domain invariance without destroying early learning
    - In practice (for our zero-shot case), the schedule doesn't help — but it's
      the standard approach we need to compare against

    Args:
        epoch: Current training epoch
        max_epochs: Total number of epochs

    Returns:
        Alpha value in [0, 1]
    """
    p = float(epoch) / max_epochs
    return float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)


def domain_adversarial_loss(
    label_output: torch.Tensor,
    domain_output: torch.Tensor,
    labels: torch.Tensor,
    domains: torch.Tensor,
    source_mask: Optional[torch.Tensor] = None,
    lambda_domain: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Standard DANN loss: task loss + lambda * domain loss.

    L = L_task(source samples only) + lambda * L_domain(all samples)

    The domain loss uses CrossEntropy over domain predictions.
    Gradient reversal is applied inside the model (not here).

    Args:
        label_output: Task predictions, shape (batch, num_classes)
        domain_output: Domain predictions, shape (batch, 2)
        labels: Ground truth labels, shape (batch,)
        domains: Domain labels, shape (batch,)
        source_mask: Boolean mask for source samples (where labels are valid)
        lambda_domain: Weight for domain loss

    Returns:
        total_loss: Combined scalar loss
        loss_dict: Individual loss components for logging
    """
    # Task loss — only on source samples where we have real labels
    if source_mask is not None:
        label_loss = F.cross_entropy(label_output[source_mask], labels[source_mask])
    else:
        # Fallback: assume all samples have valid labels (source-only training)
        label_loss = F.cross_entropy(label_output, labels)

    # Domain loss — on all samples
    domain_loss = F.cross_entropy(domain_output, domains)

    # Combined loss
    # Note: gradient reversal is already applied in the model's forward pass
    # so we ADD domain_loss (the reversal makes it adversarial)
    total_loss = label_loss + lambda_domain * domain_loss

    loss_dict = {
        "label_loss": label_loss.item(),
        "domain_loss": domain_loss.item(),
        "total_loss": total_loss.item(),
    }

    return total_loss, loss_dict


def prior_informed_loss(
    label_output: torch.Tensor,
    features: torch.Tensor,
    labels: torch.Tensor,
    prior_means: Optional[torch.Tensor] = None,
    prior_stds: Optional[torch.Tensor] = None,
    lambda_prior: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    EXPERIMENTAL: Loss function encoding physiological priors.

    Idea: Instead of adversarial domain alignment (which fails zero-shot),
    penalize features that deviate from known physiological constraints.

    L = L_task + lambda * L_prior

    where L_prior = KL(feature_distribution || prior_distribution)

    Status: DOES NOT WORK as implemented. The prior distributions are
    too vague to provide meaningful gradients. Documenting for future
    reference — a proper version would need:
    1. Feature-specific physiological bounds (not just means/stds)
    2. Temporal consistency constraints
    3. Cross-feature correlation priors

    Args:
        label_output: Task predictions
        features: Extracted features from feature extractor
        labels: Ground truth labels
        prior_means: Prior mean for each feature dimension
        prior_stds: Prior std for each feature dimension
        lambda_prior: Weight for prior loss

    Returns:
        total_loss, loss_dict
    """
    # Task loss
    label_loss = F.cross_entropy(label_output, labels)

    # Prior regularization
    if prior_means is not None and prior_stds is not None:
        # Gaussian prior: penalize deviation from expected distribution
        feature_mean = features.mean(dim=0)
        feature_std = features.std(dim=0) + 1e-8

        # KL divergence between N(feature_mean, feature_std) and N(prior_mean, prior_std)
        kl = torch.log(prior_stds / feature_std) + \
             (feature_std ** 2 + (feature_mean - prior_means) ** 2) / (2 * prior_stds ** 2) - 0.5
        prior_loss = kl.mean()
    else:
        # Without explicit priors, regularize towards unit normal
        prior_loss = (features.mean(dim=0) ** 2).mean() + \
                     ((features.std(dim=0) - 1) ** 2).mean()

    total_loss = label_loss + lambda_prior * prior_loss

    loss_dict = {
        "label_loss": label_loss.item(),
        "prior_loss": prior_loss.item(),
        "total_loss": total_loss.item(),
    }

    return total_loss, loss_dict
