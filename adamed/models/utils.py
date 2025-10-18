"""
Model utilities for AdaMed experiments.

Helper functions for model inspection, feature extraction, and analysis.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters.

    Args:
        model: PyTorch model
        trainable_only: If True, count only parameters with requires_grad=True

    Returns:
        Total parameter count
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_parameter_summary(model: nn.Module) -> Dict[str, int]:
    """
    Get per-module parameter counts.
    Useful for understanding where model capacity is allocated.
    """
    summary = {}
    for name, module in model.named_children():
        n_params = sum(p.numel() for p in module.parameters())
        summary[name] = n_params
    summary["total"] = count_parameters(model)
    return summary


def compute_gradient_norms(model: nn.Module) -> Dict[str, float]:
    """
    Compute gradient L2 norms for each named parameter.

    Critical for diagnosing DANN training issues:
    - If domain classifier gradients >> label predictor gradients,
      the adversarial signal is overwhelming the task signal
    - If feature extractor gradients → 0, we have vanishing gradients
      (common when GRL alpha is too high)

    Returns:
        Dictionary mapping parameter names to gradient norms
    """
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norms[name] = param.grad.data.norm(2).item()
        else:
            norms[name] = 0.0
    return norms


def get_module_gradient_norms(model: nn.Module) -> Dict[str, float]:
    """
    Aggregate gradient norms by module (feature_extractor, label_predictor, domain_classifier).
    Higher-level view than per-parameter norms.
    """
    module_norms = {}
    for module_name, module in model.named_children():
        total_norm = 0.0
        n_params = 0
        for param in module.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
                n_params += 1
        module_norms[module_name] = np.sqrt(total_norm) if n_params > 0 else 0.0
    return module_norms


def alpha_schedule_ganin(epoch: int, max_epochs: int = 100) -> float:
    """
    Alpha schedule from Ganin et al. (2016).

    Gradually increases gradient reversal strength from 0 to 1.
    Uses sigmoid schedule: alpha = 2 / (1 + exp(-10 * p)) - 1
    where p = epoch / max_epochs.

    Args:
        epoch: Current epoch
        max_epochs: Total epochs

    Returns:
        Alpha value in [0, 1]
    """
    p = epoch / max_epochs
    return float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)


def alpha_schedule_linear(epoch: int, max_epochs: int = 100) -> float:
    """Simple linear alpha schedule. Alternative to Ganin schedule."""
    return min(1.0, epoch / max_epochs)


def alpha_schedule_constant(value: float = 1.0):
    """Constant alpha (for ablation studies)."""
    def schedule(epoch: int, max_epochs: int = 100) -> float:
        return value
    return schedule


@torch.no_grad()
def extract_features(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """
    Extract features, labels, and domains from all samples in a dataloader.

    Used for t-SNE visualization and distribution analysis.

    Args:
        model: DANN model (or any model with .get_features() method)
        dataloader: DataLoader to extract from
        device: Device for inference

    Returns:
        Dictionary with 'features', 'labels', 'domains' arrays
    """
    model.eval()
    model.to(device)

    all_features = []
    all_labels = []
    all_domains = []

    for data, labels, domains in dataloader:
        data = data.to(device)
        features = model.get_features(data)
        all_features.append(features.cpu().numpy())
        all_labels.append(labels.numpy())
        all_domains.append(domains.numpy())

    return {
        "features": np.concatenate(all_features, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
        "domains": np.concatenate(all_domains, axis=0),
    }
