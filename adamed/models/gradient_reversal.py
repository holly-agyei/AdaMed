"""
Gradient Reversal Layer for Domain-Adversarial Neural Networks.

Reference: Ganin et al., "Domain-Adversarial Training of Neural Networks" (JMLR 2016)

The gradient reversal layer (GRL) is the key mechanism in DANN. During the
forward pass, it acts as an identity function. During the backward pass,
it reverses the gradient direction and scales by alpha.

This forces the feature extractor to learn representations that are
INFORMATIVE for the task but UNINFORMATIVE about the domain — the
minimax game at the heart of adversarial domain adaptation.

Why it fails in zero-shot:
    Without any target samples, the domain classifier only sees source data.
    The reversed gradients push features away from source-specific patterns,
    but there's no target distribution to guide WHERE features should go.
    Result: feature collapse to random noise.
"""

import torch
from torch.autograd import Function


class GradientReversalFn(Function):
    """
    Autograd function implementing gradient reversal.

    Forward: f(x) = x  (identity)
    Backward: f'(x) = -alpha * grad  (reversed, scaled gradient)

    The alpha parameter controls reversal strength:
    - alpha = 0: no reversal (standard forward pass)
    - alpha = 1: full reversal
    - alpha > 1: amplified reversal (can cause instability)

    Typical schedule: alpha increases from 0 to 1 over training
    using the formula from Ganin et al.:
        alpha = 2 / (1 + exp(-10 * p)) - 1
    where p is the training progress ratio [0, 1].
    """

    @staticmethod
    def forward(ctx, x, alpha):
        # Store alpha for backward pass
        ctx.alpha = alpha
        # Identity in forward direction
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse gradient and scale by alpha
        # The .neg() is what makes this "adversarial"
        return grad_output.neg() * ctx.alpha, None


def gradient_reversal(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Functional interface for gradient reversal.

    Usage:
        features = feature_extractor(x)
        reversed_features = gradient_reversal(features, alpha=0.5)
        domain_pred = domain_classifier(reversed_features)

    Args:
        x: Input tensor
        alpha: Reversal strength (0 = no reversal, 1 = full reversal)

    Returns:
        Same tensor in forward, reversed gradients in backward
    """
    return GradientReversalFn.apply(x, alpha)
