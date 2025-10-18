"""
Model architectures for AdaMed domain adaptation experiments.

Modules:
    dann: Domain-Adversarial Neural Network
    gradient_reversal: Gradient reversal layer for adversarial training
    utils: Model utilities (parameter counting, feature extraction, etc.)
"""

from .dann import DANN, create_dann_for_adamed
from .gradient_reversal import GradientReversalFn, gradient_reversal

__all__ = [
    "DANN",
    "create_dann_for_adamed",
    "GradientReversalFn",
    "gradient_reversal",
]
