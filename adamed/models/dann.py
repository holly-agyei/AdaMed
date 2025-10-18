"""
Domain-Adversarial Neural Network (DANN) for AdaMed experiments.

Architecture overview:
    Input (time_steps * n_features) → FeatureExtractor → shared features
        → LabelPredictor → task labels (source only)
        → GradientReversal → DomainClassifier → domain labels (source + target)

The adversarial objective:
    min_{F,C} max_{D} L_task(C(F(x)), y) - lambda * L_domain(D(GRL(F(x))), d)

Where:
    F = feature extractor, C = label classifier, D = domain classifier
    GRL = gradient reversal layer

In the zero-shot setting (our case), the domain classifier only sees
source data, which causes the adversarial signal to be uninformative.
This is the fundamental limitation we document.

Reference: Ganin et al., "Domain-Adversarial Training of Neural Networks" (JMLR 2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List

from .gradient_reversal import GradientReversalFn


class FeatureExtractor(nn.Module):
    """
    Shared feature extractor for both task and domain heads.

    Architecture: stacked linear layers with BatchNorm, ReLU, and Dropout.

    Design decision: We use a flat MLP rather than 1D-CNN or LSTM because:
    1. The time-series are short (48 steps) — temporal convolutions add
       complexity without clear benefit at this scale
    2. MLPs are easier to analyze for gradient flow diagnostics
    3. If the MLP fails at domain adaptation, the issue is fundamental
       (not architectural), which is what we want to demonstrate

    Future work could compare against temporal architectures.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.3,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        self.network = nn.Sequential(*layers)
        self.output_dim = prev_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten time-series if 3D input: (batch, time, features) → (batch, time*features)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)


class LabelPredictor(nn.Module):
    """
    Task-specific head: predicts clinical outcome from shared features.

    Binary classification: good vs. poor glycemic control.
    Only trained on source domain samples (where labels exist).
    """

    def __init__(self, input_dim: int, num_classes: int = 2, hidden_dim: int = 64):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DomainClassifier(nn.Module):
    """
    Domain discrimination head with gradient reversal.

    Predicts whether a sample comes from source or target domain.
    Gradient reversal ensures the feature extractor learns to FOOL
    this classifier — producing domain-invariant features.

    In zero-shot: this classifier trivially achieves 100% accuracy
    (all training samples are source), so reversed gradients provide
    no useful signal for domain alignment.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2),  # Binary: source vs target
        )

    def forward(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        # Apply gradient reversal before domain classification
        x = GradientReversalFn.apply(x, alpha)
        return self.network(x)


class DANN(nn.Module):
    """
    Complete Domain-Adversarial Neural Network.

    Combines feature extractor, label predictor, and domain classifier
    into a single end-to-end trainable model.

    Args:
        input_dim: Flattened input dimension (time_steps * n_features)
        num_classes: Number of task classes (default 2: good/poor control)
        feature_dims: Hidden dimensions for feature extractor
        domain_hidden: Hidden dimension for domain classifier
        label_hidden: Hidden dimension for label predictor
        dropout: Dropout rate for feature extractor
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        feature_dims: List[int] = [256, 128],
        domain_hidden: int = 64,
        label_hidden: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.feature_extractor = FeatureExtractor(
            input_dim, hidden_dims=feature_dims, dropout=dropout
        )

        self.label_predictor = LabelPredictor(
            self.feature_extractor.output_dim,
            num_classes=num_classes,
            hidden_dim=label_hidden,
        )

        self.domain_classifier = DomainClassifier(
            self.feature_extractor.output_dim,
            hidden_dim=domain_hidden,
        )

        # Xavier initialization for stable training
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Xavier uniform initialization — standard for adversarial architectures."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(
        self, x: torch.Tensor, alpha: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through all three components.

        Args:
            x: Input tensor (batch, time_steps, features) or (batch, time_steps*features)
            alpha: Gradient reversal strength (increases during training)

        Returns:
            label_output: Task predictions, shape (batch, num_classes)
            domain_output: Domain predictions, shape (batch, 2)
            features: Shared features, shape (batch, feature_dims[-1])
        """
        features = self.feature_extractor(x)
        label_output = self.label_predictor(features)
        domain_output = self.domain_classifier(features, alpha)

        return label_output, domain_output, features

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without predictions (for visualization/analysis)."""
        return self.feature_extractor(x)

    def predict_labels(self, x: torch.Tensor) -> torch.Tensor:
        """Predict task labels only (for inference)."""
        features = self.feature_extractor(x)
        return self.label_predictor(features)


def create_dann_for_adamed(
    time_steps: int = 48,
    n_features: int = 5,
    **kwargs,
) -> DANN:
    """
    Factory function to create DANN with correct input dimensions for AdaMed.

    Default configuration:
    - Input: 48 time steps × 5 features = 240 dimensions
    - Feature extractor: 240 → 256 → 128
    - Label predictor: 128 → 64 → 2
    - Domain classifier: 128 → 64 → 2

    Total parameters: ~100K (intentionally small for fast iteration)
    """
    input_dim = time_steps * n_features
    return DANN(input_dim, **kwargs)
