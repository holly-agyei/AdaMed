"""
Tests for model architectures.

Verifies:
1. DANN produces correct output shapes
2. Gradient reversal layer reverses gradients
3. Model components work independently
4. Factory function creates valid models
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from adamed.models.dann import (
    DANN,
    FeatureExtractor,
    LabelPredictor,
    DomainClassifier,
    create_dann_for_adamed,
)
from adamed.models.gradient_reversal import GradientReversalFn, gradient_reversal
from adamed.models.utils import count_parameters, get_parameter_summary


class TestGradientReversal:
    """Tests for gradient reversal layer."""

    def test_forward_identity(self):
        """Forward pass should be identity."""
        x = torch.randn(10, 5, requires_grad=True)
        y = GradientReversalFn.apply(x, 1.0)
        assert torch.allclose(x, y)

    def test_backward_reversal(self):
        """Backward pass should reverse gradients."""
        x = torch.randn(10, 5, requires_grad=True)
        y = gradient_reversal(x, alpha=1.0)
        loss = y.sum()
        loss.backward()

        # Gradient should be -1 for all elements (reversed from +1)
        expected_grad = -torch.ones_like(x)
        assert torch.allclose(x.grad, expected_grad)

    def test_alpha_scaling(self):
        """Alpha should scale the reversed gradient."""
        x = torch.randn(10, 5, requires_grad=True)
        alpha = 0.5
        y = gradient_reversal(x, alpha=alpha)
        loss = y.sum()
        loss.backward()

        expected_grad = -alpha * torch.ones_like(x)
        assert torch.allclose(x.grad, expected_grad)

    def test_zero_alpha(self):
        """Alpha=0 should produce zero gradients (no reversal effect)."""
        x = torch.randn(10, 5, requires_grad=True)
        y = gradient_reversal(x, alpha=0.0)
        loss = y.sum()
        loss.backward()

        expected_grad = torch.zeros_like(x)
        assert torch.allclose(x.grad, expected_grad)


class TestFeatureExtractor:
    """Tests for the feature extractor module."""

    def test_output_shape(self):
        """Feature extractor should produce correct output dimensions."""
        fe = FeatureExtractor(input_dim=240, hidden_dims=[256, 128])
        x = torch.randn(32, 240)
        out = fe(x)
        assert out.shape == (32, 128)

    def test_3d_input_flatten(self):
        """Should handle 3D input by flattening."""
        fe = FeatureExtractor(input_dim=240, hidden_dims=[256, 128])
        x = torch.randn(32, 48, 5)  # 48*5 = 240
        out = fe(x)
        assert out.shape == (32, 128)

    def test_output_dim_attribute(self):
        """output_dim attribute should match actual output."""
        fe = FeatureExtractor(input_dim=100, hidden_dims=[64, 32])
        assert fe.output_dim == 32


class TestDANN:
    """Tests for the complete DANN model."""

    def setup_method(self):
        """Create a standard DANN for testing."""
        self.model = create_dann_for_adamed(time_steps=48, n_features=5)
        self.batch = torch.randn(16, 240)

    def test_forward_shapes(self):
        """Forward pass should return correct shapes."""
        label_out, domain_out, features = self.model(self.batch)
        assert label_out.shape == (16, 2)
        assert domain_out.shape == (16, 2)
        assert features.shape == (16, 128)

    def test_forward_with_alpha(self):
        """Forward should work with different alpha values."""
        for alpha in [0.0, 0.5, 1.0, 2.0]:
            label_out, domain_out, features = self.model(self.batch, alpha=alpha)
            assert label_out.shape == (16, 2)

    def test_get_features(self):
        """get_features should return only features."""
        features = self.model.get_features(self.batch)
        assert features.shape == (16, 128)

    def test_predict_labels(self):
        """predict_labels should return only label predictions."""
        preds = self.model.predict_labels(self.batch)
        assert preds.shape == (16, 2)

    def test_parameter_count(self):
        """Model should have reasonable number of parameters."""
        n_params = count_parameters(self.model)
        assert n_params > 10000  # should be ~100K
        assert n_params < 1000000  # not too large

    def test_parameter_summary(self):
        """Parameter summary should include all modules."""
        summary = get_parameter_summary(self.model)
        assert "feature_extractor" in summary
        assert "label_predictor" in summary
        assert "domain_classifier" in summary
        assert "total" in summary

    def test_gradient_flow(self):
        """Gradients should flow through all components."""
        label_out, domain_out, _ = self.model(self.batch)
        loss = label_out.sum() + domain_out.sum()
        loss.backward()

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_3d_input(self):
        """Model should handle 3D input (batch, time, features)."""
        x = torch.randn(16, 48, 5)
        label_out, domain_out, features = self.model(x)
        assert label_out.shape == (16, 2)

    def test_factory_function(self):
        """create_dann_for_adamed should produce valid model."""
        model = create_dann_for_adamed(
            time_steps=24, n_features=3,
            feature_dims=[64, 32], dropout=0.5
        )
        x = torch.randn(8, 24 * 3)
        label_out, domain_out, features = model(x)
        assert label_out.shape == (8, 2)
        assert features.shape == (8, 32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
