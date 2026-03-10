"""
Tests for the training pipeline.

Verifies:
1. Trainer can complete a training loop
2. Loss computation is correct
3. History is recorded properly
4. Checkpointing works
"""

import pytest
import torch
import numpy as np
import os
import json
import tempfile

from adamed.data.synthetic_generator import ClinicalTimeSeriesGenerator
from adamed.data.preprocessing import simple_dataloader
from adamed.models.dann import create_dann_for_adamed
from adamed.training.trainer import DANNTrainer
from adamed.training.losses import (
    compute_alpha,
    domain_adversarial_loss,
    prior_informed_loss,
)


class TestAlphaSchedule:
    """Tests for alpha scheduling."""

    def test_alpha_starts_low(self):
        """Alpha should be near 0 at epoch 0."""
        assert compute_alpha(0, max_epochs=100) < 0.1

    def test_alpha_ends_high(self):
        """Alpha should approach 1 at final epoch."""
        assert compute_alpha(99, max_epochs=100) > 0.9

    def test_alpha_monotonic(self):
        """Alpha should be monotonically increasing."""
        alphas = [compute_alpha(e, 50) for e in range(50)]
        for i in range(1, len(alphas)):
            assert alphas[i] >= alphas[i - 1]

    def test_alpha_range(self):
        """Alpha should stay in [0, 1]."""
        for e in range(100):
            alpha = compute_alpha(e, 100)
            assert 0 <= alpha <= 1


class TestLossFunctions:
    """Tests for loss functions."""

    def test_domain_adversarial_loss(self):
        """DANN loss should return scalar and dict."""
        label_out = torch.randn(16, 2)
        domain_out = torch.randn(16, 2)
        labels = torch.randint(0, 2, (16,))
        domains = torch.randint(0, 2, (16,))

        loss, loss_dict = domain_adversarial_loss(
            label_out, domain_out, labels, domains
        )
        assert loss.dim() == 0  # scalar
        assert "label_loss" in loss_dict
        assert "domain_loss" in loss_dict
        assert "total_loss" in loss_dict

    def test_loss_with_source_mask(self):
        """Loss should handle source mask correctly."""
        label_out = torch.randn(16, 2)
        domain_out = torch.randn(16, 2)
        labels = torch.randint(0, 2, (16,))
        domains = torch.randint(0, 2, (16,))
        source_mask = domains == 0

        loss, _ = domain_adversarial_loss(
            label_out, domain_out, labels, domains, source_mask=source_mask
        )
        assert loss.dim() == 0

    def test_prior_informed_loss(self):
        """Prior-informed loss should work without priors."""
        label_out = torch.randn(16, 2)
        features = torch.randn(16, 128)
        labels = torch.randint(0, 2, (16,))

        loss, loss_dict = prior_informed_loss(label_out, features, labels)
        assert loss.dim() == 0
        assert "label_loss" in loss_dict
        assert "prior_loss" in loss_dict

    def test_prior_informed_loss_with_priors(self):
        """Prior-informed loss should work with explicit priors."""
        label_out = torch.randn(16, 2)
        features = torch.randn(16, 128)
        labels = torch.randint(0, 2, (16,))
        prior_means = torch.zeros(128)
        prior_stds = torch.ones(128)

        loss, _ = prior_informed_loss(
            label_out, features, labels,
            prior_means=prior_means, prior_stds=prior_stds
        )
        assert loss.dim() == 0


class TestDANNTrainer:
    """Tests for the training loop."""

    def setup_method(self):
        """Create small model and data for fast testing."""
        self.gen = ClinicalTimeSeriesGenerator(
            n_source=30, n_target_proxy=10, seed=42
        )
        self.data = self.gen.generate_experimental_split()
        self.loader = simple_dataloader(self.data, batch_size=16)
        self.model = create_dann_for_adamed(
            time_steps=48, n_features=5,
            feature_dims=[32, 16], dropout=0.1
        )

    def test_training_runs(self):
        """Training should complete without errors."""
        trainer = DANNTrainer(self.model, lr=1e-3)
        history = trainer.train(self.loader, n_epochs=3, verbose=False)
        assert len(history["label_loss"]) == 3
        assert len(history["domain_loss"]) == 3

    def test_history_recorded(self):
        """History should contain all expected keys."""
        trainer = DANNTrainer(self.model, lr=1e-3)
        history = trainer.train(self.loader, n_epochs=2, verbose=False)
        expected_keys = [
            "label_loss", "domain_loss", "total_loss",
            "label_acc", "domain_acc", "alpha",
        ]
        for key in expected_keys:
            assert key in history

    def test_evaluation(self):
        """Evaluation should return metrics and features."""
        trainer = DANNTrainer(self.model, lr=1e-3)
        trainer.train(self.loader, n_epochs=2, verbose=False)
        results = trainer.evaluate(self.loader)

        assert "label_acc" in results
        assert "domain_acc" in results
        assert "features" in results
        assert results["features"].ndim == 2

    def test_save_artifacts(self):
        """Training should save model and history to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = DANNTrainer(self.model, lr=1e-3)
            trainer.train(self.loader, n_epochs=2, save_dir=tmpdir, verbose=False)

            assert os.path.exists(os.path.join(tmpdir, "model_final.pt"))
            assert os.path.exists(os.path.join(tmpdir, "history.json"))
            assert os.path.exists(os.path.join(tmpdir, "config.json"))
            assert os.path.exists(os.path.join(tmpdir, "losses.csv"))

            # Verify JSON is valid
            with open(os.path.join(tmpdir, "history.json")) as f:
                loaded = json.load(f)
            assert len(loaded["label_loss"]) == 2

    def test_custom_alpha_schedule(self):
        """Custom alpha schedule should be used."""
        custom_alpha = lambda epoch: 0.42
        trainer = DANNTrainer(self.model, lr=1e-3, alpha_schedule=custom_alpha)
        history = trainer.train(self.loader, n_epochs=2, verbose=False)
        assert all(a == pytest.approx(0.42) for a in history["alpha"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
