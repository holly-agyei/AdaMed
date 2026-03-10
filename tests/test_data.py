"""
Tests for the data generation and preprocessing modules.

These tests verify:
1. Synthetic data has correct shapes and distributions
2. Heuristics return valid parameters
3. Preprocessing handles NaN values correctly
4. DataLoaders produce valid batches
"""

import pytest
import numpy as np
import torch

from adamed.data.synthetic_generator import ClinicalTimeSeriesGenerator
from adamed.data.heuristics import get_west_african_parameters, get_glycemic_response
from adamed.data.preprocessing import ClinicalDataset, simple_dataloader


class TestClinicalTimeSeriesGenerator:
    """Tests for synthetic data generation."""

    def setup_method(self):
        """Create a small generator for fast testing."""
        self.gen = ClinicalTimeSeriesGenerator(
            n_source=50, n_target_proxy=20, time_steps=48, n_features=5, seed=42
        )

    def test_source_shape(self):
        """Source domain should have correct shape."""
        source = self.gen.generate_source_domain()
        assert source["data"].shape == (50, 48, 5)
        assert source["labels"].shape == (50,)
        assert source["domain"].shape == (50,)

    def test_target_shape(self):
        """Target proxy should have correct shape."""
        target = self.gen.generate_target_proxy()
        assert target["data"].shape == (20, 48, 5)
        assert target["labels"].shape == (20,)
        assert target["domain"].shape == (20,)

    def test_domain_labels(self):
        """Source should be domain 0, target should be domain 1."""
        source = self.gen.generate_source_domain()
        target = self.gen.generate_target_proxy()
        assert np.all(source["domain"] == 0)
        assert np.all(target["domain"] == 1)

    def test_source_labels_binary(self):
        """Source labels should be binary (0 or 1)."""
        source = self.gen.generate_source_domain()
        assert set(np.unique(source["labels"])).issubset({0, 1})

    def test_target_labels_unavailable(self):
        """Target labels should be -1 (unavailable)."""
        target = self.gen.generate_target_proxy()
        assert np.all(target["labels"] == -1)

    def test_experimental_split_combined(self):
        """Combined split should have correct total shape."""
        data = self.gen.generate_experimental_split()
        assert data["X"].shape == (70, 48, 5)
        assert data["y"].shape == (70,)
        assert data["domain"].shape == (70,)
        assert len(data["source_idx"]) == 50
        assert len(data["target_idx"]) == 20

    def test_no_nan_after_split(self):
        """NaN values should be imputed in experimental split."""
        data = self.gen.generate_experimental_split()
        assert not np.isnan(data["X"]).any()

    def test_reproducibility(self):
        """Same seed should produce same data when generated in same sequence."""
        gen1 = ClinicalTimeSeriesGenerator(n_source=10, n_target_proxy=5, seed=123)
        data1 = gen1.generate_experimental_split()

        gen2 = ClinicalTimeSeriesGenerator(n_source=10, n_target_proxy=5, seed=123)
        data2 = gen2.generate_experimental_split()

        np.testing.assert_array_equal(data1["X"], data2["X"])

    def test_feature_names(self):
        """Feature names should be provided."""
        data = self.gen.generate_experimental_split()
        assert len(data["feature_names"]) == 5
        assert "glucose_mgdl" in data["feature_names"]

    def test_source_glucose_positive(self):
        """Glucose values should be positive (normalized but non-negative)."""
        source = self.gen.generate_source_domain()
        glucose = source["data"][:, :, 0]
        assert np.all(glucose >= 0)


class TestHeuristics:
    """Tests for clinical heuristics module."""

    def test_parameters_structure(self):
        """Heuristics should return expected keys."""
        params = get_west_african_parameters()
        assert "feature_shifts" in params
        assert "dietary_factors" in params
        assert "access_constraints" in params
        assert "missing_prob" in params

    def test_feature_shifts_complete(self):
        """All 5 features should have shift parameters."""
        params = get_west_african_parameters()
        expected_features = [
            "glucose_mgdl", "bp_systolic", "bp_diastolic",
            "heart_rate", "physical_activity",
        ]
        for f in expected_features:
            assert f in params["feature_shifts"]
            assert "mean_shift" in params["feature_shifts"][f]
            assert "scale_factor" in params["feature_shifts"][f]

    def test_glycemic_response_shape(self):
        """Glycemic response should return correct number of samples."""
        response = get_glycemic_response("kenkey", hours=4, samples=48)
        assert response.shape == (48,)

    def test_glycemic_response_normalized(self):
        """Glycemic response should be in [0, 1]."""
        for food in ["kenkey", "fufu", "banku"]:
            response = get_glycemic_response(food)
            assert response.max() <= 1.0 + 1e-6
            assert response.min() >= 0.0 - 1e-6

    def test_glycemic_response_starts_zero(self):
        """Glycemic response should start near zero (no immediate effect)."""
        response = get_glycemic_response("kenkey")
        assert response[0] < 0.1


class TestPreprocessing:
    """Tests for data preprocessing and DataLoader creation."""

    def setup_method(self):
        """Generate small dataset for testing."""
        gen = ClinicalTimeSeriesGenerator(
            n_source=30, n_target_proxy=10, seed=42
        )
        self.data = gen.generate_experimental_split()

    def test_dataset_length(self):
        """ClinicalDataset should have correct length."""
        dataset = ClinicalDataset(
            self.data["X"].copy(), self.data["y"].copy(),
            self.data["domain"].copy(), normalize=True, flatten=True
        )
        assert len(dataset) == 40

    def test_dataset_output_types(self):
        """Dataset items should be tensors."""
        dataset = ClinicalDataset(
            self.data["X"].copy(), self.data["y"].copy(),
            self.data["domain"].copy()
        )
        x, y, d = dataset[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert isinstance(d, torch.Tensor)

    def test_dataloader_batches(self):
        """DataLoader should produce valid batches."""
        loader = simple_dataloader(self.data, batch_size=16)
        batch = next(iter(loader))
        x, y, d = batch
        assert x.shape[0] == 16
        assert y.shape[0] == 16
        assert d.shape[0] == 16

    def test_flattened_shape(self):
        """Flattened data should have correct dimension."""
        dataset = ClinicalDataset(
            self.data["X"].copy(), self.data["y"].copy(),
            self.data["domain"].copy(), flatten=True
        )
        x, _, _ = dataset[0]
        assert x.shape == (48 * 5,)  # time_steps * n_features


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
