"""
Preprocessing utilities for clinical time-series data.

Handles normalization, missing value imputation, and dataset construction
for PyTorch training pipelines.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Tuple, Optional, Dict
from sklearn.preprocessing import StandardScaler


class ClinicalDataset(Dataset):
    """
    PyTorch Dataset wrapper for clinical time-series data.

    Handles the domain-adversarial setup where each sample has:
    - x: multivariate time-series (time_steps, n_features)
    - y: task label (0 or 1, or -1 for unlabeled target)
    - d: domain label (0 = source, 1 = target)
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        domain: np.ndarray,
        normalize: bool = True,
        flatten: bool = True,
    ):
        """
        Args:
            X: Input data, shape (n_samples, time_steps, n_features)
            y: Labels, shape (n_samples,)
            domain: Domain labels, shape (n_samples,)
            normalize: Whether to apply per-feature standardization
            flatten: Whether to flatten time-series to 1D for MLP-based models
        """
        # Handle NaN values before normalization
        nan_mask = np.isnan(X)
        if nan_mask.any():
            # Impute with per-feature mean (computed on non-NaN values)
            for f in range(X.shape[2]):
                feature_mean = np.nanmean(X[:, :, f])
                X[:, :, f] = np.where(
                    np.isnan(X[:, :, f]), feature_mean, X[:, :, f]
                )

        if normalize:
            # Standardize each feature across all time steps and patients
            n_samples, time_steps, n_features = X.shape
            X_flat = X.reshape(-1, n_features)
            self.scaler = StandardScaler()
            X_flat = self.scaler.fit_transform(X_flat)
            X = X_flat.reshape(n_samples, time_steps, n_features)

        if flatten:
            # Flatten for MLP input: (n_samples, time_steps * n_features)
            X = X.reshape(X.shape[0], -1)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.domain = torch.tensor(domain, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx], self.domain[idx]


def create_dataloaders(
    data: Dict,
    batch_size: int = 64,
    val_split: float = 0.2,
    normalize: bool = True,
    flatten: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/validation DataLoaders from experimental split data.

    Splits source domain data into train/val while keeping all target
    domain data in both sets (since target has no labels anyway).

    Args:
        data: Output from ClinicalTimeSeriesGenerator.generate_experimental_split()
        batch_size: Batch size for DataLoaders
        val_split: Fraction of source data for validation
        normalize: Whether to normalize features
        flatten: Whether to flatten time-series
        seed: Random seed for reproducibility

    Returns:
        (train_loader, val_loader)
    """
    np.random.seed(seed)

    X = data["X"].copy()
    y = data["y"].copy()
    domain = data["domain"].copy()

    source_idx = data["source_idx"]
    target_idx = data["target_idx"]

    # Split source indices into train/val
    n_source = len(source_idx)
    n_val = int(n_source * val_split)
    perm = np.random.permutation(n_source)
    val_source_idx = source_idx[perm[:n_val]]
    train_source_idx = source_idx[perm[n_val:]]

    # Training set: train source + all target
    train_idx = np.concatenate([train_source_idx, target_idx])
    val_idx = val_source_idx  # Validation: only source (has labels)

    train_dataset = ClinicalDataset(
        X[train_idx], y[train_idx], domain[train_idx],
        normalize=normalize, flatten=flatten
    )
    val_dataset = ClinicalDataset(
        X[val_idx], y[val_idx], domain[val_idx],
        normalize=normalize, flatten=flatten
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader


def simple_dataloader(
    data: Dict,
    batch_size: int = 64,
    normalize: bool = True,
    flatten: bool = True,
) -> DataLoader:
    """
    Create a single DataLoader from experimental split data (no train/val split).
    Useful for quick experiments and debugging.
    """
    dataset = ClinicalDataset(
        data["X"].copy(), data["y"].copy(), data["domain"].copy(),
        normalize=normalize, flatten=flatten
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
