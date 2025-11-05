"""
Training loop for DANN experiments.

Encapsulates the full training pipeline: forward pass, loss computation,
gradient updates, logging, and checkpointing.

Design: The trainer is stateful — it tracks training history and can
resume from checkpoints. This is important for long experiments where
we want to analyze training dynamics post-hoc.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional, Callable, List
from tqdm import tqdm
import json
import csv
import os
from datetime import datetime

from .losses import domain_adversarial_loss, compute_alpha


class DANNTrainer:
    """
    Trainer for Domain-Adversarial Neural Networks.

    Handles:
    - Training loop with configurable alpha schedule
    - Separate tracking of label and domain losses
    - Gradient norm monitoring (critical for diagnosing DANN instability)
    - Checkpoint saving and history export

    Usage:
        model = create_dann_for_adamed()
        trainer = DANNTrainer(model, lr=1e-3)
        history = trainer.train(train_loader, n_epochs=50, save_dir='logs/exp_001')
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        lambda_domain: float = 1.0,
        alpha_schedule: Optional[Callable] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.lambda_domain = lambda_domain

        # Adam optimizer — standard choice for DANN
        # Weight decay provides mild L2 regularization
        self.optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Loss functions
        self.label_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.CrossEntropyLoss()

        # Alpha schedule for gradient reversal strength
        # Default: Ganin et al. sigmoid schedule
        if alpha_schedule is None:
            self.alpha_schedule = lambda epoch: compute_alpha(epoch, max_epochs=100)
        else:
            self.alpha_schedule = alpha_schedule

        # Training history — lists of per-epoch metrics
        self.history: Dict[str, List[float]] = {
            "label_loss": [],
            "domain_loss": [],
            "total_loss": [],
            "label_acc": [],
            "domain_acc": [],
            "alpha": [],
            "grad_norm_features": [],
            "grad_norm_labels": [],
            "grad_norm_domain": [],
        }

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        max_epochs: int = 100,
        alpha: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training data (yields (data, labels, domains))
            epoch: Current epoch number
            max_epochs: Total epochs (for alpha schedule)
            alpha: Override alpha value (None = use schedule)

        Returns:
            Dictionary of epoch-averaged metrics
        """
        self.model.train()

        total_label_loss = 0.0
        total_domain_loss = 0.0
        total_loss = 0.0
        correct_label = 0
        correct_domain = 0
        total_samples = 0
        grad_norms = {"features": [], "labels": [], "domain": []}

        if alpha is None:
            alpha = self.alpha_schedule(epoch)

        for batch_idx, (data, labels, domains) in enumerate(dataloader):
            data = data.to(self.device)
            labels = labels.to(self.device)
            domains = domains.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass through all three components
            label_output, domain_output, features = self.model(data, alpha)

            # Compute losses
            # Label loss: only on source samples (domain == 0)
            source_mask = domains == 0
            if source_mask.any():
                label_loss = self.label_criterion(
                    label_output[source_mask], labels[source_mask]
                )
            else:
                label_loss = torch.tensor(0.0, device=self.device)

            domain_loss = self.domain_criterion(domain_output, domains)

            # Combined loss
            loss = label_loss + self.lambda_domain * domain_loss

            # Backward pass
            loss.backward()

            # Track gradient norms per module (for diagnostics)
            for name, module in self.model.named_children():
                norm = 0.0
                for p in module.parameters():
                    if p.grad is not None:
                        norm += p.grad.data.norm(2).item() ** 2
                norm = np.sqrt(norm)
                if "feature" in name:
                    grad_norms["features"].append(norm)
                elif "label" in name:
                    grad_norms["labels"].append(norm)
                elif "domain" in name:
                    grad_norms["domain"].append(norm)

            self.optimizer.step()

            # Accumulate metrics
            total_label_loss += label_loss.item()
            total_domain_loss += domain_loss.item()
            total_loss += loss.item()

            _, label_pred = torch.max(label_output, 1)
            _, domain_pred = torch.max(domain_output, 1)

            if source_mask.any():
                correct_label += (
                    (label_pred[source_mask] == labels[source_mask]).sum().item()
                )
            correct_domain += (domain_pred == domains).sum().item()
            total_samples += labels.size(0)

        # Average across batches
        n_batches = max(len(dataloader), 1)
        n_source = max(correct_label, 1)  # avoid division by zero

        metrics = {
            "label_loss": total_label_loss / n_batches,
            "domain_loss": total_domain_loss / n_batches,
            "total_loss": total_loss / n_batches,
            "label_acc": correct_label / max(total_samples, 1),
            "domain_acc": correct_domain / max(total_samples, 1),
            "alpha": alpha,
            "grad_norm_features": float(np.mean(grad_norms["features"])) if grad_norms["features"] else 0.0,
            "grad_norm_labels": float(np.mean(grad_norms["labels"])) if grad_norms["labels"] else 0.0,
            "grad_norm_domain": float(np.mean(grad_norms["domain"])) if grad_norms["domain"] else 0.0,
        }

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        n_epochs: int = 50,
        save_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.

        Args:
            train_loader: Training DataLoader
            n_epochs: Number of epochs
            save_dir: Directory to save checkpoints and logs
            verbose: Whether to show progress bar

        Returns:
            Training history dictionary
        """
        if verbose:
            pbar = tqdm(range(n_epochs), desc="Training DANN")
        else:
            pbar = range(n_epochs)

        for epoch in pbar:
            metrics = self.train_epoch(train_loader, epoch, max_epochs=n_epochs)

            # Update history
            for k, v in metrics.items():
                self.history[k].append(v)

            if verbose:
                pbar.set_postfix({
                    "L_task": f"{metrics['label_loss']:.4f}",
                    "L_dom": f"{metrics['domain_loss']:.4f}",
                    "acc": f"{metrics['label_acc']:.3f}",
                    "α": f"{metrics['alpha']:.3f}",
                })

        # Save artifacts
        if save_dir:
            self._save_artifacts(save_dir, n_epochs)

        return self.history

    def _save_artifacts(self, save_dir: str, n_epochs: int):
        """Save model checkpoint, history, and config to disk."""
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "plots"), exist_ok=True)

        # Save model weights
        torch.save(
            self.model.state_dict(),
            os.path.join(save_dir, "model_final.pt"),
        )

        # Save history as JSON
        history_serializable = {}
        for k, v in self.history.items():
            history_serializable[k] = [
                float(x) if isinstance(x, (np.floating, float)) else x
                for x in v
            ]

        with open(os.path.join(save_dir, "history.json"), "w") as f:
            json.dump(history_serializable, f, indent=2)

        # Save history as CSV (for easy plotting in other tools)
        with open(os.path.join(save_dir, "losses.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch"] + list(self.history.keys()))
            for i in range(len(self.history["label_loss"])):
                row = [i] + [self.history[k][i] for k in self.history.keys()]
                writer.writerow(row)

        # Save config
        config = {
            "n_epochs": n_epochs,
            "device": self.device,
            "lambda_domain": self.lambda_domain,
            "optimizer": str(self.optimizer),
            "timestamp": datetime.now().isoformat(),
        }
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def evaluate(self, dataloader: DataLoader) -> Dict:
        """
        Evaluate model on a given dataset.

        Returns metrics and extracted features (for visualization).
        """
        self.model.eval()

        correct_label = 0
        correct_domain = 0
        total_samples = 0
        all_features = []
        all_labels = []
        all_domains = []

        with torch.no_grad():
            for data, labels, domains in dataloader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                domains = domains.to(self.device)

                label_output, domain_output, features = self.model(data, alpha=0)

                _, label_pred = torch.max(label_output, 1)
                _, domain_pred = torch.max(domain_output, 1)

                source_mask = domains == 0
                if source_mask.any():
                    correct_label += (
                        (label_pred[source_mask] == labels[source_mask]).sum().item()
                    )
                correct_domain += (domain_pred == domains).sum().item()
                total_samples += labels.size(0)

                all_features.append(features.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_domains.append(domains.cpu().numpy())

        return {
            "label_acc": correct_label / max(total_samples, 1),
            "domain_acc": correct_domain / max(total_samples, 1),
            "features": np.concatenate(all_features, axis=0),
            "labels": np.concatenate(all_labels, axis=0),
            "domains": np.concatenate(all_domains, axis=0),
        }
