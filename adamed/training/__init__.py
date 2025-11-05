"""
Training infrastructure for AdaMed domain adaptation experiments.

Modules:
    trainer: DANN training loop with logging and checkpointing
    losses: Custom loss functions for domain adaptation
"""

from .trainer import DANNTrainer
from .losses import domain_adversarial_loss, compute_alpha

__all__ = ["DANNTrainer", "domain_adversarial_loss", "compute_alpha"]
