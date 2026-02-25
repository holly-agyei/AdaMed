"""
Experiment configurations for AdaMed.

Each configuration defines a complete experiment: data generation parameters,
model architecture, training hyperparameters, and logging settings.

Using a registry pattern so experiments are reproducible and documented.
"""

from typing import Dict, Any


# Registry of all experiment configurations
EXPERIMENT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "baseline_dann": {
        "description": (
            "Baseline DANN with standard Ganin schedule. "
            "Expected outcome: domain alignment fails, label accuracy degrades."
        ),
        "data": {
            "n_source": 1000,
            "n_target_proxy": 200,
            "time_steps": 48,
            "n_features": 5,
            "seed": 42,
        },
        "model": {
            "feature_dims": [256, 128],
            "domain_hidden": 64,
            "label_hidden": 64,
            "dropout": 0.3,
        },
        "training": {
            "n_epochs": 50,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "lambda_domain": 1.0,
            "batch_size": 64,
        },
        "logging": {
            "save_dir": "logs/experiment_001",
            "save_plots": True,
        },
    },
    "high_alpha": {
        "description": (
            "DANN with constant high alpha (ablation study). "
            "Tests whether aggressive gradient reversal helps."
        ),
        "data": {
            "n_source": 1000,
            "n_target_proxy": 200,
            "time_steps": 48,
            "n_features": 5,
            "seed": 42,
        },
        "model": {
            "feature_dims": [256, 128],
            "domain_hidden": 64,
            "label_hidden": 64,
            "dropout": 0.3,
        },
        "training": {
            "n_epochs": 50,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "lambda_domain": 2.0,  # doubled
            "batch_size": 64,
            "alpha_override": 1.0,  # constant max alpha
        },
        "logging": {
            "save_dir": "logs/experiment_002_high_alpha",
            "save_plots": True,
        },
    },
    "no_reversal": {
        "description": (
            "No gradient reversal (alpha=0). Baseline showing source-only performance. "
            "This is the upper bound on source accuracy and lower bound on adaptation."
        ),
        "data": {
            "n_source": 1000,
            "n_target_proxy": 200,
            "time_steps": 48,
            "n_features": 5,
            "seed": 42,
        },
        "model": {
            "feature_dims": [256, 128],
            "domain_hidden": 64,
            "label_hidden": 64,
            "dropout": 0.3,
        },
        "training": {
            "n_epochs": 50,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "lambda_domain": 0.0,  # no domain loss
            "batch_size": 64,
        },
        "logging": {
            "save_dir": "logs/experiment_003_no_reversal",
            "save_plots": True,
        },
    },
    "small_model": {
        "description": (
            "Smaller feature extractor (ablation on capacity). "
            "Tests whether model capacity affects zero-shot failure mode."
        ),
        "data": {
            "n_source": 1000,
            "n_target_proxy": 200,
            "time_steps": 48,
            "n_features": 5,
            "seed": 42,
        },
        "model": {
            "feature_dims": [64, 32],
            "domain_hidden": 16,
            "label_hidden": 16,
            "dropout": 0.5,
        },
        "training": {
            "n_epochs": 50,
            "lr": 5e-4,
            "weight_decay": 1e-3,
            "lambda_domain": 1.0,
            "batch_size": 64,
        },
        "logging": {
            "save_dir": "logs/experiment_004_small",
            "save_plots": True,
        },
    },
}


def get_experiment_config(name: str) -> Dict[str, Any]:
    """
    Retrieve experiment configuration by name.

    Args:
        name: Experiment name (key in EXPERIMENT_REGISTRY)

    Returns:
        Configuration dictionary

    Raises:
        ValueError: If experiment name not found
    """
    if name not in EXPERIMENT_REGISTRY:
        available = list(EXPERIMENT_REGISTRY.keys())
        raise ValueError(
            f"Unknown experiment '{name}'. Available: {available}"
        )
    return EXPERIMENT_REGISTRY[name]


def list_experiments() -> None:
    """Print all registered experiments with descriptions."""
    for name, config in EXPERIMENT_REGISTRY.items():
        print(f"\n  {name}")
        print(f"    {config['description']}")
