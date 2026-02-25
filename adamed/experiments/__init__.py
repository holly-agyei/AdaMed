"""
Experiment configuration and execution for AdaMed.

Modules:
    configs: Predefined experiment configurations
    run_experiment: CLI entry point for running experiments
"""

from .configs import get_experiment_config, EXPERIMENT_REGISTRY

__all__ = ["get_experiment_config", "EXPERIMENT_REGISTRY"]
