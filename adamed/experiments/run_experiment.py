"""
CLI entry point for running AdaMed experiments.

Usage:
    python -m adamed.experiments.run_experiment --name baseline_dann
    python -m adamed.experiments.run_experiment --name high_alpha --epochs 100
"""

import argparse
import json
import os
import sys
import torch
import numpy as np

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from adamed.data.synthetic_generator import ClinicalTimeSeriesGenerator
from adamed.data.preprocessing import simple_dataloader, create_dataloaders
from adamed.models.dann import create_dann_for_adamed
from adamed.training.trainer import DANNTrainer
from adamed.evaluation.visualization import (
    plot_training_curves,
    plot_feature_space,
    plot_gradient_analysis,
    plot_accuracy_curve,
)
from adamed.experiments.configs import get_experiment_config, list_experiments


def run_experiment(config_name: str, override_epochs: int = None) -> None:
    """
    Run a complete experiment from config.

    Steps:
    1. Load config
    2. Generate synthetic data
    3. Create model
    4. Train with DANN
    5. Evaluate and save results
    6. Generate plots
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: {config_name}")
    print(f"{'='*60}\n")

    # Load configuration
    config = get_experiment_config(config_name)
    print(f"Description: {config['description']}\n")

    # Data generation
    print("Generating synthetic data...")
    gen = ClinicalTimeSeriesGenerator(**config["data"])
    data = gen.generate_experimental_split()
    print(f"  Source: {len(data['source_idx'])} patients")
    print(f"  Target: {len(data['target_idx'])} patients")
    print(f"  Shape: {data['X'].shape}")

    # Create DataLoader
    loader = simple_dataloader(
        data,
        batch_size=config["training"]["batch_size"],
        normalize=True,
        flatten=True,
    )

    # Create model
    print("\nCreating DANN model...")
    model = create_dann_for_adamed(
        time_steps=config["data"]["time_steps"],
        n_features=config["data"]["n_features"],
        **config["model"],
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Training
    n_epochs = override_epochs or config["training"]["n_epochs"]
    save_dir = config["logging"]["save_dir"]

    print(f"\nTraining for {n_epochs} epochs...")
    trainer = DANNTrainer(
        model,
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
        lambda_domain=config["training"]["lambda_domain"],
    )
    history = trainer.train(loader, n_epochs=n_epochs, save_dir=save_dir)

    # Evaluation
    print("\nEvaluating...")
    eval_results = trainer.evaluate(loader)
    print(f"  Label accuracy: {eval_results['label_acc']:.4f}")
    print(f"  Domain accuracy: {eval_results['domain_acc']:.4f}")

    # Save evaluation metrics
    metrics = {
        "label_accuracy": eval_results["label_acc"],
        "domain_accuracy": eval_results["domain_acc"],
        "n_epochs": n_epochs,
        "config_name": config_name,
    }
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Generate plots
    if config["logging"].get("save_plots", True):
        print("\nGenerating plots...")
        plots_dir = os.path.join(save_dir, "plots")

        plot_training_curves(history, save_path=os.path.join(plots_dir, "loss_curves.png"))
        plot_accuracy_curve(history, save_path=os.path.join(plots_dir, "accuracy_curve.png"))
        plot_gradient_analysis(history, save_path=os.path.join(plots_dir, "gradient_analysis.png"))

        # Feature space visualization
        plot_feature_space(
            eval_results["features"],
            eval_results["domains"],
            save_path=os.path.join(plots_dir, "feature_tsne.png"),
            title=f"Feature Space — {config_name}",
        )

        print(f"  Plots saved to {plots_dir}/")

    print(f"\n{'='*60}")
    print(f"Experiment complete. Results in: {save_dir}/")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Run AdaMed experiment")
    parser.add_argument(
        "--name", type=str, default="baseline_dann",
        help="Experiment config name (see configs.py)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available experiments",
    )

    args = parser.parse_args()

    if args.list:
        list_experiments()
        return

    run_experiment(args.name, override_epochs=args.epochs)


if __name__ == "__main__":
    main()
