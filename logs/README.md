# Experiment Logs

This directory contains logs from AdaMed experiments.

## Structure

Each experiment is saved in its own subdirectory:

```
logs/
├── experiment_001/          # Baseline DANN (Ganin schedule)
│   ├── config.json          # Hyperparameters and settings
│   ├── history.json         # Per-epoch training metrics
│   ├── losses.csv           # Training losses in CSV format
│   ├── metrics.json         # Final evaluation metrics
│   ├── model_final.pt       # Saved model weights
│   └── plots/               # Generated visualizations
│       ├── loss_curves.png
│       ├── accuracy_curve.png
│       ├── feature_tsne.png
│       └── gradient_analysis.png
├── experiment_002_high_alpha/   # High alpha ablation
├── experiment_003_no_reversal/  # No reversal baseline
└── experiment_004_small/        # Small model ablation
```

## Experiment Registry

| ID | Description | Status | Key Result |
|----|------------|--------|------------|
| 001 | Baseline DANN | Complete | Label acc degrades to ~50% |
| 002 | High alpha (constant α=1) | Planned | — |
| 003 | No reversal (α=0) | Planned | — |
| 004 | Small model | Planned | — |

## How to Run

```bash
python -m adamed.experiments.run_experiment --name baseline_dann
```

See `adamed/experiments/configs.py` for available experiment configurations.

## Notes

- All experiments use synthetic data (no real patient data in this repository)
- Model weights (`.pt` files) are excluded from git via `.gitignore`
- Plots are regenerated each time an experiment is run
