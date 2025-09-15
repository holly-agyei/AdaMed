# AdaMed: Zero-Shot Domain Adaptation for Clinical Time-Series

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

AdaMed is an independent research project addressing a critical gap in clinical AI: adapting machine learning models to populations with **zero training data**. Built on insights from BIO-DQNA (accepted to IEEE BIBM 2025), which achieved 94% success managing T1D+hypertension on Spanish CGM data but cannot generalize to low-resource settings where no continuous glucose data exists.

## The Problem

Existing domain adaptation methods (CORAL, DANN, contrastive learning) assume at least a few target examples. For many global populations—including West African diabetic patients—**no public datasets exist**. Zero-shot adaptation for multivariate time-series under unknown distribution shift remains unsolved.

## Current Status (March 2026)

- **Architecture:** Domain-Adversarial Neural Network (DANN) trained to learn invariant features between source (Spanish CGM) and target (simulated Ghanaian) domains
- **Heuristics incorporated:** Glycemic indices of fermented West African staples (kenkey, fufu), clinic stock-out frequencies (WHO survey proxies), transport delays (distance to nearest facility)
- **Result:** Model collapses to random guessing — adversarial training without any target samples leads to unstable gradient flow
- **Key finding:** Gradient reversal pushes features away from source distribution, destroying task-relevant information

## Repository Structure

```
adamed/
├── adamed/          # Core Python package
│   ├── data/        # Synthetic data generation & heuristics
│   ├── models/      # DANN architecture & gradient reversal
│   ├── training/    # Training loop & loss functions
│   ├── evaluation/  # Metrics & visualization
│   └── experiments/ # Experiment configs & runner
├── notebooks/       # Jupyter notebooks with analysis
│   ├── 01_data_synthesis.ipynb
│   ├── 02_dann_prototype.ipynb
│   ├── 03_failure_analysis.ipynb
│   └── 04_visualization_dashboard.ipynb
├── logs/            # Experiment logs and results
├── tests/           # Unit tests
├── docs/            # Technical documentation
└── results/         # Generated figures and summaries
```

## Quick Start

```bash
git clone https://github.com/holly-agyei/AdaMed.git
cd adamed
pip install -e .
jupyter notebook
```

### Run an experiment

```bash
# List available experiments
python -m adamed.experiments.run_experiment --list

# Run baseline DANN
python -m adamed.experiments.run_experiment --name baseline_dann

# Run with custom epochs
python -m adamed.experiments.run_experiment --name baseline_dann --epochs 100
```

### Run tests

```bash
pytest tests/ -v
```

## Key Results

![Loss Curves](logs/experiment_001/plots/loss_curves.png)
*Figure 1: Label and domain losses during training — domain loss minimization destroys label performance*

![Feature Space](logs/experiment_001/plots/feature_tsne.png)
*Figure 2: t-SNE visualization of learned features — no domain alignment achieved*

### Ablation Study

| Reversal Strength (α) | Label Accuracy | Domain Accuracy |
|----------------------|----------------|-----------------|
| 0.0 (no reversal) | ~62% | ~90% |
| 0.5 (moderate) | ~55% | ~82% |
| 1.0 (full) | ~50% | ~78% |
| 2.0 (aggressive) | ~48% | ~72% |

Increasing gradient reversal strength monotonically degrades task performance toward random chance.

## Technical Gap

Zero-shot adaptation for multivariate time-series requires fundamentally new approaches. Current next steps:

1. **Prior-informed loss functions:** Encode physiological constraints (glycemic response curves, vital sign correlations) directly into training objective
2. **Meta-learning:** Train on simulated distribution shifts for rapid adaptation (MAML/Reptile)
3. **Conditional generation:** Use heuristic-conditioned VAEs to synthesize plausible target samples
4. **Causal feature selection:** Identify features that are causally (not merely correlatively) related to outcomes

See [docs/next_steps.md](docs/next_steps.md) for detailed research plan.

## Documentation

- [Technical Note](docs/technical_note.md) — Problem formulation, methods, and analysis
- [Literature Review](docs/literature_review.md) — Survey of domain adaptation approaches
- [Next Steps](docs/next_steps.md) — Research directions and timeline

## Citation

If you use this code, please cite:

```bibtex
@software{adamed2026,
  author = {Holly Agyei},
  title = {AdaMed: Zero-Shot Domain Adaptation for Clinical Time-Series},
  year = {2026},
  url = {https://github.com/holly-agyei/AdaMed}
}
```

## License

MIT — see [LICENSE](LICENSE) for details.
