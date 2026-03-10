# AdaMed Results Summary

## Experiment Overview

This document summarizes results from the AdaMed zero-shot domain adaptation experiments conducted September 2025 – March 2026.

## Key Finding

**Domain-adversarial training (DANN) fails at zero-shot domain adaptation for clinical time-series.**

The gradient reversal mechanism requires both source AND target samples during training. Without target data, the domain classifier becomes degenerate, and reversed gradients destroy task-relevant features.

## Quantitative Results

### Experiment 001: Baseline DANN

| Metric | Value | Notes |
|--------|-------|-------|
| Initial label accuracy | ~62% | Before domain adversarial training kicks in |
| Final label accuracy | ~50% | Degrades to random chance |
| Domain classification accuracy | ~85-90% | Domains remain easily separable |
| Proxy A-distance | ~1.8 | Near maximum (2.0 = perfectly separable) |

### Ablation: Effect of Alpha (Gradient Reversal Strength)

| Alpha | Final Label Accuracy | Interpretation |
|-------|---------------------|----------------|
| 0.0 | ~62% | No reversal = source-only baseline |
| 0.5 | ~55% | Moderate reversal begins degrading performance |
| 1.0 | ~50% | Full reversal → random guessing |
| 2.0 | ~48% | Aggressive reversal → worse than random |

**Conclusion:** Monotonically increasing reversal strength monotonically degrades task performance. No sweet spot exists in the zero-shot setting.

## Figures

### Training Dynamics
![Loss Curves](../logs/experiment_001/plots/loss_curves.png)
*Label loss increases as domain loss decreases — the adversarial trade-off destroys task performance.*

### Feature Space
![Feature t-SNE](../logs/experiment_001/plots/feature_tsne.png)
*Source and target domains remain clearly separated after training — no alignment achieved.*

### Gradient Analysis
![Gradient Norms](../logs/experiment_001/plots/gradient_analysis.png)
*Domain classifier gradients dominate label predictor gradients, overwhelming task signal.*

### Data Distributions
![Distributions](../results/figures/data_distributions.png)
*Clear distribution shift between synthetic source and target domains.*

## Implications

1. **Negative result is informative:** Rules out standard adversarial approaches for zero-shot clinical adaptation
2. **Failure mechanism identified:** Gradient reversal without target signal → feature destruction
3. **Next directions:**
   - Prior-informed loss functions (encode domain knowledge directly)
   - Meta-learning for rapid adaptation
   - Conditional generation of synthetic target samples

## Reproducibility

All experiments can be reproduced by running:
```bash
python -m adamed.experiments.run_experiment --name baseline_dann
```

See `notebooks/` for interactive analysis and `adamed/experiments/configs.py` for experiment configurations.

---

*Last updated: March 2026*
