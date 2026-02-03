# Technical Note: Zero-Shot Domain Adaptation for Clinical Time-Series

## Problem Formulation

We consider the problem of adapting a clinical risk prediction model trained on **source domain** S (Spanish CGM patients) to a **target domain** T (Ghanaian patients) where:

1. **No target training data exists** (zero-shot constraint)
2. The feature spaces partially overlap but differ in modality (continuous CGM vs. sparse clinic visits)
3. Distribution shift arises from dietary, genetic, socioeconomic, and healthcare-access differences

Formally, given source data `{(x_s, y_s)} ~ P_S(X, Y)`, we seek a model `f: X → Y` that minimizes risk on `P_T(X, Y)` without access to any samples from `P_T`.

## Approach: Domain-Adversarial Neural Network (DANN)

### Architecture

We implemented a standard DANN (Ganin et al., 2016) with three components:

| Component | Architecture | Parameters |
|-----------|-------------|------------|
| Feature Extractor | 240 → 256 → 128 (MLP + BN + ReLU + Dropout) | ~75K |
| Label Predictor | 128 → 64 → 2 (MLP) | ~8.5K |
| Domain Classifier | 128 → GRL → 64 → 2 (MLP with gradient reversal) | ~8.5K |

**Total: ~92K parameters**

### Training Objective

```
min_{F,C} max_{D} L_task(C(F(x_s)), y_s) - λ · L_domain(D(GRL(F(x))), d)
```

Where GRL is the gradient reversal layer with strength α increasing from 0 to 1 following Ganin's sigmoid schedule.

### Synthetic Data

Due to the absence of real Ghanaian CGM data, we generated synthetic data encoding:

- **Source:** Circadian glucose patterns, regular meal responses, correlated vital signs
- **Target proxy:** Higher glucose variability (high-GI staples), missing data segments (clinic stock-outs), shifted distributions (WHO/IDF parameters)

## Results

### Key Finding: DANN Fails at Zero-Shot Adaptation

| Metric | No Reversal (α=0) | DANN (α→1) |
|--------|-------------------|------------|
| Source Label Accuracy | ~62% | ~50% (random) |
| Domain Classification | ~90% | ~85% |
| Feature Mean Distance | 2.1 | 1.8 |

### Failure Mechanism

1. **Domain classifier degeneracy:** With only source samples, the domain classifier trivially predicts all inputs as "source" with high confidence.

2. **Uninformative gradient reversal:** Reversed gradients from a degenerate classifier provide no useful signal for domain alignment. Instead, they push features away from source-specific patterns without convergence target.

3. **Task-relevant information destruction:** As α increases, the feature extractor is penalized for encoding ANY information that correlates with the source domain — but in zero-shot, this includes task-relevant features. Result: convergence to random guessing.

### Gradient Analysis

Gradient norm monitoring reveals:
- Domain classifier gradients grow with α
- Label predictor gradients diminish
- Feature extractor receives conflicting signals

The gradient ratio (domain/label) exceeds 5:1 by epoch 30, indicating the adversarial signal overwhelms the task signal.

## Implications

### For Zero-Shot Adaptation

Standard adversarial domain adaptation methods **require target domain samples**. Zero-shot adaptation for time-series requires fundamentally different approaches:

1. **Prior-informed loss functions:** Encode clinical knowledge (e.g., glycemic indices, physiological bounds) directly into the training objective rather than relying on adversarial signals.

2. **Meta-learning:** Train on simulated distribution shifts so the model learns to adapt rapidly. MAML or Reptile could enable few-shot adaptation if even small amounts of target data become available.

3. **Causal feature identification:** Learn features that are causally (not merely correlatively) related to outcomes, which should be more robust to distribution shift.

### For Clinical AI in Low-Resource Settings

This work highlights a fundamental challenge: ML models trained on high-resource populations cannot be blindly applied to low-resource populations. The absence of data IS the problem, and standard ML cannot solve it without incorporating domain expertise.

## Conclusion

We documented the failure of DANN for zero-shot clinical domain adaptation and identified the specific mechanisms of failure. This negative result is valuable because it:

1. Rules out a common approach, saving future researchers time
2. Identifies the exact failure mode (gradient reversal without target signal)
3. Points toward promising alternatives (prior-informed losses, meta-learning)

---

*Author: Holly Agyei*
*Date: February 2026*
*Status: Working draft*
