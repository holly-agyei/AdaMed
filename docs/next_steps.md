# AdaMed: Next Steps

Based on experimental results (March 2026), the following directions are being explored.

## 1. Loss Function Reformulation

Current adversarial approach fails because domain classifier has no target samples. Proposed alternative: encode physiological constraints directly into loss function.

```python
# Pseudo-code for proposed loss
L_total = L_task(x_source) + λ * L_prior(x_source, heuristics)
```

Where `L_prior` penalizes features inconsistent with known physiological constraints:
- Glucose response should follow pharmacokinetic decay curves
- Blood pressure should correlate with glucose within known bounds
- Activity patterns should follow circadian rhythms

### Status
Initial experiments with KL-divergence prior (see `adamed/training/losses.py::prior_informed_loss`) showed the prior distributions are too vague. Need to formalize feature-specific bounds rather than simple Gaussian priors.

### Specific proposals
1. **Temporal consistency loss:** Penalize feature sequences that violate physiological rates of change (e.g., glucose can't jump 200 mg/dL in 5 minutes)
2. **Cross-feature correlation loss:** Enforce known correlations (glucose ↔ BP) that should hold across domains
3. **Glycemic response prior:** Features extracted from post-meal windows should align with known GI-based response curves

## 2. Meta-Learning for Rapid Adaptation

Train on **simulated** distribution shifts so model learns to adapt with minimal samples.

### Approach
Use Model-Agnostic Meta-Learning (MAML, Finn et al. 2017):
1. Create multiple "synthetic domains" by varying heuristic parameters
2. Meta-train: for each synthetic domain, simulate few-shot adaptation
3. Meta-test: evaluate on held-out synthetic domains

### Key question
Can meta-learning help when the test distribution shift is qualitatively different from training shifts? Our shifts involve changes in **modality** (CGM → clinic visits), not just parameter variation.

## 3. Synthetic Target Data via Conditional Generation

Use the heuristics to generate more realistic synthetic target data.

### Approach
1. Train a conditional VAE on source data, conditioned on heuristic parameters
2. Generate target samples by conditioning on Ghanaian heuristic values
3. Use generated samples in standard DANN training

### Risk
Generated samples may not capture the true target distribution — we'd be training on our own assumptions.

## 4. Feature Selection Before Adaptation

Instead of adapting all features, identify a subset that should be domain-invariant by construction.

### Candidates for invariant features
- **Heart rate variability:** Physiologically similar across populations
- **Circadian rhythm phase:** Universal biological clock
- **Relative glucose changes:** Even if baseline differs, rate-of-change may be similar

### Approach
1. Compute feature importance on source
2. Analyze which features have smallest expected shift (based on heuristics)
3. Build model using only (approximately) invariant features

## 5. Collaboration Opportunities

Seeking labs/groups working on:
- Zero-shot domain adaptation (any modality)
- Clinical time-series analysis in underrepresented populations
- AI for global health equity
- West African diabetes epidemiology

## 6. Data Collection Plan

Long-term goal: collect pilot data from a Ghanaian diabetes clinic.

### Requirements
- IRB approval through collaborating institution
- Minimum 50 patients with 3+ clinic visits
- Variables: fasting glucose, BP, basic demographics
- Timeline: 6-12 months for pilot

This would enable few-shot (rather than zero-shot) adaptation, which is a much more tractable problem.

## Timeline

| Month | Milestone |
|-------|-----------|
| Mar 2026 | Current: DANN failure analysis complete |
| Apr 2026 | Implement prior-informed loss v2 |
| May 2026 | Meta-learning baseline experiments |
| Jun 2026 | Conditional VAE for target generation |
| Jul 2026 | Feature selection analysis |
| Aug 2026 | Paper draft (workshop submission) |

---

*Last updated: March 2026*
