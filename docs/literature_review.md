# Literature Review: Domain Adaptation for Clinical Time-Series

## Overview

This document surveys existing approaches to domain adaptation, with focus on methods applicable to clinical time-series data and low-resource healthcare settings.

## 1. Domain Adaptation Foundations

### 1.1 Theory

Ben-David et al. (2010) established the theoretical framework for domain adaptation, showing that target error is bounded by:

```
ε_T(h) ≤ ε_S(h) + d_A(S,T) + λ*
```

Where `d_A` is the A-distance between domains and `λ*` is the ideal joint error. This bound motivates learning domain-invariant representations.

**Key insight for AdaMed:** The bound assumes access to samples from both domains. In zero-shot settings, we cannot estimate `d_A` from data.

### 1.2 DANN (Ganin et al., 2016)

Domain-Adversarial Neural Networks use gradient reversal to learn features that are discriminative for the task but invariant to the domain.

- **Strengths:** End-to-end training, no explicit distribution matching
- **Limitations:** Requires target domain samples during training
- **Our finding:** Fails completely in zero-shot setting (see experiments)

### 1.3 CORAL (Sun & Saenko, 2016)

Correlation Alignment matches second-order statistics between domains.

- **Strengths:** Simple, no adversarial training needed
- **Limitations:** Assumes Gaussian feature distributions, requires target samples
- **Relevance:** Could potentially work with heuristic-derived target statistics

### 1.4 Maximum Mean Discrepancy (Long et al., 2015)

MMD-based methods minimize distribution distance in reproducing kernel Hilbert space.

- **Strengths:** Non-parametric, theoretically grounded
- **Limitations:** Computational cost, kernel selection, needs target samples

## 2. Clinical Time-Series Specific

### 2.1 Transfer Learning for CGM Data

- Zhu et al. (2022): Transfer learning for glucose prediction across patients
- Limited to within-population transfer (same CGM device, same country)
- No work on cross-population adaptation with missing modalities

### 2.2 BIO-DQNA (IEEE BIBM 2025)

Our prior work achieved 94% success in T1D+hypertension management using Spanish CGM data. Key limitations:
- Trained and evaluated on Spanish population only
- Requires continuous glucose monitoring (not available in target setting)
- No adaptation mechanism for distribution shift

## 3. Zero-Shot Domain Adaptation

### 3.1 Existing Approaches

Very limited literature on zero-shot domain adaptation:

- **Attribute-based:** Use semantic attributes to bridge domains (Lampert et al., 2014)
  - Not applicable: clinical features don't have clear semantic attributes
- **Generative:** Use GANs to synthesize target samples (Xian et al., 2018)
  - Promising but requires knowing target distribution characteristics
- **Meta-learning:** Learn to adapt from simulated shifts (Finn et al., 2017)
  - Most promising direction for our setting

### 3.2 The Gap

No published work addresses zero-shot domain adaptation for:
- Multivariate clinical time-series
- Settings where the target modality itself differs (CGM vs. clinic visits)
- Low-resource populations with no training data

**This is the gap AdaMed addresses.**

## 4. Related Concepts

### 4.1 Few-Shot vs Zero-Shot

Most "low-resource" adaptation work assumes at least 5-50 target samples. True zero-shot (0 target samples) is fundamentally harder because:
- Cannot estimate target distribution
- Cannot validate adaptation quality
- Must rely on prior knowledge and heuristics

### 4.2 Domain Generalization

Different from domain adaptation: train on multiple source domains to generalize to unseen target.
- Requires multiple source domains (we have only one)
- Promising if combined with simulated domain shifts

## 5. Key References

1. Ben-David, S. et al. "A theory of learning from different domains." Machine Learning, 2010.
2. Ganin, Y. et al. "Domain-adversarial training of neural networks." JMLR, 2016.
3. Sun, B. & Saenko, K. "Deep CORAL: Correlation alignment for deep domain adaptation." ECCV, 2016.
4. Long, M. et al. "Learning transferable features with deep adaptation networks." ICML, 2015.
5. Finn, C. et al. "Model-agnostic meta-learning for fast adaptation of deep networks." ICML, 2017.
6. International Diabetes Federation. "IDF Diabetes Atlas, 10th Edition." 2021.
7. WHO. "Ghana Diabetes Country Profile." 2023.

---

*Last updated: October 2025*
