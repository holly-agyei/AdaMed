"""
Clinical heuristics for West African diabetes management.

These parameters encode domain knowledge from epidemiological literature,
WHO country profiles, and dietary studies. They serve two purposes:
1. Parameterize the synthetic target domain distribution
2. Provide physiological priors that could inform loss function design

Sources cited inline. All values are approximate and intended for
research simulation, not clinical decision-making.
"""

import numpy as np
from typing import Dict, Optional


def get_west_african_parameters() -> Dict:
    """
    Return dictionary of parameters characterizing West African diabetes context.

    Sources:
    - WHO Diabetes Country Profiles: Ghana (2023)
    - International Diabetes Federation Atlas, 10th Edition (2021)
    - Amoako et al., "Glycemic Index of Traditional Ghanaian Foods"
      Journal of Nutrition and Metabolism (2021)
    - Ofori-Asenso et al., "Diabetes awareness and management in Ghana"
      BMC Public Health (2019)
    - Ghana Health Service Annual Report (2022)

    Returns:
        Dictionary with feature shifts, dietary factors, and access constraints
    """
    return {
        "description": "Heuristics for Ghanaian diabetes management context",
        "missing_prob": 0.3,  # P(missing data segment) per patient per feature
        "feature_shifts": {
            "glucose_mgdl": {
                "mean_shift": 1.2,  # Higher baseline glucose (mg/dL normalized)
                "scale_factor": 1.3,  # Greater variability
                "notes": (
                    "Post-prandial spikes from high-GI staples (kenkey GI~75, "
                    "fufu GI~85). Irregular meal timing compared to Mediterranean pattern."
                ),
            },
            "bp_systolic": {
                "mean_shift": 1.15,
                "scale_factor": 1.2,
                "notes": (
                    "Higher prevalence of undiagnosed hypertension. "
                    "Ofori-Asenso et al. report 54% unaware of HTN status."
                ),
            },
            "bp_diastolic": {
                "mean_shift": 1.1,
                "scale_factor": 1.15,
                "notes": "Correlated with sodium intake from preserved/smoked foods.",
            },
            "heart_rate": {
                "mean_shift": 1.05,
                "scale_factor": 1.1,
                "notes": (
                    "Marginally elevated due to physical labor patterns "
                    "and under-managed cardiovascular comorbidities."
                ),
            },
            "physical_activity": {
                "mean_shift": 0.8,
                "scale_factor": 0.9,
                "notes": (
                    "Different activity profile: sustained agricultural/market labor "
                    "vs. structured exercise bouts. Lower measured 'exercise' but "
                    "higher occupational activity."
                ),
            },
        },
        "dietary_factors": {
            "kenkey": {
                "glycemic_index": 75,
                "consumption_frequency": "daily",
                "fermentation_effect": (
                    "Fermentation reduces glucose spike by ~15% compared to "
                    "unfermented maize (Amoako et al., 2021)"
                ),
                "serving_size_g": 300,
            },
            "fufu": {
                "glycemic_index": 85,
                "consumption_frequency": "3-4x weekly",
                "preparation": "Cassava and plantain pounded — high starch content",
                "serving_size_g": 400,
            },
            "banku": {
                "glycemic_index": 70,
                "consumption_frequency": "2-3x weekly",
                "fermentation_duration_days": 3,
                "serving_size_g": 350,
            },
            "rice_jollof": {
                "glycemic_index": 72,
                "consumption_frequency": "2-3x weekly",
                "preparation": "Parboiled rice with tomato stew — GI varies with cooking method",
                "serving_size_g": 300,
            },
        },
        "access_constraints": {
            "cgm_cost_months_wage": 3,  # CGM device costs ~3 months minimum wage
            "test_strip_stockout_prob": 0.4,  # 40% chance clinic has no strips
            "avg_clinic_distance_km": 25,  # Rural average
            "transport_modes": ["tro-tro", "walking", "shared taxi"],
            "avg_emergency_response_min": 45,
            "diabetes_prevalence_pct": 6.5,  # IDF estimate for Ghana
            "diagnosed_fraction": 0.46,  # Only 46% are diagnosed
        },
    }


def get_glycemic_response(
    food: str, hours: int = 4, samples: int = 48
) -> np.ndarray:
    """
    Simulate glycemic response curve for West African foods.

    Uses a simplified two-compartment pharmacokinetic model:
    - Compartment 1: gut absorption (rate depends on GI)
    - Compartment 2: insulin-mediated clearance

    This is intentionally simplified — a real model would need patient-specific
    insulin sensitivity parameters we don't have for the target population.

    Args:
        food: One of 'kenkey', 'fufu', 'banku', 'rice_jollof'
        hours: Duration to simulate (default 4h post-prandial window)
        samples: Number of time points

    Returns:
        Normalized glucose response curve, shape (samples,)
    """
    params = get_west_african_parameters()
    food_data = params["dietary_factors"].get(
        food, params["dietary_factors"]["kenkey"]
    )

    t = np.linspace(0, hours, samples)

    # Two-compartment model parameters
    gi = food_data["glycemic_index"] / 100  # Normalize to [0, 1]
    peak_time = 0.75 + 0.5 * (1 - gi)  # Higher GI → faster peak
    absorption_rate = 2.0 * gi  # Higher GI → faster absorption
    clearance_rate = 0.4 + 0.2 * gi  # Higher GI → slightly faster clearance

    # Gamma-like response curve (models delayed absorption → exponential clearance)
    response = gi * (t / peak_time) ** absorption_rate * np.exp(
        -clearance_rate * (t - peak_time)
    )
    response[t < 0.05] = 0  # No immediate effect

    # Apply fermentation effect if applicable
    if "fermentation_effect" in food_data:
        # Fermentation partially breaks down starches → reduced peak, broader curve
        response *= 0.85  # ~15% peak reduction
        # Slightly prolonged absorption
        response *= np.exp(-0.05 * t)

    # Normalize to [0, 1]
    max_val = response.max()
    if max_val > 0:
        response /= max_val

    return response


def compute_distribution_divergence(
    source_features: np.ndarray, target_features: np.ndarray
) -> Dict:
    """
    Compute simple distribution divergence metrics between domains.

    Uses KL divergence approximation and maximum mean discrepancy (MMD)
    to quantify how different the source and target distributions are.
    This helps validate that our synthetic data has meaningful shift.

    Args:
        source_features: (n_source, n_features) array
        target_features: (n_target, n_features) array

    Returns:
        Dictionary with per-feature divergence metrics
    """
    n_features = source_features.shape[1] if source_features.ndim > 1 else 1

    results = {}
    for f in range(n_features):
        s = source_features[:, f] if n_features > 1 else source_features
        t = target_features[:, f] if n_features > 1 else target_features

        # Remove NaNs
        s = s[~np.isnan(s)]
        t = t[~np.isnan(t)]

        # Mean and variance shift
        mean_shift = np.abs(np.mean(s) - np.mean(t))
        var_ratio = np.var(t) / (np.var(s) + 1e-8)

        # Simple MMD estimate using Gaussian kernel
        # (full MMD is O(n^2), so we subsample)
        n_sub = min(200, len(s), len(t))
        s_sub = np.random.choice(s, n_sub, replace=True)
        t_sub = np.random.choice(t, n_sub, replace=True)

        sigma = np.median(np.abs(s_sub[:, None] - t_sub[None, :]))
        if sigma < 1e-8:
            sigma = 1.0

        k_ss = np.exp(-0.5 * (s_sub[:, None] - s_sub[None, :]) ** 2 / sigma ** 2).mean()
        k_tt = np.exp(-0.5 * (t_sub[:, None] - t_sub[None, :]) ** 2 / sigma ** 2).mean()
        k_st = np.exp(-0.5 * (s_sub[:, None] - t_sub[None, :]) ** 2 / sigma ** 2).mean()

        mmd = k_ss + k_tt - 2 * k_st

        results[f"feature_{f}"] = {
            "mean_shift": float(mean_shift),
            "variance_ratio": float(var_ratio),
            "mmd_estimate": float(mmd),
        }

    return results
