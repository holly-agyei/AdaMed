"""
Synthetic data generator for AdaMed experiments.

Simulates Spanish CGM data (source) and Ghanaian proxy inputs (target).
Uses clinically plausible distributions and correlations to create
multivariate time-series that mirror real patient trajectories.

Design choice: We use synthetic data because no public Ghanaian CGM dataset exists.
The source domain mimics patterns from Spanish CGM studies (BIO-DQNA baseline),
while the target domain encodes heuristics from WHO reports and dietary literature.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import warnings


class ClinicalTimeSeriesGenerator:
    """
    Generate synthetic multivariate time-series data with realistic clinical patterns.

    Source domain: Simulates Spanish patients with continuous glucose monitors
    Target domain: Simulates Ghanaian patients with sparse measurements and heuristics

    Why two domains?
    - Source has dense, well-labeled data (CGM readings every 5 min)
    - Target has sparse, unlabeled data (clinic visits every few weeks)
    - The gap between these is what zero-shot adaptation must bridge

    Attributes:
        n_source: Number of source domain patients
        n_target_proxy: Number of target domain proxy patients
        time_steps: Number of time points per patient (48 = 24h at 30min intervals)
        n_features: Number of clinical features per time point
        feature_names: Human-readable feature names
    """

    def __init__(
        self,
        n_source: int = 1000,
        n_target_proxy: int = 200,
        time_steps: int = 48,
        n_features: int = 5,
        seed: int = 42,
    ):
        self.n_source = n_source
        self.n_target_proxy = n_target_proxy
        self.time_steps = time_steps
        self.n_features = n_features
        self.seed = seed
        np.random.seed(seed)

        # Feature names for interpretability
        # These mirror the variables BIO-DQNA used for T1D+hypertension management
        self.feature_names = [
            "glucose_mgdl",
            "bp_systolic",
            "bp_diastolic",
            "heart_rate",
            "physical_activity",
        ]

    def _generate_circadian_base(self) -> np.ndarray:
        """
        Generate a circadian rhythm base signal.
        Glucose, BP, and HR all follow ~24h oscillations in real patients.
        """
        t = np.linspace(0, 24, self.time_steps)
        return 0.5 * np.sin(2 * np.pi * t / 24) + 1.0

    def _simulate_meals(self, n_meals: int = 3) -> np.ndarray:
        """
        Simulate meal-induced glucose spikes.
        Spanish patients: regular meal times (breakfast ~8h, lunch ~14h, dinner ~21h).
        """
        meal_response = np.zeros(self.time_steps)
        # Approximate meal indices for a 24h window with 48 steps (30 min each)
        typical_meals = [16, 28, 42]  # ~8h, ~14h, ~21h
        for idx in typical_meals[:n_meals]:
            # Gaussian-shaped meal response peaking ~1h after eating
            for offset in range(min(6, self.time_steps - idx)):
                decay = np.exp(-0.5 * (offset - 2) ** 2)
                meal_response[idx + offset] += 0.4 * decay
        return meal_response

    def generate_source_domain(self) -> Dict:
        """
        Generate synthetic source domain data (Spanish CGM).

        Patterns modeled:
        - Regular meal times with predictable glucose spikes
        - Controlled variability (well-managed patients)
        - Complete data (no missing values)
        - Correlated features (BP tracks with glucose, HR with activity)

        Returns:
            Dictionary with 'data', 'labels', 'domain', 'feature_names', 'metadata'
        """
        t = np.linspace(0, 24, self.time_steps)
        circadian = self._generate_circadian_base()

        X = []
        y = []  # binary outcome: good (1) / poor (0) glycemic control

        for i in range(self.n_source):
            # Patient-specific baseline — some patients run higher/lower
            baseline_glucose = np.random.normal(1.0, 0.15)
            baseline_bp = np.random.normal(1.0, 0.1)

            patient_data = np.zeros((self.time_steps, self.n_features))

            # --- Glucose (feature 0) ---
            # Circadian base + meal responses + patient noise
            glucose = baseline_glucose * circadian.copy()
            meal_response = self._simulate_meals(n_meals=3)
            glucose += meal_response
            glucose += np.random.normal(0, 0.08, self.time_steps)
            # Clamp to physiological range (normalized)
            glucose = np.clip(glucose, 0.3, 2.5)
            patient_data[:, 0] = glucose

            # --- BP systolic (feature 1) ---
            # Correlated with glucose (shared autonomic drivers)
            bp_sys = baseline_bp * circadian * 0.7 + glucose * 0.3
            bp_sys += np.random.normal(0, 0.06, self.time_steps)
            patient_data[:, 1] = bp_sys

            # --- BP diastolic (feature 2) ---
            # Tracks systolic but with narrower range
            bp_dia = bp_sys * 0.65 + np.random.normal(0, 0.04, self.time_steps)
            patient_data[:, 2] = bp_dia

            # --- Heart rate (feature 3) ---
            # Own circadian pattern + activity-driven spikes
            hr = 0.7 + 0.15 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 0.08, self.time_steps)
            patient_data[:, 3] = hr

            # --- Physical activity (feature 4) ---
            # Sparse events: most patients have 1-3 activity bouts per day
            activity = np.zeros(self.time_steps)
            n_bouts = np.random.randint(1, 5)
            active_hours = np.random.choice(self.time_steps, n_bouts, replace=False)
            activity[active_hours] = np.random.uniform(0.3, 1.0, n_bouts)
            patient_data[:, 4] = activity

            X.append(patient_data)

            # Label: good control if glucose coefficient of variation < threshold
            # CV is a standard metric in diabetes management (< 36% = stable)
            glucose_cv = np.std(glucose) / (np.mean(glucose) + 1e-8)
            y.append(1 if glucose_cv < 0.25 else 0)

        X = np.array(X)
        y = np.array(y)

        return {
            "data": X,
            "labels": y,
            "domain": np.zeros(self.n_source, dtype=int),  # domain 0 = source
            "feature_names": self.feature_names,
            "metadata": {
                "domain": "Spanish CGM",
                "sampling": "continuous",
                "time_steps": self.time_steps,
                "n_patients": self.n_source,
                "label_distribution": {
                    "good_control": int(y.sum()),
                    "poor_control": int(len(y) - y.sum()),
                },
            },
        }

    def generate_target_proxy(self) -> Dict:
        """
        Generate synthetic target proxy data (Ghanaian heuristics).

        Key differences from source:
        - Higher glucose variability (high-GI staples like kenkey, fufu)
        - Missing data segments (clinic stock-outs, transport delays)
        - Different distribution parameters from WHO/IDF reports
        - No reliable labels (the core zero-shot challenge)

        Returns:
            Dictionary with 'data', 'labels' (dummy), 'domain', 'feature_names', 'metadata'
        """
        X = []

        # Load heuristics for distribution shift
        from .heuristics import get_west_african_parameters

        params = get_west_african_parameters()

        for i in range(self.n_target_proxy):
            patient_data = np.zeros((self.time_steps, self.n_features))

            # Apply distribution shift based on heuristics
            for f in range(self.n_features):
                feature_name = self.feature_names[f]
                shift = params["feature_shifts"][feature_name]["mean_shift"]
                scale = params["feature_shifts"][feature_name]["scale_factor"]

                # Base signal: shifted distribution relative to source
                if feature_name == "glucose_mgdl":
                    # Irregular meal patterns + high-GI foods = more variability
                    base = np.random.normal(shift, scale * 0.3, self.time_steps)
                    # Add high-GI meal spikes at irregular times
                    n_meals = np.random.randint(2, 5)
                    meal_times = np.random.choice(self.time_steps, n_meals, replace=False)
                    for mt in meal_times:
                        end = min(mt + 4, self.time_steps)
                        base[mt:end] += np.random.uniform(0.3, 0.8)
                elif feature_name == "physical_activity":
                    # Agricultural workers: sustained activity rather than gym bouts
                    base = np.random.uniform(0, 0.5, self.time_steps)
                    # Work hours: roughly 6am-6pm
                    work_start, work_end = 12, 36  # indices for ~6am to ~6pm
                    base[work_start:work_end] += np.random.uniform(0.2, 0.6)
                    base *= scale
                else:
                    base = np.random.normal(shift, scale * 0.2, self.time_steps)

                # Simulate missing data (clinic stock-outs, device unavailability)
                if np.random.random() < params["missing_prob"]:
                    missing_start = np.random.randint(0, max(1, self.time_steps - 10))
                    missing_len = np.random.randint(3, min(10, self.time_steps - missing_start))
                    base[missing_start : missing_start + missing_len] = np.nan

                patient_data[:, f] = base

            X.append(patient_data)

        X = np.array(X)

        return {
            "data": X,
            "labels": np.full(self.n_target_proxy, -1, dtype=int),  # -1 = unavailable
            "domain": np.ones(self.n_target_proxy, dtype=int),  # domain 1 = target
            "feature_names": self.feature_names,
            "metadata": {
                "domain": "Ghanaian Proxy",
                "sampling": "sparse",
                "time_steps": self.time_steps,
                "n_patients": self.n_target_proxy,
                "heuristics_applied": params["description"],
                "missing_data_prob": params["missing_prob"],
            },
        }

    def generate_experimental_split(self) -> Dict:
        """
        Generate complete dataset for experiments.

        Combines source and target, providing indices for each domain.
        Target labels are set to 0 (dummy) for combined array compatibility,
        but should NOT be used for training — this is the zero-shot constraint.

        Returns:
            Dictionary with combined data, labels, domains, and split indices
        """
        source = self.generate_source_domain()
        target = self.generate_target_proxy()

        # For target samples, replace -1 labels with 0 for tensor compatibility
        # IMPORTANT: these are dummy labels and must NOT be used in loss computation
        target_labels = np.zeros(self.n_target_proxy, dtype=int)

        X = np.concatenate([source["data"], target["data"]], axis=0)
        y = np.concatenate([source["labels"], target_labels], axis=0)
        domain = np.concatenate([source["domain"], target["domain"]], axis=0)

        # Handle NaN values in target data: fill with feature-wise source mean
        # This is a deliberate choice — more sophisticated imputation is future work
        for f in range(self.n_features):
            source_mean = np.nanmean(source["data"][:, :, f])
            feature_slice = X[:, :, f]
            nan_mask = np.isnan(feature_slice)
            feature_slice[nan_mask] = source_mean
            X[:, :, f] = feature_slice

        return {
            "X": X,
            "y": y,
            "domain": domain,
            "source_idx": np.arange(self.n_source),
            "target_idx": np.arange(self.n_target_proxy) + self.n_source,
            "feature_names": self.feature_names,
            "metadata": {"source": source["metadata"], "target": target["metadata"]},
        }


if __name__ == "__main__":
    # Quick sanity check
    gen = ClinicalTimeSeriesGenerator()
    data = gen.generate_experimental_split()
    print(f"Generated data shape: {data['X'].shape}")
    print(f"Source samples: {len(data['source_idx'])}")
    print(f"Target proxy samples: {len(data['target_idx'])}")
    print(f"Feature names: {data['feature_names']}")
    print(f"Label distribution (source only): "
          f"{np.bincount(data['y'][:len(data['source_idx'])])}")
    print(f"NaN values remaining: {np.isnan(data['X']).sum()}")
