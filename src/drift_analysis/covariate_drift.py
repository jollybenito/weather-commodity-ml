"""Technological Data Drift: Covariate Shift Analysis.

Measures feature distribution drift across technological eras using:
- Population Stability Index (PSI)
- Two-Sample Kolmogorov-Smirnov (KS) Test
- 1D Wasserstein Distance (Earth Mover's Distance)
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def calculate_psi(
    reference: pd.Series,
    current: pd.Series,
    num_buckets: int = 10,
    epsilon: float = 1e-4
) -> float:
    """Computes the Population Stability Index (PSI) between reference and current distributions.
    
    Formula:
        PSI = Sum ( (Actual% - Expected%) * ln(Actual% / Expected%) )
    
    Interpretation:
        PSI < 0.10  : No significant distribution change (Stable)
        0.10 - 0.25 : Moderate distribution change (Warning)
        PSI >= 0.25 : Significant structural drift (Action required)
    """
    ref_clean = reference.dropna().values
    cur_clean = current.dropna().values

    if len(ref_clean) < 10 or len(cur_clean) < 10:
        return 0.0

    # Determine quantile bins based on reference distribution
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bins = np.percentile(ref_clean, percentiles)
    # Ensure strictly monotonically increasing bin edges
    bins[0] = -np.inf
    bins[-1] = np.inf
    bins = np.unique(bins)

    ref_counts, _ = np.histogram(ref_clean, bins=bins)
    cur_counts, _ = np.histogram(cur_clean, bins=bins)

    ref_pct = ref_counts / len(ref_clean)
    cur_pct = cur_counts / len(cur_clean)

    # Apply epsilon smoothing to prevent div by zero / log(0)
    ref_pct = np.clip(ref_pct, epsilon, 1.0)
    cur_pct = np.clip(cur_pct, epsilon, 1.0)

    # Normalize to strictly sum to 1
    ref_pct /= ref_pct.sum()
    cur_pct /= cur_pct.sum()

    psi_val = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(np.maximum(0.0, psi_val))


def classify_psi(psi: float) -> str:
    if psi < 0.10:
        return "Stable (PSI < 0.10)"
    elif psi < 0.25:
        return "Moderate Drift (0.10 <= PSI < 0.25)"
    else:
        return "Significant Structural Drift (PSI >= 0.25)"


class CovariateDriftAnalyzer:
    """Runs statistical covariate shift tests across technological eras."""

    def __init__(self, features: List[str]):
        self.features = features

    def compare_distributions(
        self,
        reference_series: pd.Series,
        current_series: pd.Series,
        feature_name: str
    ) -> dict:
        """Runs KS test, Wasserstein distance, and PSI for a single feature."""
        ref = reference_series.dropna()
        cur = current_series.dropna()

        if len(ref) < 10 or len(cur) < 10:
            return {
                "feature": feature_name,
                "psi": 0.0,
                "psi_status": "Insufficient Data",
                "ks_stat": 0.0,
                "ks_pvalue": 1.0,
                "wasserstein": 0.0,
                "mean_ref": 0.0,
                "mean_cur": 0.0,
                "std_ref": 0.0,
                "std_cur": 0.0
            }

        psi = calculate_psi(ref, cur)
        ks_res = stats.ks_2samp(ref, cur)
        w_dist = stats.wasserstein_distance(ref, cur)

        return {
            "feature": feature_name,
            "psi": round(psi, 4),
            "psi_status": classify_psi(psi),
            "ks_stat": round(ks_res.statistic, 4),
            "ks_pvalue": round(ks_res.pvalue, 6),
            "wasserstein": round(w_dist, 4),
            "mean_ref": round(float(ref.mean()), 4),
            "mean_cur": round(float(cur.mean()), 4),
            "std_ref": round(float(ref.std()), 4),
            "std_cur": round(float(cur.std()), 4)
        }

    def run_pairwise_era_drift(
        self,
        era_subsets: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Compares Era 1 vs Era 2, Era 2 vs Era 3, and Era 1 vs Era 3."""
        pairs = [
            ("era_1", "era_2", "Era 1 (2000-09) vs Era 2 (2010-17)"),
            ("era_2", "era_3", "Era 2 (2010-17) vs Era 3 (2018-24)"),
            ("era_1", "era_3", "Era 1 (2000-09) vs Era 3 (2018-24) [Full Macro Drift]")
        ]

        results = {}
        for ref_era, cur_era, comparison_title in pairs:
            if ref_era not in era_subsets or cur_era not in era_subsets:
                continue

            ref_df = era_subsets[ref_era]
            cur_df = era_subsets[cur_era]
            
            rows = []
            for feat in self.features:
                if feat in ref_df.columns and feat in cur_df.columns:
                    stat_dict = self.compare_distributions(ref_df[feat], cur_df[feat], feat)
                    rows.append(stat_dict)

            res_df = pd.DataFrame(rows).sort_values("psi", ascending=False)
            results[comparison_title] = res_df

        return results

