"""Technological Data Drift: Comprehensive Report Generator.

Synthesizes covariate shift, concept drift, Chow structural break tests,
and information lead-time decay into an executive research report.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np

from src.drift_analysis.data_splitter import TechnologicalEraSplitter
from src.drift_analysis.covariate_drift import CovariateDriftAnalyzer
from src.drift_analysis.concept_drift import ConceptDriftAnalyzer


def run_comprehensive_drift_analysis(
    master_df: pd.DataFrame,
    features_to_monitor: list[str] = None
) -> Dict[str, Any]:
    """Runs end-to-end data drift analysis on the master panel dataset."""
    splitter = TechnologicalEraSplitter()
    era_subsets = splitter.split_by_eras(master_df)

    if features_to_monitor is None:
        features_to_monitor = [
            "weighted_temp_anomaly_level",
            "weighted_temp_anomaly_30d_mean",
            "weighted_extreme_heat_30d_sum",
            "weighted_moisture_deficit_30d_cum",
            "term_roll_yield",
            "price_vol_21d",
            "price_mom_21d"
        ]
        # Filter only existing columns
        features_to_monitor = [f for f in features_to_monitor if f in master_df.columns]

    # 1. Covariate Shift Analysis
    cov_analyzer = CovariateDriftAnalyzer(features=features_to_monitor)
    pairwise_drift = cov_analyzer.run_pairwise_era_drift(era_subsets)

    # 2. Concept Drift & Structural Break Analysis
    heat_feature = "weighted_extreme_heat_30d_mean"
    if heat_feature not in master_df.columns:
        # Fallback to level or available column
        heat_feature = [c for c in master_df.columns if "extreme_heat" in c or "temp_anomaly" in c][0]

    concept_analyzer = ConceptDriftAnalyzer(weather_feature=heat_feature)
    chow_results = concept_analyzer.evaluate_era_structural_breaks(era_subsets)
    rolling_elasticity = concept_analyzer.compute_rolling_elasticity(master_df)
    lead_time_decay = concept_analyzer.compute_lead_time_decay(era_subsets, max_lag_days=15)

    return {
        "pairwise_covariate_drift": pairwise_drift,
        "chow_structural_breaks": chow_results,
        "rolling_elasticity": rolling_elasticity,
        "lead_time_decay": lead_time_decay,
        "monitored_features": features_to_monitor
    }


def generate_drift_summary_table(drift_results: Dict[str, Any]) -> str:
    """Formats drift results as a clear GitHub-flavored markdown report."""
    md = []
    md.append("### Technological Data Drift Analysis Summary\n")
    md.append("**Research Hypothesis Tested**: Technological advances in seed genetics (drought tolerance) ")
    md.append("and orbital satellite remote sensing have induced both *covariate shift* (distribution drift) ")
    md.append("and *concept drift* (structural relationship attenuation and lead-time compression).\n")

    # 1. Chow Tests
    md.append("#### 1. Econometric Structural Break Tests (Chow Test)")
    md.append("| Inflection Boundary | Breakpoint | F-Statistic | p-Value | Structural Break Detected? |")
    md.append("| :--- | :--- | :---: | :---: | :---: |")
    for key, res in drift_results["chow_structural_breaks"].items():
        break_str = "**YES (p < 0.05)**" if res["structural_break"] else "NO"
        md.append(f"| {key} | {res['breakpoint']} | {res['f_stat']:.2f} | {res['p_value']:.4f} | {break_str} |")
    md.append("\n")

    # 2. Covariate Drift (Era 1 vs Era 3)
    macro_key = [k for k in drift_results["pairwise_covariate_drift"].keys() if "Era 1" in k and "Era 3" in k]
    if macro_key:
        df_macro = drift_results["pairwise_covariate_drift"][macro_key[0]]
        md.append("#### 2. Macro Feature Drift (Era 1: 2000-09 vs Era 3: 2018-24)")
        md.append("| Feature | PSI | Status | KS Stat | KS p-Value | Wasserstein Dist |")
        md.append("| :--- | :---: | :--- | :---: | :---: | :---: |")
        for _, row in df_macro.head(8).iterrows():
            md.append(f"| `{row['feature']}` | {row['psi']:.3f} | {row['psi_status']} | {row['ks_stat']:.3f} | {row['ks_pvalue']:.4f} | {row['wasserstein']:.3f} |")
        md.append("\n")

    # 3. Lead-time decay
    decay_df = drift_results["lead_time_decay"]
    md.append("#### 3. Information Lead-Time Compression (Weather-Return Correlation by Forward Lag)")
    md.append("| Lag (Days) | Era 1 (2000-09) | Era 2 (2010-17) | Era 3 (2018-24) | Market Efficiency Interpretation |")
    md.append("| :---: | :---: | :---: | :---: | :--- |")
    for lag in [1, 3, 5, 10, 15]:
        if lag in decay_df.index:
            e1 = decay_df.loc[lag].get("era_1", np.nan)
            e2 = decay_df.loc[lag].get("era_2", np.nan)
            e3 = decay_df.loc[lag].get("era_3", np.nan)
            interp = "Immediate reaction (satellite)" if lag <= 3 else "Delayed USDA reaction (pre-modern)"
            md.append(f"| {lag}d | {e1:+.4f} | {e2:+.4f} | {e3:+.4f} | {interp} |")
    md.append("\n")

    return "\n".join(md)

