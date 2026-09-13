"""Master CLI Execution Script: Weather -> Commodity Price Regime Model.

Runs the complete institutional research workflow:
1. Production-weighted weather anomaly feature construction
2. Technological data drift analysis (PSI, KS, Chow tests, lead-time decay)
3. Latent market & weather regime detection (HMM) & conditional predictability hypothesis test
4. Multi-horizon model bake-off (Ridge vs LightGBM vs XGBoost vs TCN vs LSTM)
5. The 5-stage ablation benchmark (Price-only -> Weather-only -> +Weather -> +Term -> +Regime)
"""

import sys
import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from src.data_ingestion.usda_weights import get_corn_spatial_config
from src.feature_engineering.panel_builder import build_master_panel
from src.drift_analysis.drift_report import run_comprehensive_drift_analysis, generate_drift_summary_table
from src.experiments.regime_hypothesis_test import run_regime_hypothesis_test, format_regime_hypothesis_markdown
from src.experiments.run_experiment_suite import ExperimentSuiteRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    print("=" * 80)
    print("  WEATHER -> COMMODITY PRICE REGIME MODEL")
    print("  Hypothesis: Extreme weather anomalies affecting major agricultural production")
    print("  regions create predictable changes in commodity futures prices.")
    print("=" * 80)

    # 1. Build Master Panel
    print("\n[STEP 1/5] Ingesting Data & Building Production-Weighted Weather Exposure Panel...")
    config = get_corn_spatial_config()
    print(f"Target Commodity: {config.commodity} ({config.ticker})")
    print(f"Monitored Production Regions (Top US Corn Belt States):")
    for code, reg in config.regions.items():
        print(f"  - {code} ({reg.name}): {reg.weight * 100:.1f}% production share | coords: ({reg.lat}, {reg.lon})")

    panel = build_master_panel(spatial_config=config, start_date="2000-01-01", end_date="2024-12-31")
    print(f"Master panel shape: {panel.df.shape}")
    print(f"Total features: {len(panel.all_features)} (Weather: {len(panel.weather_cols)}, Price: {len(panel.price_cols)}, Term: {len(panel.term_cols)})")

    # 2. Technological Data Drift Analysis
    print("\n" + "=" * 80)
    print("[STEP 2/5] Running Technological Data Drift Analysis across Eras...")
    print("=" * 80)
    drift_results = run_comprehensive_drift_analysis(panel.df)
    drift_md = generate_drift_summary_table(drift_results)
    print(drift_md)

    # 3. Regime Hypothesis Test
    print("\n" + "=" * 80)
    print("[STEP 3/5] Testing Core Regime Hypothesis: Does Weather IC Change by Regime?")
    print("=" * 80)
    hyp_results = run_regime_hypothesis_test(panel.df)
    hyp_md = format_regime_hypothesis_markdown(hyp_results)
    print(hyp_md)

    # 4. Multi-Horizon Model Bake-off
    print("\n" + "=" * 80)
    print("[STEP 4/5] Multi-Model Algorithm Bake-off (Ridge vs LightGBM vs XGBoost vs TCN vs LSTM)...")
    print("=" * 80)
    runner = ExperimentSuiteRunner(panel_dataset=panel, tx_cost_bps=5.0, n_cv_splits=5)
    bakeoff_df = runner.run_algorithm_bakeoff()
    print("\nAlgorithm Comparison Table (Walk-Forward Out-of-Sample):")
    print(bakeoff_df.to_string(index=False))

    # Determine winning algorithm
    # Best by Sharpe ratio
    best_algo_row = bakeoff_df.sort_values("Sharpe", ascending=False).iloc[0]
    best_algo = best_algo_row["Model"]
    print(f"\nEmpirically Selected Best Performer: {best_algo} (Sharpe: {best_algo_row['Sharpe']}, IC: {best_algo_row['IC']})")

    # 5. The Killer Experiment: 5-Stage Ablation Benchmark
    print("\n" + "=" * 80)
    print("[STEP 5/5] Executing The Killer Experiment: 5-Stage Ablation Benchmark...")
    print("=" * 80)
    ablation_df, curves = runner.run_5_stage_ablation(core_algorithm=best_algo)
    
    print("\n" + "=" * 80)
    print("FINAL INSTITUTIONAL BENCHMARK RESULTS")
    print("=" * 80)
    print(ablation_df.to_string(index=False))
    print("=" * 80)

    # Save output to csv
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    ablation_df.to_csv(out_dir / "ablation_results.csv", index=False)
    bakeoff_df.to_csv(out_dir / "algorithm_bakeoff.csv", index=False)
    print(f"\nResults successfully saved to {out_dir}/")


if __name__ == "__main__":
    main()

