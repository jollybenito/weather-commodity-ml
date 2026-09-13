"""Technological Data Drift: Concept Drift & Structural Break Analysis.

Evaluates structural shifts in the relationship between Weather and Commodity Returns:
1. Rolling Elasticity beta_weather,t: Measures whether physical crop resilience (biotech seeds)
   has attenuated price sensitivity to heat/drought anomalies.
2. Chow Structural Break Test: Formal econometric F-test proving parameter instability
   across technological inflection dates (2010 and 2018).
3. Signal Lead-Time Compression: Quantifies whether modern satellite remote sensing
   compressed the pricing half-life of weather anomalies from weeks to days.
"""

from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
from scipy import stats


def run_chow_test(
    X_pre: np.ndarray,
    y_pre: np.ndarray,
    X_post: np.ndarray,
    y_post: np.ndarray
) -> Dict[str, float]:
    """Conducts a Chow Structural Break Test across a candidate regime/era breakpoint.
    
    Formula:
        F = [ (SSR_pooled - (SSR_pre + SSR_post)) / k ] / [ (SSR_pre + SSR_post) / (N_pre + N_post - 2k) ]
    """
    # Add constant column for intercept
    X_pre_const = np.column_stack([np.ones(len(X_pre)), X_pre])
    X_post_const = np.column_stack([np.ones(len(X_post)), X_post])
    
    X_pooled = np.vstack([X_pre_const, X_post_const])
    y_pooled = np.concatenate([y_pre, y_post])

    k = X_pooled.shape[1]  # number of parameters including intercept
    n_pre = len(y_pre)
    n_post = len(y_post)
    n_total = n_pre + n_post

    if n_total <= 2 * k:
        return {"f_stat": 0.0, "p_value": 1.0, "structural_break": False}

    # OLS estimation via least squares
    beta_pooled, res_pooled, _, _ = np.linalg.lstsq(X_pooled, y_pooled, rcond=None)
    beta_pre, res_pre, _, _ = np.linalg.lstsq(X_pre_const, y_pre, rcond=None)
    beta_post, res_post, _, _ = np.linalg.lstsq(X_post_const, y_post, rcond=None)

    ssr_pooled = np.sum((y_pooled - X_pooled @ beta_pooled) ** 2)
    ssr_pre = np.sum((y_pre - X_pre_const @ beta_pre) ** 2)
    ssr_post = np.sum((y_post - X_post_const @ beta_post) ** 2)

    numerator = (ssr_pooled - (ssr_pre + ssr_post)) / k
    denominator = (ssr_pre + ssr_post) / (n_total - 2 * k)

    if denominator <= 0:
        f_stat = 0.0
    else:
        f_stat = numerator / denominator

    df1 = k
    df2 = n_total - 2 * k
    p_value = 1.0 - stats.f.cdf(f_stat, df1, df2) if f_stat > 0 else 1.0

    return {
        "f_stat": round(float(f_stat), 4),
        "p_value": round(float(p_value), 6),
        "df1": df1,
        "df2": df2,
        "structural_break": bool(p_value < 0.05)
    }


class ConceptDriftAnalyzer:
    """Analyzes time-varying elasticity and structural breaks in weather-to-price relationships."""

    def __init__(self, weather_feature: str = "weighted_extreme_heat_30d_mean"):
        self.weather_feature = weather_feature

    def compute_rolling_elasticity(
        self,
        df: pd.DataFrame,
        target_col: str = "target_fwd_ret_5d",
        window_days: int = 756  # 3-year rolling window
    ) -> pd.DataFrame:
        """Computes rolling OLS elasticity beta_weather,t and R^2 over time.
        
        Tests hypothesis: Did advances in seed drought resistance reduce beta over time?
        """
        valid = df[[self.weather_feature, target_col]].dropna()
        x = valid[self.weather_feature].values
        y = valid[target_col].values
        dates = valid.index

        betas = np.full(len(x), np.nan)
        r_squared = np.full(len(x), np.nan)

        for i in range(window_days, len(x)):
            x_win = x[i - window_days:i]
            y_win = y[i - window_days:i]

            x_var = np.var(x_win)
            if x_var > 1e-6:
                slope, intercept, r_val, _, _ = stats.linregress(x_win, y_win)
                betas[i] = slope
                r_squared[i] = r_val ** 2

        res_df = pd.DataFrame({
            "beta_weather": betas,
            "r_squared": r_squared
        }, index=dates)
        return res_df

    def evaluate_era_structural_breaks(
        self,
        era_subsets: Dict[str, pd.DataFrame],
        target_col: str = "target_fwd_ret_5d"
    ) -> Dict[str, Dict]:
        """Tests for structural breaks at the 2010 (Era 1 -> 2) and 2018 (Era 2 -> 3) boundaries."""
        results = {}

        if "era_1" in era_subsets and "era_2" in era_subsets:
            df1 = era_subsets["era_1"][[self.weather_feature, target_col]].dropna()
            df2 = era_subsets["era_2"][[self.weather_feature, target_col]].dropna()
            res_1_2 = run_chow_test(
                df1[[self.weather_feature]].values,
                df1[target_col].values,
                df2[[self.weather_feature]].values,
                df2[target_col].values
            )
            res_1_2["breakpoint"] = "2010 (Era 1 -> Era 2: Early Satellite & Drought Benchmark)"
            results["era_1_vs_2"] = res_1_2

        if "era_2" in era_subsets and "era_3" in era_subsets:
            df2 = era_subsets["era_2"][[self.weather_feature, target_col]].dropna()
            df3 = era_subsets["era_3"][[self.weather_feature, target_col]].dropna()
            res_2_3 = run_chow_test(
                df2[[self.weather_feature]].values,
                df2[target_col].values,
                df3[[self.weather_feature]].values,
                df3[target_col].values
            )
            res_2_3["breakpoint"] = "2018 (Era 2 -> Era 3: Biotech Resilient Seeds & High-Cadence Satellites)"
            results["era_2_vs_3"] = res_2_3

        return results

    def compute_lead_time_decay(
        self,
        era_subsets: Dict[str, pd.DataFrame],
        max_lag_days: int = 21
    ) -> pd.DataFrame:
        """Computes cross-correlation Corr(R_{t+k}, Weather_t) across lags k=1..max_lag.
        
        Tests hypothesis: Has technological advancement (satellite imagery) compressed 
        the lead time of weather signals, discounting information much earlier?
        """
        lags = list(range(1, max_lag_days + 1))
        corr_data = {}

        for era_name, sub_df in era_subsets.items():
            if self.weather_feature not in sub_df.columns:
                continue
            
            # Using 1-day future return shifted by lag k
            ret_1d = sub_df["price_mom_5d"] if "price_mom_5d" in sub_df.columns else sub_df["target_fwd_ret_5d"]
            weather_sig = sub_df[self.weather_feature]
            
            era_corrs = []
            for lag in lags:
                fwd_ret = ret_1d.shift(-lag)
                c = weather_sig.corr(fwd_ret)
                era_corrs.append(round(float(c), 4) if not np.isnan(c) else 0.0)

            corr_data[era_name] = era_corrs

        decay_df = pd.DataFrame(corr_data, index=lags)
        decay_df.index.name = "lag_days"
        return decay_df

