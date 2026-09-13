"""Feature Engineering: Production-Weighted Weather Anomalies.

Calculates agronomically meaningful weather metrics:
- Growing Degree Days (GDD, base 50F cap 86F)
- Standardized Climatological Temperature Anomalies (z-scores by day of year)
- Cumulative Precipitation Deficits / Surpluses
- Extreme Heat Days (>35C / 95F, critical pollination threshold)
- Extreme Rainfall Days (>95th percentile events)
- Drought & Moisture Stress Index (P - PET deficit)
- Geographically weighted production exposure across US Corn Belt states
"""

from typing import List, Dict, Optional
import numpy as np
import pandas as pd

from src.data_ingestion.usda_weights import CommoditySpatialConfig, compute_spatially_weighted_weather


def celsius_to_fahrenheit(c: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    return c * 9.0 / 5.0 + 32.0


def calculate_gdd(t_max_c: pd.Series, t_min_c: pd.Series) -> pd.Series:
    """Calculate agricultural Growing Degree Days (GDD) for corn.
    
    Standard US Corn GDD formula (base 50F, cap 86F):
    T_max_adj = min(max(T_max, 50), 86)
    T_min_adj = min(max(T_min, 50), 86)
    GDD = max(0, (T_max_adj + T_min_adj)/2 - 50)
    """
    t_max_f = celsius_to_fahrenheit(t_max_c)
    t_min_f = celsius_to_fahrenheit(t_min_c)

    t_max_adj = np.clip(t_max_f, 50.0, 86.0)
    t_min_adj = np.clip(t_min_f, 50.0, 86.0)
    
    gdd = (t_max_adj + t_min_adj) / 2.0 - 50.0
    return pd.Series(np.maximum(0.0, gdd), index=t_max_c.index)


def compute_climatological_anomalies(
    series: pd.Series,
    window_doy: int = 15
) -> pd.Series:
    """Computes standardized climatological anomalies (z-scores) relative to the day-of-year mean and std."""
    df = pd.DataFrame({"val": series, "doy": series.index.dayofyear})
    # Smooth DOY climatology
    doy_stats = df.groupby("doy")["val"].agg(["mean", "std"]).reset_index()
    
    # Rolling smoothing of daily normals across DOY cycle
    doy_stats["mean_smooth"] = (
        pd.concat([doy_stats["mean"], doy_stats["mean"], doy_stats["mean"]])
        .rolling(window_doy, center=True, min_periods=3)
        .mean()
        .iloc[366:366 + len(doy_stats)]
        .values
    )
    doy_stats["std_smooth"] = (
        pd.concat([doy_stats["std"], doy_stats["std"], doy_stats["std"]])
        .rolling(window_doy, center=True, min_periods=3)
        .mean()
        .iloc[366:366 + len(doy_stats)]
        .values
    )
    doy_stats["std_smooth"] = np.maximum(doy_stats["std_smooth"], 1e-3)

    merged = df.merge(doy_stats[["doy", "mean_smooth", "std_smooth"]], on="doy", how="left")
    anomaly = (merged["val"].values - merged["mean_smooth"].values) / merged["std_smooth"].values
    return pd.Series(anomaly, index=series.index)


def compute_state_weather_features(
    state_df: pd.DataFrame,
    state_code: str
) -> pd.DataFrame:
    """Computes all agronomic weather metrics for a single state."""
    features = pd.DataFrame(index=state_df.index)
    
    t_max = state_df[f"{state_code}_temp_max"]
    t_min = state_df[f"{state_code}_temp_min"]
    t_mean = state_df[f"{state_code}_temp_mean"]
    precip = state_df[f"{state_code}_precip"]
    et0 = state_df.get(f"{state_code}_evapotranspiration", t_max * 0.1)

    # 1. GDD
    features[f"{state_code}_gdd"] = calculate_gdd(t_max, t_min)

    # 2. Temperature Anomalies (z-score vs seasonal norm)
    features[f"{state_code}_temp_anomaly"] = compute_climatological_anomalies(t_mean)

    # 3. Precipitation Anomaly & Moisture Deficit (P - ET0)
    features[f"{state_code}_precip_anomaly"] = compute_climatological_anomalies(precip)
    moisture_balance = precip - et0
    features[f"{state_code}_moisture_deficit"] = compute_climatological_anomalies(moisture_balance)

    # 4. Extreme Heat Days (T_max > 35C / 95F)
    features[f"{state_code}_extreme_heat"] = (t_max >= 35.0).astype(float)

    # 5. Extreme Rainfall Days (precip > 25mm / 1 inch)
    features[f"{state_code}_extreme_rain"] = (precip >= 25.0).astype(float)

    return features


def build_production_weighted_weather_panel(
    regional_weather_df: pd.DataFrame,
    spatial_config: CommoditySpatialConfig
) -> pd.DataFrame:
    """Builds full panel of state-level and geographically production-weighted weather anomalies.
    
    Implements:
        Weather_t = Sum_{r} (ProductionShare_r * WeatherMetric_{r,t})
    """
    all_state_features = []
    
    for code in spatial_config.regions.keys():
        if f"{code}_temp_max" in regional_weather_df.columns:
            state_feats = compute_state_weather_features(regional_weather_df, code)
            all_state_features.append(state_feats)

    state_features_df = pd.concat(all_state_features, axis=1)

    # Compute production-weighted composites
    core_metrics = [
        "gdd",
        "temp_anomaly",
        "precip_anomaly",
        "moisture_deficit",
        "extreme_heat",
        "extreme_rain"
    ]

    weighted_df = pd.DataFrame(index=regional_weather_df.index)
    weights = spatial_config.get_weights_series()

    for metric in core_metrics:
        weighted_val = pd.Series(0.0, index=regional_weather_df.index)
        w_sum = 0.0
        for code, w in weights.items():
            col = f"{code}_{metric}"
            if col in state_features_df.columns:
                weighted_val += state_features_df[col].fillna(0.0) * w
                w_sum += w
        if w_sum > 0:
            weighted_df[f"weighted_{metric}"] = weighted_val / w_sum

    # Merge individual state features and weighted composites
    complete_panel = pd.concat([weighted_df, state_features_df], axis=1)
    return complete_panel

