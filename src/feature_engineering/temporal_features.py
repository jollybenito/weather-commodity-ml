"""Feature Engineering: Multi-Horizon Temporal Weather Signals.

Constructs rolling lookback features across tradeable windows:
- 7-day history (short-term shock / heatwave impact)
- 14-day history (bi-weekly weather pattern persistence)
- 30-day history (monthly soil moisture accumulation / drought)
- 60-day history (seasonal crop development window)
- Crop phenology / critical pollination window interactions (July-August)
"""

from typing import List
import numpy as np
import pandas as pd


def build_temporal_weather_features(
    weather_panel: pd.DataFrame,
    windows: List[int] = [7, 14, 30, 60]
) -> pd.DataFrame:
    """Computes multi-horizon rolling aggregations for production-weighted weather metrics.

    Features generated:
    - `{metric}_{w}d_mean`: Rolling average anomaly
    - `{metric}_{w}d_sum`: Cumulative stress (e.g. total extreme heat days in past w days)
    - `{metric}_{w}d_std`: Weather volatility / instability
    - `{metric}_{w}d_zscore`: Current anomaly vs recent lookback window
    - Seasonal pollination interaction: Pollination window (DOY 180-240) multiplier
    """
    df = pd.DataFrame(index=weather_panel.index)
    
    # Focus primary temporal aggregations on the production-weighted series
    base_metrics = [
        "weighted_temp_anomaly",
        "weighted_precip_anomaly",
        "weighted_moisture_deficit",
        "weighted_extreme_heat",
        "weighted_extreme_rain",
        "weighted_gdd"
    ]

    # Crop phenology: July 1 to August 31 (DOY 182 to 243) is the critical silking/tasseling phase
    doy = weather_panel.index.dayofyear
    is_pollination = ((doy >= 182) & (doy <= 243)).astype(float)
    df["season_is_pollination"] = is_pollination
    df["season_is_summer"] = ((doy >= 152) & (doy <= 273)).astype(float)

    for metric in base_metrics:
        if metric not in weather_panel.columns:
            continue
        
        series = weather_panel[metric]
        df[f"{metric}_level"] = series

        for w in windows:
            roll = series.rolling(window=w, min_periods=max(2, w // 3))
            
            # Rolling mean
            roll_mean = roll.mean()
            df[f"{metric}_{w}d_mean"] = roll_mean
            
            # Cumulative stress for count-like metrics (heat, rain)
            if "extreme" in metric:
                df[f"{metric}_{w}d_sum"] = roll.sum()
            elif "deficit" in metric or "precip" in metric:
                df[f"{metric}_{w}d_cum"] = roll.sum()
            
            # Weather volatility
            df[f"{metric}_{w}d_std"] = roll.std().fillna(0.0)

        # Rate of change / acceleration (e.g. 7d shock vs 30d background)
        df[f"{metric}_shock_7v30"] = df[f"{metric}_7d_mean"] - df[f"{metric}_30d_mean"]
        df[f"{metric}_shock_14v60"] = df[f"{metric}_14d_mean"] - df[f"{metric}_60d_mean"]

        # Critical seasonal interaction: Heat stress during pollination creates asymmetric crop damage!
        if "extreme_heat" in metric or "temp_anomaly" in metric:
            df[f"{metric}_pollination_stress_30d"] = df[f"{metric}_30d_mean"] * is_pollination
            df[f"{metric}_pollination_stress_14d"] = df[f"{metric}_14d_mean"] * is_pollination

    return df

