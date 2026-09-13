"""USDA Crop Production Shares and Spatial Exposure Matrix.

Provides state-level production weights and geographic coordinates
for major agricultural commodity producing regions (Corn Belt, Wheat Belt, Soybean Belt).
"""

from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np
import pandas as pd


@dataclass
class RegionInfo:
    code: str
    name: str
    lat: float
    lon: float
    weight: float


@dataclass
class CommoditySpatialConfig:
    commodity: str
    ticker: str
    regions: Dict[str, RegionInfo] = field(default_factory=dict)

    def __post_init__(self):
        # Normalize weights so they strictly sum to 1.0
        total_w = sum(r.weight for r in self.regions.values())
        if not np.isclose(total_w, 1.0, atol=1e-4):
            for r in self.regions.values():
                r.weight = r.weight / total_w

    def get_weights_series(self) -> pd.Series:
        return pd.Series({code: r.weight for code, r in self.regions.items()})

    def get_coordinates(self) -> Dict[str, tuple[float, float]]:
        return {code: (r.lat, r.lon) for code, r in self.regions.items()}


def get_corn_spatial_config() -> CommoditySpatialConfig:
    """Return production weights and coordinates for major US Corn Belt states based on USDA NASS."""
    regions = {
        "IA": RegionInfo("IA", "Iowa", 41.8780, -93.0977, 0.180),
        "IL": RegionInfo("IL", "Illinois", 40.6331, -89.3985, 0.155),
        "NE": RegionInfo("NE", "Nebraska", 41.4925, -99.9018, 0.125),
        "MN": RegionInfo("MN", "Minnesota", 46.7296, -94.6859, 0.100),
        "IN": RegionInfo("IN", "Indiana", 40.2672, -86.1349, 0.075),
        "OH": RegionInfo("OH", "Ohio", 40.4173, -82.9071, 0.055),
        "SD": RegionInfo("SD", "South Dakota", 43.9695, -99.9018, 0.050),
        "KS": RegionInfo("KS", "Kansas", 39.0119, -98.4842, 0.045),
        "MO": RegionInfo("MO", "Missouri", 37.9643, -91.8318, 0.045),
        "WI": RegionInfo("WI", "Wisconsin", 43.7844, -88.7879, 0.040),
    }
    return CommoditySpatialConfig(
        commodity="Corn",
        ticker="ZC=F",
        regions=regions
    )


def get_soybeans_spatial_config() -> CommoditySpatialConfig:
    """Return production weights and coordinates for major US Soybean states based on USDA NASS."""
    regions = {
        "IL": RegionInfo("IL", "Illinois", 40.6331, -89.3985, 0.180),
        "IA": RegionInfo("IA", "Iowa", 41.8780, -93.0977, 0.165),
        "MN": RegionInfo("MN", "Minnesota", 46.7296, -94.6859, 0.095),
        "IN": RegionInfo("IN", "Indiana", 40.2672, -86.1349, 0.085),
        "NE": RegionInfo("NE", "Nebraska", 41.4925, -99.9018, 0.075),
        "MO": RegionInfo("MO", "Missouri", 37.9643, -91.8318, 0.070),
        "OH": RegionInfo("OH", "Ohio", 40.4173, -82.9071, 0.065),
        "SD": RegionInfo("SD", "South Dakota", 43.9695, -99.9018, 0.055),
        "ND": RegionInfo("ND", "North Dakota", 47.5515, -101.0020, 0.050),
    }
    return CommoditySpatialConfig(
        commodity="Soybeans",
        ticker="ZS=F",
        regions=regions
    )


def get_wheat_spatial_config() -> CommoditySpatialConfig:
    """Return production weights and coordinates for major US Wheat states based on USDA NASS."""
    regions = {
        "KS": RegionInfo("KS", "Kansas", 39.0119, -98.4842, 0.220),
        "ND": RegionInfo("ND", "North Dakota", 47.5515, -101.0020, 0.180),
        "MT": RegionInfo("MT", "Montana", 46.8797, -110.3626, 0.110),
        "WA": RegionInfo("WA", "Washington", 47.7511, -120.7401, 0.100),
        "OK": RegionInfo("OK", "Oklahoma", 35.0078, -97.0929, 0.075),
        "TX": RegionInfo("TX", "Texas", 31.9686, -99.9018, 0.065),
        "CO": RegionInfo("CO", "Colorado", 39.5501, -105.7821, 0.055),
    }
    return CommoditySpatialConfig(
        commodity="Wheat",
        ticker="ZW=F",
        regions=regions
    )


def compute_spatially_weighted_weather(
    regional_weather_df: pd.DataFrame,
    spatial_config: CommoditySpatialConfig,
    metric_cols: list[str]
) -> pd.DataFrame:
    """Computes geographically weighted weather exposure across production regions.

    Formula:
        W_t = Sum_{r} (ProductionShare_r * WeatherMetric_{r,t})

    Parameters
    ----------
    regional_weather_df : pd.DataFrame
        DataFrame indexed by date, with columns named `{region_code}_{metric}`
        e.g., 'IA_temp_max', 'IL_precip_sum', etc.
    spatial_config : CommoditySpatialConfig
        Config containing state weights.
    metric_cols : list[str]
        List of base metrics, e.g. ['temp_max', 'temp_min', 'temp_mean', 'precip', 'gdd']

    Returns
    -------
    pd.DataFrame
        DataFrame with weighted columns, e.g. 'weighted_temp_max', 'weighted_precip', etc.
    """
    weights = spatial_config.get_weights_series()
    result = pd.DataFrame(index=regional_weather_df.index)

    for metric in metric_cols:
        weighted_series = pd.Series(0.0, index=regional_weather_df.index)
        valid_weight_sum = 0.0
        
        for code, w in weights.items():
            col_name = f"{code}_{metric}"
            if col_name in regional_weather_df.columns:
                weighted_series += regional_weather_df[col_name].fillna(method="ffill") * w
                valid_weight_sum += w
        
        if valid_weight_sum > 0:
            result[f"weighted_{metric}"] = weighted_series / valid_weight_sum
        else:
            result[f"weighted_{metric}"] = np.nan

    return result

