"""Feature Engineering: Master Panel Builder.

Aligns weather anomalies, temporal lookbacks, futures prices, and term structure
into a unified, strictly aligned time-series feature matrix.
"""

from typing import Tuple, List, Dict
import pandas as pd
import numpy as np

from src.data_ingestion.usda_weights import get_corn_spatial_config, CommoditySpatialConfig
from src.data_ingestion.weather_fetcher import WeatherFetcher
from src.data_ingestion.price_fetcher import CommodityPriceFetcher
from src.feature_engineering.weather_anomalies import build_production_weighted_weather_panel
from src.feature_engineering.temporal_features import build_temporal_weather_features
from src.feature_engineering.term_structure import compute_term_structure_and_price_features


class PanelDataset:
    """Container holding the aligned feature matrix, feature group lists, and targets."""

    def __init__(
        self,
        df: pd.DataFrame,
        weather_cols: List[str],
        price_cols: List[str],
        term_cols: List[str],
        target_col: str = "target_fwd_ret_5d"
    ):
        self.df = df
        self.weather_cols = weather_cols
        self.price_cols = price_cols
        self.term_cols = term_cols
        self.target_col = target_col

    @property
    def all_features(self) -> List[str]:
        return self.weather_cols + self.price_cols + self.term_cols

    def get_X_y(self, feature_group: str = "all") -> Tuple[pd.DataFrame, pd.Series]:
        """Returns X feature matrix and y target vector for the chosen feature group."""
        if feature_group == "price_only":
            cols = self.price_cols
        elif feature_group == "weather_only":
            cols = self.weather_cols
        elif feature_group == "price_weather":
            cols = self.price_cols + self.weather_cols
        elif feature_group == "price_weather_term":
            cols = self.price_cols + self.weather_cols + self.term_cols
        else:
            cols = self.all_features

        valid_df = self.df.dropna(subset=cols + [self.target_col])
        return valid_df[cols], valid_df[self.target_col]


def build_master_panel(
    spatial_config: CommoditySpatialConfig = None,
    start_date: str = "2000-01-01",
    end_date: str = "2024-12-31",
    force_refresh: bool = False
) -> PanelDataset:
    """Builds complete, aligned time series dataset for quantitative modeling."""
    if spatial_config is None:
        spatial_config = get_corn_spatial_config()

    weather_fetcher = WeatherFetcher()
    raw_weather = weather_fetcher.get_regional_weather_panel(
        config=spatial_config,
        start_date=start_date,
        end_date=end_date,
        force_refresh=force_refresh
    )

    # 1. Production-weighted weather anomalies
    weather_weighted = build_production_weighted_weather_panel(raw_weather, spatial_config)
    
    # 2. Multi-horizon temporal features (7d, 14d, 30d, 60d)
    temporal_weather = build_temporal_weather_features(weather_weighted)

    # 3. Commodity futures prices and term structure
    price_fetcher = CommodityPriceFetcher()
    futures_df = price_fetcher.get_futures_panel(
        ticker=spatial_config.ticker,
        commodity_name=spatial_config.commodity,
        start_date=start_date,
        end_date=end_date,
        weather_df=weather_weighted,
        force_refresh=force_refresh
    )
    price_term_df = compute_term_structure_and_price_features(futures_df)

    # Align on futures trading calendar (business days)
    # Forward fill weekend weather to Monday
    weather_aligned = temporal_weather.reindex(price_term_df.index).ffill()

    # Combine everything
    master_df = pd.concat([price_term_df, weather_aligned], axis=1)

    # Drop warm-up period (first 65 trading days for 60d rolling window)
    master_df = master_df.iloc[65:]

    weather_cols = [c for c in weather_aligned.columns if not c.startswith("target_")]
    price_cols = [c for c in price_term_df.columns if c.startswith("price_") and not c.startswith("target_")]
    term_cols = [c for c in price_term_df.columns if c.startswith("term_") and not c.startswith("target_")]

    return PanelDataset(
        df=master_df,
        weather_cols=weather_cols,
        price_cols=price_cols,
        term_cols=term_cols,
        target_col="target_fwd_ret_5d"
    )

