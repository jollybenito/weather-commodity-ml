"""Data Splitting Framework: Macro-Technological Eras and Purged Walk-Forward Cross-Validation.

Splits data according to:
1. Macro-Technological Eras:
   - Era 1 (2000-2009): Pre-modern precision ag, legacy USDA weekly reports, floor pit transition.
   - Era 2 (2010-2017): Satellite remote sensing adoption, 2012 mega-drought, rise of systematic CTAs.
   - Era 3 (2018-2024): Modern drought-tolerant GMO hybrids (DroughtGard), high-res satellite NDVI, algo weather trading.
2. Purged Walk-Forward CV:
   - Expanding and rolling windows with strict embargo periods to eliminate lookahead bias from forward targets.
"""

from dataclasses import dataclass
from typing import List, Tuple, Generator, Optional
import pandas as pd
import numpy as np


@dataclass
class EraDefinition:
    name: str
    label: str
    start_date: str
    end_date: str
    description: str


ERA_DEFINITIONS = [
    EraDefinition(
        name="era_1",
        label="Era 1: Pre-Modern Precision Ag (2000-2009)",
        start_date="2000-01-01",
        end_date="2009-12-31",
        description="Baseline crop genetics; reliance on delayed USDA weekly progress reports; floor pit to early electronic trading."
    ),
    EraDefinition(
        name="era_2",
        label="Era 2: Satellite Adoption & Benchmark Drought (2010-2017)",
        start_date="2010-01-01",
        end_date="2017-12-31",
        description="Early commercial satellite imagery; historic 2012 Midwest mega-drought benchmark; emergence of systematic CTAs."
    ),
    EraDefinition(
        name="era_3",
        label="Era 3: Modern Precision Ag & Algo Weather Trading (2018-2024)",
        start_date="2018-01-01",
        end_date="2024-12-31",
        description="High-resolution satellite NDVI (Planet, Sentinel); drought-resistant GMO hybrids; real-time algorithmic weather execution."
    ),
]


class TechnologicalEraSplitter:
    """Splits time series data into macroeconomic and agronomic technological eras."""

    def __init__(self, eras: List[EraDefinition] = ERA_DEFINITIONS):
        self.eras = eras

    def split_by_eras(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Splits DataFrame into era-specific subsets indexed by era name."""
        subsets = {}
        for era in self.eras:
            mask = (df.index >= era.start_date) & (df.index <= era.end_date)
            sub_df = df.loc[mask]
            if not sub_df.empty:
                subsets[era.name] = sub_df
        return subsets

    def get_era_for_date(self, dt: pd.Timestamp) -> str:
        """Returns the era name for a given timestamp."""
        dt_str = dt.strftime("%Y-%m-%d")
        for era in self.eras:
            if era.start_date <= dt_str <= era.end_date:
                return era.name
        return "unknown"


class PurgedWalkForwardCV:
    """Purged Walk-Forward Cross-Validation with Embargo.
    
    Ensures that for a target with horizon H (e.g. 5 or 21 days),
    an embargo of at least H days is placed between the end of the training set
    and the start of the validation/test set to strictly eliminate lookahead leakage.
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_days: int = 21,
        rolling: bool = False,
        rolling_window_days: int = 252 * 5  # 5 years rolling
    ):
        self.n_splits = n_splits
        self.embargo_days = embargo_days
        self.rolling = rolling
        self.rolling_window_days = rolling_window_days

    def split(
        self,
        df: pd.DataFrame
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Yields (train_indices, test_indices) for each walk-forward fold."""
        n = len(df)
        indices = np.arange(n)
        
        # Reserve first 30% for minimum initial training window
        min_train_size = int(n * 0.35)
        test_size = (n - min_train_size) // self.n_splits

        for i in range(self.n_splits):
            test_start = min_train_size + i * test_size
            test_end = min_train_size + (i + 1) * test_size if i < self.n_splits - 1 else n
            
            # Apply embargo: training set must end at least embargo_days before test_start
            train_end = max(0, test_start - self.embargo_days)
            
            if self.rolling:
                train_start = max(0, train_end - self.rolling_window_days)
            else:
                train_start = 0

            train_idx = indices[train_start:train_end]
            test_idx = indices[test_start:test_end]

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

