"""Meta-Model: Adaptive Signal Allocator & Regime-Conditioned Gating Network.

Combines base forecasts:
1. Price ML Model
2. Weather ML Model
3. Term Structure Carry Model
4. Momentum Model

Determines dynamically: 'Which signal should I trust right now?'
- In calm contango regimes, allocates capital primarily to carry and trend.
- In high-stress weather shock regimes, shifts allocation decisively to weather alpha.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class AdaptiveSignalAllocator:
    """Regime-conditioned gating allocator dynamically adjusting signal weights."""

    def __init__(
        self,
        base_weights: Optional[Dict[str, float]] = None,
        regime_multipliers: Optional[Dict[int, Dict[str, float]]] = None
    ):
        # Default static weights
        self.base_weights = base_weights or {
            "price_ml": 0.35,
            "weather_ml": 0.25,
            "term_carry": 0.25,
            "momentum": 0.15
        }
        
        # Multipliers by regime state:
        # Regime 0 (Calm): Carry and momentum dominate; weather is low weight (mostly noise)
        # Regime 1 (Moderate Anomaly): Weather weight increases to 40%
        # Regime 2 (Supply Shock / Crisis): Weather weight surges to 65%, carry flips
        self.regime_multipliers = regime_multipliers or {
            0: {"price_ml": 1.1, "weather_ml": 0.3, "term_carry": 1.4, "momentum": 1.2},
            1: {"price_ml": 0.9, "weather_ml": 1.5, "term_carry": 1.0, "momentum": 0.8},
            2: {"price_ml": 0.6, "weather_ml": 2.5, "term_carry": 0.8, "momentum": 0.5}
        }

    def allocate(
        self,
        signal_df: pd.DataFrame,
        regime_states: np.ndarray,
        regime_proba: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """Computes dynamic allocation weights w_{t, i} for each base signal."""
        signals = ["price_ml", "weather_ml", "term_carry", "momentum"]
        valid_signals = [s for s in signals if s in signal_df.columns]
        
        n_samples = len(signal_df)
        weight_matrix = np.zeros((n_samples, len(valid_signals)))

        for t in range(n_samples):
            if regime_proba is not None:
                # Expectation across posterior regime probabilities
                p = regime_proba[t]
                w_vec = np.zeros(len(valid_signals))
                for r_idx in range(len(p)):
                    multipliers = self.regime_multipliers.get(r_idx, {})
                    r_weights = np.array([self.base_weights.get(s, 0.25) * multipliers.get(s, 1.0) for s in valid_signals])
                    w_vec += p[r_idx] * r_weights
            else:
                # Discrete state
                r_idx = int(regime_states[t])
                multipliers = self.regime_multipliers.get(r_idx, {})
                w_vec = np.array([self.base_weights.get(s, 0.25) * multipliers.get(s, 1.0) for s in valid_signals])

            # Normalize weights to sum to 1.0
            sum_w = np.sum(w_vec)
            if sum_w > 0:
                w_vec /= sum_w
            weight_matrix[t] = w_vec

        weights_df = pd.DataFrame(weight_matrix, index=signal_df.index, columns=[f"weight_{s}" for s in valid_signals])
        return weights_df

    def blend_forecasts(
        self,
        signal_df: pd.DataFrame,
        weights_df: pd.DataFrame
    ) -> pd.Series:
        """Blends signals using dynamic weights: r_pred_t = Sum ( w_{t, i} * s_{t, i} )."""
        signals = [c.replace("weight_", "") for c in weights_df.columns]
        combined = pd.Series(0.0, index=signal_df.index)
        for s in signals:
            if s in signal_df.columns and f"weight_{s}" in weights_df.columns:
                combined += signal_df[s] * weights_df[f"weight_{s}"]
        return combined


class MetaLearner:
    """Supervised meta-model (Stacking Regressor) that learns optimal regime-dependent weights."""

    def __init__(self, alpha: float = 1.0):
        self.meta_model = Ridge(alpha=alpha, fit_intercept=True)
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []

    def fit(
        self,
        base_signals: pd.DataFrame,
        regime_proba: np.ndarray,
        y: pd.Series
    ):
        """Fits meta-learner on base signal predictions multiplied by regime probabilities."""
        self.feature_names = []
        X_meta_list = [base_signals.values]
        self.feature_names.extend(list(base_signals.columns))

        # Add interaction terms between base signals and regime probabilities
        for r in range(regime_proba.shape[1]):
            prob_r = regime_proba[:, r:r+1]
            interact = base_signals.values * prob_r
            X_meta_list.append(interact)
            self.feature_names.extend([f"{c}_regime_{r}" for c in base_signals.columns])

        X_meta = np.hstack(X_meta_list)
        y_vals = y.values

        X_scaled = self.scaler.fit_transform(X_meta)
        self.meta_model.fit(X_scaled, y_vals)
        return self

    def predict(
        self,
        base_signals: pd.DataFrame,
        regime_proba: np.ndarray
    ) -> np.ndarray:
        X_meta_list = [base_signals.values]
        for r in range(regime_proba.shape[1]):
            prob_r = regime_proba[:, r:r+1]
            interact = base_signals.values * prob_r
            X_meta_list.append(interact)

        X_meta = np.hstack(X_meta_list)
        X_scaled = self.scaler.transform(X_meta)
        return self.meta_model.predict(X_scaled)

