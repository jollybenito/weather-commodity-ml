"""Unsupervised Regime Detection: Gaussian Hidden Markov Model (HMM) & GMM.

Discovers joint market & weather regimes:
- Regime 0: Low Volatility / Normal Weather / Stable Contango
- Regime 1: Moderate Anomaly / Weather Stress / Elevated Volatility
- Regime 2: Extreme Anomaly / Physical Supply Shock / Severe Backwardation

Provides posterior regime probabilities and evaluates the core ML research question:
'Does the predictive power of the weather signal change depending on the regime?'
"""

from typing import Tuple, Dict, List, Optional
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy import stats


class RegimeDetector:
    """Discovers latent economic and weather regimes using unsupervised probabilistic models."""

    def __init__(
        self,
        n_regimes: int = 3,
        method: str = "hmm",  # 'hmm' or 'gmm'
        random_state: int = 42
    ):
        self.n_regimes = n_regimes
        self.method = method.lower()
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        
        if self.method == "gmm":
            self.model = GaussianMixture(
                n_components=n_regimes,
                covariance_type="full",
                random_state=random_state
            )
        else:
            self.model = GaussianHMM(
                n_components=n_regimes,
                covariance_type="full",
                n_iter=200,
                random_state=random_state
            )

        self.regime_labels: Dict[int, str] = {}
        self.state_order_map: Dict[int, int] = {}

    def _select_regime_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        candidate_cols = [
            "price_vol_21d",
            "term_roll_yield",
            "weighted_extreme_heat_30d_sum",
            "weighted_temp_anomaly_30d_mean",
            "weighted_moisture_deficit_30d_cum"
        ]
        chosen_cols = [c for c in candidate_cols if c in df.columns]
        if not chosen_cols:
            # Fallback to available numeric columns
            chosen_cols = [c for c in df.columns if "vol" in c or "yield" in c or "anomaly" in c][:4]
        
        self.feature_names = chosen_cols
        return df[chosen_cols].fillna(0.0).values, chosen_cols

    def fit(self, df: pd.DataFrame):
        X_vals, _ = self._select_regime_features(df)
        X_scaled = self.scaler.fit_transform(X_vals)
        self.model.fit(X_scaled)

        # Order states by average stress / volatility so Regime 0 is Calm and Regime 2 is Crisis
        if hasattr(self.model, "means_"):
            means = self.model.means_
            # Composite stress score = higher volatility + higher heat + higher roll yield
            stress_scores = np.mean(means, axis=1)
            sorted_indices = np.argsort(stress_scores)
            self.state_order_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_indices)}
        else:
            self.state_order_map = {i: i for i in range(self.n_regimes)}

        self.regime_labels = {
            0: "Regime 0: Low Volatility / Calm / Contango",
            1: "Regime 1: Moderate Weather Stress / Transition",
            2: "Regime 2: High Volatility / Weather Supply Shock / Backwardation"
        }
        return self

    def predict_regimes(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts regime state IDs and posterior state probabilities.
        
        Returns:
            states: [n_samples] array of mapped regime IDs (0, 1, 2)
            proba: [n_samples, n_regimes] posterior probability matrix
        """
        X_vals, _ = self._select_regime_features(df)
        X_scaled = self.scaler.transform(X_vals)

        if self.method == "gmm":
            raw_states = self.model.predict(X_scaled)
            raw_proba = self.model.predict_proba(X_scaled)
        else:
            raw_states = self.model.predict(X_scaled)
            raw_proba = self.model.predict_proba(X_scaled)

        # Map to calibrated ordered regimes
        ordered_states = np.array([self.state_order_map.get(s, s) for s in raw_states])
        
        ordered_proba = np.zeros_like(raw_proba)
        for old_idx, new_idx in self.state_order_map.items():
            ordered_proba[:, new_idx] = raw_proba[:, old_idx]

        return ordered_states, ordered_proba

    def evaluate_regime_conditional_predictability(
        self,
        df: pd.DataFrame,
        weather_signal_col: str,
        target_col: str = "target_fwd_ret_5d"
    ) -> pd.DataFrame:
        """Empirically tests if the predictive power (IC and R^2) of weather signals varies by regime.
        
        Answers: 'Does the predictive power of the weather signal change depending on the regime?'
        """
        states, proba = self.predict_regimes(df)
        sub_df = df[[weather_signal_col, target_col]].copy()
        sub_df["regime"] = states

        results = []
        for r_id in range(self.n_regimes):
            r_data = sub_df[sub_df["regime"] == r_id].dropna()
            label = self.regime_labels.get(r_id, f"Regime {r_id}")
            
            if len(r_data) < 15:
                results.append({
                    "regime_id": r_id,
                    "regime_name": label,
                    "sample_count": len(r_data),
                    "share_pct": round(100.0 * len(r_data) / max(1, len(sub_df)), 1),
                    "ic_spearman": 0.0,
                    "p_value": 1.0,
                    "mean_fwd_return": 0.0,
                    "volatility": 0.0
                })
                continue

            x = r_data[weather_signal_col]
            y = r_data[target_col]

            # Spearman rank correlation (Information Coefficient)
            ic, p_val = stats.spearmanr(x, y)
            
            results.append({
                "regime_id": r_id,
                "regime_name": label,
                "sample_count": len(r_data),
                "share_pct": round(100.0 * len(r_data) / len(sub_df), 1),
                "ic_spearman": round(float(ic), 4) if not np.isnan(ic) else 0.0,
                "p_value": round(float(p_val), 6) if not np.isnan(p_val) else 1.0,
                "mean_fwd_return": round(float(y.mean() * 100.0), 3),
                "volatility": round(float(y.std() * np.sqrt(252 / 5) * 100.0), 2)
            })

        return pd.DataFrame(results)

