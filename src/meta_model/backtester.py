"""Institutional Quantitative Backtesting Engine.

Simulates strategy execution, calculates transaction costs,
and computes institutional portfolio metrics:
- Information Coefficient (IC)
- Annualized Sharpe Ratio (net of transaction costs)
- Maximum Drawdown (Max DD)
- Annualized Turnover
- Hit Rate / Win Rate
- Calmar Ratio
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class BacktestMetrics:
    model_name: str
    information_coefficient: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    annualized_turnover: float
    hit_rate: float
    calmar_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Model": self.model_name,
            "IC": round(self.information_coefficient, 4),
            "Sharpe": round(self.sharpe_ratio, 2),
            "Max DD": f"{round(self.max_drawdown * 100, 2)}%",
            "Turnover": round(self.annualized_turnover, 2),
            "Return": f"{round(self.annualized_return * 100, 2)}%",
            "Vol": f"{round(self.annualized_volatility * 100, 2)}%",
            "Hit Rate": f"{round(self.hit_rate * 100, 1)}%",
            "Calmar": round(self.calmar_ratio, 2)
        }


class InstitutionalBacktester:
    """Simulates realistic out-of-sample portfolio execution."""

    def __init__(
        self,
        transaction_cost_bps: float = 5.0,  # 5 basis points (0.0005) per unit turnover
        volatility_target: float = 0.15      # 15% annualized target vol
    ):
        self.transaction_cost = transaction_cost_bps / 10000.0
        self.volatility_target = volatility_target

    def compute_positions(
        self,
        predictions: pd.Series,
        realized_vol: Optional[pd.Series] = None
    ) -> pd.Series:
        """Converts raw model return forecasts into bounded target portfolio positions [-1, +1]."""
        # Normalized z-score of predictions
        roll_mean = predictions.rolling(63, min_periods=10).mean().fillna(0.0)
        roll_std = predictions.rolling(63, min_periods=10).std().fillna(1.0)
        pred_z = (predictions - roll_mean) / np.maximum(roll_std, 1e-4)

        # Non-linear squashing via tanh
        raw_pos = np.tanh(pred_z * 1.5)

        # Volatility targeting if realized vol provided
        if realized_vol is not None:
            vol_scalar = self.volatility_target / np.maximum(realized_vol, 0.05)
            pos = raw_pos * np.clip(vol_scalar, 0.3, 2.0)
        else:
            pos = raw_pos

        return np.clip(pos, -1.0, 1.0)

    def run_backtest(
        self,
        predictions: pd.Series,
        realized_returns: pd.Series,
        model_name: str = "Strategy",
        realized_vol: Optional[pd.Series] = None
    ) -> Tuple[BacktestMetrics, pd.DataFrame]:
        """Executes walk-forward backtest accounting for execution delay and trading costs."""
        # Align series
        df = pd.DataFrame({
            "pred": predictions,
            "fwd_ret": realized_returns
        }).dropna()

        if realized_vol is not None:
            vol_aligned = realized_vol.reindex(df.index).fillna(0.15)
        else:
            vol_aligned = None

        # 1. Information Coefficient (IC) - Spearman rank correlation
        ic, _ = stats.spearmanr(df["pred"], df["fwd_ret"])
        if np.isnan(ic):
            ic = 0.0

        # 2. Portfolio Position (Shifted by 1 step to eliminate lookahead)
        positions = self.compute_positions(df["pred"], vol_aligned)
        df["target_position"] = positions
        df["exec_position"] = positions.shift(1).fillna(0.0)

        # 3. Turnover and Transaction Costs
        df["delta_position"] = (df["exec_position"] - df["exec_position"].shift(1)).abs().fillna(0.0)
        df["tx_cost"] = df["delta_position"] * self.transaction_cost

        # 4. Strategy Net Returns
        df["gross_return"] = df["exec_position"] * df["fwd_ret"]
        df["net_return"] = df["gross_return"] - df["tx_cost"]

        # 5. Cumulative Equity Curve
        df["cum_return"] = (1.0 + df["net_return"]).cumprod()
        peak = df["cum_return"].cummax()
        df["drawdown"] = (peak - df["cum_return"]) / peak

        # 6. Summary Metrics
        n_periods = len(df)
        annual_factor = 252.0 / 5.0  # Since fwd_ret is 5-day return
        
        mean_ret = df["net_return"].mean() * annual_factor
        std_ret = df["net_return"].std() * np.sqrt(annual_factor)
        sharpe = mean_ret / max(std_ret, 1e-4)

        max_dd = float(df["drawdown"].max())
        calmar = mean_ret / max(max_dd, 1e-4) if max_dd > 0 else 0.0

        ann_turnover = float(df["delta_position"].sum() * (252.0 / 5.0) / max(1, n_periods))
        hit_rate = float((df["net_return"] > 0).mean())

        metrics = BacktestMetrics(
            model_name=model_name,
            information_coefficient=float(ic),
            annualized_return=float(mean_ret),
            annualized_volatility=float(std_ret),
            sharpe_ratio=float(sharpe),
            max_drawdown=float(max_dd),
            annualized_turnover=float(ann_turnover),
            hit_rate=float(hit_rate),
            calmar_ratio=float(calmar)
        )

        return metrics, df
