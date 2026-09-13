"""Feature Engineering: Commodity Term Structure, Price Momentum & Target Formulation.

Calculates:
- Annualized Roll Yield & Calendar Spreads (Contango vs. Backwardation)
- Price Momentum (5d, 21d, 63d)
- Realized Volatility (Close-to-Close & Parkinson High-Low)
- Mean-Reversion z-score, RSI, MACD
- Target Variables: 5-day and 21-day forward returns
"""

import numpy as np
import pandas as pd


def compute_term_structure_and_price_features(
    futures_df: pd.DataFrame,
    contract_gap_days: float = 60.0
) -> pd.DataFrame:
    """Computes carry, term structure, and technical price signals from futures OHLCV data."""
    df = pd.DataFrame(index=futures_df.index)
    close = futures_df["close"]
    high = futures_df["high"]
    low = futures_df["low"]
    
    # 1. Term Structure: Roll Yield
    if "roll_yield" in futures_df.columns:
        roll_yield = futures_df["roll_yield"]
    elif "second_month" in futures_df.columns:
        f1 = futures_df["front_month"]
        f2 = futures_df["second_month"]
        roll_yield = ((f1 - f2) / f1) * (365.0 / contract_gap_days)
    else:
        roll_yield = pd.Series(0.0, index=close.index)

    df["term_roll_yield"] = roll_yield
    df["term_is_backwardation"] = (roll_yield > 0.0).astype(float)
    df["term_roll_yield_21d_ma"] = roll_yield.rolling(21).mean().fillna(0.0)
    df["term_roll_yield_zscore"] = (
        (roll_yield - roll_yield.rolling(63).mean()) / 
        (roll_yield.rolling(63).std() + 1e-4)
    ).fillna(0.0)

    # 2. Price Momentum Signals
    df["price_mom_5d"] = close.pct_change(5).fillna(0.0)
    df["price_mom_21d"] = close.pct_change(21).fillna(0.0)
    df["price_mom_63d"] = close.pct_change(63).fillna(0.0)
    df["price_mom_126d"] = close.pct_change(126).fillna(0.0)

    # 3. Volatility Signals
    ret_1d = close.pct_change(1).fillna(0.0)
    df["price_vol_21d"] = ret_1d.rolling(21).std().fillna(0.0) * np.sqrt(252.0)
    df["price_vol_63d"] = ret_1d.rolling(63).std().fillna(0.0) * np.sqrt(252.0)

    # Parkinson Volatility (High-Low estimator)
    hl_ratio = np.log(np.maximum(high, close) / np.maximum(1e-4, np.minimum(low, close)))
    parkinson_var = (hl_ratio ** 2) / (4.0 * np.log(2.0))
    df["price_vol_parkinson_21d"] = np.sqrt(parkinson_var.rolling(21).mean().fillna(0.0) * 252.0)

    # 4. Mean-Reversion z-score (Distance from 20d and 50d moving average)
    ma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std() + 1e-4
    df["price_mean_reversion_20d"] = ((close - ma_20) / std_20).fillna(0.0)

    # 5. RSI (14-day)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-6)
    df["price_rsi_14d"] = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0) / 100.0  # Normalized [0, 1]

    # 6. MACD (12d EMA - 26d EMA)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = (ema_12 - ema_26) / close
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df["price_macd"] = macd_line
    df["price_macd_signal"] = signal_line
    df["price_macd_hist"] = macd_line - signal_line

    # 7. Targets (Forward Returns)
    # 5-day forward return: (P_{t+5} - P_t) / P_t
    df["target_fwd_ret_5d"] = close.shift(-5) / close - 1.0
    # 21-day forward return: (P_{t+21} - P_t) / P_t
    df["target_fwd_ret_21d"] = close.shift(-21) / close - 1.0
    # Binary direction targets
    df["target_dir_5d"] = (df["target_fwd_ret_5d"] > 0).astype(float)
    df["target_dir_21d"] = (df["target_fwd_ret_21d"] > 0).astype(float)

    return df

