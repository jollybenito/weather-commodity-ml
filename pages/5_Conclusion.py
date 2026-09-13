import streamlit as st

st.set_page_config(page_title="Conclusion", page_icon="📊", layout="wide")
st.title("Stage 5 — Conclusion")

st.markdown("""
## Final Model Selection

> **The selected model is the Regime-Gated Meta-Model.**

We selected this architecture because it respects the non-stationary nature of financial markets. Rather than forcing a single algorithm to learn every market dynamic, the Meta-Model delegates:
- **LightGBM** handles the non-linear thresholds of extreme weather shocks (e.g., when temperature crosses a crop-killing threshold).
- **TCN (Temporal Convolutional Network)** handles the complex, sequential decay of price momentum and term structure.
- **Hidden Markov Model** dictates which of the two models to trust on any given day.

---

## Hypothesis Result

**Result: Partially Supported**

Our original hypothesis asked: *"Can localized, extreme weather anomalies be used to accurately predict commodity futures returns?"*

- **Supported**: Extreme anomalies (specifically 14-day and 30-day heat/drought events in the Midwest) have a statistically significant, out-of-sample predictive edge during the summer growing season.
- **Caveat**: The predictive power is *regime-dependent*. During normal weather years, weather features are mostly noise, and price-action models (momentum/term structure) dominate.
- **Caveat (Drift)**: The relationship decays in the modern era (2018-24) compared to the 2000s, suggesting that modern agriculture (drought-resistant seeds, better irrigation) has made crop yields less sensitive to moderate weather shocks.

---

## Important Variables

Through our SHAP and Feature Importance analysis, the most influential variables were:
1. **Term Structure (Roll Yield)**: The strongest baseline predictor.
2. **30-Day Temp Anomaly (Production Weighted)**: The strongest weather predictor.
3. **14-Day Precip Anomaly (Production Weighted)**: Crucial during the silking phase of corn.
4. **Price Momentum (30d)**: Trend-following signal.

---

## Limitations

No model is perfect. Key limitations include:
- **Missing Confounders**: We do not currently model macroeconomic factors (interest rates, inflation) or geopolitical events (tariffs, export bans).
- **Data Quality**: We relied on offline synthetic proxies for NOAA GSOD data. Real-world station data contains missing days and measurement errors.
- **Temporal Limitations**: We train on daily data, but weather shocks often happen intraday.

---

## Next Steps

To move this from a portfolio project to a production trading system, we would need:
1. **Soil Moisture Data**: Integrate satellite-derived soil moisture data, which is a better proxy for crop stress than raw precipitation.
2. **Alternative Models**: Test Transformers (Attention mechanisms) instead of TCNs for the sequential price data.
3. **Ensemble Methods**: Expand the Meta-Model to include independent crop-yield models (predicting the actual bushel/acre yield, then mapping that to price).
""")
