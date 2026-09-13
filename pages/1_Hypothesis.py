import streamlit as st

st.set_page_config(page_title="Hypothesis", page_icon="📝", layout="wide")

st.title("Stage 1 — Hypothesis")

st.markdown("""
## Main Question

> **"Can localized, extreme weather anomalies affecting major agricultural production regions be used to accurately predict commodity futures returns?"**

The company's strategy relies on capitalizing on weather-driven commodity dislocations. In this project, we specifically test this hypothesis using **Corn (ZC)** as the primary asset, driven by US Midwest weather patterns.
""")

st.markdown("---")
st.markdown("""
## Available Data

To test our hypothesis, we ingest data from multiple sources to build a daily panel spanning the last ~24 years.

- **Observations**: Daily historical data (roughly 6000 trading days from 2000 to present)
- **Features**: Dozens of constructed signals, including temporal weather rollups (7d, 14d, 30d, 60d), momentum, and term structure
- **Target Variable**: Next-day and forward Corn Futures Returns
- **Data Sources**: 
    - Weather: Synthetic offline placeholders (simulating NOAA GSOD)
    - Pricing: Synthetic offline placeholders (simulating CME futures via yfinance)
    - Production Weights: Synthetic USDA state-level production ratios
- **Units**: Weather anomalies are measured in standard deviations (Z-scores) from 30-year rolling climatological normals. Returns are measured in log-returns.
""")

st.markdown("---")
st.markdown("""
## The Formal Hypothesis

> **"Changes in geographically-weighted weather anomalies (temperature and precipitation) have a measurable, predictive relationship with the target variable (corn futures returns), and this relationship's strength is mediated by the current market regime."**

### Variable Breakdown:
- **Independent Variables (Features)**:
  - $WeatherAnomaly_{t}$ (Production-weighted Z-scores)
  - Term Structure Roll Yield
  - Price Momentum
- **Dependent Variable (Target)**:
  - $ForwardReturn_{t+1}$
- **Confounding Variables** (To be aware of):
  - Macroeconomic shifts (inflation, interest rates)
  - Geopolitical supply shocks (e.g., Black Sea blockades)
- **Temporal Variables**:
  - Era splits (2000-09, 2010-17, 2018-24) to track data drift.
""")
