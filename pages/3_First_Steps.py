import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.app_utils.data_loader import get_features_and_target

st.set_page_config(page_title="First Steps", page_icon="🧪", layout="wide")
st.title("Stage 3 — First Steps")

st.markdown("Before deploying complex Neural Networks, we must establish rigorous baselines and validate our methodology.")

st.header("3.1 Train/Test Methodology")
st.markdown("""
For financial time-series data, **we cannot randomly shuffle observations**. Random shuffling leaks future information into the training set.
Instead, we use a strict chronological split (Walk-Forward or Era-based).

```text
Past ───────────────────────────────► Future

Train          Validation       Test
██████████████ ███████           █████
```
""")

panel, features, target = get_features_and_target()

# Chronological split
train_idx = int(len(panel) * 0.7)
val_idx = int(len(panel) * 0.85)

train = panel.iloc[:train_idx]
val = panel.iloc[train_idx:val_idx]
test = panel.iloc[val_idx:]

st.write(f"**Training Set**: {len(train)} days")
st.write(f"**Validation Set**: {len(val)} days")
st.write(f"**Test Set**: {len(test)} days")

st.header("3.2 Model Families: Why we chose them")
st.markdown("""
Agricultural commodities are influenced by complex, non-linear weather interactions. We selected specific model families to target different types of signals:

1. **Linear Models (Ridge Regression)**: 
   - *Why*: Establishes a rigorous baseline. Weather features are often highly collinear (e.g., 7-day temp and 14-day temp). Ridge regression penalizes large coefficients, preventing multicollinearity from breaking the model.
2. **Tree-Based (LightGBM & XGBoost)**: 
   - *Why*: Crops don't die linearly. A 35°C day is infinitely more damaging than a 30°C day. Tree-based models naturally discover these **non-linear thresholds** and feature interactions (e.g., "if temperature > 35C AND precipitation < 2mm, then yield drops").
3. **Neural Networks (Temporal Convolutional Networks - TCN)**: 
   - *Why*: Sequence models remember the past. A drought on day 1 is bad, but a drought lasting 30 consecutive days causes compounding damage. TCNs capture this temporal decay and memory better than tabular tree models.
""")

st.header("3.3 The Ensemble Architecture (Meta-Model)")
st.markdown("""
Rather than choosing a single "best" model, we acknowledge that financial markets operate in different **regimes** (e.g., normal low-volatility conditions vs. panic-driven weather shocks). 

To handle this, we construct a **Regime-Gated Ensemble Meta-Model**.

### How the Ensemble is Structured:
1. **The Base Learners**: We train the Tree-based model (LightGBM) purely on weather data, and the Neural Network (TCN) purely on price momentum and term structure.
2. **The Regime Detector (HMM)**: We use an unsupervised Hidden Markov Model (HMM) to constantly monitor market volatility and classify the current environment as either "Normal" or "Shock".
3. **The Adaptive Allocator**: Instead of a simple average, a meta-layer dynamically shifts weight between the base models. 
   - During a *Normal Regime*, it assigns 80% weight to the Price/TCN model.
   - During a *Shock Regime* (drought/heatwave), it shifts 80% weight to the Weather/LightGBM model.
""")

st.info("By ensembling these models through a regime lens, the system ignores weather noise during the winter, but hypersensitizes to it during the summer growing season.")

st.header("3.4 Preliminary Baselines Evaluation")

with st.spinner("Training Baselines..."):
    X_train, y_train = train[features], train[target]
    X_val, y_val = val[features], val[target]
    X_test, y_test = test[features], test[target]
    
    # Baseline (Mean Predictor)
    mean_pred = np.mean(y_train)
    base_preds = np.full(len(y_val), mean_pred)
    
    # Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_preds = ridge.predict(X_val)

    def get_metrics(y_true, y_pred):
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R²": r2_score(y_true, y_pred)
        }

    results = pd.DataFrame([
        {"Model": "Baseline (Mean)", **get_metrics(y_val, base_preds)},
        {"Model": "Ridge Regression", **get_metrics(y_val, ridge_preds)}
    ])

st.dataframe(results)

st.markdown("""
The Ridge model barely edges out the baseline. This is common in financial data! Linear relationships are usually arbitraged away. To capture the actual weather shocks, we need the ensemble and threshold models detailed above.
""")
