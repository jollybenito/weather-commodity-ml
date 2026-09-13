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

st.header("3.2 Candidate Models")
st.markdown("""
We consider several model families:
1. **Baseline**: Always predict the mean of the training set.
2. **Linear Models**: Ridge Regression (to avoid multicollinearity among weather variables).
3. **Tree-Based**: LightGBM and XGBoost (can handle non-linear thresholds like 'if temp > 30C AND rain < 10mm').
4. **Neural Networks**: TCN and LSTM (for complex temporal sequences).
""")

st.header("3.3 Preliminary Baselines Evaluation")

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
The Ridge model barely edges out the baseline. This is common in financial data! Linear relationships are usually arbitraged away. To capture the actual weather shocks, we need models that understand *thresholds* and *regimes*.
""")
