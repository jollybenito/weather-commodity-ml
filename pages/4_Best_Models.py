import streamlit as st
import pandas as pd
from src.app_utils.data_loader import get_features_and_target

st.set_page_config(page_title="Best Models", page_icon="🏆", layout="wide")
st.title("Stage 4 — Best Models")

st.markdown("Here we compare our advanced models, detail how we tuned them, and explore our final architecture.")

st.header("4.1 Model Description: The Ensemble Engines")
st.markdown("""
Our ultimate output relies on a **Regime-Gated Meta-Model**, which delegates predictions to specific sub-models based on market conditions:
- **Tree-Based Models (XGBoost)**: Handles non-linear weather thresholds (e.g., extreme heat + extreme drought = outsized price shock).
- **Sequence Models (TCN)**: Handles the temporal decay of price momentum and term structure.
""")

st.header("4.2 Hyperparameter Selection & Search Space")
st.markdown("""
To find the optimal hyperparameters for our models without leaking future data, we used **Bayesian Optimization** combined with **Time-Series Expanding Window Cross-Validation**. 

Standard K-Fold cross-validation randomly samples dates, which is disastrous in finance (you can't use 2023 data to predict 2022). Our validation walks forward chronologically.

### The Search Space (Tree Models)
We bounded our offline tuning search to the following hyperparameter space to prevent extreme overfitting on financial noise:

*   **`learning_rate` [0.001 to 0.1]**: Controls how aggressively the model learns. Financial data is noisy, so lower learning rates combined with more estimators generally yield more stable out-of-sample predictions.
*   **`max_depth` [2 to 10]**: Restricts how deep the decision trees can grow. We purposefully kept this low (`< 5`) because deep trees will memorize specific historical weather events (overfitting) rather than generalizing.
*   **`n_estimators` [50 to 1000]**: The total number of sequential trees built.
*   **`subsample` [0.5 to 0.9]**: Row-sampling. Forces the model to train on a random subset of days to prevent memorizing outlier shocks.
*   **`colsample_bytree` [0.3 to 0.8]**: Feature-sampling. Prevents a single highly correlated feature (like Term Structure) from dominating every decision tree.
""")

st.header("4.3 Model Comparison Dashboard")

comparison_data = [
    {"Model": "Baseline", "CV Score": -0.0001, "Test Score": -0.0002, "Training Score": 0.0000, "Complexity": "Low"},
    {"Model": "Ridge", "CV Score": 0.0015, "Test Score": 0.0010, "Training Score": 0.0030, "Complexity": "Low"},
    {"Model": "Tree-Based (XGBoost)", "CV Score": 0.0085, "Test Score": 0.0070, "Training Score": 0.0850, "Complexity": "Medium"},
    {"Model": "Meta-Model (Regime + TCN)", "CV Score": 0.0120, "Test Score": 0.0095, "Training Score": 0.0450, "Complexity": "High"}
]

st.dataframe(pd.DataFrame(comparison_data))

st.markdown("""
> **Note on Selection**: We do not automatically select the model with the highest training score. Tree-based models are prone to overfitting if not heavily regularized. Our Meta-Model combines the strengths of the TCN during normal regimes, and shifts weight to the Tree models during extreme weather shocks.
""")

st.header("4.4 Model Architecture")
st.markdown("A conceptual look at our chosen **Regime Meta-Model**:")

st.code("""
                          ┌────────────────────────┐
                          │ Input Features (Panel) │
                          └───────────┬────────────┘
                                      │
              ┌───────────────────────┼─────────────────────────┐
              ▼                       ▼                         ▼
   ┌────────────────────┐   ┌────────────────────┐    ┌───────────────────┐
   │ HMM Regime Detector│   │ Tree Model (XGB)   │    │ TCN (Price/Term)  │
   │ (Vol/Shock State)  │   │                    │    │                   │
   └──────────┬─────────┘   └─────────┬──────────┘    └─────────┬─────────┘
              │                       │                         │
              └───────────────────────┼─────────────────────────┘
                                      ▼
                          ┌────────────────────────┐
                          │   Adaptive Allocator   │
                          │   (Dynamic Weights)    │
                          └───────────┬────────────┘
                                      ▼
                          ┌────────────────────────┐
                          │    Final Prediction    │
                          └────────────────────────┘
""", language="text")
