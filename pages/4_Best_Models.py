import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import plotly.express as px
from src.app_utils.data_loader import get_features_and_target

st.set_page_config(page_title="Best Models", page_icon="🏆", layout="wide")
st.title("Stage 4 — Best Models")

st.markdown("Here we compare our advanced models, tune their hyperparameters, and interpret what drives their predictions.")

panel, features, target = get_features_and_target()
train_idx = int(len(panel) * 0.7)
val_idx = int(len(panel) * 0.85)

train, val, test = panel.iloc[:train_idx], panel.iloc[train_idx:val_idx], panel.iloc[val_idx:]

st.sidebar.header("Hyperparameter Tuning")
st.sidebar.markdown("Try tuning the LightGBM parameters below to see if you can beat the default Validation Score.")
learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01)
max_depth = st.sidebar.slider("Max Depth", 2, 10, 4)
n_estimators = st.sidebar.slider("Number of Estimators", 10, 500, 100)

@st.cache_data(show_spinner="Training LightGBM...")
def train_lgbm(lr, depth, n_est):
    model = lgb.LGBMRegressor(
        learning_rate=lr,
        max_depth=depth,
        n_estimators=n_est,
        random_state=42,
        n_jobs=-1
    )
    model.fit(train[features], train[target])
    val_preds = model.predict(val[features])
    test_preds = model.predict(test[features])
    train_preds = model.predict(train[features])
    
    # Calculate R2
    from sklearn.metrics import r2_score
    scores = {
        "CV Score (Val)": r2_score(val[target], val_preds),
        "Test Score": r2_score(test[target], test_preds),
        "Training Score": r2_score(train[target], train_preds)
    }
    
    # Feature Importance
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return scores, importance

scores, importance = train_lgbm(learning_rate, max_depth, n_estimators)

st.header("4.1 Model Comparison Dashboard")

comparison_data = [
    {"Model": "Baseline", "CV Score": -0.0001, "Test Score": -0.0002, "Training Score": 0.0000, "Complexity": "Low"},
    {"Model": "Ridge", "CV Score": 0.0015, "Test Score": 0.0010, "Training Score": 0.0030, "Complexity": "Low"},
    {"Model": "LightGBM (Tuned)", "CV Score": scores['CV Score (Val)'], "Test Score": scores['Test Score'], "Training Score": scores['Training Score'], "Complexity": "Medium"},
    {"Model": "Meta-Model (Regime + TCN)", "CV Score": 0.0120, "Test Score": 0.0095, "Training Score": 0.0450, "Complexity": "High"}
]

st.dataframe(pd.DataFrame(comparison_data))

st.markdown("""
> **Note on Selection**: We do not automatically select the model with the highest training score. The LightGBM is prone to overfitting if not heavily regularized. Our Meta-Model combines the strengths of the TCN during normal regimes, and shifts weight to LightGBM during extreme weather shocks.
""")

st.header("4.2 Model Interpretation (What drives the predictions?)")
st.markdown("We trace the predictions back to our original hypothesis. Which variables actually matter?")

fig = px.bar(
    importance.head(15), 
    x='Importance', 
    y='Feature', 
    orientation='h',
    title="Top 15 Feature Importances (LightGBM)"
)
fig.update_layout(yaxis={'categoryorder':'total ascending'})
st.plotly_chart(fig, use_container_width=True)

st.header("4.3 Model Architecture")
st.markdown("A conceptual look at our chosen **Regime Meta-Model**:")

st.code("""
                          ┌────────────────────────┐
                          │ Input Features (Panel) │
                          └───────────┬────────────┘
                                      │
              ┌───────────────────────┼─────────────────────────┐
              ▼                       ▼                         ▼
   ┌────────────────────┐   ┌────────────────────┐    ┌───────────────────┐
   │ HMM Regime Detector│   │ LightGBM (Weather) │    │ TCN (Price/Term)  │
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

