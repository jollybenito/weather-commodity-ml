import streamlit as st
import pandas as pd
from src.app_utils.data_loader import load_dataset, get_features_and_target
from src.app_utils.visualizations import plot_correlation_matrix, plot_top_correlations_boxplots, plot_all_yearly_correlations

st.set_page_config(page_title="Exploration", page_icon="🔎", layout="wide")
st.title("Stage 2 — Exploration")

st.markdown("Before building models, we must understand the shape, quality, and relationships in our dataset.")

with st.spinner("Loading Panel Data..."):
    panel, features, target = get_features_and_target()

st.header("2.1 Dataset Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Observations (Days)", len(panel))
col2.metric("Total Features", len(features))
col3.metric("Target Variable", target)

st.dataframe(panel.head())

st.header("2.2 Data Quality")
st.markdown("We run automated checks to ensure our offline simulation data doesn't contain breaking errors.")

# Calculate quality metrics
missing_values = panel.isna().sum().sum()
duplicate_rows = panel.duplicated().sum()
num_vars = len(panel.select_dtypes(include=['number']).columns)

quality_data = {
    "Check": ["Rows", "Features", "Missing values", "Duplicate rows", "Numerical variables"],
    "Result": [len(panel), len(features), missing_values, duplicate_rows, num_vars]
}
st.table(pd.DataFrame(quality_data))

st.header("2.3 Correlation Analysis")
st.markdown("Let's identify linear and monotonic relationships between our independent variables (features) and the dependent variable (target).")

corr_method = st.radio("Select Correlation Method:", ["pearson", "spearman"], horizontal=True)

# Calculate correlations
all_corrs = panel[features + [target]].corr(method=corr_method)[target].drop(target).abs().sort_values(ascending=False)
top_20 = all_corrs.head(20).index.tolist()
top_5_features = all_corrs.head(5).index.tolist()

fig_corr = plot_correlation_matrix(panel[top_20 + [target]], method=corr_method)
st.plotly_chart(fig_corr, use_container_width=True)

st.header("2.4 Top 5 Most Correlated Variables")
st.markdown(f"The 5 variables with the strongest absolute correlation to `{target}`.")

# Plot the 5 boxplots in tabs for a cleaner UI
box_figs, _ = plot_top_correlations_boxplots(panel[features + [target]], target, top_n=5)
tabs = st.tabs(top_5_features)

for i, tab in enumerate(tabs):
    with tab:
        st.plotly_chart(box_figs[i], use_container_width=True)

st.header("2.5 Temporal Analysis (Correlation Stability)")
st.markdown("""
A feature might look strongly correlated over the entire 24-year dataset, but is that relationship stable? 
Does it break down over time? Here we track the yearly correlation of our top 5 variables against the target.
""")

temporal_df = panel.copy()
if 'date' not in temporal_df.columns:
    temporal_df['date'] = temporal_df.index

# Plot all 5 yearly trends on the same chart
fig_temporal_all = plot_all_yearly_correlations(temporal_df, top_5_features, target, method=corr_method)
st.plotly_chart(fig_temporal_all, use_container_width=True)

st.info("""
**Important Finding on Data Drift**: 
If you notice that some weather features lose their predictive correlation in recent years (Era 3: 2018-2024), this supports our hypothesis that technological advances (drought-resistant seeds, precision agriculture) alter the relationship between extreme weather and yield.
""")
