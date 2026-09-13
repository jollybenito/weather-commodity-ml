import streamlit as st

st.set_page_config(
    page_title="Weather-Commodity ML Portfolio",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Streamlit ML Portfolio: Commodity Price Regime Model")

st.markdown("""
Welcome to the end-to-end Machine Learning Workflow demonstration!

This application explores the hypothesis:
**"Can extreme weather anomalies predict corn futures returns, and does this predictive power change under different market regimes?"**

### Navigation
Please use the sidebar to navigate through the 5 stages of this project:
1. **Hypothesis**: The core question and available data.
2. **Exploration**: Exploratory data analysis, data quality, and correlations.
3. **First Steps**: Baselines, linear models, and our train/validation/test methodology.
4. **Best Models**: Hyperparameter tuning, model comparison, interpretation, and visualization.
5. **Conclusion**: Final answers, interpretations, and limitations.

---
*Use the sidebar on the left to begin with **1. Hypothesis**.*
""")
