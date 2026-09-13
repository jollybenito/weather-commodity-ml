import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_correlation_matrix(df, method='pearson'):
    corr = df.corr(method=method)
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        title=f"{method.capitalize()} Correlation Matrix"
    )
    return fig

def plot_top_correlations_boxplots(df, target, top_n=5):
    corrs = df.corr()[target].drop(target).abs().sort_values(ascending=False).head(top_n)
    top_features = corrs.index.tolist()
    
    # We can use plotly subplots or just return a list of figs
    figs = []
    for feat in top_features:
        fig = px.box(df, x=feat, title=f"Distribution of {feat} (Corr: {df.corr()[target][feat]:.2f})")
        figs.append(fig)
    return figs, top_features

def plot_yearly_correlation(df, feature, target, method='pearson'):
    df = df.copy()
    if 'date' not in df.columns:
        df['date'] = df.index
    df['year'] = pd.to_datetime(df['date']).dt.year
    
    yearly_corr = df.groupby('year').apply(lambda x: x[[feature, target]].corr(method=method).iloc[0, 1]).reset_index()
    yearly_corr.columns = ['Year', 'Correlation']
    
    fig = px.line(yearly_corr, x='Year', y='Correlation', title=f"Yearly Correlation: {feature} vs {target}")
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    return fig

def plot_all_yearly_correlations(df, features, target, method='pearson'):
    df = df.copy()
    if 'date' not in df.columns:
        df['date'] = df.index
    df['year'] = pd.to_datetime(df['date']).dt.year
    
    # Calculate yearly correlation for each feature
    records = []
    for year, group in df.groupby('year'):
        for feat in features:
            # Drop NaNs to ensure correlation can be calculated
            valid_data = group[[feat, target]].dropna()
            if len(valid_data) > 1:
                corr_val = valid_data.corr(method=method).iloc[0, 1]
            else:
                corr_val = np.nan
            records.append({'Year': year, 'Feature': feat, 'Correlation': corr_val})
            
    yearly_corr = pd.DataFrame(records)
    
    fig = px.line(
        yearly_corr, 
        x='Year', 
        y='Correlation', 
        color='Feature',
        title=f"Yearly Trend of {method.capitalize()} Correlation (Top {len(features)} Variables vs {target})",
        markers=True
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.3,
        xanchor="center",
        x=0.5
    ))
    return fig
