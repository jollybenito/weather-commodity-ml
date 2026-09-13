"""Streamlit-cached data loader.

Reads the pre-materialized SQLite database so the app never calls
yfinance or Open-Meteo at runtime.  Falls back to the live pipeline
if the database doesn't exist yet (developer convenience).
"""

import json
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

DB_PATH = Path("data/weather_commodities.db")


@st.cache_data(show_spinner="Loading data from database...")
def load_dataset():
    """Load the pre-built panel from SQLite and return (df, metadata dict)."""
    if not DB_PATH.exists():
        # Fallback: build live (for local dev before materializing)
        from src.feature_engineering.panel_builder import build_master_panel

        panel_ds = build_master_panel()
        meta = {
            "weather_cols": panel_ds.weather_cols,
            "price_cols": panel_ds.price_cols,
            "term_cols": panel_ds.term_cols,
            "target_col": panel_ds.target_col,
        }
        return panel_ds.df, meta

    conn = sqlite3.connect(str(DB_PATH))
    try:
        df = pd.read_sql("SELECT * FROM panel", conn)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        rows = conn.execute("SELECT key, value FROM metadata").fetchall()
        meta = {k: json.loads(v) for k, v in rows}
    finally:
        conn.close()

    return df, meta


@st.cache_data(show_spinner="Preparing features and target...")
def get_features_and_target():
    df, meta = load_dataset()
    target = meta["target_col"]
    features = meta["weather_cols"] + meta["price_cols"] + meta["term_cols"]

    # Keep only columns that actually exist
    features = [c for c in features if c in df.columns]

    # Drop rows where target is NaN
    df = df.dropna(subset=[target])

    return df, features, target
