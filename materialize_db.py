"""Materialize the full ML pipeline into a SQLite database.

Run this script once (offline) to populate data/weather_commodities.db.
The Streamlit app then reads from SQLite — zero network calls at runtime.

Usage:
    python materialize_db.py
"""

import sqlite3
import json
from pathlib import Path

from src.feature_engineering.panel_builder import build_master_panel


DB_PATH = Path("data/weather_commodities.db")


def materialize():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Building master panel (weather + price + term structure)...")
    panel_ds = build_master_panel(force_refresh=True)

    df = panel_ds.df.copy()
    # SQLite doesn't have a native datetime type; store the index as a TEXT column
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = df["date"].astype(str)

    # Store metadata about feature groups so the app can reconstruct PanelDataset
    meta = {
        "weather_cols": panel_ds.weather_cols,
        "price_cols": panel_ds.price_cols,
        "term_cols": panel_ds.term_cols,
        "target_col": panel_ds.target_col,
    }

    conn = sqlite3.connect(str(DB_PATH))
    try:
        # Write the panel
        df.to_sql("panel", conn, if_exists="replace", index=False)
        print(f"  Wrote {len(df)} rows × {len(df.columns)} columns to 'panel' table.")

        # Write metadata
        conn.execute("DROP TABLE IF EXISTS metadata")
        conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        for k, v in meta.items():
            conn.execute(
                "INSERT INTO metadata (key, value) VALUES (?, ?)",
                (k, json.dumps(v)),
            )
        conn.commit()
        print(f"  Wrote metadata ({len(meta)} keys) to 'metadata' table.")
    finally:
        conn.close()

    size_mb = DB_PATH.stat().st_size / (1024 * 1024)
    print(f"Done. Database saved to {DB_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    materialize()

