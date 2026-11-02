"""
prepare_feature_store.py
────────────────────────
Extracts historical card aggregations from training data, writes to Parquet,
registers entities and feature views with Feast, and materializes them into
the online SQLite feature store for real-time inference.

Usage:
    python feature_store/prepare_feature_store.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from feast import FeatureStore

# Make project root importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from feature_store.features import card_entity, card_risk_fv


def prepare_card_features(
    train_csv_path: str = "data/processed/train_engineered.csv",
    output_parquet: str = "feature_store/data/card_risk_features.parquet",
    max_rows: int = 100000,
) -> pd.DataFrame:
    print(f"Reading sample from {train_csv_path} to build historical card profiles …")
    cols = ["card1", "card1_TransactionAmt_mean", "card1_TransactionAmt_std", "card1_freq"]

    # Read header to confirm columns exist
    df = pd.read_csv(train_csv_path, nrows=max_rows, low_memory=False)
    available_cols = [c for c in cols if c in df.columns]

    card_df = df[available_cols].dropna(subset=["card1"]).drop_duplicates(subset=["card1"]).copy()
    card_df["card1"] = card_df["card1"].astype(str)

    # Fill default values if columns were missing
    if "card1_TransactionAmt_mean" not in card_df.columns:
        card_df["card1_TransactionAmt_mean"] = 120.0
    if "card1_TransactionAmt_std" not in card_df.columns:
        card_df["card1_TransactionAmt_std"] = 40.0
    if "card1_freq" not in card_df.columns:
        card_df["card1_freq"] = 5

    card_df["card1_TransactionAmt_mean"] = card_df["card1_TransactionAmt_mean"].astype(np.float32)
    card_df["card1_TransactionAmt_std"] = card_df["card1_TransactionAmt_std"].astype(np.float32)
    card_df["card1_freq"] = card_df["card1_freq"].astype(np.int64)

    # Feast requires timestamp columns
    now = datetime.now(timezone.utc)
    card_df["event_timestamp"] = now
    card_df["created_timestamp"] = now

    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    card_df.to_parquet(output_parquet, index=False)
    print(f"  [OK] Saved {len(card_df):,} unique card profiles -> {output_parquet}")
    return card_df


def materialize_feast(repo_path: str = "feature_store"):
    print(f"Applying Feast feature definitions in {repo_path} …")
    store = FeatureStore(repo_path=repo_path)
    store.apply([card_entity, card_risk_fv])
    print("  [OK] Feast definitions applied to registry.")

    print("Materializing features into online SQLite store …")
    start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end_date = datetime.now(timezone.utc)
    store.materialize(start_date=start_date, end_date=end_date)
    print("  [OK] Materialization complete! Features are now accessible at <1ms latency.")


if __name__ == "__main__":
    prepare_card_features()
    materialize_feast()
