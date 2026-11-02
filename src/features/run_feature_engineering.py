"""
run_feature_engineering.py
──────────────────────────
Entry-point script for the full feature-engineering pipeline.

Usage:
    python src/features/run_feature_engineering.py
    make featurize
"""

import sys
import os
import yaml
import pandas as pd
import numpy as np

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.features.feature_engineering import (
    create_transaction_amount_features,
    create_time_features,
    create_card_features,
    create_email_features,
    create_device_features,
    create_address_features,
    create_v_features,
    create_c_d_features,
    create_m_features,
    create_aggregation_features,
)
from src.utils import reduce_mem_usage


# ─── Load Config ──────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ─── Merge raw CSVs ──────────────────────────────────────────────────────────

def load_and_merge(transaction_path: str, identity_path: str) -> pd.DataFrame:
    print(f"  Loading {transaction_path} ...")
    txn = pd.read_csv(transaction_path, low_memory=False)
    print(f"  Loading {identity_path} ...")
    try:
        idn = pd.read_csv(identity_path, low_memory=False)
        df = txn.merge(idn, on="TransactionID", how="left")
    except FileNotFoundError:
        print(f"  ⚠  Identity file not found — skipping merge.")
        df = txn
    print(f"  Merged shape: {df.shape}")
    return df


# ─── Apply all feature modules ────────────────────────────────────────────────

def apply_all_features(df: pd.DataFrame, ref_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Apply every feature-engineering step in order.
    If ``ref_df`` is None the aggregation step is run without a reference (test-only).
    """
    df = create_transaction_amount_features(df)
    df = create_time_features(df)
    df = create_card_features(df)
    df = create_email_features(df)
    df = create_device_features(df)
    df = create_address_features(df)
    df = create_v_features(df)
    df = create_c_d_features(df)
    df = create_m_features(df)
    # Aggregation features are leak-aware: pass ref_df (train) when engineering test
    if ref_df is not None:
        df, _ = create_aggregation_features(df, test_df=None)
    else:
        df, _ = create_aggregation_features(df, test_df=None)
    return df


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    raw_dir = cfg["paths"]["data_raw"]
    out_dir = cfg["paths"]["data_processed"]
    os.makedirs(out_dir, exist_ok=True)

    # ── Train ──────────────────────────────────────────────────────────────────
    print("\n▶ Engineering TRAIN set …")
    train_df = load_and_merge(
        os.path.join(raw_dir, "train_transaction.csv"),
        os.path.join(raw_dir, "train_identity.csv"),
    )
    train_df = apply_all_features(train_df, ref_df=None)
    train_df = reduce_mem_usage(train_df)

    out_train = os.path.join(out_dir, "train_engineered.csv")
    print(f"  Saving → {out_train}  ({train_df.shape})")
    train_df.to_csv(out_train, index=False)

    # ── Test ───────────────────────────────────────────────────────────────────
    print("\n▶ Engineering TEST set …")
    test_df = load_and_merge(
        os.path.join(raw_dir, "test_transaction.csv"),
        os.path.join(raw_dir, "test_identity.csv"),
    )
    # Pass train_df so aggregations are learned from train only
    test_df = create_transaction_amount_features(test_df)
    test_df = create_time_features(test_df)
    test_df = create_card_features(test_df)
    test_df = create_email_features(test_df)
    test_df = create_device_features(test_df)
    test_df = create_address_features(test_df)
    test_df = create_v_features(test_df)
    test_df = create_c_d_features(test_df)
    test_df = create_m_features(test_df)
    _, test_df = create_aggregation_features(train_df, test_df=test_df)
    test_df = reduce_mem_usage(test_df)

    out_test = os.path.join(out_dir, "test_engineered.csv")
    print(f"  Saving → {out_test}  ({test_df.shape})")
    test_df.to_csv(out_test, index=False)

    print("\n✅ Feature engineering complete.")


if __name__ == "__main__":
    main()
