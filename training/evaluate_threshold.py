"""
evaluate_threshold.py
─────────────────────
Load the trained ensemble pipeline and sweep fraud-probability thresholds
to find the optimal operating point.

Outputs a threshold sweep table and saves the best threshold back to config.yaml.

Usage:
    python training/evaluate_threshold.py
    make evaluate
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.structured.evaluation import sweep_thresholds, plot_precision_recall_curve
from src.models.structured.pipeline import load_pipeline
from src.utils import reduce_mem_usage
from sklearn.model_selection import train_test_split


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


import argparse

def main():
    parser = argparse.ArgumentParser(description="Evaluate threshold sweep for fraud detection")
    parser.add_argument("--nrows", type=int, default=50000, help="Number of rows to sample for sweep")
    parser.add_argument("--update-config", action="store_true", help="Automatically write optimal threshold to config.yaml")
    args = parser.parse_args()

    cfg = load_config()

    # ── Load pipeline ─────────────────────────────────────────────────────────
    pipeline_path = cfg["model"]["ensemble_path"]
    feature_cols_path = cfg["model"]["feature_cols_path"]

    if not os.path.exists(pipeline_path):
        print(f"[FAIL] No pipeline found at {pipeline_path}. Run `make ensemble` first.")
        sys.exit(1)

    print(f"Loading pipeline from {pipeline_path} …")
    pipeline, feature_cols = load_pipeline(pipeline_path, feature_cols_path)

    # ── Load held-out validation set ─────────────────────────────────────────
    data_path = cfg["training"]["data_path"]
    print(f"Loading data from {data_path} (nrows={args.nrows}) …")
    df = pd.read_csv(data_path, nrows=args.nrows, low_memory=False)
    df = reduce_mem_usage(df, verbose=False)

    drop_cols = ["isFraud", "TransactionID", "TransactionDT"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["isFraud"]

    _, X_val, _, y_val = train_test_split(
        X, y,
        test_size=cfg["training"]["test_size"],
        random_state=cfg["training"]["random_state"],
        stratify=y,
    )

    # ── Score ─────────────────────────────────────────────────────────────────
    print("Scoring validation set …")
    y_proba = pipeline.predict_proba(X_val)[:, 1]

    # ── Sweep thresholds ──────────────────────────────────────────────────────
    import numpy as np
    sweep_df = sweep_thresholds(y_val, y_proba, thresholds=np.linspace(0.1, 0.99, 180))
    sweep_df = sweep_df.round(4)

    print("\n── Threshold Sweep (sample every 10 rows) ──")
    print(sweep_df.iloc[::10].to_string(index=False))

    # Best by F1
    best_f1_row = sweep_df.loc[sweep_df["f1"].idxmax()]
    # Best by Precision ≥ 0.5 and max Recall
    high_prec = sweep_df[sweep_df["precision"] >= 0.5]
    best_recall_row = high_prec.loc[high_prec["recall"].idxmax()] if not high_prec.empty else best_f1_row

    print(f"\n📌 Best by F1:                    threshold={best_f1_row['threshold']:.2f}  "
          f"P={best_f1_row['precision']:.3f}  R={best_f1_row['recall']:.3f}  F1={best_f1_row['f1']:.3f}")
    print(f"📌 Best Recall (Precision ≥ 0.5): threshold={best_recall_row['threshold']:.2f}  "
          f"P={best_recall_row['precision']:.3f}  R={best_recall_row['recall']:.3f}  F1={best_recall_row['f1']:.3f}")

    # ── Save sweep to CSV ─────────────────────────────────────────────────────
    metrics_dir = os.path.dirname(cfg["training"]["metrics_path"]) or "."
    os.makedirs(metrics_dir, exist_ok=True)
    sweep_path = os.path.join(metrics_dir, "threshold_sweep.csv")
    sweep_df.to_csv(sweep_path, index=False)
    print(f"\n✅ Sweep saved → {sweep_path}")

    # ── Optionally update config.yaml with the best threshold ─────────────────
    recommended = float(best_f1_row["threshold"])
    print(f"\nRecommended threshold (best F1): {recommended:.2f}")
    if args.update_config:
        with open("config.yaml") as f:
            raw = f.read()
        import re
        new_raw = re.sub(
            r"(threshold:\s*)[\d.]+",
            f"threshold: {recommended}",
            raw,
            count=1,
        )
        with open("config.yaml", "w") as f:
            f.write(new_raw)
        print(f"  [OK] config.yaml updated -> threshold: {recommended}")


if __name__ == "__main__":
    main()
