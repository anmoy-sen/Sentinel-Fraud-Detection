"""
train_ensemble.py
─────────────────
Build a soft-voting ensemble from the three individually-trained models,
then save a full inference pipeline ready for the dashboard.

Usage:
    python training/train_ensemble.py
    make ensemble

Prerequisites:
    models/random_forest_model.pkl
    models/xgboost_model.pkl
    models/lightgbm_model.pkl   ← at minimum (others optional)
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.structured.models import get_ensemble
from src.models.structured.pipeline import build_inference_pipeline, save_pipeline
from src.models.structured.evaluation import evaluate_classification
from src.utils import reduce_mem_usage


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


import argparse

def main():
    parser = argparse.ArgumentParser(description="Build and evaluate the soft-voting ensemble")
    parser.add_argument("--nrows", type=int, default=None, help="Max rows to read from train_engineered.csv (default: all)")
    args = parser.parse_args()

    cfg = load_config()
    models_dir = cfg["training"]["models_dir"]
    data_path = cfg["training"]["data_path"]
    rs = cfg["training"]["random_state"]

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading training data from {data_path} (nrows={args.nrows}) …")
    df = pd.read_csv(data_path, nrows=args.nrows, low_memory=False)
    df = reduce_mem_usage(df, verbose=True)

    drop_cols = ["isFraud", "TransactionID", "TransactionDT"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["isFraud"]
    feature_cols = list(X.columns)

    _, X_val, _, y_val = train_test_split(
        X, y,
        test_size=cfg["training"]["test_size"],
        random_state=rs,
        stratify=y,
    )

    # ── Build ensemble from saved individual models ───────────────────────────
    print("\nBuilding soft-voting ensemble (LGBM + XGB + CatBoost) …")
    ensemble = get_ensemble(
        rf_path=cfg["model"]["random_forest_path"],
        xgb_path=cfg["model"]["xgboost_path"],
        lgbm_path=cfg["model"]["lightgbm_path"],
        catboost_path=cfg["model"].get("catboost_path", "models/catboost_model.pkl"),
    )

    # ── Wrap in full inference pipeline ──────────────────────────────────────
    print("Fitting ensemble pipeline (pre-processing + ensemble) …")
    pipeline = build_inference_pipeline(ensemble, feature_cols)
    pipeline.fit(X, y)   # fit on FULL training set

    # ── Evaluate on held-out val ──────────────────────────────────────────────
    y_proba = pipeline.predict_proba(X_val)[:, 1]
    threshold = cfg["model"]["threshold"]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = evaluate_classification(y_val, y_pred, y_proba, model_name="Ensemble")
    metrics["threshold"] = threshold

    # ── Save ──────────────────────────────────────────────────────────────────
    save_pipeline(
        pipeline,
        feature_cols,
        cfg["model"]["ensemble_path"],
        cfg["model"]["feature_cols_path"],
    )

    metrics_path = cfg["training"]["metrics_path"]
    os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
    existing: dict = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                pass
    existing["ensemble"] = metrics
    with open(metrics_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\n[OK] Ensemble pipeline saved.  Val AUC: {metrics.get('ROC-AUC', 'N/A')}")


if __name__ == "__main__":
    main()
