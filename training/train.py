"""
train.py
────────
Production training script for Sentinel Fraud Detection.

Usage:
    python training/train.py --model lgbm
    python training/train.py --model lgbm --optimize      # Optuna tuning
    python training/train.py --model xgb
    python training/train.py --model rf
    python training/train.py --model catboost

What it does:
    1. Loads data/processed/train_engineered.csv
    2. Splits into train/val (stratified)
    3. Encodes categoricals + handles inf/NaN via the inference pipeline
    4. Trains the chosen model (with optional Optuna tuning)
    5. Evaluates: AUC, F1, Precision, Recall
    6. Saves: models/<name>_model.pkl  and  (for lgbm) the full ensemble pipeline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import yaml

# Make project root importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.model_selection import train_test_split

from src.models.structured.evaluation import evaluate_classification, sweep_thresholds
from src.models.structured.models import get_model_by_name
from src.models.structured.pipeline import build_inference_pipeline, save_pipeline
from src.utils import reduce_mem_usage


# ─── Config ───────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_data(cfg: dict, nrows: int | None = None) -> tuple[pd.DataFrame, pd.Series]:
    data_path = cfg["training"]["data_path"]
    print(f"Loading data from {data_path} (nrows={nrows}) …")
    df = pd.read_csv(data_path, nrows=nrows, low_memory=False)
    df = reduce_mem_usage(df, verbose=True)

    drop_cols = ["isFraud", "TransactionID", "TransactionDT"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["isFraud"]
    print(f"  Dataset: {X.shape[0]:,} rows × {X.shape[1]:,} features  |  fraud rate: {y.mean():.2%}")
    return X, y


# ─── Optuna tuning ────────────────────────────────────────────────────────────

def _optuna_tune_lgbm(X_train, y_train, X_val, y_val, cfg: dict) -> dict:
    import optuna
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n_trials = cfg["model"]["optuna"]["n_trials"]
    timeout = cfg["model"]["optuna"]["timeout_sec"]
    study_dir = cfg["model"]["optuna"]["study_dir"]
    os.makedirs(study_dir, exist_ok=True)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
        mdl = LGBMClassifier(
            **params,
            class_weight="balanced",
            random_state=cfg["training"]["random_state"],
            n_jobs=-1,
            verbosity=-1,
        )
        mdl.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[optuna.integration.lightgbm.early_stopping(50, verbose=False)],
        )
        proba = mdl.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, proba)

    study = optuna.create_study(
        direction="maximize",
        study_name="lgbm_fraud",
        storage=f"sqlite:///{study_dir}/lgbm_fraud.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    print(f"\n  Best AUC: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params


def _optuna_tune_xgb(X_train, y_train, X_val, y_val, cfg: dict) -> dict:
    import optuna
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n_trials = cfg["model"]["optuna"]["n_trials"]
    timeout = cfg["model"]["optuna"]["timeout_sec"]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
        mdl = XGBClassifier(
            **params,
            scale_pos_weight=3,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=-1,
            random_state=cfg["training"]["random_state"],
        )
        mdl.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                early_stopping_rounds=50, verbose=False)
        proba = mdl.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, proba)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    return study.best_params


def _optuna_tune_catboost(X_train, y_train, X_val, y_val, cfg: dict) -> dict:
    import optuna
    from catboost import CatBoostClassifier
    from sklearn.metrics import roc_auc_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n_trials = cfg["model"]["optuna"]["n_trials"]

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 200, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10, log=True),
        }
        mdl = CatBoostClassifier(**params, auto_class_weights="Balanced", verbose=0)
        mdl.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
        proba = mdl.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, proba)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params


# ─── Main training loop ───────────────────────────────────────────────────────

def train(model_name: str, optimize: bool = False, cfg_path: str = "config.yaml", nrows: int | None = None):
    cfg = load_config(cfg_path)
    rs = cfg["training"]["random_state"]
    models_dir = cfg["training"]["models_dir"]
    metrics_path = cfg["training"]["metrics_path"]
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)

    # ── Load & split ──────────────────────────────────────────────────────────
    X, y = load_data(cfg, nrows=nrows)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=cfg["training"]["test_size"],
        random_state=rs,
        stratify=y,
    )
    print(f"  Train: {len(X_train):,}  |  Val: {len(X_val):,}")

    feature_cols = list(X_train.columns)

    # ── Build pipeline & pre-process (fit encoder on train only) ─────────────
    model = get_model_by_name(model_name, random_state=rs)
    pipeline = build_inference_pipeline(model, feature_cols)

    # Fit only the pre-processing steps, then get transformed data for tuning
    from sklearn.pipeline import Pipeline as SkPipeline
    pre_pipeline = SkPipeline(pipeline.steps[:-1])  # all but the model
    pre_pipeline.fit(X_train, y_train)
    X_train_enc = pre_pipeline.transform(X_train)
    X_val_enc = pre_pipeline.transform(X_val)

    # ── Optional Optuna tuning ────────────────────────────────────────────────
    if optimize:
        print(f"\n🔍 Running Optuna hyper-parameter search for {model_name.upper()} …")
        tuners = {
            "lgbm": _optuna_tune_lgbm,
            "lightgbm": _optuna_tune_lgbm,
            "xgb": _optuna_tune_xgb,
            "xgboost": _optuna_tune_xgb,
            "catboost": _optuna_tune_catboost,
        }
        tuner = tuners.get(model_name.lower())
        if tuner:
            best_params = tuner(X_train_enc, y_train, X_val_enc, y_val, cfg)
            model = get_model_by_name(model_name, random_state=rs)
            model.set_params(**best_params)
            pipeline = build_inference_pipeline(model, feature_cols)
        else:
            print(f"  ⚠  No Optuna tuner for {model_name}; skipping.")

    # ── Train the full pipeline ───────────────────────────────────────────────
    print(f"\n🚀 Training {model_name.upper()} …")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_proba = pipeline.predict_proba(X_val)[:, 1]
    threshold = cfg["model"]["threshold"]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = evaluate_classification(y_val, y_pred, y_proba, model_name=model_name.upper())
    metrics["threshold"] = threshold
    metrics["train_time_sec"] = round(elapsed, 2)

    # Best threshold (by F1)
    sweep_df = sweep_thresholds(y_val, y_proba)
    best_row = sweep_df.loc[sweep_df["f1"].idxmax()]
    print(f"\n  📐 Best threshold by F1: {best_row['threshold']:.2f}  "
          f"(P={best_row['precision']:.3f}, R={best_row['recall']:.3f}, F1={best_row['f1']:.3f})")

    # ── Save artefacts ────────────────────────────────────────────────────────
    # Individual model pickle (for ensemble building)
    model_pkl = os.path.join(models_dir, f"{model_name}_model.pkl")
    joblib.dump(pipeline.named_steps["model"], model_pkl)
    print(f"  ✅ Model  saved → {model_pkl}")

    # Full pipeline (encoder + model)
    ensemble_path = cfg["model"]["ensemble_path"]
    feature_cols_path = cfg["model"]["feature_cols_path"]
    if model_name.lower() in ("lgbm", "lightgbm"):
        save_pipeline(pipeline, feature_cols, ensemble_path, feature_cols_path)

    # Metrics JSON
    existing_metrics: dict = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            try:
                existing_metrics = json.load(f)
            except json.JSONDecodeError:
                pass
    existing_metrics[model_name] = metrics
    with open(metrics_path, "w") as f:
        json.dump(existing_metrics, f, indent=2)
    print(f"  ✅ Metrics saved → {metrics_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train a Sentinel fraud-detection model")
    p.add_argument(
        "--model",
        default="lgbm",
        choices=["rf", "xgb", "lgbm", "catboost"],
        help="Which model to train (default: lgbm)",
    )
    p.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Number of rows to load from train CSV (default: all)",
    )
    p.add_argument(
        "--optimize",
        action="store_true",
        help="Run Optuna hyper-parameter search before training",
    )
    p.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml in CWD)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(model_name=args.model, optimize=args.optimize, cfg_path=args.config, nrows=args.nrows)
