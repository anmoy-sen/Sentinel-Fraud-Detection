"""
models.py
─────────
Model factory — initialises and returns classifier instances.

All training hyper-parameters live here so that training scripts
stay clean.  The Optuna-tuned best params for LightGBM are the defaults;
fall back to sane baselines for the other models.
"""

from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier


# ─── Individual model factories ───────────────────────────────────────────────

def get_random_forest(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        random_state=random_state,
        class_weight="balanced",
        n_estimators=200,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        n_jobs=-1,
    )


def get_xgboost(random_state: int = 42) -> XGBClassifier:
    return XGBClassifier(
        random_state=random_state,
        scale_pos_weight=3,
        eval_metric="logloss",
        max_depth=6,
        n_estimators=500,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        verbosity=0,
        n_jobs=-1,
    )


def get_lightgbm(random_state: int = 42) -> LGBMClassifier:
    """LightGBM with tuned parameters (best single-model AUC on this dataset)."""
    return LGBMClassifier(
        random_state=random_state,
        class_weight="balanced",
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_jobs=-1,
        verbosity=-1,
    )


def get_catboost(random_state: int = 42) -> CatBoostClassifier:
    return CatBoostClassifier(
        random_seed=random_state,
        iterations=500,
        learning_rate=0.05,
        depth=6,
        auto_class_weights="Balanced",
        verbose=0,
    )


# ─── Pre-fitted soft voting classifier ────────────────────────────────────────

class PreFittedSoftVotingClassifier(BaseEstimator, ClassifierMixin):
    """
    Wraps already-fitted estimators for soft-voting without retraining.
    """

    def __init__(self, estimators: list[tuple[str, Any]], weights: list[float] | None = None):
        self.estimators = estimators
        self.weights = weights
        self.estimators_ = [est for _, est in estimators]
        self.classes_ = np.array([0, 1])

    def fit(self, X=None, y=None):
        return self

    def predict_proba(self, X):
        probas = [est.predict_proba(X) for est in self.estimators_]
        return np.average(probas, axis=0, weights=self.weights)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    @property
    def feature_importances_(self):
        all_imp = []
        for est in self.estimators_:
            if hasattr(est, "feature_importances_"):
                all_imp.append(est.feature_importances_)
        if all_imp:
            return np.mean(all_imp, axis=0)
        raise AttributeError("None of the base estimators expose feature_importances_.")


# ─── Named registry ───────────────────────────────────────────────────────────

def get_tree_models(random_state: int = 42) -> dict:
    """Return a dict of all four model instances (backward-compatible)."""
    return {
        "Random Forest": get_random_forest(random_state),
        "XGBoost": get_xgboost(random_state),
        "LightGBM": get_lightgbm(random_state),
        "CatBoost": get_catboost(random_state),
    }


def get_model_by_name(name: str, random_state: int = 42):
    """
    Return a single model instance by short name.

    Accepted names:  'rf', 'xgb', 'lgbm', 'catboost'
    """
    name = name.lower()
    if name in ("rf", "random_forest", "randomforest"):
        return get_random_forest(random_state)
    if name in ("xgb", "xgboost"):
        return get_xgboost(random_state)
    if name in ("lgbm", "lightgbm"):
        return get_lightgbm(random_state)
    if name in ("catboost",):
        return get_catboost(random_state)
    raise ValueError(f"Unknown model name: {name!r}.  Choose rf / xgb / lgbm / catboost.")


def get_best_model(random_state: int = 42) -> LGBMClassifier:
    """Return the recommended production model (LightGBM)."""
    return get_lightgbm(random_state)


def get_ensemble(
    rf_path: str | None = None,
    xgb_path: str | None = None,
    lgbm_path: str | None = None,
    catboost_path: str | None = None,
    include_rf: bool = False,
    random_state: int = 42,
):
    """
    Build a soft-voting ensemble from pre-trained models.
    By default uses the high-performance gradient boosted trio (LGBM + XGB + CatBoost),
    replacing Random Forest for superior precision and recall.
    """
    candidates = [
        ("lgbm", lgbm_path, lambda: get_lightgbm(random_state)),
        ("xgb", xgb_path, lambda: get_xgboost(random_state)),
    ]

    # Include CatBoost if path provided or if models/catboost_model.pkl exists
    cb_path = catboost_path or "models/catboost_model.pkl"
    if cb_path and os.path.exists(cb_path):
        candidates.append(("catboost", cb_path, lambda: get_catboost(random_state)))
    elif catboost_path:
        candidates.append(("catboost", catboost_path, lambda: get_catboost(random_state)))

    # Only include Random Forest if explicitly requested or if CatBoost is missing
    if include_rf or len(candidates) < 3:
        candidates.insert(0, ("rf", rf_path, lambda: get_random_forest(random_state)))

    estimators = []
    all_pre_trained = True

    for name, path, factory in candidates:
        if path and os.path.exists(path):
            model = joblib.load(path)
            print(f"  Loaded {name.upper()} from {path}")
        else:
            model = factory()
            all_pre_trained = False
            print(f"  Using fresh {name.upper()} (no saved model at {path!r})")
        estimators.append((name, model))

    if all_pre_trained:
        return PreFittedSoftVotingClassifier(estimators=estimators)
    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)


def train_model(model, X_train, y_train):
    """Fit a model to training data (kept for notebook back-compat)."""
    model.fit(X_train, y_train)
    return model
