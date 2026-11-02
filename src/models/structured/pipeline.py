"""
pipeline.py
───────────
Builds and serialises a reproducible, end-to-end sklearn inference pipeline.

Responsibilities
----------------
* Column selection  – keep only the features seen at train time
* Ordinal encoding  – convert object/category columns to integers
* Inf/NaN handling  – custom transformer that replaces ±inf → NaN
* Model            – any sklearn-compatible classifier
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


# ─── Custom transformers ──────────────────────────────────────────────────────

class ColumnSelector(BaseEstimator, TransformerMixin):
    """Keep only the columns used during training and fill missing ones."""

    def __init__(self, feature_cols: list[str]):
        self.feature_cols = feature_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_cols if c not in X.columns]
        if missing:
            # Add missing columns as NaN so the pipeline stays robust
            for col in missing:
                X = X.copy()
                X[col] = np.nan
        return X[self.feature_cols]


class InfToNaN(BaseEstimator, TransformerMixin):
    """Replace +/-inf with NaN so downstream steps don't break."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.replace([np.inf, -np.inf], np.nan)


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Encode object/category columns with OrdinalEncoder, leave numerics alone.

    ``handle_unknown='use_encoded_value'`` with ``unknown_value=-1`` means
    categories seen only at inference time get a safe sentinel value.
    """

    def __init__(self):
        self._encoder: OrdinalEncoder | None = None
        self._cat_cols: list[str] = []
        self._other_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        self._cat_cols = list(X.select_dtypes(include=["object", "category"]).columns)
        self._other_cols = [c for c in X.columns if c not in self._cat_cols]

        if self._cat_cols:
            filled = X[self._cat_cols].fillna("__missing__").astype(str)
            self._encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            self._encoder.fit(filled)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self._cat_cols and self._encoder is not None:
            filled = X[self._cat_cols].fillna("__missing__").astype(str)
            X[self._cat_cols] = self._encoder.transform(filled)

        # Coerce any remaining non-encoded object columns to numeric
        remaining_obj = X.select_dtypes(include=["object", "category"]).columns
        for col in remaining_obj:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        # Cast float16 → float32  (XGBoost / LightGBM don't handle float16)
        f16 = X.select_dtypes(include=["float16"]).columns
        if len(f16):
            X[f16] = X[f16].astype(np.float32)

        return X


# ─── Pipeline builder ─────────────────────────────────────────────────────────

def build_inference_pipeline(model: Any, feature_cols: list[str]) -> Pipeline:
    """
    Wrap *model* in a full sklearn Pipeline.

    Parameters
    ----------
    model       : Any sklearn-compatible classifier (already fitted or fresh).
    feature_cols: Ordered list of column names the model was trained on.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Steps: column_selector → inf_to_nan → encoder → model
    """
    return Pipeline(
        steps=[
            ("column_selector", ColumnSelector(feature_cols)),
            ("inf_to_nan", InfToNaN()),
            ("encoder", CategoricalEncoder()),
            ("model", model),
        ]
    )


# ─── Save / Load helpers ──────────────────────────────────────────────────────

def save_pipeline(
    pipeline: Pipeline,
    feature_cols: list[str],
    pipeline_path: str,
    feature_cols_path: str,
) -> None:
    """Persist pipeline artefact and the feature-column list side-car."""
    os.makedirs(os.path.dirname(pipeline_path) or ".", exist_ok=True)
    joblib.dump(pipeline, pipeline_path)
    with open(feature_cols_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"  [OK] Pipeline saved  -> {pipeline_path}")
    print(f"  [OK] Feature cols    -> {feature_cols_path}")


def load_pipeline(pipeline_path: str, feature_cols_path: str) -> tuple[Pipeline, list[str]]:
    """Load pipeline + feature cols from disk."""
    pipeline = joblib.load(pipeline_path)
    with open(feature_cols_path, "r") as f:
        feature_cols = json.load(f)
    return pipeline, feature_cols
