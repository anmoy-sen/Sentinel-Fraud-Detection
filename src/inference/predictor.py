"""
predictor.py
────────────
Thread-safe FraudPredictor — the single point of contact between
the dashboard (or any other caller) and the trained model pipeline.

Features
--------
* Lazy-loads the pipeline from disk on first call.
* Thread-safe prediction via a re-entrant lock.
* Optional SHAP explanations (lazy-loaded on first explain call).
* Returns structured dicts — no raw numpy required by callers.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import pandas as pd


class FraudPredictor:
    """
    Singleton-friendly wrapper around the Sentinel inference pipeline.

    Usage
    -----
    predictor = FraudPredictor()
    predictor.load("models/ensemble_pipeline.pkl", "models/feature_cols.json")
    result = predictor.predict(row_df)
    explanation = predictor.explain(row_df)
    """

    def __init__(self):
        self._pipeline = None
        self._feature_cols: list[str] = []
        self._explainer = None
        self._lock = threading.RLock()
        self._predict_count = 0
        self._total_latency_ms = 0.0

    # ── Load ─────────────────────────────────────────────────────────────────

    def load(self, pipeline_path: str, feature_cols_path: str) -> "FraudPredictor":
        """Load the pipeline from disk (safe to call multiple times)."""
        from src.models.structured.pipeline import load_pipeline

        with self._lock:
            self._pipeline, self._feature_cols = load_pipeline(
                pipeline_path, feature_cols_path
            )
            self._explainer = None  # reset SHAP on reload
        return self

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Score one or more transactions.

        Parameters
        ----------
        X : pd.DataFrame
            Raw (pre-encoding) feature rows.  Column set may be a superset of
            what the model was trained on — extras are silently ignored.

        Returns
        -------
        list of dicts, one per row:
            {
                "fraud_proba": float,
                "is_fraud":    bool,
                "latency_ms":  float,
            }
        """
        if not self.is_loaded:
            raise RuntimeError("Call FraudPredictor.load() before predict().")

        t0 = time.perf_counter()
        with self._lock:
            probas = self._pipeline.predict_proba(X)[:, 1]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._predict_count += len(X)
        self._total_latency_ms += elapsed_ms

        return [
            {
                "fraud_proba": float(p),
                "is_fraud": bool(p >= self._threshold()),
                "latency_ms": round(elapsed_ms / len(X), 2),
            }
            for p in probas
        ]

    def predict_single(self, row: dict | pd.Series) -> dict[str, Any]:
        """Convenience wrapper for a single transaction (as a dict or Series)."""
        if isinstance(row, dict):
            row = pd.DataFrame([row])
        elif isinstance(row, pd.Series):
            row = row.to_frame().T
        return self.predict(row)[0]

    # ── SHAP Explain ──────────────────────────────────────────────────────────

    def explain(self, X: pd.DataFrame, max_display: int = 15) -> list[dict[str, Any]]:
        """
        Compute SHAP values for each row in X.

        Returns a list of dicts, one per row:
            {
                "shap_values":  np.ndarray  (one value per feature),
                "feature_names": list[str],
                "base_value":   float,
            }

        Note: first call is slow (~1–2 s) because the SHAP explainer is built lazily.
        """
        if not self.is_loaded:
            raise RuntimeError("Call FraudPredictor.load() before explain().")

        import shap

        with self._lock:
            model_step = self._pipeline.named_steps["model"]
            pre_steps = list(self._pipeline.steps[:-1])

        from sklearn.pipeline import Pipeline as SkPipeline
        pre = SkPipeline(pre_steps)
        X_enc = pre.transform(X)
        feature_names = list(X_enc.columns) if hasattr(X_enc, "columns") else self._feature_cols

        with self._lock:
            if self._explainer is None:
                print("Building SHAP explainer (one-time) …")
                self._explainer = shap.TreeExplainer(model_step)
            explainer = self._explainer

        shap_vals = explainer.shap_values(X_enc)
        # For binary classifiers some models return a list [class0, class1]
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        results = []
        for i in range(len(X)):
            sv = shap_vals[i] if shap_vals.ndim == 2 else shap_vals
            results.append(
                {
                    "shap_values": sv,
                    "feature_names": feature_names,
                    "base_value": float(explainer.expected_value)
                    if not isinstance(explainer.expected_value, (list, np.ndarray))
                    else float(explainer.expected_value[1]),
                }
            )
        return results

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def avg_latency_ms(self) -> float:
        if self._predict_count == 0:
            return 0.0
        return self._total_latency_ms / self._predict_count

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _threshold(self) -> float:
        """Read threshold from config at call-time (supports live config edits)."""
        try:
            import yaml
            with open("config.yaml") as f:
                return float(yaml.safe_load(f)["model"]["threshold"])
        except Exception:
            return 0.75
