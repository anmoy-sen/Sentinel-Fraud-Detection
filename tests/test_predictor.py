"""
test_predictor.py
─────────────────
Unit tests for the FraudPredictor inference wrapper.
"""

import tempfile
import os

import joblib
import numpy as np
import pandas as pd
import pytest

from src.inference.predictor import FraudPredictor


# ── Helpers ──────────────────────────────────────────────────────────────────

def _save_pipeline(trained_pipeline, tmp_dir):
    """Save pipeline + feature_cols to temp files, return paths."""
    import json
    pipeline, feature_cols = trained_pipeline
    pipe_path = os.path.join(tmp_dir, "pipeline.pkl")
    cols_path = os.path.join(tmp_dir, "feature_cols.json")
    joblib.dump(pipeline, pipe_path)
    with open(cols_path, "w") as f:
        json.dump(feature_cols, f)
    return pipe_path, cols_path


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_predictor_loads(trained_pipeline):
    """Predictor should load without error."""
    with tempfile.TemporaryDirectory() as tmp:
        pipe_path, cols_path = _save_pipeline(trained_pipeline, tmp)
        predictor = FraudPredictor()
        predictor.load(pipe_path, cols_path)
        assert predictor.is_loaded


def test_predict_returns_correct_shape(trained_pipeline, synthetic_X):
    """predict() should return one dict per row."""
    with tempfile.TemporaryDirectory() as tmp:
        pipe_path, cols_path = _save_pipeline(trained_pipeline, tmp)
        predictor = FraudPredictor()
        predictor.load(pipe_path, cols_path)

        results = predictor.predict(synthetic_X.head(10))
        assert len(results) == 10


def test_predict_proba_in_range(trained_pipeline, synthetic_X):
    """Probabilities should be in [0, 1]."""
    with tempfile.TemporaryDirectory() as tmp:
        pipe_path, cols_path = _save_pipeline(trained_pipeline, tmp)
        predictor = FraudPredictor()
        predictor.load(pipe_path, cols_path)

        results = predictor.predict(synthetic_X)
        for r in results:
            assert 0.0 <= r["fraud_proba"] <= 1.0, f"Out of range: {r['fraud_proba']}"


def test_predict_result_keys(trained_pipeline, synthetic_X):
    """Each result dict must have the expected keys."""
    with tempfile.TemporaryDirectory() as tmp:
        pipe_path, cols_path = _save_pipeline(trained_pipeline, tmp)
        predictor = FraudPredictor()
        predictor.load(pipe_path, cols_path)

        result = predictor.predict(synthetic_X.head(1))[0]
        assert "fraud_proba" in result
        assert "is_fraud" in result
        assert "latency_ms" in result


def test_predict_without_load_raises():
    """Predicting before loading should raise RuntimeError."""
    predictor = FraudPredictor()
    with pytest.raises(RuntimeError, match="load"):
        predictor.predict(pd.DataFrame({"a": [1]}))


def test_predict_single(trained_pipeline, synthetic_X):
    """predict_single() should accept a dict."""
    with tempfile.TemporaryDirectory() as tmp:
        pipe_path, cols_path = _save_pipeline(trained_pipeline, tmp)
        predictor = FraudPredictor()
        predictor.load(pipe_path, cols_path)

        row_dict = synthetic_X.iloc[0].to_dict()
        result = predictor.predict_single(row_dict)
        assert isinstance(result["fraud_proba"], float)


def test_extra_columns_ignored(trained_pipeline, synthetic_X):
    """Predictor should handle DataFrames with extra columns gracefully."""
    with tempfile.TemporaryDirectory() as tmp:
        pipe_path, cols_path = _save_pipeline(trained_pipeline, tmp)
        predictor = FraudPredictor()
        predictor.load(pipe_path, cols_path)

        X_extra = synthetic_X.copy()
        X_extra["unknown_column_xyz"] = 999
        results = predictor.predict(X_extra.head(5))
        assert len(results) == 5


def test_avg_latency_accumulates(trained_pipeline, synthetic_X):
    """avg_latency_ms should be > 0 after predictions."""
    with tempfile.TemporaryDirectory() as tmp:
        pipe_path, cols_path = _save_pipeline(trained_pipeline, tmp)
        predictor = FraudPredictor()
        predictor.load(pipe_path, cols_path)

        predictor.predict(synthetic_X.head(20))
        assert predictor.avg_latency_ms > 0
