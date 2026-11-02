"""
conftest.py  —  Shared pytest fixtures
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd

# Make project root importable from tests/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


# ── Tiny synthetic dataset ────────────────────────────────────────────────────
N = 200
RNG = np.random.default_rng(42)

@pytest.fixture(scope="session")
def synthetic_X() -> pd.DataFrame:
    """200-row DataFrame mimicking engineered feature columns."""
    return pd.DataFrame({
        "TransactionAmt": RNG.uniform(1, 5000, N),
        "TransactionAmt_Log": RNG.uniform(0, 8, N),
        "TransactionAmt_cents": RNG.integers(0, 100, N),
        "TransactionAmt_is_round": RNG.choice([True, False], N),
        "Transaction_hour": RNG.integers(0, 24, N),
        "Transaction_dow": RNG.integers(0, 7, N),
        "card1": RNG.integers(1000, 9999, N).astype(str),
        "card2": RNG.integers(100, 999, N).astype(float),
        "P_emaildomain": RNG.choice(["gmail.com", "yahoo.com", None], N),
        "R_emaildomain": RNG.choice(["gmail.com", "outlook.com", None], N),
        **{f"V{i}": RNG.uniform(0, 10, N) for i in range(1, 21)},
        **{f"C{i}": RNG.integers(0, 100, N).astype(float) for i in range(1, 6)},
    })


@pytest.fixture(scope="session")
def synthetic_y(synthetic_X) -> pd.Series:
    """Imbalanced labels (~5% fraud)."""
    return pd.Series(RNG.choice([0, 1], len(synthetic_X), p=[0.95, 0.05]), name="isFraud")


@pytest.fixture(scope="session")
def trained_pipeline(synthetic_X, synthetic_y):
    """A minimal trained pipeline for inference tests."""
    from sklearn.ensemble import RandomForestClassifier
    from src.models.structured.pipeline import build_inference_pipeline

    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    feature_cols = list(synthetic_X.columns)
    pipeline = build_inference_pipeline(model, feature_cols)
    pipeline.fit(synthetic_X, synthetic_y)
    return pipeline, feature_cols
