"""
tests/test_financial_cost.py
─────────────────────────────
Unit tests for Financial Cost-Benefit Matrix & Profit Optimization.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.structured.financial_cost import (
    compute_financial_cost_matrix,
    plot_financial_cost_curve,
)


def test_compute_financial_cost_matrix_structure():
    np.random.seed(42)
    n = 200
    y_true = np.random.choice([0, 1], size=n, p=[0.9, 0.1])
    y_proba = np.random.uniform(0, 1, size=n)
    amounts = np.random.uniform(10, 500, size=n)

    cost_df = compute_financial_cost_matrix(
        y_true=y_true,
        y_pred_proba=y_proba,
        amounts=amounts,
        cost_fp=15.0,
        fixed_chargeback_fee=25.0,
        review_cost=2.0,
    )

    assert isinstance(cost_df, pd.DataFrame)
    assert not cost_df.empty
    expected_cols = [
        "threshold",
        "net_savings_usd",
        "fraud_prevented_usd",
        "chargeback_loss_usd",
        "friction_cost_usd",
        "tp_count",
        "fp_count",
        "fn_count",
        "tn_count",
        "precision",
        "recall",
        "f1",
    ]
    for col in expected_cols:
        assert col in cost_df.columns

    # Check bounds
    assert (cost_df["threshold"] >= 0.0).all() and (cost_df["threshold"] <= 1.0).all()
    assert (cost_df["precision"] >= 0.0).all() and (cost_df["precision"] <= 1.0).all()
    assert (cost_df["recall"] >= 0.0).all() and (cost_df["recall"] <= 1.0).all()


def test_plot_financial_cost_curve():
    cost_df = pd.DataFrame({
        "threshold": [0.1, 0.5, 0.9],
        "net_savings_usd": [1000.0, 5000.0, 2000.0],
        "f1": [0.3, 0.75, 0.4],
    })
    fig = plot_financial_cost_curve(cost_df)
    assert fig is not None
    assert len(fig.data) >= 3
