"""
financial_cost.py
─────────────────
Financial Cost-Benefit Matrix & Profit Optimization for Fraud Detection.

Quantifies the dollar impact of ML predictions:
- False Negatives (Missed Fraud): Direct loss of transaction amount + bank chargeback penalty fee.
- False Positives (Good User Blocked): Customer support friction cost + customer churn penalty.
- True Positives (Prevented Fraud): Net fraud loss recovered minus minor review cost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def compute_financial_cost_matrix(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    amounts: np.ndarray,
    cost_fp: float = 15.0,
    fixed_chargeback_fee: float = 25.0,
    review_cost: float = 2.0,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Simulate financial outcomes across a spectrum of decision thresholds.

    Parameters
    ----------
    y_true               : Actual ground-truth fraud flags (0 or 1)
    y_pred_proba         : Model predicted fraud probabilities [0, 1]
    amounts              : Transaction dollar amounts
    cost_fp              : Friction cost per false positive ($)
    fixed_chargeback_fee : Bank penalty fee per missed fraud ($)
    review_cost          : Cost per manual alert investigation ($)
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)

    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    amounts = np.asarray(amounts)

    # Baseline cost without model (every fraud is approved -> FN loss)
    is_fraud_bool = (y_true == 1)
    no_model_loss = np.sum(amounts[is_fraud_bool] + fixed_chargeback_fee)

    rows = []
    for t in thresholds:
        pred_fraud = (y_pred_proba >= t)

        # TP: Blocked Fraud (Savings)
        tp_mask = pred_fraud & is_fraud_bool
        tp_count = np.sum(tp_mask)
        fraud_loss_prevented = np.sum(amounts[tp_mask])
        tp_review_cost = tp_count * review_cost
        gross_fraud_savings = fraud_loss_prevented - tp_review_cost

        # FP: Blocked Good Customer (Friction)
        fp_mask = pred_fraud & (~is_fraud_bool)
        fp_count = np.sum(fp_mask)
        fp_friction_loss = fp_count * cost_fp

        # FN: Missed Fraud (Loss)
        fn_mask = (~pred_fraud) & is_fraud_bool
        fn_count = np.sum(fn_mask)
        fn_chargeback_loss = np.sum(amounts[fn_mask] + fixed_chargeback_fee)

        # TN: Allowed Good Customer
        tn_count = np.sum((~pred_fraud) & (~is_fraud_bool))

        # Net Financial Savings vs doing nothing
        total_model_cost = fn_chargeback_loss + fp_friction_loss + tp_review_cost
        net_savings = no_model_loss - total_model_cost

        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        rows.append({
            "threshold": round(float(t), 3),
            "net_savings_usd": round(float(net_savings), 2),
            "fraud_prevented_usd": round(float(fraud_loss_prevented), 2),
            "chargeback_loss_usd": round(float(fn_chargeback_loss), 2),
            "friction_cost_usd": round(float(fp_friction_loss), 2),
            "tp_count": int(tp_count),
            "fp_count": int(fp_count),
            "fn_count": int(fn_count),
            "tn_count": int(tn_count),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
        })

    return pd.DataFrame(rows)


def plot_financial_cost_curve(cost_df: pd.DataFrame) -> go.Figure:
    """
    Generate an executive-ready Plotly figure showing Financial Savings ($)
    and F1-Score across thresholds.
    """
    best_cost_row = cost_df.loc[cost_df["net_savings_usd"].idxmax()]
    best_f1_row = cost_df.loc[cost_df["f1"].idxmax()]

    fig = go.Figure()

    # Financial Savings Trace
    fig.add_trace(go.Scatter(
        x=cost_df["threshold"],
        y=cost_df["net_savings_usd"],
        name="Net Financial Savings ($)",
        mode="lines",
        line=dict(color="#10b981", width=3),
        hovertemplate="Threshold: %{x:.2f}<br>Net Savings: $%{y:,.0f}<extra></extra>",
    ))

    # F1 Score Trace (Secondary y-axis)
    fig.add_trace(go.Scatter(
        x=cost_df["threshold"],
        y=cost_df["f1"],
        name="F1 Score",
        yaxis="y2",
        mode="lines",
        line=dict(color="#6366f1", width=2, dash="dash"),
        hovertemplate="Threshold: %{x:.2f}<br>F1: %{y:.3f}<extra></extra>",
    ))

    # Cost-optimal point annotation
    fig.add_trace(go.Scatter(
        x=[best_cost_row["threshold"]],
        y=[best_cost_row["net_savings_usd"]],
        mode="markers+text",
        name=f"Cost-Optimal (${best_cost_row['net_savings_usd']:,.0f})",
        marker=dict(size=12, color="#10b981", symbol="star"),
        text=[f"Max ROI (${best_cost_row['net_savings_usd']:,.0f})"],
        textposition="top center",
    ))

    fig.update_layout(
        title="Financial ROI & Cost-Optimal Threshold vs Machine Learning F1",
        xaxis_title="Decision Threshold",
        yaxis=dict(title="Net Financial Savings (USD $)", tickformat="$,.0f"),
        yaxis2=dict(title="F1-Score", overlaying="y", side="right", range=[0, 1]),
        template="plotly_dark",
        legend=dict(x=0.02, y=0.98),
        height=450,
    )
    return fig
