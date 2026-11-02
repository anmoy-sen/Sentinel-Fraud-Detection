"""
evaluation.py
─────────────
Model evaluation utilities.

Returns *Plotly* figures so they render natively in both Streamlit and notebooks.
All heavy sklearn imports stay inside functions to keep load time low.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ─── Core classification report ───────────────────────────────────────────────

def evaluate_classification(
    y_true,
    y_pred,
    y_pred_proba=None,
    model_name: str = "Model",
) -> dict:
    """
    Compute and print standard classification metrics.

    Returns a dict with keys: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
    """
    metrics: dict = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_pred_proba is not None:
        metrics["ROC-AUC"] = roc_auc_score(y_true, y_pred_proba)

    print(f"\n--- {model_name} Performance ---")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")
    return metrics


# ─── Confusion matrix (Plotly) ────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, model_name: str = "Model"):
    """
    Return a Plotly heatmap of the confusion matrix.

    Falls back to a matplotlib show() when called outside Streamlit.
    """
    import plotly.figure_factory as ff

    cm = confusion_matrix(y_true, y_pred)
    labels = ["Legit", "Fraud"]
    z_text = [[str(v) for v in row] for row in cm]

    fig = ff.create_annotated_heatmap(
        z=cm.tolist(),
        x=labels,
        y=labels,
        annotation_text=z_text,
        colorscale="Blues",
        showscale=True,
    )
    fig.update_layout(
        title=f"{model_name} — Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        yaxis=dict(autorange="reversed"),
        template="plotly_dark",
    )
    return fig


# ─── ROC curve (Plotly) ───────────────────────────────────────────────────────

def plot_roc_curve(
    y_true,
    y_pred_proba,
    model_name: str = "Model",
):
    """Return a Plotly ROC curve figure."""
    import plotly.graph_objects as go

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{model_name} (AUC={auc:.4f})",
                   line=dict(color="#ef4444", width=2))
    )
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                   line=dict(color="grey", dash="dash"))
    )
    fig.update_layout(
        title=f"{model_name} — ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_dark",
        legend=dict(x=0.6, y=0.1),
    )
    return fig


# ─── Precision-Recall curve (Plotly) ─────────────────────────────────────────

def plot_precision_recall_curve(
    y_true,
    y_pred_proba,
    model_name: str = "Model",
):
    """Return a Plotly Precision-Recall curve figure."""
    import plotly.graph_objects as go

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    # Append sentinel so arrays align for hover
    thresholds = np.append(thresholds, 1.0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name=model_name,
            customdata=thresholds,
            hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<br>Threshold: %{customdata:.3f}",
            line=dict(color="#f59e0b", width=2),
        )
    )
    fig.update_layout(
        title=f"{model_name} — Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        template="plotly_dark",
    )
    return fig


# ─── Feature importance (Plotly) ─────────────────────────────────────────────

def plot_feature_importance(
    model,
    feature_names: list[str],
    top_n: int = 30,
    model_name: str = "Model",
):
    """
    Return a Plotly horizontal bar chart of feature importances.

    Works with RF, XGBoost, LightGBM, CatBoost (all expose `feature_importances_`).
    """
    import plotly.graph_objects as go

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        # VotingClassifier: average the sub-estimator importances
        all_imp = []
        for _, est in model.estimators_:
            if hasattr(est, "feature_importances_"):
                all_imp.append(est.feature_importances_)
        if not all_imp:
            raise AttributeError("Model does not expose feature_importances_.")
        importances = np.mean(all_imp, axis=0)

    idx = np.argsort(importances)[-top_n:]
    names = [feature_names[i] for i in idx]
    vals = importances[idx]

    fig = go.Figure(
        go.Bar(x=vals, y=names, orientation="h",
               marker=dict(color="#3b82f6"))
    )
    fig.update_layout(
        title=f"{model_name} — Top {top_n} Feature Importances",
        xaxis_title="Importance",
        yaxis_title="Feature",
        template="plotly_dark",
        height=max(400, top_n * 18),
    )
    return fig


# ─── Threshold sweep helper ───────────────────────────────────────────────────

def sweep_thresholds(
    y_true,
    y_pred_proba,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Return a DataFrame of Precision/Recall/F1 at each threshold.

    Useful for picking the operational decision boundary.
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.99, 100)

    rows = []
    for t in thresholds:
        y_pred = (y_pred_proba >= t).astype(int)
        rows.append(
            {
                "threshold": round(t, 3),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
            }
        )
    return pd.DataFrame(rows)
