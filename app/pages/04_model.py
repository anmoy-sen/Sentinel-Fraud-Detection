"""
04_model.py  —  Model Performance & Financial Cost Matrix
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

st.set_page_config(page_title="Model Performance · Sentinel", page_icon="📊", layout="wide")

st.title("📊 Model Performance & Financial ROI")
st.markdown("AUC curves, confusion matrix, feature importance, and dollar-quantified financial optimization.")

predictor = st.session_state.get("predictor")
if predictor is None:
    st.error("Model not loaded.")
    st.stop()

# ── Load saved metrics ─────────────────────────────────────────────────────────
metrics_path = os.path.join(ROOT, "metrics", "eval_metrics.json")
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        all_metrics = json.load(f)
else:
    all_metrics = {}

# ── Metrics table ─────────────────────────────────────────────────────────────
if all_metrics:
    st.subheader("Ensemble & Candidate Validation Benchmarks")
    rows = []
    for model_name, m in all_metrics.items():
        rows.append({
            "Model": model_name.upper(),
            "AUC": round(m.get("ROC-AUC", 0), 4),
            "F1": round(m.get("F1-Score", 0), 4),
            "Precision": round(m.get("Precision", 0), 4),
            "Recall": round(m.get("Recall", 0), 4),
            "Threshold": m.get("threshold", "—"),
        })
    metrics_df = pd.DataFrame(rows)
    st.dataframe(metrics_df, use_container_width=True)
    st.divider()

# ── Live evaluation on a test sample ──────────────────────────────────────────
st.subheader("Live Evaluation on Validation Sample")
test_path = os.path.join(ROOT, "data", "processed", "test_engineered.csv")
train_path = os.path.join(ROOT, "data", "processed", "train_engineered.csv")

col_s1, col_s2 = st.columns([2, 1])
sample_size = col_s1.slider("Sample size for evaluation", 1000, 20000, 5000, step=1000)

if col_s2.button("▶ Run Evaluation & ROI Analysis", use_container_width=True):
    if not os.path.exists(train_path):
        st.error(f"Training data not found at {train_path}")
    else:
        with st.spinner("Loading data, scoring ensemble, and computing cost matrix …"):
            from sklearn.model_selection import train_test_split
            import yaml
            from src.utils import reduce_mem_usage

            with open(os.path.join(ROOT, "config.yaml")) as f:
                cfg = yaml.safe_load(f)

            df = pd.read_csv(train_path, low_memory=False, nrows=sample_size * 5)
            df = reduce_mem_usage(df, verbose=False)
            X = df.drop(columns=[c for c in ["isFraud", "TransactionID", "TransactionDT"] if c in df.columns])
            y = df["isFraud"]
            _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            X_val = X_val.head(sample_size)
            y_val = y_val.iloc[:sample_size]

            results = predictor.predict(X_val)
            y_proba = np.array([r["fraud_proba"] for r in results])
            threshold = cfg["model"]["threshold"]
            y_pred = np.array([int(p >= threshold) for p in y_proba])

            amounts = X_val["TransactionAmt"].values if "TransactionAmt" in X_val.columns else np.full(len(X_val), 135.0)

            # Store in session state for slider reactivity
            st.session_state["eval_y_val"] = y_val.values
            st.session_state["eval_y_proba"] = y_proba
            st.session_state["eval_y_pred"] = y_pred
            st.session_state["eval_amounts"] = amounts

if "eval_y_val" in st.session_state:
    y_val = st.session_state["eval_y_val"]
    y_proba = st.session_state["eval_y_proba"]
    y_pred = st.session_state["eval_y_pred"]
    amounts = st.session_state["eval_amounts"]

    from src.models.structured.evaluation import (
        evaluate_classification, plot_roc_curve,
        plot_precision_recall_curve, plot_confusion_matrix
    )
    from src.models.structured.financial_cost import (
        compute_financial_cost_matrix, plot_financial_cost_curve
    )

    metrics = evaluate_classification(y_val, y_pred, y_proba, model_name="Ensemble")
    st.success(f"Ensemble ROC-AUC: **{metrics.get('ROC-AUC', 0):.4f}** | F1-Score: **{metrics.get('F1-Score', 0):.4f}** | Precision: **{metrics.get('Precision', 0):.4f}** | Recall: **{metrics.get('Recall', 0):.4f}**")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(plot_confusion_matrix(y_val, y_pred, "Ensemble"), use_container_width=True)
    with c2:
        st.plotly_chart(plot_roc_curve(y_val, y_proba, "Ensemble"), use_container_width=True)
    with c3:
        st.plotly_chart(plot_precision_recall_curve(y_val, y_proba, "Ensemble"), use_container_width=True)

    # ── Financial Cost Matrix ─────────────────────────────────────────────────
    st.divider()
    st.subheader("💰 Financial Cost-Benefit Matrix & Profit Optimization")
    st.markdown("""
    Quantifies the true **dollar impact** of machine learning decisions:
    - **False Negatives (Missed Fraud)**: Direct fraud amount lost + bank chargeback penalty fee.
    - **False Positives (Customer Friction)**: Good customer blocked, churn risk & support cost.
    - **True Positives (Prevented Fraud)**: Gross fraud dollars saved minus manual alert review cost.
    """)

    fc1, fc2, fc3 = st.columns(3)
    cost_fp = fc1.slider("False Positive Friction Cost ($)", 0.0, 50.0, 15.0, step=1.0)
    chargeback_fee = fc2.slider("Bank Chargeback Penalty ($)", 10.0, 100.0, 25.0, step=5.0)
    review_cost = fc3.slider("Manual Review Cost ($)", 0.0, 10.0, 2.0, step=0.5)

    cost_df = compute_financial_cost_matrix(
        y_true=y_val,
        y_pred_proba=y_proba,
        amounts=amounts,
        cost_fp=cost_fp,
        fixed_chargeback_fee=chargeback_fee,
        review_cost=review_cost,
    )

    best_cost_row = cost_df.loc[cost_df["net_savings_usd"].idxmax()]
    best_f1_row = cost_df.loc[cost_df["f1"].idxmax()]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Max Net Financial Savings", f"${best_cost_row['net_savings_usd']:,.2f}")
    m2.metric("Cost-Optimal Threshold", f"{best_cost_row['threshold']:.3f}")
    m3.metric("F1-Optimal Threshold", f"{best_f1_row['threshold']:.3f}")
    m4.metric("Fraud Loss Prevented", f"${best_cost_row['fraud_prevented_usd']:,.2f}")

    st.plotly_chart(plot_financial_cost_curve(cost_df), use_container_width=True)

# ── Feature Importance ────────────────────────────────────────────────────────
st.divider()
st.subheader("Global Feature Importance (GBDT Ensemble)")

if st.button("📊 Show Feature Importance"):
    with st.spinner("Computing feature importance …"):
        try:
            from src.models.structured.evaluation import plot_feature_importance
            from src.models.structured.pipeline import load_pipeline

            pipeline_path = os.path.join(ROOT, "models", "ensemble_pipeline.pkl")
            feat_cols_path = os.path.join(ROOT, "models", "feature_cols.json")
            pipeline, feature_cols = load_pipeline(pipeline_path, feat_cols_path)
            model = pipeline.named_steps["model"]
            fig = plot_feature_importance(model, feature_cols, top_n=30, model_name="Ensemble")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not compute feature importance: {e}")
