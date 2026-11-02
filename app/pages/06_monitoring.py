"""
06_monitoring.py  —  Feature Drift & Model Monitoring
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

st.set_page_config(page_title="Monitoring · Sentinel", page_icon="📡", layout="wide")

st.title("📡 Monitoring & Drift Detection")
st.markdown("Population Stability Index (PSI) comparison: **training distribution vs live transactions**.")

# ── Data loading ──────────────────────────────────────────────────────────────
train_path = os.path.join(ROOT, "data", "processed", "train_engineered.csv")
test_path = os.path.join(ROOT, "data", "processed", "test_engineered.csv")


@st.cache_data(show_spinner=False)
def load_reference(n=10000):
    return pd.read_csv(train_path, nrows=n, low_memory=False)


@st.cache_data(show_spinner=False, ttl=60)
def load_current(n=5000):
    return pd.read_csv(test_path, nrows=n, low_memory=False)


with st.spinner("Loading data …"):
    ref_df = load_reference()
    cur_df = load_current()

# ── Drift detector ────────────────────────────────────────────────────────────
import yaml
with open(os.path.join(ROOT, "config.yaml")) as f:
    cfg = yaml.safe_load(f)

top_n = st.sidebar.slider("Features to monitor", 10, 50, int(cfg["monitoring"]["top_features_to_monitor"]))
psi_warn = float(cfg["monitoring"]["psi_warning"])
psi_thresh = float(cfg["monitoring"]["psi_threshold"])

from src.monitoring.drift_detector import DriftDetector

detector = DriftDetector(
    ref_df.drop(columns=[c for c in ["isFraud", "TransactionID", "TransactionDT"] if c in ref_df.columns]),
    top_features=top_n,
    psi_warning=psi_warn,
    psi_threshold=psi_thresh,
)

cur_clean = cur_df.drop(columns=[c for c in ["TransactionID", "TransactionDT"] if c in cur_df.columns])
summary = detector.summary(cur_clean)
report = summary["report"]

# ── KPI cards ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Features Monitored", summary["total_features"])
c2.metric("🔴 Drifted", summary["drifted"])
c3.metric("🟡 Warning", summary["warning"])
c4.metric("🟢 Stable", summary["ok"])

if summary["drifted"] > 0:
    st.error(f"⚠️ **{summary['drifted']} feature(s) show significant drift** (PSI > {psi_thresh})")
elif summary["warning"] > 0:
    st.warning(f"⚡ {summary['warning']} feature(s) show moderate drift (PSI > {psi_warn})")
else:
    st.success("✅ No significant drift detected.")

st.divider()

# ── PSI Heatmap ───────────────────────────────────────────────────────────────
if not report.empty:
    color_map = {"ok": "#166534", "warning": "#854d0e", "drift": "#7f1d1d"}
    report["color"] = report["status"].map(color_map)

    fig = go.Figure(go.Bar(
        x=report["psi"],
        y=report["feature"],
        orientation="h",
        marker_color=report["color"],
        hovertemplate="<b>%{y}</b><br>PSI: %{x:.4f}<extra></extra>",
    ))
    fig.add_vline(x=psi_warn, line_dash="dash", line_color="#facc15",
                  annotation_text="Warning", annotation_position="top right")
    fig.add_vline(x=psi_thresh, line_dash="dash", line_color="#ef4444",
                  annotation_text="Drift", annotation_position="top right")
    fig.update_layout(
        title="PSI by Feature (🟢 OK  🟡 Warning  🔴 Drift)",
        xaxis_title="Population Stability Index",
        template="plotly_dark",
        height=max(400, len(report) * 22),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Feature Distribution Comparison")

    # Pick a feature to drill into
    selected_feat = st.selectbox(
        "Compare distribution for feature:",
        report["feature"].tolist()
    )

    col_ref, col_cur = st.columns(2)
    ref_vals = ref_df[selected_feat].dropna()
    cur_vals = cur_df[selected_feat].dropna() if selected_feat in cur_df.columns else pd.Series(dtype=float)

    with col_ref:
        fig_r = go.Figure(go.Histogram(x=ref_vals, nbinsx=40, marker_color="#3b82f6", name="Train"))
        fig_r.update_layout(title="Training (Reference)", template="plotly_dark", height=280)
        st.plotly_chart(fig_r, use_container_width=True)

    with col_cur:
        fig_c = go.Figure(go.Histogram(x=cur_vals, nbinsx=40, marker_color="#f59e0b", name="Live"))
        fig_c.update_layout(title="Live (Current)", template="plotly_dark", height=280)
        st.plotly_chart(fig_c, use_container_width=True)

    # Full PSI report table
    with st.expander("📋 Full PSI Report"):
        st.dataframe(report[["feature", "psi", "status"]], use_container_width=True)
