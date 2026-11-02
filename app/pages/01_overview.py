"""
01_overview.py  —  KPI Overview Page
"""

import os
import sys
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

st.set_page_config(page_title="Overview · Sentinel", page_icon="🏠", layout="wide")

st.title("🏠 Overview")
st.markdown("Live KPI cards and fraud rate trend.")

alert_store = st.session_state.get("alert_store")
simulator = st.session_state.get("simulator")

if alert_store is None:
    st.error("Model not loaded — return to the home page first.")
    st.stop()

# ── Auto-refresh ──────────────────────────────────────────────────────────────
refresh_sec = st.sidebar.slider("Auto-refresh (s)", 2, 30, 5)
placeholder = st.empty()

def render():
    stats = alert_store.get_stats()
    sim_proc = simulator.processed if simulator else stats["total"]
    sim_fraud = simulator.fraud_detected if simulator else stats["fraud_count"]

    with placeholder.container():
        # KPI row
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Transactions Processed", f"{sim_proc:,}")
        c2.metric("Fraud Detected", f"{sim_fraud:,}")
        rate_pct = (sim_fraud / sim_proc * 100) if sim_proc else 0
        c3.metric("Fraud Rate", f"{rate_pct:.2f}%")
        fraud_prevented = stats.get("fraud_dollars_prevented", 0.0)
        c4.metric("Losses Prevented ($)", f"${fraud_prevented:,.2f}")
        c5.metric("Avg Fraud Prob.", f"{stats['avg_fraud_proba']:.3f}")
        c6.metric("Avg Latency", f"{stats['avg_latency_ms']:.1f} ms")

        st.divider()

        # Hourly fraud chart
        hourly = stats["hourly_counts"]
        if not hourly.empty:
            hourly["fraud_rate"] = (hourly["frauds"] / hourly["total"].replace(0, 1)) * 100

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=hourly["hour"], y=hourly["total"],
                name="Total Transactions", marker_color="#3b82f6", opacity=0.6,
            ))
            fig.add_trace(go.Scatter(
                x=hourly["hour"], y=hourly["fraud_rate"],
                name="Fraud Rate (%)", yaxis="y2",
                mode="lines+markers", line=dict(color="#ef4444", width=2),
            ))
            fig.update_layout(
                title="Hourly Transaction Volume & Fraud Rate",
                xaxis_title="Hour",
                yaxis=dict(title="Transactions"),
                yaxis2=dict(title="Fraud Rate (%)", overlaying="y", side="right"),
                template="plotly_dark",
                legend=dict(x=0, y=1.1, orientation="h"),
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Waiting for transactions to accumulate …")

        # Fraud Probability Distribution
        recent_df = alert_store.get_recent(2000)
        if not recent_df.empty:
            col1, col2 = st.columns(2)
            with col1:
                fig2 = go.Figure()
                fig2.add_trace(go.Histogram(
                    x=recent_df["fraud_proba"],
                    nbinsx=50,
                    name="Score Distribution",
                    marker_color="#6366f1",
                ))
                fig2.update_layout(
                    title="Fraud Score Distribution (last 2,000)",
                    xaxis_title="Fraud Probability",
                    yaxis_title="Count",
                    template="plotly_dark",
                    height=320,
                )
                st.plotly_chart(fig2, use_container_width=True)

            with col2:
                # Gauge — current threat level (rolling avg of last 200 scores)
                last200 = recent_df["fraud_proba"].head(200).mean()
                fig3 = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(float(last200) * 100, 1),
                    title={"text": "Threat Level (last 200)", "font": {"color": "#e2e8f0"}},
                    number={"suffix": "%", "font": {"color": "#e2e8f0"}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#ef4444"},
                        "steps": [
                            {"range": [0, 30], "color": "#166534"},
                            {"range": [30, 60], "color": "#854d0e"},
                            {"range": [60, 100], "color": "#7f1d1d"},
                        ],
                        "threshold": {"line": {"color": "white", "width": 3}, "value": 75},
                    },
                ))
                fig3.update_layout(template="plotly_dark", height=320)
                st.plotly_chart(fig3, use_container_width=True)


render()
time.sleep(refresh_sec)
st.rerun()
