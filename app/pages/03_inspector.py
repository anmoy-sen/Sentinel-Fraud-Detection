"""
03_inspector.py  —  Transaction Inspector with SHAP
"""

import os
import sys

import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

st.set_page_config(page_title="Inspector · Sentinel", page_icon="🔬", layout="wide")

st.title("🔬 Transaction Inspector")
st.markdown("Pick any recent alert to see a SHAP waterfall explanation.")

alert_store = st.session_state.get("alert_store")
predictor = st.session_state.get("predictor")

if alert_store is None or predictor is None:
    st.error("Model not loaded.")
    st.stop()

# ── Pick a transaction ────────────────────────────────────────────────────────
recent = alert_store.get_recent(200)
if recent.empty:
    st.info("No alerts to inspect yet. Let the simulator run for a moment.")
    st.stop()

fraud_only = recent[recent["is_fraud"] == 1]
display_pool = fraud_only if not fraud_only.empty else recent

tx_ids = display_pool["transaction_id"].tolist()
selected_id = st.selectbox("Select Transaction ID", tx_ids)
selected_row = display_pool[display_pool["transaction_id"] == selected_id].iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Fraud Probability", f"{selected_row['fraud_proba']:.4f}")
col2.metric("Amount", f"${selected_row['amount']:.2f}")
col3.metric("Card1", str(selected_row["card1"]))
col4.metric("Time", str(selected_row["timestamp"])[:19])

st.divider()

# ── SHAP explanation ──────────────────────────────────────────────────────────
with st.spinner("Computing SHAP values (may take 2–3 s on first call) …"):
    try:
        import pandas as pd
        test_path = os.path.join(ROOT, "data", "processed", "test_engineered.csv")

        @st.cache_data(show_spinner=False)
        def _load_test_sample(n=5000):
            return pd.read_csv(test_path, nrows=n, low_memory=False)

        test_df = _load_test_sample()

        # Try to find matching row; fall back to first row
        match = test_df[test_df["TransactionID"].astype(str) == str(selected_id)]
        if match.empty:
            match = test_df.iloc[[0]]

        drop = [c for c in ["isFraud", "TransactionID", "TransactionDT"] if c in match.columns]
        row_features = match.drop(columns=drop)

        explanations = predictor.explain(row_features, max_display=20)
        exp = explanations[0]

        sv = exp["shap_values"]
        names = exp["feature_names"]
        base = exp["base_value"]

        # Top 20 by absolute value
        idx = np.argsort(np.abs(sv))[-20:][::-1]
        top_names = [names[i] for i in idx]
        top_vals = [sv[i] for i in idx]

        # Waterfall chart
        colors = ["#ef4444" if v > 0 else "#22c55e" for v in top_vals]
        fig = go.Figure(go.Waterfall(
            name="SHAP",
            orientation="h",
            measure=["relative"] * len(top_vals),
            x=top_vals,
            y=top_names,
            connector={"line": {"color": "#334155"}},
            decreasing={"marker": {"color": "#22c55e"}},
            increasing={"marker": {"color": "#ef4444"}},
            totals={"marker": {"color": "#6366f1"}},
        ))
        fig.update_layout(
            title=f"SHAP Waterfall — {selected_id}  (base: {base:.3f})",
            xaxis_title="SHAP Value (→ increases fraud risk)",
            template="plotly_dark",
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Feature table
        import pandas as _pd
        shap_df = _pd.DataFrame({"Feature": top_names, "SHAP Value": top_vals}).sort_values(
            "SHAP Value", key=abs, ascending=False
        )
        shap_df["Direction"] = shap_df["SHAP Value"].apply(
            lambda v: "🔴 Increases Risk" if v > 0 else "🟢 Decreases Risk"
        )
        shap_df["SHAP Value"] = shap_df["SHAP Value"].round(5)
        st.dataframe(shap_df, use_container_width=True)

        # Export investigation report
        st.divider()
        from src.inference.pdf_report import generate_fraud_report_bytes

        top_pairs = list(zip(top_names, top_vals))
        raw_flags = str(selected_row.get("rule_flags", ""))
        rule_flags_list = [f.strip() for f in raw_flags.split(",") if f.strip()]
        report_bytes, mime_type = generate_fraud_report_bytes(
            transaction_id=str(selected_id),
            fraud_proba=float(selected_row["fraud_proba"]),
            amount=float(selected_row["amount"]),
            card1=str(selected_row["card1"]),
            timestamp=str(selected_row["timestamp"]),
            top_features=top_pairs,
            rule_flags=rule_flags_list,
        )
        ext = "pdf" if mime_type == "application/pdf" else "md"
        st.download_button(
            label=f"📄 Export Fraud Investigation Report (.{ext})",
            data=report_bytes,
            file_name=f"investigation_report_{selected_id}.{ext}",
            mime=mime_type,
        )

    except ImportError:
        st.warning("SHAP not installed. Run `pip install shap` to enable explanations.")
    except Exception as e:
        st.error(f"Could not compute SHAP values: {e}")
