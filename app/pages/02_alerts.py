"""
02_alerts.py  —  Live Alerts Feed & Analyst Review Workbench
"""

import os
import sys
import time

import pandas as pd
import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

st.set_page_config(page_title="Alerts & Analyst Queue · Sentinel", page_icon="🚨", layout="wide")

st.title("🚨 Alerts & Analyst Investigation Workbench")
st.markdown("Real-time fraud alert stream coupled with a human-in-the-loop analyst review queue.")

alert_store = st.session_state.get("alert_store")
if alert_store is None:
    st.error("Model not loaded — return to the home page first.")
    st.stop()

tab_stream, tab_queue, tab_audit = st.tabs([
    "📡 Live Alerts Stream",
    "👨‍💼 Analyst Action Queue",
    "📜 Decision Audit Trail"
])

# ─── TAB 1: Live Alerts Stream ────────────────────────────────────────────────
with tab_stream:
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 2])
    refresh_sec = col_ctrl1.slider("Auto-refresh interval (s)", 1, 30, 3, key="feed_refresh")
    max_rows = col_ctrl2.slider("Max alerts shown", 50, 500, 100, step=50, key="feed_max")
    only_fraud = col_ctrl3.checkbox("Show fraud alerts only", value=True, key="feed_fraud_only")

    st.divider()

    df = alert_store.get_recent(max_rows * 3)
    if only_fraud and not df.empty:
        df = df[df["is_fraud"] == 1]
    df = df.head(max_rows)

    if df.empty:
        st.info("No alerts yet. Ensure the simulator is running from the Overview or Home page.")
    else:
        fraud_cnt = int(df["is_fraud"].sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Showing Alerts", f"{len(df):,}")
        c2.metric("Fraud Count", f"{fraud_cnt:,}")
        c3.metric("Avg Fraud Prob.", f"{df['fraud_proba'].mean():.4f}")
        c4.metric("Avg Latency", f"{df['latency_ms'].mean():.1f} ms")

        # Display table
        display = df[["timestamp", "transaction_id", "amount", "card1",
                      "fraud_proba", "is_fraud", "status", "rule_flags", "latency_ms"]].copy()
        display.columns = ["Time", "Transaction ID", "Amount ($)", "Card1",
                           "Fraud Prob.", "Is Fraud", "Status", "Rule Flags", "Latency (ms)"]
        display["Fraud Prob."] = display["Fraud Prob."].round(4)
        display["Amount ($)"] = display["Amount ($)"].round(2)
        display["Time"] = pd.to_datetime(display["Time"]).dt.strftime("%H:%M:%S")

        def color_row(row):
            p = row["Fraud Prob."]
            if p >= 0.85:
                return ["background-color: #7f1d1d"] * len(row)
            elif p >= 0.65:
                return ["background-color: #450a0a"] * len(row)
            return [""] * len(row)

        st.dataframe(
            display.style.apply(color_row, axis=1),
            use_container_width=True,
            height=480,
        )

# ─── TAB 2: Analyst Action Queue ──────────────────────────────────────────────
with tab_queue:
    st.subheader("Pending Transaction Reviews")
    st.markdown("Human-in-the-loop review queue for transactions flagged as suspicious.")

    pending_df = alert_store.get_pending_review(limit=50)

    if pending_df.empty:
        st.success("🎉 Inbox zero! No pending alerts awaiting analyst review.")
    else:
        st.info(f"**{len(pending_df)}** transactions pending review. Select a transaction below to take action.")

        col_q1, col_q2 = st.columns([1, 1])

        with col_q1:
            st.markdown("### Queue Items")
            q_display = pending_df[["id", "transaction_id", "fraud_proba", "amount", "rule_flags"]].copy()
            q_display.columns = ["#", "Transaction ID", "Fraud Prob.", "Amount ($)", "Triggered Rules"]
            q_display["Fraud Prob."] = q_display["Fraud Prob."].round(4)
            q_display["Amount ($)"] = q_display["Amount ($)"].round(2)

            st.dataframe(q_display, use_container_width=True, height=360)

            selected_tx = st.selectbox(
                "Select Transaction to Investigate:",
                options=pending_df["transaction_id"].tolist(),
                key="select_pending_tx",
            )

        with col_q2:
            st.markdown("### Transaction Dossier & Investigation")
            if selected_tx:
                tx_row = pending_df[pending_df["transaction_id"] == selected_tx].iloc[0]

                st.markdown(f"""
                - **Transaction ID**: `{tx_row['transaction_id']}`
                - **Amount**: `${tx_row['amount']:,.2f}`
                - **Card Identifier**: `{tx_row['card1']}`
                - **Fraud Probability**: `{tx_row['fraud_proba']:.4f}`
                - **Rule Triggers**: `{tx_row['rule_flags'] or 'None (Model Flagged)'}`
                - **Ingestion Timestamp**: `{tx_row['timestamp']}`
                """)

                analyst_name = st.text_input("Analyst Name / ID", value="analyst_jane", key="analyst_id_input")
                notes = st.text_area("Investigation Notes", placeholder="Enter notes or justification...", key="analyst_notes_input")

                btn_c1, btn_c2, btn_c3 = st.columns(3)
                if btn_c1.button("✅ Approve (Legit)", use_container_width=True):
                    alert_store.record_analyst_action(
                        transaction_id=str(selected_tx),
                        action="APPROVED_LEGIT",
                        analyst_name=analyst_name,
                        notes=notes or "Verified cardholder activity.",
                    )
                    st.success(f"Transaction {selected_tx} marked as APPROVED_LEGIT.")
                    st.rerun()

                if btn_c2.button("🚫 Confirm Fraud (Block)", use_container_width=True):
                    alert_store.record_analyst_action(
                        transaction_id=str(selected_tx),
                        action="CONFIRMED_FRAUD",
                        analyst_name=analyst_name,
                        notes=notes or "Confirmed unauthorized charge pattern.",
                    )
                    st.error(f"Transaction {selected_tx} BLOCKED and marked as CONFIRMED_FRAUD.")
                    st.rerun()

                if btn_c3.button("🔍 Escalate", use_container_width=True):
                    alert_store.record_analyst_action(
                        transaction_id=str(selected_tx),
                        action="ESCALATED",
                        analyst_name=analyst_name,
                        notes=notes or "Escalated for senior fraud investigator review.",
                    )
                    st.warning(f"Transaction {selected_tx} ESCALATED to Tier-2.")
                    st.rerun()

# ─── TAB 3: Decision Audit Trail ──────────────────────────────────────────────
with tab_audit:
    st.subheader("Analyst Decision Audit Trail")
    st.markdown("Immutable audit log tracking all human decisions, analyst IDs, and notes.")

    audit_df = alert_store.get_audit_trail(limit=100)
    if audit_df.empty:
        st.info("No analyst decisions recorded yet.")
    else:
        st.dataframe(audit_df, use_container_width=True, height=450)
