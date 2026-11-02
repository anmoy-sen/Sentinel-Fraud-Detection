"""
05_settings.py  —  Settings & Simulator Controls
"""

import os
import re
import sys

import streamlit as st
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

st.set_page_config(page_title="Settings · Sentinel", page_icon="⚙️", layout="wide")

st.title("⚙️ Settings")
st.markdown("Tune thresholds, simulator speed, and manage the alert history.")

# ── Load config ────────────────────────────────────────────────────────────────
cfg_path = os.path.join(ROOT, "config.yaml")
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

simulator = st.session_state.get("simulator")
alert_store = st.session_state.get("alert_store")

# ── Simulator controls ────────────────────────────────────────────────────────
st.subheader("🎮 Simulator Controls")
col1, col2, col3 = st.columns(3)

if simulator:
    if simulator.is_running and not simulator.is_paused:
        if col1.button("⏸ Pause"):
            simulator.pause()
            st.rerun()
    elif simulator.is_paused:
        if col1.button("▶ Resume"):
            simulator.resume()
            st.rerun()
    else:
        if col1.button("▶ Start"):
            simulator.start()
            st.rerun()

    if col2.button("🔄 Restart"):
        simulator.reset()
        simulator.start()
        st.session_state.sim_started = True
        st.rerun()

    with col3.expander("Status"):
        st.write(f"Running: {simulator.is_running}")
        st.write(f"Paused: {simulator.is_paused}")
        st.write(f"Processed: {simulator.processed:,}")
        st.write(f"Fraud detected: {simulator.fraud_detected:,}")
else:
    st.warning("Simulator not available (model not loaded).")

st.divider()

# ── Threshold ─────────────────────────────────────────────────────────────────
st.subheader("🎚️ Fraud Threshold")
current_threshold = float(cfg["model"]["threshold"])
new_threshold = st.slider(
    "Fraud probability threshold",
    min_value=0.3, max_value=0.99, value=current_threshold, step=0.01,
    help="Predictions above this value are flagged as fraud."
)
st.markdown(
    f"**Current:** `{current_threshold}`  →  **New:** `{new_threshold}`  "
    f"({'↑ fewer alerts, higher precision' if new_threshold > current_threshold else '↓ more alerts, higher recall'})"
)

if new_threshold != current_threshold:
    if st.button("💾 Save Threshold to config.yaml"):
        with open(cfg_path) as f:
            raw = f.read()
        raw = re.sub(r"(threshold:\s*)[\d.]+", f"threshold: {new_threshold}", raw, count=1)
        with open(cfg_path, "w") as f:
            f.write(raw)
        st.success(f"Threshold updated to {new_threshold}. Restart the dashboard to apply.")

st.divider()

# ── Simulator speed ────────────────────────────────────────────────────────────
st.subheader("⚡ Simulator Speed")
current_mode = cfg["simulator"]["mode"]
mode = st.selectbox("Mode", ["batch", "custom"], index=["batch", "custom"].index(current_mode))

if mode == "batch":
    batch_size = st.number_input("Batch size (transactions per flush)", 10, 500,
                                  int(cfg["simulator"]["batch_size"]))
    batch_interval = st.number_input("Flush interval (seconds)", 1, 60,
                                      int(cfg["simulator"]["batch_interval_sec"]))
else:
    custom_speed = st.number_input("Custom speed (transactions/sec)", 1, 500,
                                    int(cfg["simulator"]["custom_speed"]))

if st.button("💾 Save Simulator Config"):
    cfg["simulator"]["mode"] = mode
    if mode == "batch":
        cfg["simulator"]["batch_size"] = int(batch_size)
        cfg["simulator"]["batch_interval_sec"] = int(batch_interval)
    else:
        cfg["simulator"]["custom_speed"] = int(custom_speed)

    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    st.success("Simulator config saved. Restart the simulator to apply.")

st.divider()

# ── Alert DB management ───────────────────────────────────────────────────────
st.subheader("🗑️ Alert Database")
if alert_store:
    count = alert_store.count()
    fraud_n = alert_store.fraud_count()
    st.write(f"**{count:,}** total alerts  |  **{fraud_n:,}** fraud  |  **{count - fraud_n:,}** legit")

    col_dl, col_clr = st.columns(2)

    # Download
    df_all = alert_store.get_all()
    csv_bytes = df_all.to_csv(index=False).encode()
    col_dl.download_button("📥 Export alerts.csv", csv_bytes, "alerts.csv", "text/csv")

    # Clear
    if col_clr.button("🗑️ Clear all alerts", type="primary"):
        alert_store.clear()
        st.success("Alert history cleared.")
        st.rerun()
else:
    st.warning("Alert store not available.")
