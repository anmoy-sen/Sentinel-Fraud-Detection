"""
main.py  —  Sentinel Fraud Detection Dashboard
───────────────────────────────────────────────
Entry point.  Run with:
    streamlit run app/main.py

Architecture
------------
* Loads the trained pipeline + predictor once, stores them in st.session_state.
* Starts the TransactionSimulator background thread (if model is available).
* Uses Streamlit's multi-page structure (pages/ directory).
* Shared state lives in st.session_state so all pages can access it.
"""

import os
import sys

import streamlit as st
import yaml

# ── Make project root importable ─────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Sentinel · Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — dark theme + red accent ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Main background */
    .stApp { background-color: #0f172a; color: #e2e8f0; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #1e293b; }

    /* KPI metric cards */
    div[data-testid="metric-container"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 12px 18px;
    }

    /* Fraud alert rows */
    .fraud-row { color: #ef4444 !important; font-weight: bold; }

    /* Buttons */
    .stButton > button {
        background-color: #1d4ed8;
        color: white;
        border-radius: 6px;
        border: none;
    }
    .stButton > button:hover { background-color: #2563eb; }

    /* DataFrames */
    .dataframe { font-size: 13px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Load config ───────────────────────────────────────────────────────────────
@st.cache_resource
def _load_config():
    with open(os.path.join(ROOT, "config.yaml")) as f:
        return yaml.safe_load(f)


cfg = _load_config()


# ── Init shared state ─────────────────────────────────────────────────────────
def _init_state():
    if "predictor" not in st.session_state:
        st.session_state.predictor = None
    if "simulator" not in st.session_state:
        st.session_state.simulator = None
    if "alert_store" not in st.session_state:
        st.session_state.alert_store = None
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    if "sim_started" not in st.session_state:
        st.session_state.sim_started = False


_init_state()


# ── Bootstrap predictor + simulator (once per session) ────────────────────────
@st.cache_resource
def _bootstrap(pipeline_path: str, feature_cols_path: str, db_path: str):
    """
    Heavy init: load model, create alert store, create simulator.
    Cached so it runs only once per Streamlit server lifecycle.
    """
    from src.inference.predictor import FraudPredictor
    from src.simulation.alert_store import AlertStore
    from src.simulation.simulator import create_simulator

    predictor = FraudPredictor()
    predictor.load(pipeline_path, feature_cols_path)

    alert_store = AlertStore(db_path)
    simulator = create_simulator(predictor, alert_store, os.path.join(ROOT, "config.yaml"))

    return predictor, alert_store, simulator


pipeline_path = os.path.join(ROOT, cfg["model"]["ensemble_path"])
feature_cols_path = os.path.join(ROOT, cfg["model"]["feature_cols_path"])
db_path = os.path.join(ROOT, cfg["alerts"]["db_path"])

if os.path.exists(pipeline_path) and os.path.exists(feature_cols_path):
    try:
        predictor, alert_store, simulator = _bootstrap(
            pipeline_path, feature_cols_path, db_path
        )
        st.session_state.predictor = predictor
        st.session_state.alert_store = alert_store
        st.session_state.simulator = simulator
        st.session_state.model_loaded = True

        # Auto-start simulator
        if not st.session_state.sim_started:
            simulator.start()
            st.session_state.sim_started = True

    except Exception as e:
        st.session_state.model_loaded = False
        st.warning(f"⚠️ Could not load model: {e}")
else:
    st.session_state.model_loaded = False


# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("🛡️ Sentinel")
    st.caption("Fraud Detection System")
    st.divider()

    if st.session_state.model_loaded:
        st.success("✅ Model loaded")
        sim = st.session_state.simulator
        if sim and sim.is_running:
            status = "⏸ Paused" if sim.is_paused else "🟢 Running"
            st.info(f"Simulator: {status}")
            st.metric("Transactions", f"{sim.processed:,}")
            st.metric("Fraud Detected", f"{sim.fraud_detected:,}")
    else:
        st.error("❌ No model found")
        st.info("Run `make ensemble` to train the model first.")

    st.divider()
    st.caption("Pages ↓ (use sidebar navigation)")


# ── Home page content ─────────────────────────────────────────────────────────
st.title("🛡️ Sentinel Fraud Detection Dashboard")
st.markdown(
    """
    Welcome to **Sentinel** — a real-time fraud detection system powered by a soft-voting
    ensemble of Random Forest, XGBoost, and LightGBM.

    Use the **sidebar** to navigate between pages:

    | Page | Description |
    |------|-------------|
    | 🏠 **Overview** | KPI cards & live fraud-rate chart |
    | 🚨 **Live Alerts** | Real-time alert feed |
    | 🔬 **Inspector** | SHAP waterfall for any transaction |
    | 📊 **Model Performance** | AUC, confusion matrix, feature importance |
    | 📡 **Monitoring** | Feature drift & data quality |
    | ⚙️ **Settings** | Threshold & simulator controls |
    """
)

if not st.session_state.model_loaded:
    st.warning(
        "**Model not found.** "
        "Run `make train-all && make ensemble` to train the ensemble pipeline, "
        "then restart the dashboard."
    )
else:
    st.info(
        "The transaction simulator is running in the background. "
        "Navigate to **Live Alerts** or **Overview** to see real-time results."
    )
