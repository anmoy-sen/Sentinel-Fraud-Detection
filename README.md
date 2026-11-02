# 🛡️ Sentinel — Enterprise Real-Time Fraud Detection & Risk Decisioning Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42+-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Feast](https://img.shields.io/badge/Feature%20Store-Feast%200.67-orange.svg)](https://feast.dev/)
[![DVC](https://img.shields.io/badge/MLOps-DVC%20Tracked-945DD6.svg?logo=dvc&logoColor=white)](https://dvc.org)
[![Tests](https://img.shields.io/badge/Pytest-42%2F42%20Passing-brightgreen.svg)](https://docs.pytest.org)
[![AUC](https://img.shields.io/badge/ROC--AUC-0.9729-success.svg)]()

Sentinel is an industry-grade, end-to-end **Fraud Detection & Transaction Risk Decisioning Platform** built on the 590,000+ transaction [IEEE-CIS Fraud Detection benchmark](https://www.kaggle.com/competitions/ieee-fraud-detection). 

Engineered with production-first principles, Sentinel combines a **Soft-Voting GBDT Trinity Ensemble (LightGBM + XGBoost + CatBoost)**, an ultra-low-latency **Feast Feature Store**, real-time **Velocity Rule Heuristics**, an executive **Financial Cost-Benefit Matrix ($ ROI)**, and a **Human-in-the-Loop Analyst Investigation Workbench**.

---

## 🏗️ End-to-End System Architecture

```
                                  [ INCOMING TRANSACTION ]
                                             │
                                             ▼
                     ┌───────────────────────────────────────────────┐
                     │          FastAPI Microservice / Stream        │
                     │                 (p95 < 15ms)                  │
                     └───────────────────────┬───────────────────────┘
                                             │
                     ┌───────────────────────┴───────────────────────┐
                     │                                               │
                     ▼                                               ▼
     ┌───────────────────────────────┐               ┌───────────────────────────────┐
     │      Feast Feature Store      │               │     Velocity Rules Engine     │
     │  (Sub-ms Online Profile Sync) │               │   (Sliding-Window Frequency,  │
     │   • card1_TransactionAmt_mean │               │    Nighttime High-Value Spikes│
     │   • card1_TransactionAmt_std  │               │    Domain Discrepancy Checks) │
     │   • card1_freq velocity       │               └───────────────┬───────────────┘
     └───────────────┬───────────────┘                               │
                     │                                               │
                     └───────────────────────┬───────────────────────┘
                                             ▼
                     ┌───────────────────────────────────────────────┐
                     │         GBDT Trinity Soft-Voting Ensemble     │
                     │  ┌─────────────┐ ┌─────────────┐ ┌──────────┐ │
                     │  │  LightGBM   │ │   XGBoost   │ │ CatBoost │ │
                     │  │  (Hist GBDT)│ │(Exact Split)│ │(Cat High)│ │
                     │  └──────┬──────┘ └──────┬──────┘ └────┬─────┘ │
                     │         └───────────────┼─────────────┘       │
                     │                         ▼                     │
                     │             Ensemble AUC: 0.9729              │
                     └───────────────────────┬───────────────────────┘
                                             │
                                             ▼
                     ┌───────────────────────────────────────────────┐
                     │           Explainable AI & Decision Sink      │
                     │  • SHAP Waterfall Attribution (Top Features)  │
                     │  • Financial ROI Cost-Optimal Threshold (0.52)│
                     │  • SQLite Alert Store & Audit Log Persistence │
                     └───────────────────────┬───────────────────────┘
                                             │
                     ┌───────────────────────┴───────────────────────┐
                     │                                               │
                     ▼                                               ▼
┌─────────────────────────────────────────┐     ┌─────────────────────────────────────────┐
│     Human-in-the-Loop Analyst Queue     │     │      Model Monitoring & Drift (PSI)     │
│   • PENDING_REVIEW Triage Stream        │     │   • Population Stability Index Check    │
│   • APPROVE / BLOCK / ESCALATE Actions  │     │   • Alert if PSI > 0.20                 │
│   • Immutable Audit Trail               │     │   • Triggers Manual 'dvc repro' Retrain │
└─────────────────────────────────────────┘     └─────────────────────────────────────────┘
```

---

## 💡 Key Architectural Highlights

### 1. 🌲 GBDT Tournament & Ablation Study
Rather than relying on a generic single model, Sentinel pitted four distinct tree algorithms across 630+ engineered features. Baseline Random Forest suffered from extreme false positive rates on high-cardinality interaction features. Replacing Random Forest with **CatBoost** formed the production **GBDT Trinity**, elevating ROC-AUC to **0.9729** and precision to **84.9%**.

| Model | ROC-AUC | F1-Score | Precision | Recall | Architecture & Role |
|:------|:-------:|:--------:|:---------:|:------:|:--------------------|
| Random Forest (Baseline) | 0.8778 | 0.2552 | 0.1534 | 0.7590 | Bagging baseline (high FP rate on card interactions) |
| CatBoost | **0.9340** | **0.6503** | **0.7007** | **0.6067** | High-cardinality interaction specialist |
| XGBoost | **0.9491** | **0.6924** | **0.7826** | **0.6209** | Exact greedy split gradient booster |
| LightGBM | **0.9620** | **0.7010** | **0.8120** | **0.6350** | Fast histogram-based gradient booster |
| **Sentinel Ensemble (LGBM + XGB + CatBoost)** | **0.9729** | **0.7236** | **0.8490** | **0.6568** | **Production Soft-Voting GBDT Trinity** |

---

### 2. 💰 Financial Cost-Benefit Matrix & Profit Optimization
In enterprise fraud operations, optimizing solely for traditional ML metrics (like F1-score or accuracy) is a critical anti-pattern because errors carry **asymmetric financial penalties**:
- **False Negative (Missed Fraud)**: Direct loss of the stolen transaction dollar amount plus an unavoidable **\$25 bank chargeback penalty fee**.
- **False Positive (Good User Blocked)**: Customer friction, support overhead, and churn penalty (modeled at **\$15**).
- **True Positive (Blocked Fraud)**: Gross dollar savings recovered minus a minor investigation triage cost (**\$2**).

Sentinel computes an exact financial cost curve across all decision thresholds:
$$\text{Net Savings} = \text{Baseline Fraud Loss} - \left(\sum \text{Chargeback Losses} + \sum \text{Friction Costs} + \sum \text{Investigation Costs}\right)$$

> **ROI Insight**: The cost-optimal decision threshold (**~0.52**) differs significantly from the naive F1-optimal threshold (**0.63**), generating thousands of dollars in additional net fraud recovery by preventing costly chargeback penalties without inducing customer churn.

---

### 3. ⚡ Feast Real-Time Feature Store
Production fraud engines cannot afford expensive aggregations over millions of rows during a sub-50ms checkout flow.
- **Offline Store**: Historical Parquet feature store (`feature_store/data/card_risk_features.parquet`) tracking cardholder baseline metrics (`card1_TransactionAmt_mean`, `card1_TransactionAmt_std`, `card1_freq`).
- **Online Store**: Materialized low-latency SQLite key-value store (`feature_store/data/online_store.db`).
- **Inference Hydration**: Incoming raw transaction requests query Feast in sub-millisecond time (`FeastFeatureService.enrich_transaction`), feeding the pipeline pre-computed entity risk profiles without latency overhead.

---

### 4. 👨‍💼 Human-in-the-Loop Analyst Investigation Workbench
Critical alerts should not simply sit in raw database logs. Sentinel's Streamlit operations center features an integrated **Analyst Action Queue**:
- **Triage Queue**: Displays transactions flagged by the model or velocity heuristics (`PENDING_REVIEW`).
- **Investigation Dossier**: Full metadata display (Transaction Amount, Cardholder ID, Model Probability, Rule Triggers, Timestamp).
- **One-Click Actions**:
  - `✅ Approve (Legit)`: Clears transaction and marks status as `APPROVED_LEGIT`.
  - `🚫 Confirm Fraud (Block)`: Confirms fraud, blacklists cardholder, marks status as `CONFIRMED_FRAUD`.
  - `🔍 Escalate`: Re-routes to Tier-2 financial crimes investigators (`ESCALATED`).
- **Immutable Audit Trail**: Logs reviewer identity, action timestamp, and rationale notes for regulatory compliance (SOX/PCI-DSS).

---

### 5. 📦 MLOps: DVC Lineage & Drift-Triggered Manual Retraining
Model training, threshold optimization, and dataset dependencies are version-controlled via **Data Version Control (DVC)**:
- **Continuous Monitoring**: Streaming features are monitored via **Population Stability Index (PSI)** against training distributions.
- **Drift Action Threshold**: If feature drift exceeds the critical boundary ($\text{PSI} > 0.20$), the monitoring suite fires a high-priority drift alert.
- **Controlled Manual Retraining**: Rather than unsafe opaque automated re-training, engineers trigger reproducible pipeline retraining with:
  ```bash
  dvc repro
  ```
  DVC automatically evaluates dependency hashes, executes updated training stages, recalculates optimal decision thresholds, and logs new candidate metrics into `metrics/eval_metrics.json`.

---

## ⚡ Production Latency & Performance Benchmarks

Measured on single-node standard hardware (batch size = 1, single transaction inference):

| Pipeline Stage | Implementation | P50 Latency | P95 Latency |
|:---------------|:---------------|:-----------:|:-----------:|
| Online Feature Store Query | Feast (SQLite Key-Value) | 0.8 ms | 1.4 ms |
| Velocity Rule Evaluation | In-Memory Sliding Window | 0.2 ms | 0.4 ms |
| Preprocessing & Imputation | Vectorized Pandas / Scikit-Learn | 1.5 ms | 2.6 ms |
| GBDT Trinity Inference | Soft-Voting (LGBM + XGB + CatBoost) | 6.8 ms | 10.2 ms |
| Persistence & Audit Sink | Thread-Safe SQLite WAL Mode | 0.9 ms | 1.8 ms |
| **Total End-to-End Latency** | **FastAPI Synchronous Path** | **~10.2 ms** | **< 16.5 ms** |

---

## 🚀 Quickstart

### 1. Prerequisites & Environment Setup

```bash
# Clone repository
git clone https://github.com/your-username/sentinel-fraud-detection.git
cd Sentinel-Fraud-Detection

# Create virtual environment & install dependencies
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Feature Store & Model Preparation

```bash
# Materialize Feast online feature store
python feature_store/prepare_feature_store.py

# (Optional) Retrain or reproduce pipeline via DVC
dvc repro
```

### 3. Launch Services

#### Option A: Interactive Operations Dashboard (Streamlit)
```bash
streamlit run app/main.py
# → Visit http://localhost:8501
```

#### Option B: Real-Time High-Throughput REST API (FastAPI)
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# → Swagger UI: http://localhost:8000/docs
```

#### Option C: Production Docker Deployment
```bash
docker-compose -f docker/docker-compose.yml up --build
```

---

## 🧪 Comprehensive Test Suite

Sentinel includes **42 automated unit and integration tests** verifying service health, real-time inference, feature store enrichment, financial cost calculations, simulator thread safety, and velocity rule triggers:

```bash
pytest tests/ -v
```

```
tests/test_api.py::test_health_service PASSED                            [  2%]
tests/test_api.py::test_predict_single_service PASSED                    [  4%]
tests/test_api.py::test_predict_batch_service PASSED                     [  7%]
tests/test_api.py::test_feedback_service PASSED                          [  9%]
tests/test_feature_engineering.py::test_amount_features_added PASSED     [ 11%]
...
tests/test_feature_service.py::test_feast_online_feature_retrieval PASSED [ 38%]
tests/test_feature_service.py::test_feast_enrich_transaction PASSED      [ 40%]
tests/test_financial_cost.py::test_compute_financial_cost_matrix_structure PASSED [ 42%]
tests/test_financial_cost.py::test_plot_financial_cost_curve PASSED      [ 45%]
tests/test_predictor.py::test_predictor_loads PASSED                     [ 47%]
tests/test_simulator.py::TestAlertStore::test_thread_safety PASSED       [ 83%]
tests/test_velocity_rules.py::test_card_velocity_triggered PASSED        [ 92%]
============================= 42 passed in 12.07s =============================
```

---

## 📁 Repository Structure

```
Sentinel-Fraud-Detection/
├── api/                        # FastAPI Production Microservice
│   ├── main.py                 # REST endpoints (/v1/predict, /v1/feedback)
│   └── schemas.py              # Pydantic v2 validation contracts
├── app/                        # Streamlit Operations Command Center
│   ├── main.py                 # Multi-page dashboard entry
│   └── pages/
│       ├── 01_overview.py      # Real-time KPIs & Dollar Losses Prevented ($)
│       ├── 02_alerts.py        # Live Alert Feed & Analyst Investigation Queue
│       ├── 03_inspector.py     # SHAP Waterfall Attribution Drill-Down
│       ├── 04_model.py         # Performance Curves & Financial Cost-Benefit Matrix
│       ├── 05_settings.py      # Threshold Calibration & Simulator Controls
│       └── 06_monitoring.py    # Population Stability Index (PSI) Drift Monitor
├── data/
│   ├── raw/                    # Raw IEEE-CIS datasets
│   └── processed/              # Memory-optimized feature engineered datasets
├── docker/                     # Docker & compose infrastructure
├── feature_store/              # Feast Feature Store
│   ├── feature_store.yaml      # Feature store registry configuration
│   ├── features.py             # Entity & FeatureView definitions
│   └── prepare_feature_store.py# Offline parquet generation & online sync
├── metrics/                    # DVC tracked model metrics & threshold sweeps
├── models/                     # GBDT Trinity serialized pipelines
├── src/
│   ├── features/               # Modular feature engineering (time, card, V-cols)
│   ├── inference/              # FraudPredictor & FeastFeatureService
│   ├── models/structured/      # GBDT Trinity architectures & Financial Cost Matrix
│   ├── monitoring/             # PSI Feature Drift Engine
│   ├── rules/                  # Sliding-window velocity heuristic rules
│   └── simulation/             # TransactionSimulator & SQLite AlertStore
├── tests/                      # Pytest test suite (42 tests)
├── config.yaml                 # Central system configuration
├── dvc.yaml                    # DVC pipeline DAG definition
├── Makefile                    # Unified developer task runner
└── requirements.txt            # Locked project dependencies
```

---

## 🛠️ Tech Stack & Tooling

- **Machine Learning**: LightGBM, XGBoost, CatBoost, Scikit-Learn, Optuna, SHAP
- **Serving & Microservices**: FastAPI, Uvicorn, Pydantic v2
- **Feature Store & Storage**: Feast, Apache Parquet, SQLite (WAL mode)
- **MLOps & Experiment Tracking**: DVC (Data Version Control), Git
- **UI & Analytics**: Streamlit, Plotly, Pandas, NumPy
- **Testing & Quality Assurance**: Pytest, Pytest-Cov

---

## 👤 Author & Recruiter Contact

Developed by **Anmoy** — Machine Learning & MLOps Engineer.
Open to production machine learning engineering, risk intelligence, and data platform roles.
Feel free to reach out via [GitHub](https://github.com/anmoy-sen) or [LinkedIn](https://www.linkedin.com/in/anmoy-sen-8270aa380?utm_source=share_via&utm_content=profile&utm_medium=member_android).
