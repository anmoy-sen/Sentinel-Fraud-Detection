"""
api/main.py
───────────
High-throughput, production-grade REST API for Sentinel Fraud Detection.

Architecture
------------
Separates service logic from HTTP routing so functions can be unit-tested
directly without HTTP mocking overhead.
"""

from __future__ import annotations

import os
import sys
import time

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from api.schemas import (
    AnalystFeedbackRequest,
    BatchTransactionRequest,
    HealthResponse,
    PredictionResponse,
    TransactionRequest,
)
from src.inference.feature_service import FeastFeatureService
from src.inference.predictor import FraudPredictor
from src.rules.velocity_rules import VelocityRulesEngine
from src.simulation.alert_store import AlertStore

# Global lazy singletons
_predictor = None
_alert_store = None
_rules_engine = None
_feature_service = None


def _get_services():
    global _predictor, _alert_store, _rules_engine, _feature_service
    if _predictor is None:
        cfg_path = os.path.join(ROOT, "config.yaml")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        pipeline_path = os.path.join(ROOT, cfg["model"]["ensemble_path"])
        feat_cols_path = os.path.join(ROOT, cfg["model"]["feature_cols_path"])
        db_path = os.path.join(ROOT, cfg["alerts"]["db_path"])

        _predictor = FraudPredictor()
        if os.path.exists(pipeline_path) and os.path.exists(feat_cols_path):
            _predictor.load(pipeline_path, feat_cols_path)

        _alert_store = AlertStore(db_path)
        _rules_engine = VelocityRulesEngine(cfg)
        _feature_service = FeastFeatureService()

    return _predictor, _alert_store, _rules_engine, _feature_service


# ─── Service Functions ────────────────────────────────────────────────────────

def service_health_check() -> HealthResponse:
    predictor, _, _, _ = _get_services()
    return HealthResponse(
        status="healthy" if predictor.is_loaded else "model_unloaded",
        model_loaded=predictor.is_loaded,
        model_type="Soft-Voting GBDT Ensemble (LGBM + XGB + CatBoost)",
        version="2.0.0",
        ensemble_auc=0.9729,
    )


def service_predict_transaction(tx: TransactionRequest) -> PredictionResponse:
    predictor, alert_store, rules_engine, feature_service = _get_services()
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline is not loaded on this server.",
        )

    t0 = time.perf_counter()
    row_data = tx.model_dump()
    row_features = row_data.pop("features", {}) or {}
    combined = {**row_data, **row_features}

    # Enrich with real-time Feast features if available
    if feature_service and feature_service.is_available:
        try:
            combined = feature_service.enrich_transaction(combined)
        except Exception:
            pass

    df_row = pd.DataFrame([combined])

    # Rule evaluation
    triggered_rules = rules_engine.evaluate_transaction(combined)

    # Model inference
    pred_res = predictor.predict(df_row)[0]
    latency_ms = (time.perf_counter() - t0) * 1000

    fraud_p = pred_res["fraud_proba"]
    is_fraud = pred_res["is_fraud"] or bool(triggered_rules)

    risk_level = "CRITICAL" if fraud_p >= 0.85 else ("HIGH" if fraud_p >= 0.65 else ("MEDIUM" if fraud_p >= 0.40 else "LOW"))
    tx_id = tx.TransactionID or f"TX_{int(time.time() * 1000)}"

    # Record alert
    alert_store.insert_alert({
        "transaction_id": tx_id,
        "fraud_proba": fraud_p,
        "is_fraud": is_fraud,
        "amount": tx.TransactionAmt,
        "card1": str(tx.card1 or ""),
        "rule_flags": "; ".join(triggered_rules),
        "latency_ms": latency_ms,
    })

    return PredictionResponse(
        transaction_id=tx_id,
        fraud_probability=round(fraud_p, 4),
        is_fraud=is_fraud,
        risk_level=risk_level,
        rule_flags=triggered_rules,
        latency_ms=round(latency_ms, 2),
    )


def service_predict_batch(batch: BatchTransactionRequest) -> list[PredictionResponse]:
    return [service_predict_transaction(tx) for tx in batch.transactions]


def service_submit_feedback(feedback: AnalystFeedbackRequest) -> dict:
    _, alert_store, _, _ = _get_services()
    if alert_store is None:
        raise HTTPException(status_code=500, detail="Alert store not initialized.")

    action = "CONFIRMED_FRAUD" if feedback.confirmed_fraud else "APPROVED_LEGIT"
    alert_store.record_analyst_action(
        transaction_id=feedback.transaction_id,
        action=action,
        analyst_name=feedback.analyst_id,
        notes=feedback.notes or "",
    )

    return {
        "status": "success",
        "message": f"Feedback for transaction {feedback.transaction_id} recorded successfully.",
        "analyst_id": feedback.analyst_id,
        "action": action,
        "confirmed_fraud": feedback.confirmed_fraud,
    }


# ─── FastAPI Application ──────────────────────────────────────────────────────

app = FastAPI(
    title="Sentinel Fraud Detection API",
    description="Real-time transaction fraud scoring powered by a Soft-Voting GBDT Ensemble (LightGBM + XGBoost + CatBoost)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health_endpoint():
    return service_health_check()


@app.post("/v1/predict", response_model=PredictionResponse)
def predict_endpoint(tx: TransactionRequest):
    return service_predict_transaction(tx)


@app.post("/v1/predict/batch", response_model=list[PredictionResponse])
def predict_batch_endpoint(batch: BatchTransactionRequest):
    return service_predict_batch(batch)


@app.post("/v1/feedback")
def feedback_endpoint(feedback: AnalystFeedbackRequest):
    return service_submit_feedback(feedback)
