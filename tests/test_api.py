"""
test_api.py
───────────
Tests for the Sentinel API service logic and schemas.
"""

import pytest
from api.main import (
    service_health_check,
    service_predict_transaction,
    service_predict_batch,
    service_submit_feedback,
)
from api.schemas import (
    TransactionRequest,
    BatchTransactionRequest,
    AnalystFeedbackRequest,
)


def test_health_service():
    res = service_health_check()
    assert res.status == "healthy"
    assert res.model_loaded is True
    assert "GBDT" in res.model_type
    assert res.ensemble_auc > 0.95


def test_predict_single_service():
    req = TransactionRequest(
        TransactionID="TX_TEST_001",
        TransactionAmt=150.00,
        card1="10024",
        P_emaildomain="gmail.com",
        R_emaildomain="gmail.com",
    )
    res = service_predict_transaction(req)
    assert res.transaction_id == "TX_TEST_001"
    assert 0.0 <= res.fraud_probability <= 1.0
    assert isinstance(res.is_fraud, bool)
    assert res.risk_level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert res.latency_ms > 0


def test_predict_batch_service():
    batch = BatchTransactionRequest(
        transactions=[
            TransactionRequest(TransactionID="T1", TransactionAmt=45.0, card1="1111"),
            TransactionRequest(TransactionID="T2", TransactionAmt=999.0, card1="2222"),
        ]
    )
    res_list = service_predict_batch(batch)
    assert len(res_list) == 2
    assert res_list[0].transaction_id == "T1"
    assert res_list[1].transaction_id == "T2"


def test_feedback_service():
    feedback = AnalystFeedbackRequest(
        transaction_id="TX_TEST_001",
        confirmed_fraud=True,
        analyst_id="investigator_alice",
        notes="Cardholder confirmed unauthorized transaction.",
    )
    res = service_submit_feedback(feedback)
    assert res["status"] == "success"
    assert res["confirmed_fraud"] is True
