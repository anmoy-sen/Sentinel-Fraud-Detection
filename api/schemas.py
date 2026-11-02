"""
schemas.py
──────────
Pydantic schemas for the Sentinel REST API.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    TransactionID: Optional[str] = Field(None, description="Unique transaction ID")
    TransactionAmt: float = Field(..., gt=0, description="Transaction dollar amount")
    card1: Optional[str] = Field(None, description="Primary card identifier")
    card2: Optional[float] = Field(None, description="Secondary card identifier")
    P_emaildomain: Optional[str] = Field(None, description="Purchaser email domain")
    R_emaildomain: Optional[str] = Field(None, description="Recipient email domain")
    features: Optional[dict[str, Any]] = Field(default_factory=dict, description="Additional engineered features")

    model_config = {
        "json_schema_extra": {
            "example": {
                "TransactionID": "T_987654",
                "TransactionAmt": 350.00,
                "card1": "12345",
                "P_emaildomain": "gmail.com",
                "R_emaildomain": "yahoo.com",
            }
        }
    }


class BatchTransactionRequest(BaseModel):
    transactions: list[TransactionRequest]


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    rule_flags: list[str]
    latency_ms: float


class AnalystFeedbackRequest(BaseModel):
    transaction_id: str
    confirmed_fraud: bool
    analyst_id: str = "analyst_1"
    notes: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    version: str
    ensemble_auc: float
