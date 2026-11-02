"""
test_velocity_rules.py
───────────────────────
Unit tests for the VelocityRulesEngine.
"""

import time
import pytest
from src.rules.velocity_rules import VelocityRulesEngine


@pytest.fixture
def rules_engine():
    cfg = {
        "velocity_rules": {
            "card_tx_limit": 3,
            "card_tx_window_sec": 2,
            "amount_zscore_threshold": 3.0,
            "night_round_amount": True,
            "email_mismatch_amount_threshold": 500,
            "email_variety_limit": 2,
            "email_variety_window_sec": 5,
        }
    }
    return VelocityRulesEngine(cfg)


def test_no_rules_triggered_on_normal_tx(rules_engine):
    tx = {
        "card1": "1001",
        "TransactionAmt": 45.50,
        "P_emaildomain": "gmail.com",
        "R_emaildomain": "gmail.com",
        "Transaction_hour": 14,
    }
    flags = rules_engine.evaluate_transaction(tx)
    assert flags == []


def test_card_velocity_triggered(rules_engine):
    card = "9999"
    # Submit 3 transactions (under limit)
    for _ in range(3):
        rules_engine.evaluate_transaction({
            "card1": card,
            "TransactionAmt": 20.0,
            "Transaction_hour": 12,
        })

    # 4th transaction within window should trigger velocity rule
    flags = rules_engine.evaluate_transaction({
        "card1": card,
        "TransactionAmt": 20.0,
        "Transaction_hour": 12,
    })
    assert any("CARD_VELOCITY_EXCEEDED" in f for f in flags)


def test_night_round_amount(rules_engine):
    tx = {
        "card1": "5555",
        "TransactionAmt": 200.0,
        "Transaction_hour": 3,  # night
    }
    flags = rules_engine.evaluate_transaction(tx)
    assert any("NIGHT_ROUND_AMOUNT" in f for f in flags)


def test_high_value_email_mismatch(rules_engine):
    tx = {
        "card1": "7777",
        "TransactionAmt": 1200.0,
        "P_emaildomain": "gmail.com",
        "R_emaildomain": "yahoo.com",
        "Transaction_hour": 15,
    }
    flags = rules_engine.evaluate_transaction(tx)
    assert any("HIGH_VALUE_EMAIL_MISMATCH" in f for f in flags)


def test_clear_resets_history(rules_engine):
    card = "8888"
    for _ in range(3):
        rules_engine.evaluate_transaction({
            "card1": card,
            "TransactionAmt": 10.0,
        })
    rules_engine.clear()
    flags = rules_engine.evaluate_transaction({
        "card1": card,
        "TransactionAmt": 10.0,
    })
    assert not any("CARD_VELOCITY_EXCEEDED" in f for f in flags)
