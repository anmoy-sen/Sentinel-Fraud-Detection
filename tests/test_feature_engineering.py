"""
test_feature_engineering.py
───────────────────────────
Smoke tests for every feature-creation module.
Each function must run without errors and return the expected columns.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import (
    create_transaction_amount_features,
    create_time_features,
    create_card_features,
    create_email_features,
    create_device_features,
    create_address_features,
    create_v_features,
    create_c_d_features,
    create_m_features,
    create_aggregation_features,
)


# ─── Shared minimal DataFrame ─────────────────────────────────────────────────

@pytest.fixture
def base_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 50
    return pd.DataFrame(
        {
            "TransactionID": range(n),
            "TransactionAmt": rng.uniform(1, 1000, n),
            "TransactionDT": rng.integers(86400, 86400 * 180, n),
            "ProductCD": rng.choice(["W", "H", "C", "S", "R"], n),
            "card1": rng.integers(1000, 9999, n),
            "card2": rng.integers(100, 999, n).astype(float),
            "card4": rng.choice(["visa", "mastercard", None], n),
            "card6": rng.choice(["credit", "debit", None], n),
            "addr1": rng.integers(100, 500, n).astype(float),
            "addr2": rng.integers(10, 100, n).astype(float),
            "dist1": rng.uniform(0, 1000, n),
            "dist2": rng.uniform(0, 500, n),
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", None], n),
            "R_emaildomain": rng.choice(["hotmail.com", None], n),
            "DeviceType": rng.choice(["mobile", "desktop", None], n),
            "DeviceInfo": rng.choice(["Samsung/Galaxy", "iOS/iPhone", "Windows/PC", None], n),
            "id_31": rng.choice(["chrome 80.0", "safari 13", None], n),
            "id_30": rng.choice(["Windows 10", "iOS 13", "Android 9", None], n),
            "id_33": rng.choice(["1920x1080", "375x812", None], n),
            **{f"V{i}": rng.uniform(0, 5, n) for i in range(1, 30)},
            **{f"C{i}": rng.integers(0, 50, n).astype(float) for i in range(1, 6)},
            **{f"D{i}": rng.uniform(0, 100, n) for i in range(1, 5)},
            **{f"M{i}": rng.choice(["T", "F", None], n) for i in range(1, 5)},
            **{f"id_0{i}": rng.uniform(-10, 10, n) for i in range(1, 10)},
        }
    )


# ─── Amount features ──────────────────────────────────────────────────────────

def test_amount_features_added(base_df):
    df = create_transaction_amount_features(base_df.copy())
    for col in ["TransactionAmt_Log", "TransactionAmt_cents", "TransactionAmt_is_round"]:
        assert col in df.columns, f"Missing column: {col}"
    assert not df["TransactionAmt_Log"].isna().all()


# ─── Time features ────────────────────────────────────────────────────────────

def test_time_features_added(base_df):
    df = create_time_features(base_df.copy())
    for col in ["Transaction_hour", "Transaction_dow", "Transaction_is_night"]:
        assert col in df.columns
    assert df["Transaction_hour"].between(0, 23).all()


# ─── Card features ────────────────────────────────────────────────────────────

def test_card_features_added(base_df):
    df = create_card_features(base_df.copy())
    assert "card1_card2" in df.columns


# ─── Email features ───────────────────────────────────────────────────────────

def test_email_features_added(base_df):
    df = create_email_features(base_df.copy())
    assert "P_emaildomain_prefix" in df.columns
    assert "email_domain_match" in df.columns


# ─── Device features ─────────────────────────────────────────────────────────

def test_device_features_added(base_df):
    df = create_device_features(base_df.copy())
    assert "DeviceType_is_mobile" in df.columns
    assert "Browser_is_chrome" in df.columns


# ─── Address features ────────────────────────────────────────────────────────

def test_address_features_added(base_df):
    df = create_address_features(base_df.copy())
    assert "dist1_log" in df.columns
    assert "addr1_missing" in df.columns


# ─── V-column features ───────────────────────────────────────────────────────

def test_v_features_added(base_df):
    df = create_v_features(base_df.copy())
    assert "v1_auth_sum" in df.columns
    assert "V_nan_ratio" in df.columns


# ─── C/D features ────────────────────────────────────────────────────────────

def test_cd_features_added(base_df):
    df = create_c_d_features(base_df.copy())
    assert "C_sum" in df.columns
    assert "D_mean" in df.columns


# ─── M features ──────────────────────────────────────────────────────────────

def test_m_features_added(base_df):
    df = create_m_features(base_df.copy())
    assert "M_true_count" in df.columns
    assert "M_true_ratio" in df.columns


# ─── Aggregation (no data leakage) ───────────────────────────────────────────

def test_aggregation_no_leakage(base_df):
    train = base_df.iloc[:30].copy()
    test = base_df.iloc[30:].copy()
    train_out, test_out = create_aggregation_features(train, test_df=test)
    assert "card1_freq" in train_out.columns
    assert "card1_freq" in test_out.columns
    # Unseen test cards should get 0 frequency, not NaN
    assert not test_out["card1_freq"].isna().any()
