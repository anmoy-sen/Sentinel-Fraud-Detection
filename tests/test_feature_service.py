"""
tests/test_feature_service.py
─────────────────────────────
Unit tests for Feast Feature Store service.
"""

from src.inference.feature_service import FeastFeatureService


def test_feast_feature_service_initialization():
    service = FeastFeatureService()
    assert service is not None


def test_feast_online_feature_retrieval():
    service = FeastFeatureService()
    # Test with sample card ids
    features = service.get_online_card_features(["12345", "99999"])
    assert len(features) == 2
    for f in features:
        assert "card1_TransactionAmt_mean" in f
        assert "card1_TransactionAmt_std" in f
        assert "card1_freq" in f
        assert isinstance(f["card1_TransactionAmt_mean"], float)
        assert isinstance(f["card1_TransactionAmt_std"], float)
        assert isinstance(f["card1_freq"], (int, float))


def test_feast_enrich_transaction():
    service = FeastFeatureService()
    tx = {
        "TransactionID": "TX_TEST_01",
        "TransactionAmt": 250.0,
        "card1": "12345",
    }
    enriched = service.enrich_transaction(tx)
    assert enriched["TransactionID"] == "TX_TEST_01"
    assert "card1_TransactionAmt_mean" in enriched
    assert "card1_TransactionAmt_std" in enriched
    assert "card1_freq" in enriched
