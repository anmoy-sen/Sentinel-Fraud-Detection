"""
feature_service.py
──────────────────
Online Feature Service client powered by Feast.
Enriches real-time transaction streams with sub-millisecond historical features.
"""

from __future__ import annotations

import os
from typing import Any
import pandas as pd
from feast import FeatureStore

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
REPO_PATH = os.path.join(ROOT, "feature_store")


class FeastFeatureService:
    """
    Feast client wrapper for retrieving online features during inference.
    """

    def __init__(self, repo_path: str = REPO_PATH):
        self._store = None
        self._repo_path = repo_path
        if os.path.exists(os.path.join(repo_path, "feature_store.yaml")):
            try:
                self._store = FeatureStore(repo_path=repo_path)
            except Exception as e:
                print(f"[Feast] Could not initialize FeatureStore: {e}")

    @property
    def is_available(self) -> bool:
        return self._store is not None

    def get_online_card_features(self, card_ids: list[str]) -> list[dict[str, Any]]:
        """
        Query Feast online store for card risk profiles.
        Returns a list of feature dicts aligned with card_ids.
        """
        if not self.is_available or not card_ids:
            return [{"card1_TransactionAmt_mean": 120.0, "card1_TransactionAmt_std": 40.0, "card1_freq": 1} for _ in card_ids]

        entity_rows = [{"card1": str(c)} for c in card_ids]
        feature_refs = [
            "card_risk_profile:card1_TransactionAmt_mean",
            "card_risk_profile:card1_TransactionAmt_std",
            "card_risk_profile:card1_freq",
        ]

        try:
            response = self._store.get_online_features(
                features=feature_refs,
                entity_rows=entity_rows,
            )
            response_dict = response.to_dict()

            results = []
            for i in range(len(card_ids)):
                mean_val = response_dict.get("card1_TransactionAmt_mean", [None])[i]
                std_val = response_dict.get("card1_TransactionAmt_std", [None])[i]
                freq_val = response_dict.get("card1_freq", [None])[i]

                results.append({
                    "card1_TransactionAmt_mean": float(mean_val) if mean_val is not None else 120.0,
                    "card1_TransactionAmt_std": float(std_val) if std_val is not None else 40.0,
                    "card1_freq": int(freq_val) if freq_val is not None else 1,
                })
            return results
        except Exception as e:
            print(f"[Feast] Error retrieving online features: {e}")
            return [{"card1_TransactionAmt_mean": 120.0, "card1_TransactionAmt_std": 40.0, "card1_freq": 1} for _ in card_ids]

    def enrich_transaction(self, tx_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Enrich a single transaction with historical card features from Feast.
        """
        card1 = str(tx_dict.get("card1", ""))
        features = self.get_online_card_features([card1])[0]
        return {**tx_dict, **features}
