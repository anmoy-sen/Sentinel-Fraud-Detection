"""
test_simulator.py
─────────────────
Unit tests for the TransactionSimulator and AlertStore.
"""

import os
import tempfile
import time

import numpy as np
import pandas as pd
import pytest

from src.simulation.alert_store import AlertStore


# ── AlertStore tests ──────────────────────────────────────────────────────────

class TestAlertStore:
    def setup_method(self):
        self._tmp = tempfile.mkdtemp()
        self.store = AlertStore(db_path=os.path.join(self._tmp, "test_alerts.db"))

    def test_insert_and_retrieve(self):
        self.store.insert_alert({
            "transaction_id": "TX001",
            "fraud_proba": 0.92,
            "is_fraud": True,
            "amount": 250.0,
            "card1": "1234",
        })
        df = self.store.get_recent(10)
        assert len(df) == 1
        assert df.iloc[0]["transaction_id"] == "TX001"
        assert abs(df.iloc[0]["fraud_proba"] - 0.92) < 1e-5

    def test_insert_many(self):
        records = [
            {"transaction_id": f"TX{i:03d}", "fraud_proba": float(i) / 100,
             "is_fraud": i > 80, "amount": float(i * 10)}
            for i in range(100)
        ]
        self.store.insert_many(records)
        assert self.store.count() == 100

    def test_fraud_count(self):
        self.store.insert_many([
            {"transaction_id": "A", "fraud_proba": 0.9, "is_fraud": True},
            {"transaction_id": "B", "fraud_proba": 0.1, "is_fraud": False},
            {"transaction_id": "C", "fraud_proba": 0.85, "is_fraud": True},
        ])
        assert self.store.fraud_count() == 2

    def test_get_stats_empty(self):
        stats = self.store.get_stats()
        assert stats["total"] == 0
        assert stats["fraud_count"] == 0
        assert stats["fraud_rate"] == 0.0

    def test_get_stats_populated(self):
        self.store.insert_many([
            {"transaction_id": f"T{i}", "fraud_proba": 0.9, "is_fraud": True}
            for i in range(5)
        ] + [
            {"transaction_id": f"L{i}", "fraud_proba": 0.1, "is_fraud": False}
            for i in range(15)
        ])
        stats = self.store.get_stats()
        assert stats["total"] == 20
        assert stats["fraud_count"] == 5
        assert abs(stats["fraud_rate"] - 0.25) < 1e-5

    def test_clear(self):
        self.store.insert_alert({"transaction_id": "X", "fraud_proba": 0.5, "is_fraud": False})
        assert self.store.count() == 1
        self.store.clear()
        assert self.store.count() == 0

    def test_get_recent_limit(self):
        records = [{"transaction_id": f"T{i}", "fraud_proba": 0.5, "is_fraud": False}
                   for i in range(50)]
        self.store.insert_many(records)
        df = self.store.get_recent(10)
        assert len(df) == 10

    def test_thread_safety(self):
        """Concurrent inserts should not corrupt the database."""
        import threading

        def insert_batch(start):
            for i in range(start, start + 20):
                self.store.insert_alert({
                    "transaction_id": f"TH{i}",
                    "fraud_proba": 0.5,
                    "is_fraud": False,
                })

        threads = [threading.Thread(target=insert_batch, args=(i * 20,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert self.store.count() == 100


# ── Simulator smoke test ──────────────────────────────────────────────────────

class TestSimulatorSmoke:
    """
    Lightweight smoke test — uses a mock predictor and a tiny CSV.
    Does NOT require a trained model or real data on disk.
    """

    def _make_csv(self, tmp_dir, n=100):
        import os
        rng = np.random.default_rng(7)
        df = pd.DataFrame({
            "TransactionID": [f"T{i}" for i in range(n)],
            "TransactionAmt": rng.uniform(1, 1000, n),
            "card1": rng.integers(1000, 9999, n),
            **{f"V{i}": rng.uniform(0, 5, n) for i in range(1, 10)},
        })
        path = os.path.join(tmp_dir, "test_sim.csv")
        df.to_csv(path, index=False)
        return path

    def test_simulator_starts_and_produces_records(self):
        from unittest.mock import MagicMock
        from src.simulation.simulator import TransactionSimulator

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._make_csv(tmp, n=60)
            store = AlertStore(db_path=os.path.join(tmp, "sim_alerts.db"))

            # Mock predictor
            predictor = MagicMock()
            predictor.predict.side_effect = lambda df: [
                {"fraud_proba": 0.9, "is_fraud": True, "latency_ms": 1.0}
                for _ in range(len(df))
            ]

            sim_cfg = {
                "mode": "batch",
                "batch_size": 50,
                "batch_interval_sec": 0,   # no sleep in test
                "test_data_path": csv_path,
                "chunk_size": 50,
                "custom_speed": 100,
            }
            sim = TransactionSimulator(predictor, store, sim_cfg)
            sim.start()

            # Wait up to 3 s for records to appear
            deadline = time.time() + 3
            while store.count() == 0 and time.time() < deadline:
                time.sleep(0.1)

            sim.stop()
            assert store.count() > 0, "No records inserted by simulator"

    def test_simulator_pause_resume(self):
        from unittest.mock import MagicMock
        from src.simulation.simulator import TransactionSimulator

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._make_csv(tmp, n=200)
            store = AlertStore(db_path=os.path.join(tmp, "sim_pr.db"))
            predictor = MagicMock()
            predictor.predict.side_effect = lambda df: [
                {"fraud_proba": 0.5, "is_fraud": False, "latency_ms": 0.5}
                for _ in range(len(df))
            ]

            sim_cfg = {
                "mode": "batch",
                "batch_size": 100,
                "batch_interval_sec": 0,
                "test_data_path": csv_path,
                "chunk_size": 100,
                "custom_speed": 100,
            }
            sim = TransactionSimulator(predictor, store, sim_cfg)
            sim.start()
            time.sleep(0.2)
            sim.pause()
            assert sim.is_paused
            sim.resume()
            assert not sim.is_paused
            sim.stop()
            assert not sim.is_running
