"""
simulator.py
────────────
Replays test transactions through the FraudPredictor at a configurable speed
and routes results to:
  1. A thread-safe queue.Queue (for the dashboard's live feed)
  2. The AlertStore (for persistence)

Architecture
------------
  TransactionSimulator.start()
      └─ daemon thread → _run_loop()
              └─ reads CSV in chunks → scores batch → flushes queue
"""

from __future__ import annotations

import queue
import threading
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pandas as pd
import yaml

if TYPE_CHECKING:
    from src.inference.predictor import FraudPredictor
    from src.simulation.alert_store import AlertStore


class TransactionSimulator:
    """
    Replays ``test_data_path`` transactions through the predictor.

    Parameters
    ----------
    predictor   : FraudPredictor  (must be loaded)
    alert_store : AlertStore      (SQLite sink)
    config      : dict            (from config.yaml["simulator"])

    Attributes
    ----------
    output_queue : queue.Queue
        Every scored transaction is pushed here as a dict.
        Dashboard consumers can call .get_nowait() to drain it.
    """

    def __init__(
        self,
        predictor: "FraudPredictor",
        alert_store: "AlertStore",
        config: dict,
    ):
        self._predictor = predictor
        self._store = alert_store
        self._cfg = config

        from src.rules.velocity_rules import VelocityRulesEngine
        self._rules_engine = VelocityRulesEngine(config)

        self.output_queue: queue.Queue = queue.Queue(maxsize=5000)

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # not paused initially

        self._processed = 0
        self._fraud_count = 0
        self._lock = threading.Lock()

    # ── Public interface ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background simulation thread (idempotent)."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._pause_event.set()
        self._thread = threading.Thread(
            target=self._run_loop, name="sentinel-simulator", daemon=True
        )
        self._thread.start()

    def pause(self) -> None:
        self._pause_event.clear()

    def resume(self) -> None:
        self._pause_event.set()

    def stop(self) -> None:
        self._stop_event.set()
        self._pause_event.set()  # Unblock thread if paused
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)

    def reset(self) -> None:
        """Stop, clear the queue and counters, ready to restart."""
        self.stop()
        if hasattr(self, "_rules_engine"):
            self._rules_engine.clear()
        with self.output_queue.mutex:
            self.output_queue.queue.clear()
        with self._lock:
            self._processed = 0
            self._fraud_count = 0

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive() and not self._stop_event.is_set())

    @property
    def is_paused(self) -> bool:
        return not self._pause_event.is_set()

    @property
    def processed(self) -> int:
        with self._lock:
            return self._processed

    @property
    def fraud_detected(self) -> int:
        with self._lock:
            return self._fraud_count

    # ── Internal simulation loop ──────────────────────────────────────────────

    def _run_loop(self) -> None:
        mode = self._cfg.get("mode", "batch")
        data_path = self._cfg["test_data_path"]
        chunk_size = self._cfg.get("chunk_size", 500)

        while not self._stop_event.is_set():
            # Replay infinitely (loop back after exhausting the file)
            reader = pd.read_csv(data_path, chunksize=chunk_size, low_memory=False)
            for chunk in reader:
                if self._stop_event.is_set():
                    return

                # Block if paused
                self._pause_event.wait()
                if self._stop_event.is_set():
                    return

                self._process_chunk(chunk, mode)

    def _process_chunk(self, chunk: pd.DataFrame, mode: str) -> None:
        if mode == "batch":
            self._flush_batch(chunk)
            interval = self._cfg.get("batch_interval_sec", 3)
            # Sleep with periodic check for stop/pause
            if interval > 0:
                step = 0.2
                elapsed = 0.0
                while elapsed < interval and not self._stop_event.is_set():
                    time.sleep(step)
                    elapsed += step
        else:
            # Custom speed: transactions per second
            speed = max(1, self._cfg.get("custom_speed", 50))
            delay_per_tx = 1.0 / speed
            batch_size = self._cfg.get("batch_size", 50)

            for start in range(0, len(chunk), batch_size):
                if self._stop_event.is_set():
                    return
                self._pause_event.wait()
                if self._stop_event.is_set():
                    return
                batch = chunk.iloc[start : start + batch_size]
                self._flush_batch(batch)
                sleep_dur = delay_per_tx * len(batch)
                if sleep_dur > 0:
                    time.sleep(sleep_dur)

    def _flush_batch(self, batch: pd.DataFrame) -> None:
        """Score a batch and push results to the queue + alert store."""
        if batch.empty:
            return
        try:
            results = self._predictor.predict(batch)
        except Exception as e:
            print(f"[Simulator] prediction error: {e}")
            return

        now_iso = datetime.now(timezone.utc).isoformat()
        records = []

        n_items = min(len(results), len(batch))
        for i in range(n_items):
            res = results[i]
            row = batch.iloc[i]
            triggered_rules = self._rules_engine.evaluate_transaction(row)
            rule_flags_str = "; ".join(triggered_rules) if triggered_rules else ""

            is_fraud = res["is_fraud"] or bool(triggered_rules)

            record = {
                "transaction_id": str(row.get("TransactionID", f"T{self._processed + i}")),
                "fraud_proba": res["fraud_proba"],
                "is_fraud": is_fraud,
                "timestamp": now_iso,
                "amount": float(row.get("TransactionAmt", 0.0)),
                "card1": str(row.get("card1", "")),
                "rule_flags": rule_flags_str,
                "latency_ms": res["latency_ms"],
            }
            records.append(record)

            # Non-blocking push to queue (drop if full)
            try:
                self.output_queue.put_nowait(record)
            except queue.Full:
                pass

        # Batch-write to SQLite
        self._store.insert_many(records)

        with self._lock:
            self._processed += len(records)
            self._fraud_count += sum(1 for r in records if r["is_fraud"])


# ─── Factory helper ───────────────────────────────────────────────────────────

def create_simulator(
    predictor: "FraudPredictor",
    alert_store: "AlertStore",
    config_path: str = "config.yaml",
) -> TransactionSimulator:
    """Instantiate a TransactionSimulator from config.yaml."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return TransactionSimulator(predictor, alert_store, cfg["simulator"])
