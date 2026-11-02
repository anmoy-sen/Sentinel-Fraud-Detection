"""
alert_store.py
──────────────
SQLite-backed alert store & Human-in-the-Loop Analyst Review Queue.

Responsibilities
----------------
* Persist fraud alerts between Streamlit re-runs.
* Manage analyst action workflows (Approve, Confirm Fraud, Escalate).
* Provide fast queries for dashboard widgets and audit logs.
* Thread-safe with robust connection management.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

import pandas as pd

# Schemas
_CREATE_ALERTS_TABLE = """
CREATE TABLE IF NOT EXISTS alerts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id  TEXT,
    fraud_proba     REAL,
    is_fraud        INTEGER,
    timestamp       TEXT,
    amount          REAL,
    card1           TEXT,
    rule_flags      TEXT,
    latency_ms      REAL,
    status          TEXT DEFAULT 'PENDING_REVIEW'
)
"""

_CREATE_REVIEWS_TABLE = """
CREATE TABLE IF NOT EXISTS analyst_reviews (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id  TEXT,
    action          TEXT,
    analyst_name    TEXT,
    notes           TEXT,
    reviewed_at     TEXT
)
"""

_INSERT_ALERT = """
INSERT INTO alerts
    (transaction_id, fraud_proba, is_fraud, timestamp, amount, card1, rule_flags, latency_ms, status)
VALUES
    (:transaction_id, :fraud_proba, :is_fraud, :timestamp, :amount, :card1, :rule_flags, :latency_ms, :status)
"""


class AlertStore:
    """
    Thread-safe SQLite-backed alert store and review queue.

    Usage
    -----
    store = AlertStore("data/alerts.db")
    store.insert_alert({"transaction_id": "T123", "fraud_proba": 0.91, ...})
    store.record_analyst_action("T123", "CONFIRMED_FRAUD", analyst_name="Alice")
    df = store.get_recent(100)
    stats = store.get_stats()
    """

    def __init__(self, db_path: str = "data/alerts.db"):
        import os

        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    # ── Init ──────────────────────────────────────────────────────────────────

    @contextmanager
    def _connection(self):
        """Context manager that ensures the sqlite connection is always closed."""
        con = sqlite3.connect(self._db_path, check_same_thread=False)
        con.row_factory = sqlite3.Row
        try:
            yield con
        finally:
            con.close()

    def _init_db(self):
        with self._lock, self._connection() as con:
            con.execute(_CREATE_ALERTS_TABLE)
            con.execute(_CREATE_REVIEWS_TABLE)
            # Safe column migration if table existed before without status
            try:
                con.execute("ALTER TABLE alerts ADD COLUMN status TEXT DEFAULT 'PENDING_REVIEW'")
            except sqlite3.OperationalError:
                pass
            con.commit()

    # ── Write Alerts ──────────────────────────────────────────────────────────

    def insert_alert(self, record: dict[str, Any]) -> None:
        """Persist one alert record."""
        now_iso = datetime.now(timezone.utc).isoformat()
        is_fraud = int(bool(record.get("is_fraud", False)))
        initial_status = "PENDING_REVIEW" if is_fraud else "AUTO_APPROVED"
        row = {
            "transaction_id": str(record.get("transaction_id", "")),
            "fraud_proba": float(record.get("fraud_proba", 0.0)),
            "is_fraud": is_fraud,
            "timestamp": record.get("timestamp", now_iso),
            "amount": float(record.get("amount", 0.0)),
            "card1": str(record.get("card1", "")),
            "rule_flags": str(record.get("rule_flags", "")),
            "latency_ms": float(record.get("latency_ms", 0.0)),
            "status": str(record.get("status", initial_status)),
        }
        with self._lock, self._connection() as con:
            con.execute(_INSERT_ALERT, row)
            con.commit()

    def insert_many(self, records: list[dict[str, Any]]) -> None:
        """Batch insert."""
        now_iso = datetime.now(timezone.utc).isoformat()
        rows = [
            {
                "transaction_id": str(r.get("transaction_id", "")),
                "fraud_proba": float(r.get("fraud_proba", 0.0)),
                "is_fraud": int(bool(r.get("is_fraud", False))),
                "timestamp": r.get("timestamp", now_iso),
                "amount": float(r.get("amount", 0.0)),
                "card1": str(r.get("card1", "")),
                "rule_flags": str(r.get("rule_flags", "")),
                "latency_ms": float(r.get("latency_ms", 0.0)),
                "status": "PENDING_REVIEW" if r.get("is_fraud") else "AUTO_APPROVED",
            }
            for r in records
        ]
        with self._lock, self._connection() as con:
            con.executemany(_INSERT_ALERT, rows)
            con.commit()

    # ── Analyst Action Queue ──────────────────────────────────────────────────

    def record_analyst_action(
        self,
        transaction_id: str,
        action: str,
        analyst_name: str = "analyst_1",
        notes: str = "",
    ) -> bool:
        """
        Record manual investigator decision.
        action: 'APPROVED_LEGIT' | 'CONFIRMED_FRAUD' | 'ESCALATED'
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connection() as con:
            # 1. Update status on alerts table
            con.execute(
                "UPDATE alerts SET status = ? WHERE transaction_id = ?",
                (action, transaction_id),
            )
            # 2. Insert into audit trail
            con.execute(
                """
                INSERT INTO analyst_reviews (transaction_id, action, analyst_name, notes, reviewed_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (transaction_id, action, analyst_name, notes, now_iso),
            )
            con.commit()
        return True

    def get_pending_review(self, limit: int = 50) -> pd.DataFrame:
        """Fetch pending alerts awaiting analyst investigation."""
        sql = """
        SELECT * FROM alerts
        WHERE status = 'PENDING_REVIEW'
        ORDER BY fraud_proba DESC, id DESC
        LIMIT ?
        """
        with self._lock, self._connection() as con:
            return pd.read_sql_query(sql, con, params=(limit,))

    def get_audit_trail(self, limit: int = 100) -> pd.DataFrame:
        """Fetch audit log of past analyst decisions."""
        sql = "SELECT * FROM analyst_reviews ORDER BY id DESC LIMIT ?"
        with self._lock, self._connection() as con:
            return pd.read_sql_query(sql, con, params=(limit,))

    # ── Read Queries ──────────────────────────────────────────────────────────

    def get_recent(self, n: int = 500) -> pd.DataFrame:
        """Return the last *n* alerts as a DataFrame (newest first)."""
        sql = "SELECT * FROM alerts ORDER BY id DESC LIMIT ?"
        with self._lock, self._connection() as con:
            df = pd.read_sql_query(sql, con, params=(n,))
        return df

    def get_all(self) -> pd.DataFrame:
        with self._lock, self._connection() as con:
            return pd.read_sql_query("SELECT * FROM alerts ORDER BY id DESC", con)

    def get_stats(self) -> dict[str, Any]:
        """Return aggregate statistics for KPI cards."""
        with self._lock, self._connection() as con:
            row = con.execute(
                """
                SELECT
                    COUNT(*)                         AS total,
                    SUM(is_fraud)                    AS fraud_count,
                    COUNT(*) - SUM(is_fraud)         AS legit_count,
                    SUM(CASE WHEN is_fraud THEN amount ELSE 0.0 END) AS fraud_dollars_prevented,
                    AVG(CASE WHEN is_fraud THEN fraud_proba END) AS avg_fraud_proba,
                    AVG(latency_ms)                  AS avg_latency_ms
                FROM alerts
                """
            ).fetchone()

            hourly = pd.read_sql_query(
                """
                SELECT
                    strftime('%Y-%m-%dT%H:00:00', timestamp) AS hour,
                    COUNT(*)                                  AS total,
                    SUM(is_fraud)                             AS frauds
                FROM alerts
                GROUP BY hour
                ORDER BY hour
                """,
                con,
            )

            status_counts = pd.read_sql_query(
                """
                SELECT status, COUNT(*) as count
                FROM alerts
                GROUP BY status
                """,
                con,
            )

        total = int(row["total"] or 0)
        fraud_count = int(row["fraud_count"] or 0)
        return {
            "total": total,
            "fraud_count": fraud_count,
            "legit_count": int(row["legit_count"] or 0),
            "fraud_dollars_prevented": float(row["fraud_dollars_prevented"] or 0.0),
            "fraud_rate": (fraud_count / total) if total else 0.0,
            "avg_fraud_proba": float(row["avg_fraud_proba"] or 0.0),
            "avg_latency_ms": float(row["avg_latency_ms"] or 0.0),
            "hourly_counts": hourly,
            "status_counts": status_counts,
        }

    def count(self) -> int:
        with self._lock, self._connection() as con:
            return con.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]

    def fraud_count(self) -> int:
        with self._lock, self._connection() as con:
            return con.execute("SELECT SUM(is_fraud) FROM alerts").fetchone()[0] or 0

    # ── Maintenance ───────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Wipe all alerts and reviews — intended for dashboard 'Reset session'."""
        with self._lock, self._connection() as con:
            con.execute("DELETE FROM alerts")
            con.execute("DELETE FROM analyst_reviews")
            con.commit()
