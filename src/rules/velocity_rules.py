"""
velocity_rules.py
─────────────────
Industry-standard velocity and rule-based heuristic checks.
Layers on top of machine-learning model scores.

Rules Implemented
-----------------
1. Card Velocity Alert: > N transactions from the same card within a short time window.
2. Amount Spike Alert: Transaction amount > K standard deviations above historical mean.
3. Night + Round Amount Alert: High-risk nocturnal round amount ($100, $500, etc.).
4. Email Domain Mismatch Alert: High-value purchase with purchaser vs recipient email mismatch.
5. Email Variety Alert: Multiple distinct email domains associated with the same card recently.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any

import pandas as pd


class VelocityRulesEngine:
    """
    In-memory stateful velocity rules engine.
    Evaluates streaming transactions against operational fraud heuristics.
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        v_cfg = cfg.get("velocity_rules", {})

        # Rule 1 config
        self.card_tx_limit = v_cfg.get("card_tx_limit", 3)
        self.card_tx_window_sec = v_cfg.get("card_tx_window_sec", 300)

        # Rule 2 config
        self.amount_zscore_thresh = v_cfg.get("amount_zscore_threshold", 5)

        # Rule 3 config
        self.night_round_active = v_cfg.get("night_round_amount", True)

        # Rule 4 config
        self.email_mismatch_amount_thresh = v_cfg.get("email_mismatch_amount_threshold", 500)

        # Rule 5 config
        self.email_variety_limit = v_cfg.get("email_variety_limit", 2)
        self.email_variety_window_sec = v_cfg.get("email_variety_window_sec", 3600)

        # Internal state tracking: card1 -> deque of (timestamp, amount, email)
        self._card_history: dict[str, deque[tuple[float, float, str]]] = defaultdict(deque)

    def evaluate_transaction(self, tx: dict[str, Any] | pd.Series) -> list[str]:
        """
        Evaluate a single transaction and return a list of triggered rule names.
        """
        flags: list[str] = []
        now = time.time()

        card1 = str(tx.get("card1", ""))
        amount = float(tx.get("TransactionAmt", tx.get("amount", 0.0)))
        email_p = str(tx.get("P_emaildomain", "") or "")
        email_r = str(tx.get("R_emaildomain", "") or "")
        hour = int(tx.get("Transaction_hour", 12)) if "Transaction_hour" in tx else 12

        # ── Rule 1: Card Velocity ─────────────────────────────────────────────
        history = self._card_history[card1]
        # Evict old transactions outside window
        while history and (now - history[0][0]) > self.card_tx_window_sec:
            history.popleft()

        if len(history) >= self.card_tx_limit:
            flags.append(f"CARD_VELOCITY_EXCEEDED (>{self.card_tx_limit} in {self.card_tx_window_sec}s)")

        # ── Rule 2: Amount Z-score Spike ──────────────────────────────────────
        if len(history) >= 3:
            recent_amounts = [item[1] for item in history]
            mean_amt = sum(recent_amounts) / len(recent_amounts)
            var_amt = sum((x - mean_amt) ** 2 for x in recent_amounts) / len(recent_amounts)
            std_amt = var_amt ** 0.5
            if std_amt > 0.0:
                z = (amount - mean_amt) / std_amt
                if z >= self.amount_zscore_thresh:
                    flags.append(f"AMOUNT_SPIKE (z={z:.1f})")

        # ── Rule 3: Night + Round Amount ──────────────────────────────────────
        if self.night_round_active:
            is_night = hour < 6
            is_round = (amount > 0) and (amount % 10 == 0 or amount % 50 == 0)
            if is_night and is_round and amount >= 100:
                flags.append("NIGHT_ROUND_AMOUNT")

        # ── Rule 4: High Value + Email Mismatch ───────────────────────────────
        if email_p and email_r and email_p != "unknown" and email_r != "unknown":
            if email_p != email_r and amount >= self.email_mismatch_amount_thresh:
                flags.append(f"HIGH_VALUE_EMAIL_MISMATCH (${amount:.0f})")

        # ── Rule 5: Multiple Email Domains for Card ───────────────────────────
        if email_p and email_p != "unknown":
            recent_emails = {
                item[2] for item in history
                if (now - item[0]) <= self.email_variety_window_sec and item[2]
            }
            recent_emails.add(email_p)
            if len(recent_emails) > self.email_variety_limit:
                flags.append(f"EMAIL_DOMAIN_HOPPING ({len(recent_emails)} domains)")

        # Record this transaction into history
        history.append((now, amount, email_p))

        return flags

    def clear(self) -> None:
        """Reset historical memory."""
        self._card_history.clear()
