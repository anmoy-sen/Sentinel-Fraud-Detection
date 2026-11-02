"""
drift_detector.py
─────────────────
Population Stability Index (PSI) based feature drift detection.

PSI measures how much a distribution has shifted relative to a reference.
  PSI < 0.1   → no significant change
  PSI 0.1–0.2 → moderate shift (warning)
  PSI > 0.2   → significant drift (alert)

Usage
-----
from src.monitoring.drift_detector import DriftDetector
detector = DriftDetector(reference_df, top_features=20)
report = detector.compute_psi(current_df)
print(report)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class DriftDetector:
    """
    Computes PSI for each numeric feature between a reference and current sample.

    Parameters
    ----------
    reference_df  : pd.DataFrame   — baseline distribution (from training set)
    top_features  : int            — how many features to monitor
    n_bins        : int            — number of equal-width bins for PSI calculation
    psi_warning   : float          — yellow-alert threshold (default 0.1)
    psi_threshold : float          — red-alert threshold (default 0.2)
    """

    def __init__(
        self,
        reference_df: pd.DataFrame,
        top_features: int = 20,
        n_bins: int = 10,
        psi_warning: float = 0.1,
        psi_threshold: float = 0.2,
    ):
        # Use only numeric columns
        self._reference = reference_df.select_dtypes(include=[np.number]).copy()
        # Select the top_features with the most variance (most informative to monitor)
        variances = self._reference.var().sort_values(ascending=False)
        self._features = list(variances.head(top_features).index)
        self._n_bins = n_bins
        self._psi_warning = psi_warning
        self._psi_threshold = psi_threshold

        # Pre-compute bin edges from reference
        self._bin_edges: dict[str, np.ndarray] = {}
        for col in self._features:
            values = self._reference[col].dropna().values
            if len(values) < 2:
                continue
            edges = np.percentile(values, np.linspace(0, 100, n_bins + 1))
            edges = np.unique(edges)  # remove duplicates at boundaries
            if len(edges) < 2:
                continue
            self._bin_edges[col] = edges

    # ── PSI helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _bucket_probs(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Return normalised histogram of *values* using the given *edges*."""
        counts, _ = np.histogram(values, bins=edges)
        probs = counts / max(counts.sum(), 1)
        # Avoid log(0)
        probs = np.clip(probs, 1e-6, None)
        return probs

    def _psi_for_feature(self, col: str, current: np.ndarray) -> float:
        edges = self._bin_edges.get(col)
        if edges is None:
            return 0.0

        ref_vals = self._reference[col].dropna().values
        ref_probs = self._bucket_probs(ref_vals, edges)
        cur_probs = self._bucket_probs(current, edges)

        psi = np.sum((cur_probs - ref_probs) * np.log(cur_probs / ref_probs))
        return float(psi)

    # ── Public API ────────────────────────────────────────────────────────────

    def compute_psi(self, current_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute PSI for all monitored features against *current_df*.

        Returns a DataFrame with columns:
            feature, psi, status  ('ok' | 'warning' | 'drift')
        """
        current = current_df.select_dtypes(include=[np.number])
        rows = []
        for col in self._features:
            if col not in current.columns:
                continue
            values = current[col].dropna().values
            if len(values) < 10:
                continue
            psi = self._psi_for_feature(col, values)
            if psi >= self._psi_threshold:
                status = "drift"
            elif psi >= self._psi_warning:
                status = "warning"
            else:
                status = "ok"
            rows.append({"feature": col, "psi": round(psi, 4), "status": status})

        df = pd.DataFrame(rows).sort_values("psi", ascending=False)
        return df

    def has_drift(self, current_df: pd.DataFrame) -> bool:
        """Return True if any feature has PSI above the drift threshold."""
        report = self.compute_psi(current_df)
        return (report["status"] == "drift").any()

    def summary(self, current_df: pd.DataFrame) -> dict:
        """Return a summary dict suitable for dashboard KPI cards."""
        report = self.compute_psi(current_df)
        return {
            "total_features": len(report),
            "drifted": int((report["status"] == "drift").sum()),
            "warning": int((report["status"] == "warning").sum()),
            "ok": int((report["status"] == "ok").sum()),
            "max_psi": float(report["psi"].max()) if not report.empty else 0.0,
            "report": report,
        }
