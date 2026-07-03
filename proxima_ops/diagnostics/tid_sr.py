"""
TID-SR — Tick Integrity Drift & Synthetic Reconstruction

Build statistical manifold of expected tick behavior from the cycle log
and measure deviation (drift) across three proxy dimensions:

  1. OHLC consistency  — lifecycle errors / mof_state instability
  2. Spread stability   — mof_score volatility between cycles
  3. Price continuity   — drift_score / balance discontinuities
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterator
from typing import Any


class TickIntegrityDrift:
    """Compute tick integrity drift metrics from the wave12 cycle log."""

    def __init__(self, log_path: str = "state/wave12_cycle_log.jsonl") -> None:
        self.log_path = log_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, n_recent_cycles: int = 500) -> dict[str, Any]:
        """Produce a drift-analysis dictionary.

        Parameters
        ----------
        n_recent_cycles : int
            Number of most-recent cycles to consider.  Pass a large number
            (e.g. 1_000_000) to analyse the full log.  Default 500.

        Returns
        -------
        dict with keys:
            drift_score_trajectory  : {cycle_N: float} — per-cycle drift
            average_drift           : float (0-1)
            drift_spike_count       : int  (drift > 0.5)
            hold_correlation_with_drift : float (Pearson r) or 0.0
            drift_by_segl_state     : {state: {"mean": float, "std": float}}
            estimated_data_quality  : "GOOD" | "DEGRADED" | "POOR"
        """
        try:
            records = list(self._iter_records())
        except Exception:
            return self._empty_result()

        if not records:
            return self._empty_result()

        # Keep only the *n_recent_cycles* most recent entries
        records = records[-n_recent_cycles:]

        # Build consecutive-cycle pairs for delta-based proxies
        # and extract per-cycle drift.
        drift_trajectory: dict[str, float] = {}
        segl_states: list[str] = []
        decisions: list[str] = []
        ohlc_proxies: list[float] = []
        spread_proxies: list[float] = []
        price_proxies: list[float] = []

        for i, rec in enumerate(records):
            cycle_key = str(rec.get("cycle", i))
            segl_states.append(str(rec.get("segl_state", "UNKNOWN")))
            decisions.append(str(rec.get("decision", "")))

            # -- proxy 1 : OHLC consistency ---------------------------------
            # Use lifecycle-error density and mof_state changes as markers of
            # OHLC / tick-data inconsistency.
            ohlc_proxy = self._compute_ohlc_proxy(rec)
            ohlc_proxies.append(ohlc_proxy)

            # -- proxy 2 : spread stability ---------------------------------
            # mof_score volatility proxies for changing spread conditions.
            if i == 0:
                spread_proxies.append(0.0)
            else:
                prev_mof = _safe_float(records[i - 1].get("mof_score", 0.0))
                curr_mof = _safe_float(rec.get("mof_score", 0.0))
                spread_proxies.append(min(abs(curr_mof - prev_mof) / 0.1, 1.0))

            # -- proxy 3 : price continuity ---------------------------------
            # Large jumps in the reconciliation drift_score or balance.
            if i == 0:
                price_proxies.append(0.0)
            else:
                prev_drift = _safe_float(
                    records[i - 1].get("reconciliation", {}).get("drift_score", 0.0)
                )
                curr_drift = _safe_float(
                    rec.get("reconciliation", {}).get("drift_score", 0.0)
                )
                prev_bal = _safe_float(records[i - 1].get("balance", 0.0))
                curr_bal = _safe_float(rec.get("balance", 0.0))

                drift_delta = abs(curr_drift - prev_drift)
                bal_delta = (
                    abs(curr_bal - prev_bal) / max(abs(prev_bal), 1.0)
                    if prev_bal != 0
                    else 0.0
                )
                price_proxies.append(min(drift_delta / 0.1 + bal_delta, 1.0))

            # -- aggregate drift for this cycle -----------------------------
            avg_drift = (ohlc_proxy + spread_proxies[-1] + price_proxies[-1]) / 3.0
            drift_trajectory[cycle_key] = round(avg_drift, 6)

        # -- summary statistics -------------------------------------------
        drift_values = list(drift_trajectory.values())
        average_drift = _safe_mean(drift_values)

        spike_threshold = 0.5
        drift_spike_count = sum(1 for d in drift_values if d > spike_threshold)

        # Pearson r between drift and (decision == "HOLD")
        hold_flags = [1.0 if d.upper() == "HOLD" else 0.0 for d in decisions]
        hold_corr = self._pearson_r(drift_values, hold_flags)

        # Drift by segl_state
        groups: dict[str, list[float]] = {}
        for state, d in zip(segl_states, drift_values):
            groups.setdefault(state, []).append(d)

        drift_by_segl_state: dict[str, dict[str, float]] = {}
        for state, vals in groups.items():
            drift_by_segl_state[state] = {
                "mean": round(_safe_mean(vals), 6),
                "std": round(_safe_std(vals), 6),
            }

        # Data-quality label
        if average_drift < 0.2:
            quality = "GOOD"
        elif average_drift < 0.5:
            quality = "DEGRADED"
        else:
            quality = "POOR"

        return {
            "drift_score_trajectory": drift_trajectory,
            "average_drift": round(average_drift, 6),
            "drift_spike_count": drift_spike_count,
            "hold_correlation_with_drift": round(hold_corr, 6),
            "drift_by_segl_state": drift_by_segl_state,
            "estimated_data_quality": quality,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _iter_records(self) -> Iterator[dict[str, Any]]:
        """Yield parsed JSON objects from the log file."""
        path = self.log_path
        if not os.path.isabs(path):
            # Resolve relative to the workspace root (two levels up from
            # this module's directory).
            module_dir = os.path.dirname(os.path.abspath(__file__))
            # proxima_x/proxima_ops/diagnostics -> proxima_x -> workspace
            candidate = os.path.join(module_dir, "..", "..", "..", path)
            if os.path.isfile(candidate):
                path = candidate

        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    yield json.loads(stripped)

    def _compute_ohlc_proxy(self, rec: dict[str, Any]) -> float:
        """Derive a normalised (0-1) OHLC-inconsistency score.

        Factors:
          - lifecycle_errors > 0   => +0.5
          - _lifecycle_errors > 0   => +0.3
          - _mof_unstable_count > 0 => +0.2
        """
        lifecycle_issues = (
            rec.get("reconciliation", {}).get("lifecycle_issues") or []
        )
        raw = 0.0
        if len(lifecycle_issues) > 0:
            raw += 0.5
        if int(rec.get("_lifecycle_errors", 0)) > 0:
            raw += 0.3
        if int(rec.get("_mof_unstable_count", 0)) > 0:
            raw += 0.2
        return min(raw, 1.0)

    @staticmethod
    def _pearson_r(x: list[float], y: list[float]) -> float:
        """Compute Pearson correlation coefficient between two sequences."""
        n = len(x)
        if n < 2:
            return 0.0
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xx = sum(v * v for v in x)
        sum_yy = sum(v * v for v in y)
        sum_xy = sum(a * b for a, b in zip(x, y))
        denom = math.sqrt((n * sum_xx - sum_x * sum_x) * (n * sum_yy - sum_y * sum_y))
        if denom == 0.0:
            return 0.0
        r = (n * sum_xy - sum_x * sum_y) / denom
        return max(-1.0, min(1.0, r))

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        return {
            "drift_score_trajectory": {},
            "average_drift": 0.0,
            "drift_spike_count": 0,
            "hold_correlation_with_drift": 0.0,
            "drift_by_segl_state": {},
            "estimated_data_quality": "GOOD",
        }


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _safe_float(val: Any) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _safe_mean(vals: list[float]) -> float:
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _safe_std(vals: list[float]) -> float:
    n = len(vals)
    if n < 2:
        return 0.0
    mean = _safe_mean(vals)
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return math.sqrt(var)
