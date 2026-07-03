"""CGLD — Confirm Gate Latency Decomposition.

Measure time delays through the decision pipeline by analysing
how many consecutive cycles a signal spends in the confirm gate
before reaching full confirmation (confirm_cycles >= 2) or
dropping out.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from typing import Any


class ConfirmGateLatencyDecomp:
    """Measure and decompose confirm-gate latency from a wave cycle log.

    Parameters
    ----------
    log_path : str
        Path to the JSONL cycle log (default
        ``"state/wave12_cycle_log.jsonl"``).
    """

    def __init__(self, log_path: str = "state/wave12_cycle_log.jsonl") -> None:
        self.log_path = log_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, n_recent_cycles: int = 500) -> dict[str, Any]:
        """Run the full CGLD analysis.

        Parameters
        ----------
        n_recent_cycles : int
            Number of most-recent cycles to analyse (default 500).

        Returns
        -------
        dict
            See class docstring for schema.
        """
        try:
            rows = self._load_log()
        except FileNotFoundError:
            return self._empty_result()
        except json.JSONDecodeError:
            return self._empty_result()

        if not rows:
            return self._empty_result()

        # Keep only the N most recent cycles
        rows = rows[-n_recent_cycles:]

        try:
            return self._compute(rows)
        except Exception:
            return self._empty_result()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_log(self) -> list[dict[str, Any]]:
        """Parse every JSON line from *log_path*."""
        rows: list[dict[str, Any]] = []
        with open(self.log_path, "r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    rows.append(json.loads(stripped))
        return rows

    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        return {
            "total_latency_trajectory": {},
            "average_latency": 0.0,
            "latency_spike_count": 0,
            "latency_hold_correlation": 0.0,
            "latency_by_outcome": {
                "HOLD": {"mean": 0.0, "std": 0.0},
                "EXECUTE": {"mean": 0.0, "std": 0.0},
            },
            "decision_urgency_decay": 0.0,
        }

    # ------------------------------------------------------------------

    def _compute(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Core analysis logic on pre-loaded, trimmed *rows*."""
        # ── 1. Build per-(symbol,direction) confirm_cycles series ─────
        # Each entry: (cycle_n, confirm_cycles, decision)
        series: dict[str, list[tuple[int, int, str]]] = defaultdict(list)

        for r in rows:
            symbol = r.get("active_symbol", "?")
            direction = r.get("active_direction", "?")
            key = f"{symbol}_{direction}"
            cc = r.get("confirm_cycles", 0)
            if not isinstance(cc, int):
                try:
                    cc = int(cc)
                except (ValueError, TypeError):
                    cc = 0
            decision = r.get("decision", "HOLD")
            series[key].append((r.get("cycle", 0), cc, decision))

        # Sort each series by cycle (they should already be ordered, but
        # be defensive).
        for key in series:
            series[key].sort(key=lambda t: t[0])

        # ── 2. Extract confirm-event latency blocks ───────────────────
        # A "confirm event" starts when confirm_cycles transitions from 0
        # to >0 and ends when it either reaches >=2 (full confirm) or
        # falls back to 0.  Latency is the *number of consecutive cycles*
        # where 0 < confirm_cycles < 2 during that event.
        latencies: list[float] = []           # all latency values
        trajectory: dict[str, float] = {}     # cycle_N -> latency
        event_outcomes: list[tuple[float, str]] = []  # (latency, decision)
        survival_data: list[int] = []         # event lifetimes (cycles)

        for key, recs in series.items():
            i = 0
            while i < len(recs):
                cycle, cc, decision = recs[i]
                if cc > 0:
                    # Start of a confirm event
                    event_start = i
                    # Walk forward while confirm_cycles > 0
                    j = i
                    max_cc = cc
                    while j < len(recs) and recs[j][1] > 0:
                        max_cc = max(max_cc, recs[j][1])
                        j += 1
                    # j is now the first index where confirm_cycles == 0
                    # (or end of series)
                    event_records = recs[event_start:j]

                    # Lifetimes >= 1 cycle
                    lifetime = len(event_records)
                    if lifetime > 0:
                        survival_data.append(lifetime)

                    # The last decision of the event
                    final_decision = event_records[-1][2] if event_records else decision

                    # ── Latency extraction ──
                    # Count consecutive cycles where 0 < cc < 2
                    # within this event.
                    block_start = None
                    for pos in range(event_start, j):
                        c, cc_val, dec = recs[pos]
                        if 0 < cc_val < 2:
                            if block_start is None:
                                block_start = pos
                            # Continuation of a <2 block
                        else:
                            if block_start is not None:
                                block_len = pos - block_start
                                latencies.append(float(block_len))
                                event_outcomes.append(
                                    (float(block_len), final_decision)
                                )
                                block_start = None

                    # Flush remaining block
                    if block_start is not None:
                        block_len = j - block_start
                        latencies.append(float(block_len))
                        event_outcomes.append(
                            (float(block_len), final_decision)
                        )

                    i = j  # skip past this event
                else:
                    i += 1

        # ── 2b. Build cycle-level trajectory ──────────────────────────
        # For each cycle, compute the total active latency across all
        # keys (sum of latencies currently in progress).
        # We track active blocks per key across cycles.
        active_latency: dict[int, float] = defaultdict(float)
        for key, recs in series.items():
            for pos in range(len(recs)):
                c, cc_val, _ = recs[pos]
                if 0 < cc_val < 2:
                    active_latency[c] += 1.0

        if active_latency:
            sorted_cycles = sorted(active_latency)
            for c in sorted_cycles:
                trajectory[f"cycle_{c}"] = active_latency[c]

        # ── 3. Aggregate metrics ─────────────────────────────────────
        n = len(latencies)
        avg_latency = statistics.mean(latencies) if n > 0 else 0.0
        std_latency = statistics.stdev(latencies) if n > 1 else 0.0

        # Spikes: cycles where latency > mean + 2*std
        spike_threshold = avg_latency + 2.0 * std_latency
        spike_count = sum(1 for L in latencies if L > spike_threshold)

        # ── 4. Latency by outcome ────────────────────────────────────
        hold_lat: list[float] = []
        exec_lat: list[float] = []
        for L, dec in event_outcomes:
            if dec == "HOLD":
                hold_lat.append(L)
            elif dec == "EXECUTE":
                exec_lat.append(L)

        def _summary(vals: list[float]) -> dict[str, float]:
            if not vals:
                return {"mean": 0.0, "std": 0.0}
            return {
                "mean": statistics.mean(vals),
                "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            }

        latency_by_outcome = {
            "HOLD": _summary(hold_lat),
            "EXECUTE": _summary(exec_lat),
        }

        # ── 5. Point-biserial correlation latency ↔ HOLD ─────────────
        # HOLD=1, EXECUTE=0
        corr = _point_biserial(event_outcomes)

        # ── 6. Decision urgency decay half-life ──────────────────────
        decay_half_life = _urgency_half_life(survival_data)

        traj = dict(sorted(trajectory.items(), key=lambda kv: int(kv[0].split("_")[1])))

        return {
            "total_latency_trajectory": traj,
            "average_latency": round(avg_latency, 4),
            "latency_spike_count": spike_count,
            "latency_hold_correlation": round(corr, 4),
            "latency_by_outcome": latency_by_outcome,
            "decision_urgency_decay": round(decay_half_life, 2),
        }


# ======================================================================
# Module-level helpers (also testable in isolation)
# ======================================================================


def _point_biserial(
    outcomes: list[tuple[float, str]],
) -> float:
    """Compute point-biserial correlation between latency and binary HOLD.

    Parameters
    ----------
    outcomes
        List of ``(latency, decision)`` pairs where decision is
        ``"HOLD"`` or ``"EXECUTE"``.

    Returns
    -------
    float
        Correlation coefficient in [-1, 1].  Returns 0.0 when there are
        fewer than 3 data points or only one outcome class is present.
    """
    if len(outcomes) < 3:
        return 0.0

    holds: list[float] = []
    executes: list[float] = []
    for L, dec in outcomes:
        if dec == "HOLD":
            holds.append(L)
        else:
            executes.append(L)

    n0 = len(holds)
    n1 = len(executes)
    if n0 == 0 or n1 == 0:
        return 0.0

    all_vals = [L for L, _ in outcomes]
    m0 = statistics.mean(holds)
    m1 = statistics.mean(executes)
    std_all = statistics.stdev(all_vals)
    if std_all == 0.0:
        return 0.0

    n = n0 + n1
    r = (m1 - m0) / std_all * math.sqrt((n0 * n1) / (n * (n - 1)))
    # Clamp to [-1, 1]
    return max(-1.0, min(1.0, r))


def _urgency_half_life(lifetimes: list[int]) -> float:
    """Estimate the confirm-event half-life (urgency decay).

    Fits a Kaplan-Meier-like survival curve to confirm-event durations
    and returns the first time-point where survival probability drops
    below 50 %.

    Parameters
    ----------
    lifetimes
        Duration (in cycles) of each confirm event.

    Returns
    -------
    float
        Estimated half-life in cycles.  Returns 0.0 if *lifetimes* is
        empty.
    """
    if not lifetimes:
        return 0.0

    max_t = max(lifetimes)
    if max_t < 1:
        return 0.0

    # Survival at time t = P(T > t)
    # Count events that survived strictly longer than t
    survived: list[float] = []
    for t in range(1, max_t + 2):
        s = sum(1 for lt in lifetimes if lt > t) / len(lifetimes)
        survived.append(s)

    # Find first t where survival < 0.5
    for i, s in enumerate(survived):
        if s < 0.5:
            # Interpolate between (i, survived[i-1]) and (i+1, s)
            if i == 0:
                return float(i + 1)
            prev_s = survived[i - 1]
            t0, t1 = i, i + 1
            fraction = (0.5 - s) / (prev_s - s) if prev_s != s else 0.0
            return round(t0 + fraction * (t1 - t0), 2)

    # Never drops below 50 % → half-life >= max
    return float(max_t)
