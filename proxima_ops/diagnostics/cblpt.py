"""CBLPT — CircuitBreaker Latch Persistence Topology.

Detect hysteresis latch behaviour: CircuitBreaker that stays ON even after
the trigger condition clears.
"""

from __future__ import annotations

import json
import statistics
from typing import Any


class CBLatchPersistence:
    """Analyse CircuitBreaker latch persistence from a wave cycle log.

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
        """Run the full CBLPT analysis.

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
            "trigger_events": [],
            "latch_events": 0,
            "average_latch_duration": 0.0,
            "hysteresis_depth": 0.0,
            "reentry_probability": 0.0,
            "cb_persistence_half_life": 0.0,
            "trigger_vs_reset_asymmetry": 0.0,
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _is_cb_triggered(record: dict[str, Any]) -> bool:
        """Return True when *record* has a CircuitBreaker denial reason."""
        denial = record.get("denial_reason", "")
        return isinstance(denial, str) and "CircuitBreaker" in denial

    @staticmethod
    def _trigger_condition(record: dict[str, Any]) -> str:
        """Extract the trigger condition string from a CB record."""
        denial = record.get("denial_reason", "")
        if isinstance(denial, str) and "CircuitBreaker" in denial:
            return denial
        return ""

    # ------------------------------------------------------------------

    def _compute(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Core analysis logic on pre-loaded, trimmed *rows*."""
        # ── 1. Build CB trigger-event blocks ───────────────────────────
        # A CB trigger event starts when denial_reason first contains
        # "CircuitBreaker" and ends when denial_reason no longer mentions
        # it for a complete cycle.
        trigger_events: list[dict[str, Any]] = []

        i = 0
        while i < len(rows):
            if self._is_cb_triggered(rows[i]):
                event_start_cycle = rows[i].get("cycle", 0)
                trigger_cond = self._trigger_condition(rows[i])

                # Walk forward while CB is still triggered
                j = i
                while j < len(rows) and self._is_cb_triggered(rows[j]):
                    j += 1

                # j is the first index where CB is no longer triggered
                # (or end of data)
                event_end_cycle: int | None = None
                cleared_by_reset = False

                if j < len(rows):
                    event_end_cycle = rows[j - 1].get("cycle", 0)
                    # If there's a cycle after the CB block and the CB is
                    # cleared (not re-triggered), it was cleared by reset.
                    cleared_by_reset = True

                # Duration in cycles (inclusive of start and end)
                if event_end_cycle is not None:
                    duration = event_end_cycle - event_start_cycle + 1
                else:
                    duration = len(rows) - i

                trigger_events.append({
                    "start_cycle": event_start_cycle,
                    "end_cycle": event_end_cycle,
                    "duration": duration,
                    "trigger_condition": trigger_cond,
                    "cleared_by_reset": cleared_by_reset,
                })

                i = j  # skip past this event
            else:
                i += 1

        # ── 2. Aggregate metrics ──────────────────────────────────────
        total_events = len(trigger_events)

        # Latch events: duration > 3 cycles
        latch_events = sum(1 for ev in trigger_events if ev["duration"] > 3)

        # Average latch duration (only for events that are latched)
        latch_durations = [
            ev["duration"] for ev in trigger_events if ev["duration"] > 3
        ]
        avg_latch_duration = (
            statistics.mean(latch_durations) if latch_durations else 0.0
        )

        # Hysteresis depth = fraction of window where CB is ON
        if rows:
            first_cycle = rows[0].get("cycle", 0)
            last_cycle = rows[-1].get("cycle", 0)
            window_cycles = last_cycle - first_cycle + 1
            if window_cycles > 0:
                cb_on_cycles = sum(
                    ev["duration"] for ev in trigger_events
                )
                hysteresis_depth = cb_on_cycles / window_cycles
            else:
                hysteresis_depth = 0.0
        else:
            hysteresis_depth = 0.0

        # Re-entry probability = fraction of latch events that cleared
        # (cleared_by_reset == True) vs total latch events
        if latch_events > 0:
            cleared_latches = sum(
                1 for ev in trigger_events
                if ev["duration"] > 3 and ev["cleared_by_reset"]
            )
            reentry_probability = cleared_latches / latch_events
        else:
            reentry_probability = 0.0

        # ── 3. Persistence half-life ──────────────────────────────────
        # Cycles before 50% of CB events clear.
        # Build a synthetic survival curve from event durations.
        all_durations = [ev["duration"] for ev in trigger_events]
        cb_persistence_half_life = _persistence_half_life(all_durations)

        # ── 4. Trigger vs reset asymmetry ─────────────────────────────
        # Compare mof_state and spread at trigger vs at reset.
        # Percentage of conditions that differ.
        asymmetry = _compute_asymmetry(rows, trigger_events)

        return {
            "trigger_events": trigger_events,
            "latch_events": latch_events,
            "average_latch_duration": round(avg_latch_duration, 4),
            "hysteresis_depth": round(hysteresis_depth, 4),
            "reentry_probability": round(reentry_probability, 4),
            "cb_persistence_half_life": round(cb_persistence_half_life, 2),
            "trigger_vs_reset_asymmetry": round(asymmetry, 4),
        }


# ======================================================================
# Module-level helpers (also testable in isolation)
# ======================================================================


def _persistence_half_life(durations: list[int]) -> float:
    """Estimate the CB persistence half-life in cycles.

    Builds a survival curve from CB event durations: at time *t*, the
    survival probability is the fraction of events whose duration > *t*.
    Returns the first time-point where survival drops below 50 %.

    Parameters
    ----------
    durations
        Duration (in cycles) of each CB trigger event.

    Returns
    -------
    float
        Estimated half-life in cycles.  Returns 0.0 if *durations* is
        empty.
    """
    if not durations:
        return 0.0

    max_t = max(durations)
    if max_t < 1:
        return 0.0

    # Build survival curve
    survived: list[float] = []
    for t in range(1, max_t + 2):
        s = sum(1 for d in durations if d > t) / len(durations)
        survived.append(s)

    # Find first t where survival < 0.5
    for i, s in enumerate(survived):
        if s < 0.5:
            if i == 0:
                return float(i + 1)
            prev_s = survived[i - 1]
            t0, t1 = i, i + 1
            fraction = (0.5 - s) / (prev_s - s) if prev_s != s else 0.0
            return round(t0 + fraction * (t1 - t0), 2)

    # Never drops below 50 %
    return float(max_t)


def _compute_asymmetry(
    rows: list[dict[str, Any]],
    trigger_events: list[dict[str, Any]],
) -> float:
    """Compute trigger vs reset asymmetry percentage.

    For each trigger event, compare the ``mof_state`` and any available
    spread-like field at the trigger cycle vs the first cycle after
    the CB clears.  Returns the fraction of compared conditions that
    differ.

    Parameters
    ----------
    rows
        Full list of cycle log rows.
    trigger_events
        List of detected CB trigger events.

    Returns
    -------
    float
        Fraction of differing conditions in [0, 1].  Returns 0.0 when
        there are no resolvable comparisons.
    """
    # Build a cycle -> row lookup for O(1) access
    cycle_map: dict[int, dict[str, Any]] = {}
    for r in rows:
        cyc = r.get("cycle")
        if cyc is not None:
            cycle_map[cyc] = r

    total_comparisons = 0
    differing = 0

    for ev in trigger_events:
        start_c = ev["start_cycle"]
        end_c = ev["end_cycle"]

        if end_c is None:
            continue  # Still active, cannot compare reset

        # The reset cycle is the cycle immediately after the event end
        reset_c = end_c + 1

        start_row = cycle_map.get(start_c)
        reset_row = cycle_map.get(reset_c)

        if start_row is None or reset_row is None:
            continue

        # Compare mof_state
        start_mof = start_row.get("mof_state")
        reset_mof = reset_row.get("mof_state")

        if start_mof is not None and reset_mof is not None:
            total_comparisons += 1
            if start_mof != reset_mof:
                differing += 1

        # Compare mof_score (spread-like field)
        start_score = start_row.get("mof_score")
        reset_score = reset_row.get("mof_score")

        if start_score is not None and reset_score is not None:
            total_comparisons += 1
            # Consider different if absolute difference > 0.01
            if abs(start_score - reset_score) > 0.01:
                differing += 1

    if total_comparisons == 0:
        return 0.0

    return differing / total_comparisons
