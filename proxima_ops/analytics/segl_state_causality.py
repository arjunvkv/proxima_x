"""
segl_state_causality.py — Causal analysis of why the SEGL state machine stays
in OBSERVE when trade signals exist.

Answers: WHY does the SEGL state machine stay in OBSERVE when signals exist?

Reads from ``state/wave12_cycle_log.jsonl`` (one JSON dict per line).
Each log line contains fields such as:
    cycle, segl_state, total_signals, active_signals, decision, mof_state,
    mof_score, reconciliation.drift_score, open_positions, cb_decision,
    pipeline_trace, etc.

Output
------
Returned dict keys:
    obsere_cycles                — total cycles where segl_state == "OBSERVE"
    transition_blockers          — list of factors that correlate with OBSERVE
                                   persistence (blocker name, count, pct)
    state_transition_probability — 4 transition probabilities
    false_obsere_rate            — fraction of OBSERVE cycles with signals > 0
    signal_alignment             — count of OBSERVE cycles with / without signals

Usage
-----
    from proxima_ops.analytics.segl_state_causality import SeglStateCausality

    analyzer = SeglStateCausality()
    report = analyzer.analyze(n_recent_cycles=500)
    print(report)
"""

import json
import logging
import os
from collections import defaultdict
from typing import Any

logger = logging.getLogger("proxima_ops.analytics.segl_state_causality")

DEFAULT_LOG_PATH = "state/wave12_cycle_log.jsonl"

# Fields to inspect as potential transition blockers when state stays OBSERVE.
BLOCKER_FIELDS = [
    "mof_state",
    "rf_drift",
    "open_positions",
    "circuit_breaker",
]


class SeglStateCausality:
    """Causal analysis of SEGL state-machine persistence in OBSERVE.

    Parameters
    ----------
    log_path : str
        Path to the JSON-lines cycle log (default
        ``state/wave12_cycle_log.jsonl``).
    """

    def __init__(self, log_path: str = DEFAULT_LOG_PATH):
        self._log_path = log_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, n_recent_cycles: int = 500) -> dict[str, Any]:
        """Analyse the cycle log and return a SEGL-state causality report.

        Parameters
        ----------
        n_recent_cycles : int
            Only consider this many of the most recent cycles (default 500).
            Pass a large number (e.g. 1_000_000) to analyse all available
            cycles.

        Returns
        -------
        dict with keys:
            obsere_cycles                — count of cycles in OBSERVE state
            transition_blockers          — list of blocker summaries
            state_transition_probability — transition probability dict
            false_obsere_rate            — false-OBSERVE rate
            signal_alignment             — signal presence breakdown
        """
        try:
            return self._build_report(n_recent_cycles)
        except Exception:
            logger.exception("SeglStateCausality.analyze failed")
            return {
                "obsere_cycles": 0,
                "transition_blockers": [],
                "state_transition_probability": {
                    "OBSERVE_to_ARMED": 0.0,
                    "ARMED_to_OBSERVE": 0.0,
                    "OBSERVE_stay": 0.0,
                    "ARMED_stay": 0.0,
                },
                "false_obsere_rate": 0.0,
                "signal_alignment": {
                    "with_signals": 0,
                    "without_signals": 0,
                },
                "error": "See logs for details",
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_report(self, n_recent_cycles: int) -> dict[str, Any]:
        records = self._load_records()

        if not records:
            return {
                "obsere_cycles": 0,
                "transition_blockers": [],
                "state_transition_probability": {
                    "OBSERVE_to_ARMED": 0.0,
                    "ARMED_to_OBSERVE": 0.0,
                    "OBSERVE_stay": 0.0,
                    "ARMED_stay": 0.0,
                },
                "false_obsere_rate": 0.0,
                "signal_alignment": {
                    "with_signals": 0,
                    "without_signals": 0,
                },
                "warning": "No data found in log",
            }

        # Slice to the N most recent cycles.
        if n_recent_cycles < len(records):
            records = records[-n_recent_cycles:]

        # ---- Core counters ----
        total_obsere_cycles = 0
        obsere_with_signals = 0
        obsere_without_signals = 0

        # ---- State transition counters ----
        segl_states = [r.get("segl_state", "") for r in records]

        observe_count = segl_states.count("OBSERVE")
        armed_count = segl_states.count("ARMED")

        observe_to_armed = 0
        armed_to_observe = 0
        observe_stay = 0
        armed_stay = 0

        for i in range(len(segl_states) - 1):
            curr = segl_states[i]
            nxt = segl_states[i + 1]
            if curr == "OBSERVE" and nxt == "ARMED":
                observe_to_armed += 1
            elif curr == "OBSERVE" and nxt == "OBSERVE":
                observe_stay += 1
            elif curr == "ARMED" and nxt == "OBSERVE":
                armed_to_observe += 1
            elif curr == "ARMED" and nxt == "ARMED":
                armed_stay += 1

        # Compute probabilities with safe division.
        prob_observe_to_armed = (
            observe_to_armed / observe_count if observe_count > 0 else 0.0
        )
        prob_armed_to_observe = (
            armed_to_observe / armed_count if armed_count > 0 else 0.0
        )
        prob_observe_stay = (
            observe_stay / observe_count if observe_count > 0 else 0.0
        )
        prob_armed_stay = (
            armed_stay / armed_count if armed_count > 0 else 0.0
        )

        # ---- False OBSERVE analysis ----
        for r in records:
            segl = r.get("segl_state", "")
            if segl != "OBSERVE":
                continue
            total_obsere_cycles += 1
            total_signals = r.get("total_signals", 0) or r.get("active_signals", 0)
            if total_signals > 0:
                obsere_with_signals += 1
            else:
                obsere_without_signals += 1

        false_obsere_rate = (
            obsere_with_signals / total_obsere_cycles
            if total_obsere_cycles > 0
            else 0.0
        )

        # ---- Transition-blocker discovery ----
        # Look at every pair of consecutive entries where the current state
        # is OBSERVE.  Identify which fields appear to be correlated with
        # staying in OBSERVE.
        blocker_counter: dict[str, int] = defaultdict(int)

        for i in range(len(records) - 1):
            curr_entry = records[i]
            curr_state = curr_entry.get("segl_state", "")
            if curr_state != "OBSERVE":
                continue

            nxt_state = records[i + 1].get("segl_state", "")
            # Only count when OBSERVE fails to transition to ARMED.
            if nxt_state == "ARMED":
                continue

            # --- mof_state blocker ---
            mof = curr_entry.get("mof_state", "")
            if mof and mof != "ARMING":
                blocker_counter["mof_state_not_ARMING"] += 1

            # --- rf_drift blocker (drift_score > threshold) ---
            rec = curr_entry.get("reconciliation") or {}
            drift = rec.get("drift_score", 0.0)
            if isinstance(drift, (int, float)) and drift > 0.3:
                blocker_counter["rf_drift_high"] += 1

            # --- open_positions blocker ---
            positions = curr_entry.get("open_positions", 0)
            if positions and positions > 0:
                blocker_counter["open_positions_nonzero"] += 1

            # --- circuit_breaker blocker ---
            cb = curr_entry.get("cb_decision", "")
            denial = curr_entry.get("denial_reason", "") or ""
            if "CircuitBreaker" in denial or "circuit breaker" in cb.lower():
                blocker_counter["circuit_breaker_active"] += 1

        # Build transition_blockers list sorted by count descending.
        total_blocker_opportunities = sum(blocker_counter.values())
        transition_blockers = [
            {
                "blocker": name,
                "count": count,
                "pct": (
                    round(count / total_blocker_opportunities * 100, 2)
                    if total_blocker_opportunities > 0
                    else 0.0
                ),
            }
            for name, count in sorted(
                blocker_counter.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        ]

        return {
            "obsere_cycles": total_obsere_cycles,
            "transition_blockers": transition_blockers,
            "state_transition_probability": {
                "OBSERVE_to_ARMED": round(prob_observe_to_armed, 4),
                "ARMED_to_OBSERVE": round(prob_armed_to_observe, 4),
                "OBSERVE_stay": round(prob_observe_stay, 4),
                "ARMED_stay": round(prob_armed_stay, 4),
            },
            "false_obsere_rate": round(false_obsere_rate, 4),
            "signal_alignment": {
                "with_signals": obsere_with_signals,
                "without_signals": obsere_without_signals,
            },
        }

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_records(self) -> list[dict[str, Any]]:
        """Load and return all JSON-line records from the cycle log."""
        if not os.path.exists(self._log_path):
            logger.warning("Cycle log not found: %s", self._log_path)
            return []

        records: list[dict[str, Any]] = []
        try:
            with open(self._log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(
                            "Skipping unparseable line in %s", self._log_path
                        )
        except Exception:
            logger.exception("Failed to read cycle log: %s", self._log_path)
            return []

        return records


# ------------------------------------------------------------------
# CLI convenience
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    import sys

    n = 500
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            print("Usage: python segl_state_causality.py [n_recent_cycles]")
            sys.exit(1)

    analyzer = SeglStateCausality()
    report = analyzer.analyze(n_recent_cycles=n)
    print(json.dumps(report, indent=2))
