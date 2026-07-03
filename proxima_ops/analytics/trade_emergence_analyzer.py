"""
trade_emergence_analyzer.py — Diagnostic: what blocked the last missed trade?

Replays the pipeline trace log (state/wave12_cycle_log.jsonl) and categorises
each cycle by the reason the trade did not execute.  Answers the question:

    "What EXACT combination of conditions would have caused
     the last missed trade?"

Blocker categories
------------------
confirm   — Insufficient cross-projection confirm (e.g. 1/2 instead of 2/2)
governor  — Governor gate blocked (State=OBSERVE, !intent, circuit-breaker)
vel       — Velocity-limiter blocked (exposure_smoothing / burst_prevention)
sil       — No signals generated from the SIL universe (total_signals == 0)

A *near miss* is a cycle where every condition *except one* was satisfied
(e.g. confirm=1/2 instead of 2/2).

Usage
-----
    from proxima_ops.analytics.trade_emergence_analyzer import TradeEmergenceAnalyzer

    analyzer = TradeEmergenceAnalyzer()
    report = analyzer.analyze(n_recent_cycles=500)
    print(report)
"""

import json
import logging
import os
from collections import defaultdict
from typing import Any

logger = logging.getLogger("proxima_ops.analytics.trade_emergence_analyzer")

DEFAULT_LOG_PATH = "state/wave12_cycle_log.jsonl"


class TradeEmergenceAnalyzer:
    """Analyse the pipeline trace log and report what blocked trades."""

    def __init__(self, log_path: str = DEFAULT_LOG_PATH):
        self._log_path = log_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, n_recent_cycles: int = 500) -> dict[str, Any]:
        """Analyse the trace log and return an emergence report.

        Parameters
        ----------
        n_recent_cycles : int
            Only consider this many of the most recent cycles (default 500).
            Pass a large number (e.g. 1_000_000) to analyse all available
            cycles.

        Returns
        -------
        dict with keys:
            missed_trade_opportunities   — cycles where a trade COULD have
                                           happened (signals existed) but
                                           did not execute.
            dominant_blocker             — "confirm" | "governor" | "vel" |
                                           "sil"
            near_miss_cycles             — list of cycles just 1 step away
                                           from executing.
            blocker_distribution         — percentage breakdown of blockers.
        """
        try:
            return self._build_report(n_recent_cycles)
        except Exception:
            logger.exception("TradeEmergenceAnalyzer.analyze failed")
            return {
                "missed_trade_opportunities": 0,
                "dominant_blocker": "unknown",
                "near_miss_cycles": [],
                "blocker_distribution": {
                    "confirm": 0.0,
                    "governor": 0.0,
                    "vel": 0.0,
                    "sil": 0.0,
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
                "missed_trade_opportunities": 0,
                "dominant_blocker": "unknown",
                "near_miss_cycles": [],
                "blocker_distribution": {
                    "confirm": 0.0,
                    "governor": 0.0,
                    "vel": 0.0,
                    "sil": 0.0,
                },
                "warning": "No data found in log",
            }

        # Slice to the N most recent cycles.
        if n_recent_cycles < len(records):
            records = records[-n_recent_cycles:]

        # --- Per-cycle classification ------------------------------------
        blocker_counts: dict[str, int] = defaultdict(int)
        near_misses: list[dict[str, Any]] = []
        total_with_signals = 0

        for entry in records:
            total_signals = entry.get("total_signals", 0)

            # A "missed trade opportunity" is a cycle that HAD signals but
            # did NOT result in a trade.
            if total_signals == 0:
                continue  # nothing to trade this cycle

            total_with_signals += 1
            blocker = self._classify_blocker(entry)

            if blocker:
                blocker_counts[blocker] += 1

            # Check for near-miss conditions.
            near_miss_info = self._detect_near_miss(entry)
            if near_miss_info is not None:
                near_misses.append(near_miss_info)

        # --- Compute distribution percentages ----------------------------
        total_blocked = sum(blocker_counts.values())
        distribution: dict[str, float] = {
            "confirm": 0.0,
            "governor": 0.0,
            "vel": 0.0,
            "sil": 0.0,
        }
        if total_blocked > 0:
            for k in distribution:
                distribution[k] = round(
                    (blocker_counts.get(k, 0) / total_blocked) * 100, 2
                )

        # Dominant blocker.
        dominant: str = "unknown"
        if blocker_counts:
            dominant = max(blocker_counts, key=blocker_counts.get)  # type: ignore[arg-type]

        # Sort near misses by cycle descending, keep the most recent.
        near_misses.sort(key=lambda x: x["cycle"], reverse=True)

        return {
            "missed_trade_opportunities": total_with_signals,
            "dominant_blocker": dominant,
            "near_miss_cycles": near_misses[:20],  # keep the 20 most recent
            "blocker_distribution": distribution,
        }

    # ------------------------------------------------------------------
    # Blocker classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_blocker(entry: dict) -> str | None:
        """Return the primary blocker category for a cycle, or *None* if
        it cannot be determined."""
        denial_reason = entry.get("denial_reason") or ""
        total_signals = entry.get("total_signals", 0)

        if total_signals == 0:
            return "sil"

        # Confirm gate — insufficient cross-projection confirm.
        if "Insufficient cross-projection confirm" in denial_reason:
            return "confirm"

        # Velocity limiter.
        if "VEL blocked" in denial_reason:
            return "vel"

        # Governor gate — state / intent / circuit-breaker.
        if "State=OBSERVE" in denial_reason or "CircuitBreaker" in denial_reason:
            return "governor"

        # Fallback: check the pipeline trace for governor gate entries.
        pipeline_trace = entry.get("pipeline_trace") or {}
        governor_gate = pipeline_trace.get("governor_gate") or []
        if governor_gate and any(
            "ready_to_exec=NO" in str(g) for g in governor_gate
        ):
            return "governor"

        # Fallback: execution failed at MT5 level (not a blocker per se,
        # but we categorise it under governor since it is an infra issue).
        execution = pipeline_trace.get("execution") or ""
        if "FAILED MT5" in execution or "place_order returned None" in execution:
            return "governor"

        return None

    # ------------------------------------------------------------------
    # Near-miss detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_near_miss(entry: dict) -> dict[str, Any] | None:
        """Return a near-miss descriptor if this cycle was **one condition
        away** from executing, otherwise *None*.

        Rules
        -----
        - Confirm near-miss : confirm_cycles == 1  (needed 2/2)
        - VEL near-miss     : denial_reason starts with "VEL blocked"
        - Governor near-miss: segl_state == "OBSERVE" but signals exist
        """
        denial_reason = entry.get("denial_reason") or ""
        confirm_cycles = entry.get("confirm_cycles", 0)
        segl_state = entry.get("segl_state", "")
        total_signals = entry.get("total_signals", 0)

        if total_signals == 0:
            return None

        symbol = entry.get("active_symbol") or "UNKNOWN"
        direction = entry.get("active_direction") or "N/A"
        confidence = entry.get("active_confidence") or 0.0
        cycle = entry.get("cycle", 0)

        # --- Confirm near-miss ---
        if confirm_cycles == 1 and "Insufficient cross-projection confirm" in denial_reason:
            return {
                "cycle": cycle,
                "blocker": "confirm",
                "symbol": symbol,
                "direction": direction,
                "confidence": confidence,
            }

        # --- VEL near-miss ---
        if "VEL blocked" in denial_reason:
            return {
                "cycle": cycle,
                "blocker": "vel",
                "symbol": symbol,
                "direction": direction,
                "confidence": confidence,
            }

        # --- Governor near-miss ---
        if "State=OBSERVE" in denial_reason or "CircuitBreaker" in denial_reason:
            return {
                "cycle": cycle,
                "blocker": "governor",
                "symbol": symbol,
                "direction": direction,
                "confidence": confidence,
            }

        # --- Pipeline-based fallback ---
        pipeline_trace = entry.get("pipeline_trace") or {}
        governor_gate = pipeline_trace.get("governor_gate") or []
        if governor_gate and any(
            "ready_to_exec=NO" in str(g) for g in governor_gate
        ):
            return {
                "cycle": cycle,
                "blocker": "governor",
                "symbol": symbol,
                "direction": direction,
                "confidence": confidence,
            }

        return None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_records(self) -> list[dict[str, Any]]:
        """Load and return all JSON-line records from the trace log."""
        if not os.path.exists(self._log_path):
            logger.warning("Trace log not found: %s", self._log_path)
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
            logger.exception("Failed to read trace log: %s", self._log_path)
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
            print("Usage: python trade_emergence_analyzer.py [n_recent_cycles]")
            sys.exit(1)

    analyzer = TradeEmergenceAnalyzer()
    report = analyzer.analyze(n_recent_cycles=n)
    print(json.dumps(report, indent=2))
