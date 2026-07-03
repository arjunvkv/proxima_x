"""
stress_activation_runner.py — Light-weight diagnostic.

Simulates what WOULD have happened if the cross-projection confirm
requirement was reduced from 2 to 1.  This is ANALYSIS ONLY — no
logic changes are made or suggested.

Reads the pipeline trace log (state/wave12_cycle_log.jsonl) and
reports hypothetical trade volume vs actual trade volume.
"""

import json
import logging
import os
from collections import defaultdict
from typing import Any

logger = logging.getLogger("proxima_ops.testing.stress_activation_runner")

DEFAULT_LOG_PATH = "state/wave12_cycle_log.jsonl"

WARNING = "ANALYSIS ONLY — no logic modified"


class StressActivationRunner:
    """Simulate confirm=1 scenario for diagnostic purposes."""

    def __init__(self, log_path: str = DEFAULT_LOG_PATH):
        self._log_path = log_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, n_recent_cycles: int = 500) -> dict[str, Any]:
        """Analyse the trace log and return a diagnostic report.

        Parameters
        ----------
        n_recent_cycles : int
            Only consider this many of the most recent cycles (default 500).
            Pass a very large number (e.g. 1_000_000) to analyse all cycles.

        Returns
        -------
        dict with keys:
            hypothetical_trades,
            actual_trades,
            missed_trades_due_to_confirm,
            opportunity_cost: { missed_signals, missed_confirm1_signals,
                                by_symbol },
            warning
        """
        try:
            return self._build_report(n_recent_cycles)
        except Exception:
            logger.exception("StressActivationRunner.analyze failed")
            return {
                "hypothetical_trades": 0,
                "actual_trades": 0,
                "missed_trades_due_to_confirm": 0,
                "opportunity_cost": {
                    "missed_signals": 0,
                    "missed_confirm1_signals": 0,
                    "by_symbol": {},
                },
                "warning": f"{WARNING} (error during analysis)",
                "error": "See logs for details",
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_report(self, n_recent_cycles: int) -> dict[str, Any]:
        records = self._load_records()

        if not records:
            return {
                "hypothetical_trades": 0,
                "actual_trades": 0,
                "missed_trades_due_to_confirm": 0,
                "opportunity_cost": {
                    "missed_signals": 0,
                    "missed_confirm1_signals": 0,
                    "by_symbol": {},
                },
                "warning": f"{WARNING} (no data found in log)",
            }

        # If n_recent_cycles is smaller than the total record count, slice.
        if n_recent_cycles < len(records):
            records = records[-n_recent_cycles:]

        # --- Tally counters ---------------------------------------------------
        actual_trades = 0  # CROSS_PASS in confirm_gate
        hypothetical_trades = 0  # confirm_gate had any entry (confirm >= 1)
        missed_due_to_confirm = 0  # denied by "Insufficient cross-projection confirm"

        # Per-symbol breakdown
        by_symbol: dict[str, dict[str, int]] = defaultdict(
            lambda: {"hypothetical": 0, "actual": 0, "missed": 0}
        )

        # Opportunity-cost sub-metrics
        missed_signals: int = 0  # signals that entered confirm gate but didn't pass
        missed_confirm1_signals: int = 0  # signals stuck at confirm=1 exactly

        for entry in records:
            total_sig = entry.get("total_signals", 0)
            if total_sig == 0:
                continue

            pt = entry.get("pipeline_trace", {}) or {}
            confirm_gate = pt.get("confirm_gate", []) or []
            denial_reason = entry.get("denial_reason") or ""
            active_symbol = entry.get("active_symbol") or "UNKNOWN"
            confirm_cycles = entry.get("confirm_cycles", 0)

            entered_confirm = len(confirm_gate) > 0
            passed_confirm = any(
                "CROSS_PASS" in str(item) for item in confirm_gate
            )

            # --- Hypothetical (confirm=1) ------------------------------------
            # Any cycle that had an entry in the confirm gate counts as
            # "would have traded" if confirm required only 1 projection.
            if entered_confirm:
                hypothetical_trades += 1
                by_symbol[active_symbol]["hypothetical"] += 1

            # --- Actual (confirm=2) ------------------------------------------
            if passed_confirm:
                actual_trades += 1
                by_symbol[active_symbol]["actual"] += 1

            # --- Missed due to confirm ---------------------------------------
            if (
                entered_confirm
                and not passed_confirm
                and "Insufficient cross-projection confirm" in denial_reason
            ):
                missed_due_to_confirm += 1
                by_symbol[active_symbol]["missed"] += 1

            # --- Opportunity-cost details ------------------------------------
            if entered_confirm and not passed_confirm:
                missed_signals += 1
                if confirm_cycles == 1:
                    missed_confirm1_signals += 1

        return {
            "hypothetical_trades": hypothetical_trades,
            "actual_trades": actual_trades,
            "missed_trades_due_to_confirm": missed_due_to_confirm,
            "opportunity_cost": {
                "missed_signals": missed_signals,
                "missed_confirm1_signals": missed_confirm1_signals,
                "by_symbol": dict(by_symbol),
            },
            "warning": WARNING,
        }

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
                        logger.warning("Skipping unparseable line in %s", self._log_path)
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
            print("Usage: python stress_activation_runner.py [n_recent_cycles]")
            sys.exit(1)

    runner = StressActivationRunner()
    report = runner.analyze(n_recent_cycles=n)
    print(json.dumps(report, indent=2))
