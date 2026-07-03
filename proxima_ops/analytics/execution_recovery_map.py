"""
execution_recovery_map.py — Diagnostic: how many valid trades were lost due to over-governance?

Replays the pipeline trace log (state/wave12_cycle_log.jsonl) and identifies cycles
where a trade had valid signals, passed all non-governor gates (confirmed >= 2,
no velocity limiter block), but was blocked by the governor gate.

Answers the question:

    "How many valid trades were lost due to over-governance?"

Output categories
-----------------
lost_trades_due_to_governor   — cycles where only the governor gate blocked
recoverable_trades            — lost trades that had a valid signal (confidence > 0)
high_quality_missed           — lost trades with confidence > 0.7
time_windows_of_loss          — contiguous blocks of governor-blocked cycles
missed_by_confidence_band     — confidence distribution of missed trades

Usage
-----
    from proxima_ops.analytics.execution_recovery_map import ExecutionRecoveryMap

    analyzer = ExecutionRecoveryMap()
    report = analyzer.analyze(n_recent_cycles=500)
    print(report)
"""

import json
import logging
import os
from collections import defaultdict
from typing import Any

logger = logging.getLogger("proxima_ops.analytics.execution_recovery_map")

DEFAULT_LOG_PATH = "state/wave12_cycle_log.jsonl"


class ExecutionRecoveryMap:
    """Quantify trades lost because the governor gate blocked execution
    even though signals existed and all other gates passed."""

    def __init__(self, log_path: str = DEFAULT_LOG_PATH):
        self._log_path = log_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, n_recent_cycles: int = 500) -> dict[str, Any]:
        """Analyse the trace log and build the execution recovery map.

        Parameters
        ----------
        n_recent_cycles : int
            Only consider this many of the most recent cycles (default 500).
            Pass a large number (e.g. 1_000_000) to analyse all available
            cycles.

        Returns
        -------
        dict with keys:
            lost_trades_due_to_governor  — cycles where only governor blocked
            recoverable_trades           — lost trades that had valid signals
            high_quality_missed          — lost trades with confidence > 0.7
            time_windows_of_loss         — time windows with concentrated losses
            missed_by_confidence_band    — confidence distribution of missed trades
        """
        try:
            return self._build_report(n_recent_cycles)
        except Exception:
            logger.exception("ExecutionRecoveryMap.analyze failed")
            return {
                "lost_trades_due_to_governor": 0,
                "recoverable_trades": 0,
                "high_quality_missed": 0,
                "time_windows_of_loss": [],
                "missed_by_confidence_band": {
                    "0.0_to_0.3": 0,
                    "0.3_to_0.5": 0,
                    "0.5_to_0.7": 0,
                    "0.7_to_1.0": 0,
                },
                "error": "See logs for details",
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_report(self, n_recent_cycles: int) -> dict[str, Any]:
        records = self._load_records()

        if not records:
            return self._empty_report("No data found in log")

        # Slice to the N most recent cycles.
        if n_recent_cycles < len(records):
            records = records[-n_recent_cycles:]

        # --- Classify each cycle -------------------------------------------
        lost_cycles: list[dict[str, Any]] = []

        for entry in records:
            # A "best_signal" exists when active_signals > 0 and there is
            # an active edge (not "none").
            active_signals = entry.get("active_signals", 0)
            active_edge = entry.get("active_edge", "none")
            total_signals = entry.get("total_signals", 0)

            if active_signals == 0 or active_edge == "none" or total_signals == 0:
                continue

            # Check whether this cycle was blocked SOLELY by the governor gate.
            if not self._is_governor_only_block(entry):
                continue

            confidence = entry.get("active_confidence", 0.0) or 0.0
            cycle = entry.get("cycle", 0)
            denial_reason = entry.get("denial_reason") or ""
            pipeline_trace = entry.get("pipeline_trace") or {}
            governor_gate = pipeline_trace.get("governor_gate") or []
            blocker_reason = self._extract_blocker_reason(
                denial_reason, governor_gate
            )

            lost_cycles.append(
                {
                    "cycle": cycle,
                    "confidence": confidence,
                    "blocker_reason": blocker_reason,
                }
            )

        # --- Compute output metrics ----------------------------------------
        lost_trades = len(lost_cycles)
        recoverable = sum(1 for lc in lost_cycles if lc["confidence"] > 0)
        high_quality = sum(1 for lc in lost_cycles if lc["confidence"] > 0.7)

        # Confidence bands.
        bands: dict[str, int] = {
            "0.0_to_0.3": 0,
            "0.3_to_0.5": 0,
            "0.5_to_0.7": 0,
            "0.7_to_1.0": 0,
        }
        for lc in lost_cycles:
            c = lc["confidence"]
            if c <= 0.3:
                bands["0.0_to_0.3"] += 1
            elif c <= 0.5:
                bands["0.3_to_0.5"] += 1
            elif c <= 0.7:
                bands["0.5_to_0.7"] += 1
            else:
                bands["0.7_to_1.0"] += 1

        # Time windows: consecutive groups of cycles with losses,
        # grouped when gaps < 10 cycles.
        time_windows = self._build_time_windows(lost_cycles)

        return {
            "lost_trades_due_to_governor": lost_trades,
            "recoverable_trades": recoverable,
            "high_quality_missed": high_quality,
            "time_windows_of_loss": time_windows,
            "missed_by_confidence_band": bands,
        }

    # ------------------------------------------------------------------
    # Governor-block detection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_governor_only_block(entry: dict) -> bool:
        """Return *True* if this cycle's trade was blocked **solely** by
        the governor gate.

        Conditions for a governor-only block:
        - Confirm gate passed (confirm_cycles >= 2 AND no confirm denial)
        - No velocity-limiter block (``denial_reason`` does not contain
          ``"VEL blocked"``)
        - Governor gate actively blocked (State=OBSERVE / CircuitBreaker /
          ready_to_exec=NO)
        """
        denial_reason = entry.get("denial_reason") or ""
        confirm_cycles = entry.get("confirm_cycles", 0)
        segl_state = entry.get("segl_state", "")
        pipeline_trace = entry.get("pipeline_trace") or {}
        governor_gate = pipeline_trace.get("governor_gate") or []

        # --- Confirm gate must have passed ---
        # confirm_cycles >= 2 is a reasonable proxy; also ensure no
        # explicit confirm denial in the denial_reason.
        if confirm_cycles < 2:
            return False
        if "Insufficient cross-projection confirm" in denial_reason:
            return False

        # --- No VEL block ---
        if "VEL blocked" in denial_reason:
            return False

        # --- Check governor gate blocked ---
        if segl_state == "OBSERVE":
            return True
        if "State=OBSERVE" in denial_reason:
            return True
        if "CircuitBreaker" in denial_reason:
            return True

        # Fallback: inspect governor_gate pipeline trace.
        if governor_gate and any(
            "ready_to_exec=NO" in str(g) for g in governor_gate
        ):
            return True

        return False

    # ------------------------------------------------------------------
    # Blocker reason extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_blocker_reason(
        denial_reason: str, governor_gate: list
    ) -> str:
        """Return a short human-readable label for what the governor blocked
        on."""
        if "State=OBSERVE" in denial_reason:
            return "State=OBSERVE"
        if "CircuitBreaker" in denial_reason:
            return "CircuitBreaker"
        for g in governor_gate:
            gs = str(g)
            if "segl_state=" in gs:
                for segment in gs.split():
                    if segment.startswith("segl_state="):
                        return segment
        return "governor_gate:ready_to_exec=NO"

    # ------------------------------------------------------------------
    # Time-window builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_time_windows(
        lost_cycles: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Group consecutive cycles with losses into windows.

        Windows are separated when the gap between successive cycles is
        >= 10 cycles.  Within each window the *primary_blocker* is the
        most frequent blocker reason.
        """
        if not lost_cycles:
            return []

        sorted_cycles = sorted(lost_cycles, key=lambda x: x["cycle"])
        windows: list[dict[str, Any]] = []

        # Initialise first window.
        current_start = sorted_cycles[0]["cycle"]
        current_end = sorted_cycles[0]["cycle"]
        current_losses = 1
        current_blockers: dict[str, int] = defaultdict(int)
        current_blockers[sorted_cycles[0]["blocker_reason"]] += 1

        for i in range(1, len(sorted_cycles)):
            gap = sorted_cycles[i]["cycle"] - sorted_cycles[i - 1]["cycle"]
            if gap < 10:
                # Extend current window.
                current_end = sorted_cycles[i]["cycle"]
                current_losses += 1
                current_blockers[sorted_cycles[i]["blocker_reason"]] += 1
            else:
                # Finalise previous window.
                primary = max(
                    current_blockers, key=current_blockers.get  # type: ignore[arg-type]
                )
                windows.append(
                    {
                        "start_cycle": current_start,
                        "end_cycle": current_end,
                        "losses": current_losses,
                        "primary_blocker": primary,
                    }
                )
                # Start new window.
                current_start = sorted_cycles[i]["cycle"]
                current_end = sorted_cycles[i]["cycle"]
                current_losses = 1
                current_blockers = defaultdict(int)
                current_blockers[sorted_cycles[i]["blocker_reason"]] += 1

        # Final window.
        primary = max(
            current_blockers, key=current_blockers.get  # type: ignore[arg-type]
        )
        windows.append(
            {
                "start_cycle": current_start,
                "end_cycle": current_end,
                "losses": current_losses,
                "primary_blocker": primary,
            }
        )

        return windows

    # ------------------------------------------------------------------
    # Empty-report helper
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_report(warning: str) -> dict[str, Any]:
        return {
            "lost_trades_due_to_governor": 0,
            "recoverable_trades": 0,
            "high_quality_missed": 0,
            "time_windows_of_loss": [],
            "missed_by_confidence_band": {
                "0.0_to_0.3": 0,
                "0.3_to_0.5": 0,
                "0.5_to_0.7": 0,
                "0.7_to_1.0": 0,
            },
            "warning": warning,
        }

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
            print(
                "Usage: python execution_recovery_map.py [n_recent_cycles]"
            )
            sys.exit(1)

    analyzer = ExecutionRecoveryMap()
    report = analyzer.analyze(n_recent_cycles=n)
    print(json.dumps(report, indent=2))
