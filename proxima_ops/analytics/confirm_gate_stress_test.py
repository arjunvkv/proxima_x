"""
confirm_gate_stress_test.py — Quantify how the confirm gate requirement
affects trading by simulating confirm=1 vs confirm=2 vs confirm=3
thresholds and measuring trade delta and signal decay rate.

This is ANALYSIS ONLY. No logic changes are made or suggested.

Reads the pipeline trace log (state/wave12_cycle_log.jsonl) and
reports metrics across three confirm thresholds.
"""

import json
import logging
import os
import re
from collections import defaultdict
from typing import Any

logger = logging.getLogger("proxima_ops.analytics.confirm_gate_stress_test")

DEFAULT_LOG_PATH = "state/wave12_cycle_log.jsonl"

WARNING = "ANALYSIS ONLY — no logic modified"

# Regex to extract cross-projection cycle count from a CROSS_PASS entry,
# e.g. "edge_09: CROSS_PASS (cycles=23/2)"  -> 23
_CROSS_PASS_CYC_RE = re.compile(r"CROSS_PASS.*?\(cycles=(\d+)/")


def _signal_run_id(entry: dict[str, Any]) -> tuple[str, str, str]:
    """Extract a stable signal-run identifier from a log entry.

    Returns (symbol, direction, edge), defaulting to empty strings.
    """
    sym = entry.get("active_symbol") or ""
    direc = entry.get("active_direction") or ""
    pt = entry.get("pipeline_trace") or {}
    edge = entry.get("active_edge") or pt.get("active_edge") or ""
    return (str(sym), str(direc), str(edge))


class ConfirmGateStressTest:
    """Simulate confirm=1, confirm=2, and confirm=3 thresholds for
    diagnostic purposes."""

    def __init__(self, log_path: str = DEFAULT_LOG_PATH):
        self._log_path = log_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, n_recent_cycles: int = 500) -> dict[str, Any]:
        """Analyse the trace log and return a stress-test report.

        Parameters
        ----------
        n_recent_cycles : int
            Only consider this many of the most recent cycles (default 500).
            Pass a very large number (e.g. 1_000_000) to analyse all cycles.

        Returns
        -------
        dict with keys:
            confirm_1_trades,
            confirm_2_trades,
            confirm_3_trades,
            loss_due_to_confirm,
            optimal_confirm_level,
            signal_decay_rate,
            by_symbol,
            warning
        """
        try:
            return self._build_report(n_recent_cycles)
        except Exception:
            logger.exception("ConfirmGateStressTest.analyze failed")
            return {
                "confirm_1_trades": 0,
                "confirm_2_trades": 0,
                "confirm_3_trades": 0,
                "loss_due_to_confirm": 0,
                "optimal_confirm_level": 2,
                "signal_decay_rate": {
                    "level_0_to_1": 0.0,
                    "level_1_to_2": 0.0,
                    "level_2_to_3": 0.0,
                },
                "by_symbol": {},
                "warning": f"{WARNING} (error during analysis)",
                "error": "See logs for details",
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_report(self, n_recent_cycles: int) -> dict[str, Any]:
        records = self._load_records()

        if not records:
            return self._empty_result("no data found in log")

        # Slice to most recent N cycles
        if n_recent_cycles < len(records):
            records = records[-n_recent_cycles:]

        # ------------------------------------------------------------------
        # Phase 1: cycle-level trade counters
        #
        # A "trade at confirm=X" means: the signal reached the confirm gate
        # and the confirm threshold X would have been satisfied.  No other
        # gate is checked because:
        #   - Pre-confirm gates (threshold) were already passed (the signal
        #     reached confirm_gate).
        #   - Post-confirm gates (governor, VEL) are orthogonal — they
        #     would apply equally at any confirm threshold.
        #
        # confirm_1_trades : cycles where signal entered confirm_gate
        #                    (confirm_cycles >= 1 / threshold=1).
        # confirm_2_trades : cycles where CROSS_PASS was achieved
        #                    (current threshold=2).
        # confirm_3_trades : cycles where CROSS_PASS with cross_cyc >= 3
        #                    (hypothetical threshold=3).
        # ------------------------------------------------------------------
        total_signal_cycles = 0  # cycles with any active signal
        confirm_1_trades = 0  # would trade if threshold were 1
        confirm_2_trades = 0  # current threshold (CROSS_PASS)
        confirm_3_trades = 0  # hypothetical threshold=3

        # Per-symbol breakdown (cycle-level)
        by_symbol_cycles: dict[str, dict[str, int]] = defaultdict(
            lambda: {"confirm_1": 0, "confirm_2": 0, "confirm_3": 0}
        )

        # ------------------------------------------------------------------
        # Phase 2: signal-run tracking for decay rates
        #
        # A "signal run" is a sequence of consecutive cycles with the same
        # (symbol, direction, edge).  A gap (cycle with zero total_signals)
        # terminates the current run so that the same edge resurrecting
        # later is counted as a NEW run.
        #
        # We record the maximum confirm level reached in each run:
        #   level 0 = run exists (had signals)
        #   level 1 = at least one cycle in the run entered confirm_gate
        #   level 2 = at least one cycle achieved CROSS_PASS
        #   level 3 = at least one cycle achieved CROSS_PASS with cross_cyc>=3
        # ------------------------------------------------------------------
        run_levels: list[int] = []  # one entry per finished run
        cur_run_id: tuple[str, str, str] | None = None
        cur_max: int = 0  # max level seen in the current run

        def _finalize_run() -> None:
            """Push the current run's max level into the list."""
            nonlocal cur_max
            if cur_run_id is not None:
                run_levels.append(cur_max)
                cur_max = 0

        for entry in records:
            total_sig = entry.get("total_signals", 0)
            if total_sig == 0:
                _finalize_run()
                cur_run_id = None
                continue

            total_signal_cycles += 1

            run_id = _signal_run_id(entry)

            # Detect new run (different identity or after a gap)
            if run_id != cur_run_id:
                _finalize_run()
                cur_run_id = run_id

            pt = entry.get("pipeline_trace", {}) or {}
            confirm_gate = pt.get("confirm_gate", []) or []
            active_symbol = entry.get("active_symbol") or "UNKNOWN"

            entered_confirm = len(confirm_gate) > 0
            if not entered_confirm:
                continue

            # --- CROSS_PASS and cross_cyc extraction ---------------------
            cross_pass = False
            max_cross_cyc = 0
            for item in confirm_gate:
                item_str = str(item)
                m = _CROSS_PASS_CYC_RE.search(item_str)
                if m:
                    cross_pass = True
                    cyc = int(m.group(1))
                    if cyc > max_cross_cyc:
                        max_cross_cyc = cyc

            # Update current run's max level
            if cross_pass:
                if max_cross_cyc >= 3:
                    if cur_max < 3:
                        cur_max = 3
                else:
                    if cur_max < 2:
                        cur_max = 2
            else:
                if cur_max < 1:
                    cur_max = 1

            # --- confirm=1 (cycle-level) ---------------------------------
            confirm_1_trades += 1
            by_symbol_cycles[active_symbol]["confirm_1"] += 1

            # --- confirm=2 (cycle-level) ---------------------------------
            if cross_pass:
                confirm_2_trades += 1
                by_symbol_cycles[active_symbol]["confirm_2"] += 1

                # --- confirm=3 (cycle-level) -----------------------------
                if max_cross_cyc >= 3:
                    confirm_3_trades += 1
                    by_symbol_cycles[active_symbol]["confirm_3"] += 1

        # Finalize the last run (if any)
        _finalize_run()

        # ------------------------------------------------------------------
        # Compute signal decay rates (run-level)
        # ------------------------------------------------------------------
        total_runs = len(run_levels)
        runs_l1 = sum(1 for v in run_levels if v >= 1)
        runs_l2 = sum(1 for v in run_levels if v >= 2)
        runs_l3 = sum(1 for v in run_levels if v >= 3)

        signal_decay_rate = {
            "level_0_to_1": round(runs_l1 / max(total_runs, 1), 4),
            "level_1_to_2": round(runs_l2 / max(runs_l1, 1), 4),
            "level_2_to_3": round(runs_l3 / max(runs_l2, 1), 4),
        }

        # ------------------------------------------------------------------
        # Derived metrics
        # ------------------------------------------------------------------
        loss_due_to_confirm = confirm_1_trades - confirm_2_trades

        optimal_confirm_level = self._compute_optimal_level(
            confirm_1_trades, confirm_2_trades, confirm_3_trades,
        )

        # Per-symbol breakdown with decay rates
        by_symbol_out: dict[str, dict[str, Any]] = {}
        for sym, counts in by_symbol_cycles.items():
            c1 = counts["confirm_1"]
            c2 = counts["confirm_2"]
            by_symbol_out[sym] = {
                "confirm_1": c1,
                "confirm_2": c2,
                "confirm_3": counts["confirm_3"],
                "decay_rate_1_to_2": round(c2 / max(c1, 1), 4),
            }

        return {
            "confirm_1_trades": confirm_1_trades,
            "confirm_2_trades": confirm_2_trades,
            "confirm_3_trades": confirm_3_trades,
            "loss_due_to_confirm": loss_due_to_confirm,
            "optimal_confirm_level": optimal_confirm_level,
            "signal_decay_rate": signal_decay_rate,
            "by_symbol": by_symbol_out,
            "warning": WARNING,
        }

    # ------------------------------------------------------------------
    # Optimal-level heuristic
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_optimal_level(
        c1: int, c2: int, c3: int,
    ) -> int:
        """Heuristic to pick the best confirm threshold.

        Uses the cycle-level trade counts and the signal-run decay rates
        to decide which level best balances safety vs opportunity.

        - Level 1: no gating (max opportunity, least safety).
        - Level 2: current threshold (balances safety vs opportunity).
        - Level 3: stricter threshold (safest, least opportunity).

        Prefer level 2 if it retains at least 50 % of level-1
        opportunities; otherwise prefer level 1.  Level 3 is only
        chosen if it retains at least 80 % of level-2 opportunities.
        """
        if c1 == 0:
            return 2  # No data — default to current

        retention_1_to_2 = c2 / c1
        retention_2_to_3 = c3 / max(c2, 1)

        if retention_1_to_2 >= 0.50 and retention_2_to_3 >= 0.80:
            return 3
        elif retention_1_to_2 >= 0.50:
            return 2
        else:
            return 1

    # ------------------------------------------------------------------
    # Empty-result helper
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result(reason: str) -> dict[str, Any]:
        return {
            "confirm_1_trades": 0,
            "confirm_2_trades": 0,
            "confirm_3_trades": 0,
            "loss_due_to_confirm": 0,
            "optimal_confirm_level": 2,
            "signal_decay_rate": {
                "level_0_to_1": 0.0,
                "level_1_to_2": 0.0,
                "level_2_to_3": 0.0,
            },
            "by_symbol": {},
            "warning": f"{WARNING} ({reason})",
        }

    # ------------------------------------------------------------------
    # Log loader
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
            print("Usage: python confirm_gate_stress_test.py [n_recent_cycles]")
            sys.exit(1)

    runner = ConfirmGateStressTest()
    report = runner.analyze(n_recent_cycles=n)
    print(json.dumps(report, indent=2))
