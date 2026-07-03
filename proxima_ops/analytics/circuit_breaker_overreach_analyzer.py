"""
circuit_breaker_overreach_analyzer.py — Determine if CircuitBreaker is a genuine
safety layer or an over-suppression layer that kills valid trades unnecessarily.

Analyses the pipeline trace log (state/wave12_cycle_log.jsonl) and classifies
every CircuitBreaker block as either:

    * legitimate   — open_positions > 0 OR denial_reason mentions an order failure
    * overreach    — no open positions and no order failure context

Output
------
    total_blocks             — total CircuitBreaker blocks observed
    legitimate_blocks        — blocks that were justified
    overblocking_rate        — fraction of blocks that were *not* legitimate
    legitimate_triggers      — breakdown: consecutive_order_failures,
                               slippage_exceeded, other_risk
    recovery_gain_if_reduced — estimated trade gain if every other CB block
                               were allowed through (total_blocks // 2)
    blocks_by_cycle_range    — per-cycle-start count of CB blocks

Usage
-----
    from proxima_ops.analytics.circuit_breaker_overreach_analyzer \
        import CircuitBreakerOverreachAnalyzer

    analyzer = CircuitBreakerOverreachAnalyzer()
    report = analyzer.analyze(n_recent_cycles=500)
    print(report)
"""

import json
import logging
import os
from collections import defaultdict
from typing import Any

logger = logging.getLogger("proxima_ops.analytics.circuit_breaker_overreach_analyzer")

DEFAULT_LOG_PATH = "state/wave12_cycle_log.jsonl"


class CircuitBreakerOverreachAnalyzer:
    """Determine if CircuitBreaker is safety layer or over-suppression layer.

    Parameters
    ----------
    log_path : str
        Path to the JSON-lines cycle log (default ``state/wave12_cycle_log.jsonl``).
    """

    def __init__(self, log_path: str = DEFAULT_LOG_PATH):
        self._log_path = log_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, n_recent_cycles: int = 500) -> dict[str, Any]:
        """Classify CircuitBreaker blocks into legitimate vs. overreach.

        Parameters
        ----------
        n_recent_cycles : int
            Only consider this many of the most recent cycles (default 500).
            Pass a large number (e.g. 1_000_000) to analyse all available
            cycles.

        Returns
        -------
        dict with keys:
            total_blocks,
            legitimate_blocks,
            overblocking_rate,
            legitimate_triggers,
            recovery_gain_if_reduced,
            blocks_by_cycle_range,
        """
        try:
            return self._build_report(n_recent_cycles)
        except Exception:
            logger.exception("CircuitBreakerOverreachAnalyzer.analyze failed")
            return {
                "total_blocks": 0,
                "legitimate_blocks": 0,
                "overblocking_rate": 0.0,
                "legitimate_triggers": {
                    "consecutive_order_failures": 0,
                    "slippage_exceeded": 0,
                    "other_risk": 0,
                },
                "recovery_gain_if_reduced": 0,
                "blocks_by_cycle_range": {},
                "error": "See logs for details",
            }

    # ------------------------------------------------------------------
    # Report builder
    # ------------------------------------------------------------------

    def _build_report(self, n_recent_cycles: int) -> dict[str, Any]:
        records = self._load_records()

        if not records:
            return {
                "total_blocks": 0,
                "legitimate_blocks": 0,
                "overblocking_rate": 0.0,
                "legitimate_triggers": {
                    "consecutive_order_failures": 0,
                    "slippage_exceeded": 0,
                    "other_risk": 0,
                },
                "recovery_gain_if_reduced": 0,
                "blocks_by_cycle_range": {},
                "warning": "No data found in log",
            }

        # Slice to the N most recent cycles.
        if n_recent_cycles < len(records):
            records = records[-n_recent_cycles:]

        # --- Classification pass ----------------------------------------
        total_blocks: int = 0
        legitimate_blocks: int = 0
        legitimate_triggers: dict[str, int] = {
            "consecutive_order_failures": 0,
            "slippage_exceeded": 0,
            "other_risk": 0,
        }
        blocks_by_cycle_range: dict[str, int] = defaultdict(int)

        for entry in records:
            denial_reason = entry.get("denial_reason") or ""
            # Only consider CircuitBreaker blocks.
            if "CircuitBreaker" not in denial_reason:
                continue

            total_blocks += 1

            # Bucket by cycle range.
            cycle = entry.get("cycle", 0)
            bucket_start = (cycle // 100) * 100
            blocks_by_cycle_range[f"cycle_{bucket_start}"] += 1

            # Determine legitimacy.
            is_legitimate, trigger = self._classify_block(entry)
            if is_legitimate:
                legitimate_blocks += 1
                legitimate_triggers[trigger] += 1

        # --- Derived metrics --------------------------------------------
        overblocking_rate: float = 0.0
        if total_blocks > 0:
            overblocking_rate = round(
                (total_blocks - legitimate_blocks) / total_blocks, 4
            )

        recovery_gain_if_reduced = total_blocks // 2

        return {
            "total_blocks": total_blocks,
            "legitimate_blocks": legitimate_blocks,
            "overblocking_rate": overblocking_rate,
            "legitimate_triggers": dict(legitimate_triggers),
            "recovery_gain_if_reduced": recovery_gain_if_reduced,
            "blocks_by_cycle_range": dict(blocks_by_cycle_range),
        }

    # ------------------------------------------------------------------
    # Block classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_block(entry: dict) -> tuple[bool, str]:
        """Classify a single CircuitBreaker block as legitimate or overreach.

        A block is **legitimate** if any of the following hold:

        1. ``open_positions > 0`` — real exposure exists that needs protecting.
        2. ``denial_reason`` contains a recognised order-failure substring
           (e.g. "order failed", "consecutive failures", "slippage exceeded").

        When legitimate, the second element of the tuple is the trigger
        category:

            * ``"consecutive_order_failures"`` — order-failure keywords present
            * ``"slippage_exceeded"``           — slippage keywords present
            * ``"other_risk"``                  — legitimate but neither of above
              (e.g. open_positions > 0 without mention of failures)

        When *not* legitimate the trigger is ``"overreach"``.

        Returns
        -------
        (is_legitimate: bool, trigger: str)
        """
        open_positions = entry.get("open_positions", 0) or 0
        denial_reason = (entry.get("denial_reason") or "").lower()

        # --- Consecutive order failures ---
        if any(
            kw in denial_reason
            for kw in ["order failed", "consecutive failure", "consecutive order"]
        ):
            return True, "consecutive_order_failures"

        # --- Slippage exceeded ---
        if "slippage" in denial_reason:
            return True, "slippage_exceeded"

        # --- Open positions (real exposure) ---
        if open_positions > 0:
            return True, "other_risk"

        # --- Everything else is overreach ---
        return False, "overreach"

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_records(self) -> list[dict[str, Any]]:
        """Load and return all JSON-line records from the trace log.

        Returns an empty list if the file does not exist or is unreadable.
        """
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
            print("Usage: python circuit_breaker_overreach_analyzer.py [n_recent_cycles]")
            sys.exit(1)

    analyzer = CircuitBreakerOverreachAnalyzer()
    report = analyzer.analyze(n_recent_cycles=n)
    print(json.dumps(report, indent=2))
