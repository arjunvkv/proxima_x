"""
governance_compression_audit.py — Quantify expectancy loss due to governance.

Tracks how many raw signal opportunities survive each governance stage:

    Signal → Confirm → Governor → Execution

At each stage we compute:
  - entered : how many cycles entered the stage
  - passed  : how many cycles passed the stage
  - loss    : entered - passed
  - loss_pct: loss / entered  (0 if entered is 0)

The final output includes an overall compression ratio
(post_governance / raw_signals) and a human-readable estimate of
missed profit opportunity.

Usage
-----
    from proxima_ops.analytics.governance_compression_audit import (
        GovernanceCompressionAudit,
    )

    gca = GovernanceCompressionAudit()
    report = gca.analyze(n_recent_cycles=500)
    print(report)
"""

import json
import logging
import os
from typing import Any

logger = logging.getLogger(
    "proxima_ops.analytics.governance_compression_audit"
)

DEFAULT_LOG_PATH = "state/wave12_cycle_log.jsonl"

# Decision values that count as a non-HOLD (i.e. a trade was attempted).
_NON_HOLD_DECISIONS: frozenset[str] = frozenset({
    "BUY",
    "SELL",
    "LONG",
    "SHORT",
    "EXECUTE",
})

# Substrings that indicate the governor blocked the trade.
_GOVERNOR_BLOCK_MARKERS: tuple[str, ...] = (
    "CircuitBreaker",
    "Blocked",
    "Denied",
)


class GovernanceCompressionAudit:
    """Measure governance-stage compression from raw signals → execution."""

    def __init__(self, log_path: str = DEFAULT_LOG_PATH):
        self._log_path = log_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, n_recent_cycles: int = 500) -> dict[str, Any]:
        """Run the governance compression audit.

        Parameters
        ----------
        n_recent_cycles : int
            Only consider this many of the most recent cycles (default 500).
            Pass a large number (e.g. 1_000_000) to analyse all available
            cycles.

        Returns
        -------
        dict with top-level keys:
            raw_signals, post_governance, compression_ratio,
            loss_of_opportunity_rate, stage_breakdown,
            opportunity_loss_estimate
        """
        try:
            return self._build_report(n_recent_cycles)
        except Exception:
            logger.exception(
                "GovernanceCompressionAudit.analyze failed"
            )
            return self._empty_result()

    # ------------------------------------------------------------------
    # Report builder
    # ------------------------------------------------------------------

    def _build_report(
        self, n_recent_cycles: int
    ) -> dict[str, Any]:
        records = self._load_records()

        if not records:
            return self._empty_result("No data found in log")

        # Slice to the N most recent cycles.
        if n_recent_cycles < len(records):
            records = records[-n_recent_cycles:]

        total_cycles = len(records)

        # Per-stage counters.
        signal_passed = 0
        confirm_passed = 0
        governor_passed = 0
        execution_passed = 0

        for entry in records:
            # ---- Stage 1: Signal stage ----
            # Entered by virtue of existing as a cycle.
            # Passed when at least one signal was present.
            total_signals = entry.get("total_signals", 0)
            if total_signals == 0:
                continue
            signal_passed += 1

            # ---- Stage 2: Confirm stage ----
            # Entered = signal_passed.
            # Passed when confirm_cycles >= 2.
            confirm_cycles = entry.get("confirm_cycles", 0)
            if isinstance(confirm_cycles, dict):
                confirm_ok = any(
                    v >= 2 for v in confirm_cycles.values()
                )
            else:
                confirm_ok = bool(confirm_cycles >= 2)

            if not confirm_ok:
                continue
            confirm_passed += 1

            # ---- Stage 3: Governor stage ----
            # Entered = confirm_passed.
            # Passed when segl_state == ARMED AND no CB/VEL block.
            segl_state = entry.get("segl_state", "")
            cb_decision = entry.get("cb_decision", "")
            vel_decision = entry.get("vel_decision", "")
            denial_reason = entry.get("denial_reason") or ""

            governor_ok = (segl_state == "ARMED")

            # Check for CB block.
            if cb_decision and cb_decision.lower() in ("blocked", "denied"):
                governor_ok = False

            # Check for VEL block.
            if vel_decision and vel_decision.lower() in ("blocked", "denied"):
                governor_ok = False

            # Check denial_reason for block markers.
            if any(
                marker.lower() in denial_reason.lower()
                for marker in _GOVERNOR_BLOCK_MARKERS
            ):
                governor_ok = False

            if not governor_ok:
                continue
            governor_passed += 1

            # ---- Stage 4: Execution stage ----
            # Entered = governor_passed.
            # Passed when decision != HOLD.
            decision = entry.get("decision", "HOLD")
            if decision in _NON_HOLD_DECISIONS:
                execution_passed += 1

        # ---- Build stage breakdown ----
        def _stage_info(
            entered: int, passed: int
        ) -> dict[str, Any]:
            loss = entered - passed
            loss_pct = round(loss / max(entered, 1), 4)
            return {
                "entered": entered,
                "passed": passed,
                "loss": loss,
                "loss_pct": loss_pct,
            }

        stage_breakdown = {
            "signal_stage": _stage_info(total_cycles, signal_passed),
            "confirm_stage": _stage_info(signal_passed, confirm_passed),
            "governor_stage": _stage_info(confirm_passed, governor_passed),
            "execution_stage": _stage_info(governor_passed, execution_passed),
        }

        # ---- Overall compression ----
        raw_signals = total_cycles
        post_governance = governor_passed

        compression_ratio = round(
            post_governance / max(raw_signals, 1), 4
        )
        loss_of_opportunity_rate = round(
            1.0 - compression_ratio, 4
        )

        # ---- Opportunity loss estimate ----
        total_missed_cycles = raw_signals - post_governance

        if total_missed_cycles > 10000:
            estimated_missed_profit = "significant"
            confidence = 0.9
        elif total_missed_cycles > 1000:
            estimated_missed_profit = "moderate"
            confidence = 0.7
        else:
            estimated_missed_profit = "minimal"
            confidence = 0.5

        # Adjust confidence downward if sample is very small.
        if raw_signals < 100:
            confidence = round(confidence * 0.5, 2)
        elif raw_signals < 500:
            confidence = round(confidence * 0.8, 2)

        opportunity_loss_estimate = {
            "total_missed_cycles": total_missed_cycles,
            "estimated_missed_profit": estimated_missed_profit,
            "confidence": confidence,
        }

        return {
            "raw_signals": raw_signals,
            "post_governance": post_governance,
            "compression_ratio": compression_ratio,
            "loss_of_opportunity_rate": loss_of_opportunity_rate,
            "stage_breakdown": stage_breakdown,
            "opportunity_loss_estimate": opportunity_loss_estimate,
        }

    # ------------------------------------------------------------------
    # Empty-result helper
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result(
        reason: str = "",
    ) -> dict[str, Any]:
        """Return a safe fallback dict when analysis cannot run."""
        return {
            "raw_signals": 0,
            "post_governance": 0,
            "compression_ratio": 0.0,
            "loss_of_opportunity_rate": 0.0,
            "stage_breakdown": {
                stage: {"entered": 0, "passed": 0, "loss": 0, "loss_pct": 0.0}
                for stage in (
                    "signal_stage",
                    "confirm_stage",
                    "governor_stage",
                    "execution_stage",
                )
            },
            "opportunity_loss_estimate": {
                "total_missed_cycles": 0,
                "estimated_missed_profit": "minimal",
                "confidence": 0.0,
            },
            "warning": reason or "Analysis failed — see logs for details",
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
                            "Skipping unparseable line in %s",
                            self._log_path,
                        )
        except Exception:
            logger.exception(
                "Failed to read trace log: %s", self._log_path
            )
            return []

        return records


# ------------------------------------------------------------------
# CLI convenience
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s [%(levelname)s] "
            "%(name)s: %(message)s"
        ),
    )

    import sys

    n = 500
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            print(
                "Usage: python governance_compression_audit.py "
                "[n_recent_cycles]"
            )
            sys.exit(1)

    gca = GovernanceCompressionAudit()
    report = gca.analyze(n_recent_cycles=n)
    print(json.dumps(report, indent=2))
