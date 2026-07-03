"""
signal_execution_compression.py — Measure the full funnel:

    SIL → CORE → Confirm → Governor → EXEC

Answer: where is the bottleneck?  Which stage compresses (eliminates) the
most trading opportunities from the pipeline?

Metrics
-------
signal_generation       — cycles that had at least one signal
confirm_pass            — cycles where the confirm gate was satisfied
                          (confirm_cycles >= 2)
governor_pass           — cycles where the governor was in ARMED state
                          and did NOT block the trade
execution_count         — cycles where a trade was actually dispatched
                          (decision is BUY / SELL / etc.)

Compression ratios show how much throughput survives each stage.
The bottleneck is the stage with the lowest ratio (most compression).

Usage
-----
    from proxima_ops.analytics.signal_execution_compression import (
        SignalExecutionCompression,
    )

    sec = SignalExecutionCompression()
    report = sec.analyze(n_recent_cycles=500)
    print(report)
"""

import json
import logging
import os
from typing import Any

logger = logging.getLogger(
    "proxima_ops.analytics.signal_execution_compression"
)

DEFAULT_LOG_PATH = "state/wave12_cycle_log.jsonl"

# Decision values that count as a successfully executed trade.
_EXECUTED_DECISIONS: frozenset[str] = frozenset({
    "BUY",
    "SELL",
    "LONG",
    "SHORT",
})

# Governor-related denial substrings (the governor blocked the trade).
_GOVERNOR_DENIAL_MARKERS: tuple[str, ...] = (
    "State=OBSERVE",
    "CircuitBreaker",
)


class SignalExecutionCompression:
    """Measure pipeline-stage compression from signal → execution."""

    def __init__(self, log_path: str = DEFAULT_LOG_PATH):
        self._log_path = log_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, n_recent_cycles: int = 500) -> dict[str, Any]:
        """Run the compression analysis on the trace log.

        Parameters
        ----------
        n_recent_cycles : int
            Only consider this many of the most recent cycles (default 500).
            Pass a large number (e.g. 1_000_000) to analyse all available
            cycles.

        Returns
        -------
        dict with keys:
            signal_generation,
            confirm_pass,
            governor_pass,
            execution_count,
            execution_rate,
            compression_bottleneck_stage,
            stage_compression_ratios
        """
        try:
            return self._build_report(n_recent_cycles)
        except Exception:
            logger.exception(
                "SignalExecutionCompression.analyze failed"
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

        # Per-funnel counters.
        signal_generation = 0
        confirm_pass = 0
        governor_pass = 0
        execution_count = 0

        for entry in records:
            # ---- Stage 1: Signal generation (SIL → CORE) ----
            #
            # The spec says to check `best_signal is not None`, but the
            # field does not exist in the current log format.  We use
            # `total_signals > 0` instead, matching the convention of
            # sibling analyzers (trade_emergence_analyzer.py and
            # confirm_gate_stress_test.py).
            total_signals = entry.get("total_signals", 0)
            if total_signals == 0:
                continue
            signal_generation += 1

            # ---- Stage 2: Confirm gate ----
            #
            # The spec describes confirm_cycles as a dict, but the
            # log stores it as a plain integer.  We handle both.
            confirm_cycles = entry.get("confirm_cycles", 0)
            if isinstance(confirm_cycles, dict):
                confirm_ok = any(
                    v >= 2 for v in confirm_cycles.values()
                )
            else:
                confirm_ok = bool(confirm_cycles >= 2)

            if not confirm_ok:
                continue
            confirm_pass += 1

            # ---- Stage 3: Governor gate ----
            segl_state = entry.get("segl_state", "")
            denial_reason = entry.get("denial_reason") or ""

            governor_ok = segl_state == "ARMED"
            # Check for governor-related denial.
            if any(
                marker in denial_reason
                for marker in _GOVERNOR_DENIAL_MARKERS
            ):
                governor_ok = False

            if not governor_ok:
                continue
            governor_pass += 1

            # ---- Stage 4: Execution ----
            decision = entry.get("decision", "HOLD")
            if decision in _EXECUTED_DECISIONS:
                execution_count += 1

        # ---- Compression ratios ----
        stage_compression_ratios: dict[str, float] = {
            "confirm_compression": round(
                confirm_pass / max(signal_generation, 1), 4
            ),
            "governor_compression": round(
                governor_pass / max(confirm_pass, 1), 4
            ),
            "execution_compression": round(
                execution_count / max(governor_pass, 1), 4
            ),
        }

        execution_rate = round(
            execution_count / max(signal_generation, 1), 4
        )

        # ---- Bottleneck detection ----
        # The bottleneck is the stage with the lowest compression ratio
        # (i.e. the most throughput lost).
        #
        # When all ratios are identical (e.g. all 0 because the pipeline
        # never started) we report "unknown" rather than a misleading
        # arbitrary stage.
        bottleneck_ratios = {
            "confirm": stage_compression_ratios["confirm_compression"],
            "governor": stage_compression_ratios["governor_compression"],
            "execution": stage_compression_ratios[
                "execution_compression"
            ],
        }

        unique_ratios = set(bottleneck_ratios.values())
        if len(unique_ratios) <= 1:
            # All stages have the same ratio — no meaningful bottleneck.
            if signal_generation == 0:
                bottleneck_stage = "unknown"
            else:
                # All non-zero equal values means uniform compression
                # across every stage — still ambiguous.
                bottleneck_stage = "unknown"
        else:
            bottleneck_stage = min(
                bottleneck_ratios, key=bottleneck_ratios.get
            )

        return {
            "signal_generation": signal_generation,
            "confirm_pass": confirm_pass,
            "governor_pass": governor_pass,
            "execution_count": execution_count,
            "execution_rate": execution_rate,
            "compression_bottleneck_stage": bottleneck_stage,
            "stage_compression_ratios": stage_compression_ratios,
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
            "signal_generation": 0,
            "confirm_pass": 0,
            "governor_pass": 0,
            "execution_count": 0,
            "execution_rate": 0.0,
            "compression_bottleneck_stage": "unknown",
            "stage_compression_ratios": {
                "confirm_compression": 0.0,
                "governor_compression": 0.0,
                "execution_compression": 0.0,
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
                "Usage: python signal_execution_compression.py "
                "[n_recent_cycles]"
            )
            sys.exit(1)

    sec = SignalExecutionCompression()
    report = sec.analyze(n_recent_cycles=n)
    print(json.dumps(report, indent=2))
