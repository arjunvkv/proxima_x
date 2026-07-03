"""
ecp.py — Execution Compression Pipeline

Reduce Signal → Decision → Order to Signal → Order.
Remove redundant intermediate representations (confirm, governor, VEL).

Classes
-------
ExecutionCompressionPipeline
    analyze()  — measure current pipeline depth, delay variance, redundancy
    compress() — bypass intermediary gates and emit a direct order dict
"""

from __future__ import annotations

import json
import logging
import os
import statistics
from typing import Any

logger = logging.getLogger("proxima_ops.emancipation.ecp")

_DEFAULT_LOG_PATH = "state/wave12_cycle_log.jsonl"

# Decision values that represent a non-HOLD (executable) outcome.
_EXECUTED_DECISIONS: frozenset[str] = frozenset({"BUY", "SELL", "LONG", "SHORT"})

# Pipeline stage keys expected in pipeline_trace.
_STAGE_KEYS = ("generated", "threshold_gate", "confirm_gate", "governor_gate", "execution")


class ExecutionCompressionPipeline:
    """Measure and apply execution-pipeline compression.

    The standard pipeline runs through five stages:
        signal → threshold → confirm → governor → execution

    This class detects how many of those stages are *actually* meaningful
    and provides a ``compress()`` method that skips the intermediary
    gates (confirm, governor, VEL) and produces a direct order dict from
    the best signal.
    """

    def __init__(self, log_path: str = _DEFAULT_LOG_PATH) -> None:
        """Initialise with the path to the wave12 cycle log.

        Parameters
        ----------
        log_path : str
            Path to the JSON-lines cycle log (default
            ``state/wave12_cycle_log.jsonl``).
        """
        self._log_path = log_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, n_recent_cycles: int = 500) -> dict[str, Any]:
        """Analyse pipeline stage utilisation over recent cycles.

        Parameters
        ----------
        n_recent_cycles : int
            Number of most-recent cycles to examine (default 500).

        Returns
        -------
        dict with keys:
            current_pipeline_depth,
            compressed_pipeline_depth,
            execution_delay_variance,
            signal_to_order_correlation_lag,
            compression_gain,
            redundant_stages
        """
        try:
            return self._build_analysis(n_recent_cycles)
        except Exception:
            logger.exception("ExecutionCompressionPipeline.analyze failed")
            return self._empty_analysis_result()

    def compress(
        self,
        signals: list[dict[str, Any]],
        best_signal: dict[str, Any] | None,
        mt5_tick: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Compress the pipeline: go directly from *best_signal* to order.

        Skips confirm gate, governor check, and VEL re-evaluation.
        Uses the tick bid/ask for price when available.

        Parameters
        ----------
        signals : list[dict]
            Full list of raw signals (may be empty).
        best_signal : dict or None
            The highest-confidence signal selected by the caller.
        mt5_tick : dict or None
            Latest MT5 tick, expected to contain ``"bid"`` and ``"ask"``
            (or ``"price"``).

        Returns
        -------
        dict with keys:
            action, symbol, volume, price, compressed,
            stages_skipped, execution_readiness
        """
        try:
            return self._build_compressed_order(signals, best_signal, mt5_tick)
        except Exception:
            logger.exception("ExecutionCompressionPipeline.compress failed")
            return self._empty_compress_result()

    # ------------------------------------------------------------------
    # Analysis internals
    # ------------------------------------------------------------------

    def _build_analysis(self, n_recent_cycles: int) -> dict[str, Any]:
        records = self._load_records()
        if not records:
            return self._empty_analysis_result("No data found in log")

        if n_recent_cycles < len(records):
            records = records[-n_recent_cycles:]

        # --- current_pipeline_depth ---
        # Count how many distinct stages in pipeline_trace actually
        # appear across cycles (at least once).
        observed_stages: set[str] = set()
        total_nonempty_stages = 0
        cycle_count = 0

        # --- signal-to-order tracking ---
        # For each cycle where a signal first appears, find the number
        # of cycles until decision != HOLD.
        signal_first_cycle: dict[str, int] = {}  # symbol → cycle when first seen
        signal_to_order_lags: list[int] = []
        # Track whether a symbol has already been executed.
        executed_symbols: set[str] = set()

        # --- trivial-pass tracking for redundancy ---
        confirm_trivial = 0  # cycles where confirm_gate is trivially empty/passed
        governor_trivial = 0  # cycles where governor_gate is trivially passed
        confirm_nonempty = 0
        governor_nonempty = 0

        for entry in records:
            cycle = entry.get("cycle", 0)
            pipeline_trace = entry.get("pipeline_trace") or {}
            decision = entry.get("decision", "HOLD")

            # Count distinct non-empty stages.
            for stage in _STAGE_KEYS:
                val = pipeline_trace.get(stage)
                if val is not None and val != [] and val != "" and val != {}:
                    observed_stages.add(stage)

            # Count total non-empty stage entries across this cycle.
            stage_count = sum(
                1
                for stage in _STAGE_KEYS
                if (
                    (v := pipeline_trace.get(stage)) is not None
                    and v != []
                    and v != ""
                    and v != {}
                )
            )
            if stage_count > 0:
                total_nonempty_stages += stage_count
                cycle_count += 1

            # --- Signal-to-order lag ---
            # Detect when a symbol first appears in generated signals.
            generated = pipeline_trace.get("generated") or []
            if isinstance(generated, list):
                for sig in generated:
                    if isinstance(sig, dict):
                        sym = sig.get("symbol")
                        if sym and sym not in signal_first_cycle:
                            signal_first_cycle[sym] = cycle

            # If decision is an execution, record lag for any pending symbols.
            if decision in _EXECUTED_DECISIONS:
                active_symbol = entry.get("active_symbol")
                if active_symbol and active_symbol in signal_first_cycle:
                    lag = cycle - signal_first_cycle[active_symbol]
                    if lag >= 0 and active_symbol not in executed_symbols:
                        signal_to_order_lags.append(lag)
                        executed_symbols.add(active_symbol)

            # --- Trivial pass detection ---
            confirm_val = pipeline_trace.get("confirm_gate")
            if confirm_val is not None and confirm_val != [] and confirm_val != "" and confirm_val != {}:
                confirm_nonempty += 1
            else:
                confirm_trivial += 1

            gov_val = pipeline_trace.get("governor_gate")
            if gov_val is not None and gov_val != [] and gov_val != "" and gov_val != {}:
                governor_nonempty += 1
            else:
                governor_trivial += 1

        # Pipeline depth = average number of stages that did work.
        current_pipeline_depth = (
            round(total_nonempty_stages / max(cycle_count, 1))
            if cycle_count > 0
            else len(_STAGE_KEYS)
        )
        compressed_pipeline_depth = 2  # signal → order

        # --- Execution delay variance ---
        # Variance of signal-to-order lags.
        if len(signal_to_order_lags) >= 2:
            execution_delay_variance = round(
                statistics.variance(signal_to_order_lags), 4
            )
        elif len(signal_to_order_lags) == 1:
            execution_delay_variance = 0.0
        else:
            execution_delay_variance = 0.0

        # --- Signal-to-order correlation lag (mean) ---
        if signal_to_order_lags:
            signal_to_order_correlation_lag = round(
                statistics.mean(signal_to_order_lags), 4
            )
        else:
            signal_to_order_correlation_lag = 0.0

        # --- Compression gain ---
        compression_gain = round(
            (current_pipeline_depth - compressed_pipeline_depth)
            / max(current_pipeline_depth, 1),
            4,
        )

        # --- Redundant stages ---
        redundant_stages: list[str] = []
        total_cycles = len(records)
        if total_cycles > 0:
            confirm_trivial_ratio = confirm_trivial / total_cycles
            governor_trivial_ratio = governor_trivial / total_cycles
        else:
            confirm_trivial_ratio = 0.0
            governor_trivial_ratio = 0.0

        # If >50 % of cycles have trivially empty confirm gate, it is redundant.
        if confirm_trivial_ratio > 0.5:
            redundant_stages.append("confirm_gate")
        # If >50 % of cycles have trivially empty/non-blocking governor gate.
        if governor_trivial_ratio > 0.5:
            redundant_stages.append("governor_re_evaluation")

        return {
            "current_pipeline_depth": current_pipeline_depth,
            "compressed_pipeline_depth": compressed_pipeline_depth,
            "execution_delay_variance": execution_delay_variance,
            "signal_to_order_correlation_lag": signal_to_order_correlation_lag,
            "compression_gain": compression_gain,
            "redundant_stages": redundant_stages,
        }

    # ------------------------------------------------------------------
    # Compress internals
    # ------------------------------------------------------------------

    @staticmethod
    def _build_compressed_order(
        signals: list[dict[str, Any]],
        best_signal: dict[str, Any] | None,
        mt5_tick: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build a compressed order dict directly from *best_signal*."""
        stages_skipped = ["confirm_gate", "governor_check", "vel_check"]

        if best_signal is None or not best_signal:
            # No signal to act on — return a HOLD placeholder.
            return {
                "action": "HOLD",
                "symbol": None,
                "volume": 0.0,
                "price": None,
                "compressed": True,
                "stages_skipped": stages_skipped,
                "execution_readiness": 0.0,
            }

        # --- Determine action ---
        action = best_signal.get("direction", "HOLD")
        if action not in ("BUY", "SELL", "LONG", "SHORT"):
            action = "HOLD"

        # Normalise LONG/SHORT → BUY/SELL for order dispatch.
        if action == "LONG":
            action = "BUY"
        elif action == "SHORT":
            action = "SELL"

        # --- Symbol ---
        symbol = best_signal.get("symbol")

        # --- Confidence as volume scalar ---
        confidence = float(best_signal.get("confidence", 0.0) or 0.0)
        # Base volume = 0.01 lot, scaled by confidence (capped at 1.0 lot).
        volume = round(max(0.01, min(1.0, confidence)), 2)

        # --- Price from tick or signal ---
        price: float | None = None
        if mt5_tick and isinstance(mt5_tick, dict):
            if action in ("BUY", "LONG"):
                price = mt5_tick.get("ask") or mt5_tick.get("price")
            else:
                price = mt5_tick.get("bid") or mt5_tick.get("price")
        # Fall back to best_signal price.
        if price is None:
            price = best_signal.get("price")

        # --- Execution readiness ---
        execution_readiness = round(min(1.0, confidence * 1.5), 4)

        return {
            "action": action,
            "symbol": symbol,
            "volume": volume,
            "price": price,
            "compressed": True,
            "stages_skipped": stages_skipped,
            "execution_readiness": execution_readiness,
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
                            "Skipping unparseable line in %s",
                            self._log_path,
                        )
        except Exception:
            logger.exception("Failed to read cycle log: %s", self._log_path)
            return []

        return records

    # ------------------------------------------------------------------
    # Empty / fallback results
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_analysis_result(
        reason: str = "",
    ) -> dict[str, Any]:
        """Return a safe fallback analysis dict."""
        return {
            "current_pipeline_depth": 0,
            "compressed_pipeline_depth": 2,
            "execution_delay_variance": 0.0,
            "signal_to_order_correlation_lag": 0.0,
            "compression_gain": 0.0,
            "redundant_stages": [],
            "warning": reason or "Analysis failed — see logs for details",
        }

    @staticmethod
    def _empty_compress_result() -> dict[str, Any]:
        """Return a safe fallback compress dict."""
        return {
            "action": "HOLD",
            "symbol": None,
            "volume": 0.0,
            "price": None,
            "compressed": False,
            "stages_skipped": [],
            "execution_readiness": 0.0,
            "warning": "Compression failed — see logs for details",
        }


# ------------------------------------------------------------------
# CLI convenience
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    import sys

    ecp = ExecutionCompressionPipeline()

    if len(sys.argv) > 1 and sys.argv[1] == "compress":
        # Example: python ecp.py compress
        dummy_signals = [
            {"symbol": "EURUSD", "direction": "BUY", "confidence": 0.82},
        ]
        dummy_best = {"symbol": "EURUSD", "direction": "BUY", "confidence": 0.82}
        dummy_tick = {"bid": 1.1050, "ask": 1.1052}
        result = ecp.compress(dummy_signals, dummy_best, dummy_tick)
        print(json.dumps(result, indent=2))
    else:
        # Analyse mode.
        n = 500
        if len(sys.argv) > 1:
            try:
                n = int(sys.argv[1])
            except ValueError:
                print("Usage: python ecp.py [n_recent_cycles|compress]")
                sys.exit(1)
        report = ecp.analyze(n_recent_cycles=n)
        print(json.dumps(report, indent=2))
