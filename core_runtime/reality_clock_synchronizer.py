"""
Reality Clock Synchronizer — zero tick drift across MT5, internal cycles,
SDIL snapshots, and SAAL decisions.

Tracks multiple time sources (MT5 server time, local system time, cycle
counters, tick arrival times, layer processing timestamps) and computes
drift statistics to detect timing misalignments before they affect
decision quality.

Usage
-----
    from core_runtime.reality_clock_synchronizer import RealityClockSynchronizer

    rcs = RealityClockSynchronizer()
    rcs.register_time_source("mt5_server", lambda: get_mt5_server_time())
    rcs.register_time_source("local_clock", time.time)

    rcs.tick_received(cycle_id=42, server_time=1234567890.123)
    rcs.layer_processed(42, "SDIL")
    rcs.layer_processed(42, "SAAL")

    drift = rcs.compute_drift(42)
    report = rcs.get_drift_report(cycles=200)
    if rcs.detect_misalignment():
        logger.warning("Time sources misaligned!")
"""

import logging
import math
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_instances: Dict[str, "_RealityClockSynchronizer"] = {}


def RealityClockSynchronizer(instance_id="default"):
    """Singleton accessor for ``_RealityClockSynchronizer``.

    Parameters
    ----------
    instance_id : str
        Unique identifier.  Callers sharing the same *instance_id* share the
        same underlying synchronizer object.

    Returns
    -------
    _RealityClockSynchronizer
    """
    if instance_id not in _instances:
        _instances[instance_id] = _RealityClockSynchronizer(instance_id)
    return _instances[instance_id]


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------

class _RealityClockSynchronizer:
    """Synchronises multiple time sources and tracks tick-to-decision drift.

    The synchronizer maintains:

    * A registry of named time sources (callables returning ``float`` seconds).
    * A per-cycle record of tick arrival and layer processing timestamps.
    * A drift history used to compute aggregate statistics and trends.

    Parameters
    ----------
    instance_id : str
        Instance identifier forwarded from the singleton accessor.
    """

    def __init__(self, instance_id="default"):
        self._instance_id = instance_id

        # Registered time sources: name -> callable() -> float
        self._time_sources: Dict[str, Callable[[], float]] = {}

        # Per-cycle timing data: cycle_id -> dict
        self._cycles: Dict[int, Dict[str, Any]] = {}

        # Drift threshold in seconds (default 1.0)
        self._drift_threshold: float = 1.0

        # Ordered list of computed drift values (for trend analysis)
        self._drift_history: List[float] = []

        logger.info(
            "RealityClockSynchronizer(%r) initialised",
            instance_id,
        )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_time_source(self, name: str, get_time_fn: Callable[[], float]):
        """Register a time source.

        Parameters
        ----------
        name : str
            Arbitrary label for the time source (e.g. ``"mt5_server"``,
            ``"local_clock"``).
        get_time_fn : callable
            A zero-argument callable that returns the current time as a
            ``float`` (epoch seconds or equivalent monotonic value).
        """
        self._time_sources[name] = get_time_fn
        logger.debug("Registered time source %r", name)

    # ------------------------------------------------------------------
    # Tick / layer recording
    # ------------------------------------------------------------------

    def tick_received(
        self,
        cycle_id: int,
        server_time: Optional[float] = None,
    ):
        """Record the arrival of a new tick for *cycle_id*.

        Parameters
        ----------
        cycle_id : int
            Unique cycle identifier (sequential integer).
        server_time : float, optional
            MT5 server timestamp for this tick.  If provided, the
            server-to-local offset is also recorded.
        """
        now = time.time()

        self._cycles[cycle_id] = {
            "cycle_id": cycle_id,
            "server_time": server_time,
            "local_time": now,
            "tick_arrival_time": now,
            "layer_timings": {},
            "decision_time": None,
        }

        if server_time is not None:
            offset = server_time - now
            self._cycles[cycle_id]["offset"] = offset
            if abs(offset) > self._drift_threshold:
                logger.warning(
                    "MT5 server time differs from local time by %.3fs "
                    "(cycle %d)",
                    offset,
                    cycle_id,
                )

        logger.debug(
            "tick_received(cycle=%d, server_time=%s)",
            cycle_id,
            server_time,
        )

    def layer_processed(
        self,
        cycle_id: int,
        layer_name: str,
        processing_time: Optional[float] = None,
    ):
        """Record when *layer_name* finished processing the current tick.

        Parameters
        ----------
        cycle_id : int
            Cycle identifier previously passed to :meth:`tick_received`.
        layer_name : str
            Label for the processing layer (e.g. ``"SDIL"``, ``"SAAL"``).
        processing_time : float, optional
            Timestamp of completion.  Defaults to ``time.time()``.
        """
        if processing_time is None:
            processing_time = time.time()

        cycle = self._cycles.get(cycle_id)
        if cycle is None:
            logger.warning(
                "layer_processed(%s, %s): no tick recorded for cycle %d — "
                "ignoring",
                layer_name,
                cycle_id,
                cycle_id,
            )
            return

        cycle["layer_timings"][layer_name] = processing_time
        cycle["decision_time"] = processing_time

        # Warn if processing time exceeds 10 seconds
        arrival = cycle["tick_arrival_time"]
        duration = processing_time - arrival
        if duration > 10.0:
            logger.warning(
                "Processing time for cycle %d exceeds 10s (%.3fs) — "
                "layer %s",
                cycle_id,
                duration,
                layer_name,
            )

        logger.debug(
            "layer_processed(cycle=%d, layer=%s, time=%.3f)",
            cycle_id,
            layer_name,
            processing_time,
        )

    # ------------------------------------------------------------------
    # Drift computation
    # ------------------------------------------------------------------

    def compute_drift(self, cycle_id: int) -> dict:
        """Compute timing drift for a single cycle.

        Parameters
        ----------
        cycle_id : int
            Cycle identifier.

        Returns
        -------
        dict
            Drift report with the following keys:

            * **cycle_id** — cycle identifier
            * **server_time** — recorded server timestamp (or ``None``)
            * **local_time** — local ``time.time()`` at tick arrival
            * **offset** — ``server_time - local_time`` (or ``0.0``)
            * **processing_duration** — tick-to-decision wall time (seconds)
            * **layer_timings** — ``{layer: duration_ms}``
            * **drift** — deviation from expected timing (uses offset if
              server time is available, otherwise 0.0)
            * **drift_alert** — ``True`` if ``|drift| > threshold``
        """
        cycle = self._cycles.get(cycle_id)
        if cycle is None:
            logger.warning("compute_drift(%d): no data for cycle", cycle_id)
            return {
                "cycle_id": cycle_id,
                "server_time": None,
                "local_time": None,
                "offset": 0.0,
                "processing_duration": 0.0,
                "layer_timings": {},
                "drift": 0.0,
                "drift_alert": False,
            }

        arrival = cycle["tick_arrival_time"]
        decision = cycle["decision_time"] or arrival
        processing_duration = decision - arrival

        offset = cycle.get("offset", 0.0)

        # layer timings in milliseconds relative to tick arrival
        layer_timings_ms: Dict[str, float] = {}
        for layer, ts in cycle["layer_timings"].items():
            layer_timings_ms[layer] = round((ts - arrival) * 1000, 2)

        # Drift: use offset as the primary drift signal when server time
        # is available; otherwise drift is 0 (no reference to compare).
        drift = offset if cycle.get("server_time") is not None else 0.0

        drift_alert = abs(drift) > self._drift_threshold

        # Append to drift history for trend analysis
        self._drift_history.append(drift)
        # Trim history to avoid unbounded memory growth
        if len(self._drift_history) > 10_000:
            self._drift_history = self._drift_history[-5000:]

        report = {
            "cycle_id": cycle_id,
            "server_time": cycle.get("server_time"),
            "local_time": cycle["local_time"],
            "offset": round(offset, 6),
            "processing_duration": round(processing_duration, 6),
            "layer_timings": layer_timings_ms,
            "drift": round(drift, 6),
            "drift_alert": drift_alert,
        }

        return report

    # ------------------------------------------------------------------
    # Aggregate statistics
    # ------------------------------------------------------------------

    def get_drift_report(self, cycles: int = 100) -> dict:
        """Aggregate drift statistics over the most recent *cycles*.

        Parameters
        ----------
        cycles : int
            Number of recent cycles to analyse (default 100).  At most the
            total number of recorded drift values are used.

        Returns
        -------
        dict
            Report with:

            * **cycles_analyzed** — actual number of cycles examined
            * **mean_drift** — mean drift (seconds)
            * **max_drift** — maximum absolute drift (seconds)
            * **drift_std** — standard deviation of drift (seconds)
            * **mean_processing_time** — mean tick-to-decision time (seconds)
            * **total_drift_alerts** — count of cycles with |drift|>threshold
            * **drift_trend** — ``"stable"``, ``"increasing"``, or
              ``"decreasing"``
        """
        # Collect drift values for the requested number of recent cycles
        drift_values = self._drift_history[-cycles:] if self._drift_history else []
        n = len(drift_values)

        if n == 0:
            return {
                "cycles_analyzed": 0,
                "mean_drift": 0.0,
                "max_drift": 0.0,
                "drift_std": 0.0,
                "mean_processing_time": 0.0,
                "total_drift_alerts": 0,
                "drift_trend": "stable",
            }

        mean_drift = sum(drift_values) / n
        max_drift = max(abs(d) for d in drift_values)

        # Standard deviation
        variance = sum((d - mean_drift) ** 2 for d in drift_values) / n
        drift_std = math.sqrt(variance)

        # Mean processing time (only from cycles that have one)
        proc_times = []
        for cid, cycle in self._cycles.items():
            if cycle.get("decision_time") is not None:
                proc_times.append(
                    cycle["decision_time"] - cycle["tick_arrival_time"]
                )
        mean_proc = (
            sum(proc_times) / len(proc_times) if proc_times else 0.0
        )

        # Count drift alerts
        total_alerts = sum(
            1 for d in drift_values if abs(d) > self._drift_threshold
        )

        # Determine trend via linear regression on the drift history
        trend = self._compute_trend(drift_values)

        return {
            "cycles_analyzed": n,
            "mean_drift": round(mean_drift, 6),
            "max_drift": round(max_drift, 6),
            "drift_std": round(drift_std, 6),
            "mean_processing_time": round(mean_proc, 6),
            "total_drift_alerts": total_alerts,
            "drift_trend": trend,
        }

    # ------------------------------------------------------------------
    # Misalignment detection
    # ------------------------------------------------------------------

    def detect_misalignment(self) -> bool:
        """Check whether any registered time source is significantly
        misaligned.

        A time source is considered misaligned if its current value
        differs from ``time.time()`` by more than the drift threshold.

        Returns
        -------
        bool
            ``True`` if any source is misaligned by > threshold.
        """
        local_now = time.time()

        for name, get_time_fn in self._time_sources.items():
            try:
                source_time = get_time_fn()
            except Exception:
                logger.warning(
                    "Time source %r raised an exception — treating as "
                    "misaligned",
                    name,
                    exc_info=True,
                )
                return True

            diff = abs(source_time - local_now)
            if diff > self._drift_threshold:
                logger.warning(
                    "Time source %r is misaligned: |%.3f - %.3f| = %.3fs "
                    "(threshold=%.3fs)",
                    name,
                    source_time,
                    local_now,
                    diff,
                    self._drift_threshold,
                )
                return True

        return False

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_drift_threshold(self, seconds: float):
        """Set the drift alert threshold.

        Parameters
        ----------
        seconds : float
            New threshold in seconds.  Must be non-negative.
        """
        if seconds < 0:
            logger.warning(
                "Ignoring negative drift threshold %.3f", seconds,
            )
            return
        self._drift_threshold = seconds
        logger.debug("Drift threshold set to %.3fs", seconds)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self):
        """Clear all recorded timing data and drift history.

        Time source registrations and the drift threshold are preserved.
        """
        self._cycles.clear()
        self._drift_history.clear()
        logger.info("RealityClockSynchronizer(%r) reset", self._instance_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_trend(values: List[float]) -> str:
        """Determine whether drift values are stable, increasing, or
        decreasing using simple linear regression.

        Parameters
        ----------
        values : list of float
            Sequence of drift values in chronological order.

        Returns
        -------
        str
            ``"stable"``, ``"increasing"``, or ``"decreasing"``.
        """
        n = len(values)
        if n < 50:
            return "stable"

        # Indices 0 .. n-1
        x_avg = (n - 1) / 2.0
        y_avg = sum(values) / n

        num = 0.0
        den = 0.0
        for i, y in enumerate(values):
            dx = i - x_avg
            dy = y - y_avg
            num += dx * dy
            den += dx * dx

        if den == 0.0:
            return "stable"

        slope = num / den

        # Use a small threshold to avoid flagging negligible drift
        if slope > 1e-8:
            return "increasing"
        if slope < -1e-8:
            return "decreasing"
        return "stable"


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest():
    """Exercise the RealityClockSynchronizer with simulated time sources
    and verify drift computation, aggregation, misalignment detection, and
    singleton behaviour.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("=" * 60)
    logger.info("RealityClockSynchronizer — Self-Test")
    logger.info("=" * 60)

    passed = 0
    failed = 0

    def _check(cond: bool, msg: str) -> None:
        nonlocal passed, failed
        if cond:
            passed += 1
            logger.info("  [PASS] %s", msg)
        else:
            failed += 1
            logger.error("  [FAIL] %s", msg)

    # ==================================================================
    # Scenario 1 — basic tick recording and drift computation
    # ==================================================================
    logger.info("")
    logger.info("--- Scenario 1: Basic tick recording and drift ---")

    rcs = RealityClockSynchronizer("selftest_basic")

    # Register two time sources
    fake_server_time = [1000.0]
    rcs.register_time_source("mt5_server", lambda: fake_server_time[0])
    rcs.register_time_source("local_clock", time.time)

    # Tick with no server time
    rcs.tick_received(cycle_id=1)
    rcs.layer_processed(1, "SDIL", processing_time=time.time() + 0.005)
    rcs.layer_processed(1, "SAAL", processing_time=time.time() + 0.010)

    drift1 = rcs.compute_drift(1)
    _check(drift1["cycle_id"] == 1, "Scenario 1 cycle_id = 1")
    _check(drift1["server_time"] is None, "Scenario 1 no server_time")
    _check(drift1["drift"] == 0.0, "Scenario 1 drift = 0.0 (no server ref)")
    _check(drift1["drift_alert"] is False, "Scenario 1 no drift alert")
    _check("SDIL" in drift1["layer_timings"], "Scenario 1 SDIL timing recorded")
    _check("SAAL" in drift1["layer_timings"], "Scenario 1 SAAL timing recorded")
    _check(drift1["processing_duration"] > 0, "Scenario 1 positive proc duration")

    # Tick with server time
    rcs.tick_received(cycle_id=2, server_time=1000.5)
    rcs.layer_processed(2, "SDIL", processing_time=time.time() + 0.003)

    drift2 = rcs.compute_drift(2)
    _check(drift2["cycle_id"] == 2, "Scenario 2 cycle_id = 2")
    _check(drift2["server_time"] == 1000.5, "Scenario 2 server_time set")
    _check(drift2["drift"] != 0.0, "Scenario 2 drift non-zero (has server ref)")
    _check(isinstance(drift2["drift_alert"], bool), "Scenario 2 drift_alert bool")

    # ==================================================================
    # Scenario 2 — drift alert threshold
    # ==================================================================
    logger.info("")
    logger.info("--- Scenario 2: Drift alert threshold ---")

    rcs2 = RealityClockSynchronizer("selftest_threshold")
    rcs2.set_drift_threshold(0.1)  # 100 ms

    # Simulate server time 5 seconds ahead — should trigger alert
    rcs2.tick_received(cycle_id=10, server_time=time.time() + 5.0)
    drift10 = rcs2.compute_drift(10)
    _check(drift10["drift_alert"] is True, "Scenario 2 drift alert triggered")
    _check(abs(drift10["drift"]) > 0.1, "Scenario 2 drift > 0.1s")

    # Restore threshold for remaining tests
    rcs2.set_drift_threshold(1.0)

    # ==================================================================
    # Scenario 3 — layer_processed warning on missing cycle
    # ==================================================================
    logger.info("")
    logger.info("--- Scenario 3: Layer processed with no prior tick ---")

    rcs.layer_processed(999, "GHOST")  # should log warning, not crash
    _check(999 not in rcs._cycles, "Scenario 3 missing cycle not created")

    # ==================================================================
    # Scenario 4 — get_drift_report
    # ==================================================================
    logger.info("")
    logger.info("--- Scenario 4: Aggregate drift report ---")

    rcs4 = RealityClockSynchronizer("selftest_report")
    rcs4.set_drift_threshold(0.5)

    # Feed 60 ticks with increasing server-time offset to create trend
    base_server = time.time() + 0.01
    for i in range(60):
        server_offset = 0.01 + i * 0.002  # drift increases from 10ms
        rcs4.tick_received(
            cycle_id=100 + i,
            server_time=base_server + server_offset,
        )
        rcs4.layer_processed(100 + i, "SDIL", processing_time=time.time() + 0.002)
        rcs4.compute_drift(100 + i)  # populate drift history

    report = rcs4.get_drift_report(cycles=50)
    _check(report["cycles_analyzed"] == 50, "Scenario 4 analyzed 50 cycles")
    _check(report["mean_drift"] > 0, "Scenario 4 positive mean drift")
    _check(report["max_drift"] > 0, "Scenario 4 positive max drift")
    _check(report["drift_std"] >= 0, "Scenario 4 drift_std non-negative")
    _check(report["mean_processing_time"] > 0, "Scenario 4 positive mean proc time")
    _check(
        report["total_drift_alerts"] >= 0,
        "Scenario 4 drift alerts non-negative",
    )
    _check(
        report["drift_trend"] in ("stable", "increasing", "decreasing"),
        "Scenario 4 valid drift_trend",
    )

    # With increasing offsets, trend should be 'increasing'
    _check(
        report["drift_trend"] == "increasing",
        f"Scenario 4 drift_trend = increasing (got {report['drift_trend']})",
    )

    # ==================================================================
    # Scenario 5 — detect_misalignment
    # ==================================================================
    logger.info("")
    logger.info("--- Scenario 5: Misalignment detection ---")

    rcs5 = RealityClockSynchronizer("selftest_misalign")
    misaligned_server = [time.time() + 100.0]  # 100s off
    rcs5.register_time_source(
        "mt5_server", lambda: misaligned_server[0],
    )
    rcs5.register_time_source("local_clock", time.time)
    rcs5.set_drift_threshold(1.0)

    _check(
        rcs5.detect_misalignment() is True,
        "Scenario 5 misalignment detected (100s offset)",
    )

    # Now fix the source
    misaligned_server[0] = time.time()
    _check(
        rcs5.detect_misalignment() is False,
        "Scenario 5 no misalignment after fix",
    )

    # ==================================================================
    # Scenario 6 — reset
    # ==================================================================
    logger.info("")
    logger.info("--- Scenario 6: Reset ---")

    rcs6 = RealityClockSynchronizer("selftest_reset")
    rcs6.tick_received(1)
    rcs6.layer_processed(1, "SDIL")
    _check(len(rcs6._cycles) == 1, "Scenario 6 has 1 cycle before reset")
    _check(len(rcs6._drift_history) == 0, "Scenario 6 no drift before compute")
    rcs6.compute_drift(1)
    _check(len(rcs6._drift_history) == 1, "Scenario 6 drift history after compute")
    rcs6.reset()
    _check(len(rcs6._cycles) == 0, "Scenario 6 cycles cleared after reset")
    _check(
        len(rcs6._drift_history) == 0,
        "Scenario 6 drift history cleared after reset",
    )

    # ==================================================================
    # Scenario 7 — negative threshold rejected
    # ==================================================================
    logger.info("")
    logger.info("--- Scenario 7: Negative threshold rejected ---")

    rcs7 = RealityClockSynchronizer("selftest_negative")
    rcs7.set_drift_threshold(-5.0)
    _check(
        rcs7._drift_threshold == 1.0,
        "Scenario 7 threshold unchanged after negative set",
    )

    # ==================================================================
    # Scenario 8 — Singleton accessor
    # ==================================================================
    logger.info("")
    logger.info("--- Scenario 8: Singleton accessor ---")

    a = RealityClockSynchronizer("selftest_singleton")
    b = RealityClockSynchronizer("selftest_singleton")
    c = RealityClockSynchronizer("selftest_singleton_other")
    _check(a is b, "Same instance_id returns same object")
    _check(a is not c, "Different instance_id returns different object")

    # ==================================================================
    # Scenario 9 — compute_drift on missing cycle
    # ==================================================================
    logger.info("")
    logger.info("--- Scenario 9: compute_drift on missing cycle ---")

    rcs9 = RealityClockSynchronizer("selftest_missing")
    dr = rcs9.compute_drift(99999)
    _check(dr["cycle_id"] == 99999, "Scenario 9 missing cycle_id returned")
    _check(dr["drift"] == 0.0, "Scenario 9 missing drift = 0.0")

    # ==================================================================
    # Scenario 10 — drift trend: decreasing
    # ==================================================================
    logger.info("")
    logger.info("--- Scenario 10: Decreasing drift trend ---")

    rcs10 = RealityClockSynchronizer("selftest_decreasing")
    rcs10.set_drift_threshold(10.0)

    # Feed 60 ticks with decreasing offsets
    base_server10 = time.time() + 0.5
    for i in range(60):
        server_offset = 0.5 - i * 0.008  # drift decreases from 500ms
        if server_offset < 0:
            server_offset = 0.0
        rcs10.tick_received(
            cycle_id=200 + i,
            server_time=base_server10 + server_offset,
        )
        rcs10.layer_processed(200 + i, "SAAL", processing_time=time.time() + 0.001)
        rcs10.compute_drift(200 + i)  # populate drift history

    report10 = rcs10.get_drift_report(cycles=60)
    _check(
        report10["drift_trend"] == "decreasing",
        f"Scenario 10 drift_trend = decreasing (got {report10['drift_trend']})",
    )

    # ==================================================================
    # Summary
    # ==================================================================
    logger.info("")
    logger.info("-" * 60)
    total = passed + failed
    logger.info(
        "Results:  %d / %d passed  (%s)",
        passed,
        total,
        "ALL PASSED" if failed == 0 else f"{failed} FAILED",
    )

    if failed > 0:
        logger.error(">>> SELF-TEST FAILED <<<")
    else:
        logger.info(">>> SELF-TEST PASSED <<<")


if __name__ == "__main__":
    _selftest()
