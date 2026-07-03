"""
Decision Latency Alignment System — synchronises decision timestamps across
all cognitive layers (SDIL, CSRF, SAAL, MRSRL) that compute at different speeds.

Each layer has a characteristic latency (in ticks):

  =====  ==========  ===============================================
  Layer  Est. Lag    Reason
  =====  ==========  ===============================================
  SDIL   0           Fastest — statistical, real-time inference
  CSRF   2           Medium  — needs evidence accumulation
  SAAL   5           Slowest — needs authority history look-up
  MRSRL  1           Fast    — near real-time, slightly behind SDIL
  =====  ==========  ===============================================

The system maintains per-symbol, per-layer ring buffers of recent outputs
and aligns them to a common decision timestamp, compensating for lag so
that the final execution engine uses temporally consistent data.

Usage
-----
    from core_runtime.decision_latency_alignment_system import (
        DecisionLatencyAlignmentSystem,
    )

    dlas = DecisionLatencyAlignmentSystem()
    dlas.record_layer_output("EURUSD", "sdil", tick, {...})
    dlas.record_layer_output("EURUSD", "csfr", tick, {...})
    alignment = dlas.align_for_decision("EURUSD", current_tick=10)
    print(alignment["max_lag"])
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton registry
# ---------------------------------------------------------------------------

_instances: Dict[str, "_DecisionLatencyAlignmentSystem"] = {}


def DecisionLatencyAlignmentSystem(instance_id="default"):
    """Singleton accessor for ``_DecisionLatencyAlignmentSystem``.

    Parameters
    ----------
    instance_id : str
        Logical identifier for the alignment system instance
        (default ``"default"``).

    Returns
    -------
    _DecisionLatencyAlignmentSystem
    """
    if instance_id not in _instances:
        _instances[instance_id] = _DecisionLatencyAlignmentSystem(instance_id)
    return _instances[instance_id]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_LAYERS = frozenset({"sdil", "csfr", "saal", "mrsrl"})

_DEFAULT_LATENCY_ESTIMATES = {
    "sdil": 0,
    "csfr": 2,
    "saal": 5,
    "mrsrl": 1,
}

_DEFAULT_BUFFER_SIZE = 20


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _closest(entries: List[Tuple[float, Any]],
             target_timestamp: float) -> Optional[Tuple[float, Any]]:
    """Return the entry whose timestamp is closest to *target_timestamp*.

    Parameters
    ----------
    entries : list of (timestamp, output) tuples
    target_timestamp : float

    Returns
    -------
    (timestamp, output) or None if *entries* is empty.
    """
    if not entries:
        return None
    return min(entries, key=lambda e: abs(e[0] - target_timestamp))


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------


class _DecisionLatencyAlignmentSystem:
    """Synchronises decision timestamps across cognitive layers.

    Parameters
    ----------
    instance_id : str
        Logical instance identifier (used for logging).
    """

    def __init__(self, instance_id="default"):
        self._instance_id = instance_id
        self._latency_estimates = dict(_DEFAULT_LATENCY_ESTIMATES)
        self._buffer_size = _DEFAULT_BUFFER_SIZE

        # Per-symbol, per-layer ring buffers.
        #   self._buffers[symbol][layer] = [(timestamp, output), ...]
        self._buffers: Dict[str, Dict[str, List[Tuple[float, Any]]]] = {}

        # Cross-symbol timestamp log used by estimate_latency().
        self._all_timestamps: Dict[str, List[float]] = {
            layer: [] for layer in _VALID_LAYERS
        }

        logger.debug(
            "DecisionLatencyAlignmentSystem(%r) initialised",
            instance_id,
        )

    # ------------------------------------------------------------------
    # Public API — recording
    # ------------------------------------------------------------------

    def record_layer_output(self, symbol: str, layer_name: str,
                            timestamp: float, output: Any):
        """Store a layer's output with its production timestamp.

        Parameters
        ----------
        symbol : str
            Instrument identifier (e.g. ``"EURUSD"``).
        layer_name : str
            One of ``"sdil"``, ``"csfr"``, ``"saal"``, ``"mrsrl"``.
        timestamp : float
            The tick index or time at which this output was produced.
        output : Any
            The layer's output (signal dict, score, or other data).
        """
        layer = layer_name.lower()
        if layer not in _VALID_LAYERS:
            logger.warning(
                "record_layer_output: unknown layer '%s'. "
                "Valid layers: %s",
                layer_name, sorted(_VALID_LAYERS),
            )
            return

        # Ensure buffer structure exists for this symbol.
        if symbol not in self._buffers:
            self._buffers[symbol] = {l: [] for l in _VALID_LAYERS}

        buf = self._buffers[symbol][layer]
        buf.append((timestamp, output))

        # Trim to buffer_size (ring-buffer behaviour).
        if len(buf) > self._buffer_size:
            buf.pop(0)

        # Log timestamp for cross-symbol latency estimation.
        self._all_timestamps[layer].append(timestamp)
        if len(self._all_timestamps[layer]) > 1000:
            self._all_timestamps[layer] = self._all_timestamps[layer][-500:]

        logger.debug(
            "record_layer_output(%s, %s, ts=%.4f): buffer now %d entries",
            symbol, layer, timestamp, len(buf),
        )

    # ------------------------------------------------------------------
    # Public API — alignment
    # ------------------------------------------------------------------

    def align_for_decision(self, symbol: str,
                           current_timestamp: float) -> Dict[str, Any]:
        """Align all layer outputs to the same moment in time.

        For each layer, the output whose timestamp is closest to
        *current_timestamp* is selected.  The *max_lag* across all layers
        that have produced data is used to compute a *decision_timestamp*
        that every layer can satisfy.

        Parameters
        ----------
        symbol : str
            Instrument identifier.
        current_timestamp : float
            The nominal decision time (typically the current tick).

        Returns
        -------
        dict with keys:
            ``aligned_sdil``   — closest SDIL output (or None)
            ``aligned_csfr``   — closest CSRF output (or None)
            ``aligned_saal``   — closest SAAL output (or None)
            ``aligned_mrsrl``  — closest MRSRL output (or None)
            ``max_lag``        — max ticks of lag across present layers
            ``timestamps``     — ``{layer: timestamp_used}``
            ``decision_timestamp`` — *current_timestamp* - *max_lag*
        """
        if symbol not in self._buffers:
            return self._empty_alignment(current_timestamp)

        buffers = self._buffers[symbol]
        aligned: Dict[str, Any] = {}
        timestamps: Dict[str, Optional[float]] = {}
        lags: Dict[str, Optional[float]] = {}

        for layer in _VALID_LAYERS:
            entries = buffers[layer]
            closest_entry = _closest(entries, current_timestamp)
            if closest_entry is None:
                aligned[layer] = None
                timestamps[layer] = None
                lags[layer] = None
            else:
                ts, output = closest_entry
                aligned[layer] = output
                timestamps[layer] = ts
                lags[layer] = current_timestamp - ts

        # max_lag = max over layers that have data.
        present_lags = [lag for lag in lags.values() if lag is not None]
        max_lag = max(present_lags) if present_lags else 0
        decision_timestamp = current_timestamp - max_lag

        return {
            "aligned_sdil": aligned["sdil"],
            "aligned_csfr": aligned["csfr"],
            "aligned_saal": aligned["saal"],
            "aligned_mrsrl": aligned["mrsrl"],
            "max_lag": max_lag,
            "timestamps": {
                "sdil": timestamps["sdil"],
                "csfr": timestamps["csfr"],
                "saal": timestamps["saal"],
                "mrsrl": timestamps["mrsrl"],
            },
            "decision_timestamp": decision_timestamp,
        }

    # ------------------------------------------------------------------
    # Public API — reporting & estimation
    # ------------------------------------------------------------------

    def get_latency_report(self, symbol: str) -> Dict[str, dict]:
        """Return latency statistics per layer for a given symbol.

        For each layer, reports the average and maximum inter-output gap
        (a proxy for lag) and the number of buffered entries.

        Parameters
        ----------
        symbol : str
            Instrument identifier.

        Returns
        -------
        dict
            ``{layer: {"avg_lag": float, "max_lag": float, "count": int}}``
        """
        if symbol not in self._buffers:
            return {}

        buffers = self._buffers[symbol]
        report: Dict[str, dict] = {}

        for layer in _VALID_LAYERS:
            entries = buffers[layer]
            if not entries:
                report[layer] = {"avg_lag": 0.0, "max_lag": 0.0, "count": 0}
                continue

            # Use inter-output gaps as a proxy for layer cadence.
            timestamps = [e[0] for e in entries]
            if len(timestamps) >= 2:
                gaps = [timestamps[i + 1] - timestamps[i]
                        for i in range(len(timestamps) - 1)]
                avg_lag = sum(gaps) / len(gaps)
                max_lag = max(gaps)
            else:
                avg_lag = 0.0
                max_lag = 0.0

            report[layer] = {
                "avg_lag": round(avg_lag, 4),
                "max_lag": round(max_lag, 4),
                "count": len(entries),
            }

        return report

    def estimate_latency(self, layer_name: str) -> float:
        """Auto-estimate a layer's typical latency based on recorded data.

        Computes the **median** timestamp gap between consecutive outputs
        for the given layer across **all** symbols.  Falls back to the
        built-in default estimate if fewer than 2 data points exist.

        Parameters
        ----------
        layer_name : str
            One of ``"sdil"``, ``"csfr"``, ``"saal"``, ``"mrsrl"``.

        Returns
        -------
        float
            Estimated latency in ticks.
        """
        layer = layer_name.lower()
        if layer not in _VALID_LAYERS:
            logger.warning("estimate_latency: unknown layer '%s'", layer_name)
            return 0.0

        timestamps = self._all_timestamps.get(layer, [])
        if len(timestamps) < 2:
            return float(self._latency_estimates.get(layer, 0))

        gaps = [timestamps[i + 1] - timestamps[i]
                for i in range(len(timestamps) - 1)]
        if not gaps:
            return float(self._latency_estimates.get(layer, 0))

        sorted_gaps = sorted(gaps)
        n = len(sorted_gaps)
        if n % 2 == 1:
            median = sorted_gaps[n // 2]
        else:
            median = (sorted_gaps[n // 2 - 1] + sorted_gaps[n // 2]) / 2.0

        return round(median, 4)

    # ------------------------------------------------------------------
    # Public API — reset
    # ------------------------------------------------------------------

    def reset(self):
        """Clear all buffers and reset latency estimates to defaults."""
        self._buffers.clear()
        self._all_timestamps = {layer: [] for layer in _VALID_LAYERS}
        self._latency_estimates = dict(_DEFAULT_LATENCY_ESTIMATES)
        logger.info(
            "DecisionLatencyAlignmentSystem(%r) reset",
            self._instance_id,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _empty_alignment(self, current_timestamp: float) -> Dict[str, Any]:
        """Return a safe alignment dict when no data has been recorded."""
        return {
            "aligned_sdil": None,
            "aligned_csfr": None,
            "aligned_saal": None,
            "aligned_mrsrl": None,
            "max_lag": 0,
            "timestamps": {l: None for l in _VALID_LAYERS},
            "decision_timestamp": current_timestamp,
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("Decision Latency Alignment System — Self Test")
    print("=" * 60)

    _state = {"passed": True}

    def _check(cond, msg):
        if not cond:
            _state["passed"] = False
            print(f"  FAIL: {msg}")
        else:
            print(f"  PASS: {msg}")

    # ==================================================================
    # Scenario 1: Standard alignment at tick 10
    #
    #   SDIL  produces every tick   → 0, 1, 2, ..., 10
    #   CSRF  produces every 3 ticks → 0, 3, 6, 9
    #   SAAL  produces every 5 ticks → 0, 5, 10
    #   MRSRL produces every tick   → 0, 1, 2, ..., 10
    #
    #   At tick 10:
    #     SDIL  → tick 10, lag=0
    #     CSRF  → tick  9, lag=1  (9 is closest; 10 does not exist)
    #     SAAL  → tick 10, lag=0
    #     MRSRL → tick 10, lag=0
    #     max_lag = 1
    #     decision_timestamp = 9
    # ==================================================================
    print("\n--- SCE1: Standard alignment at tick 10 ---")
    dlas1 = DecisionLatencyAlignmentSystem("sce1")

    for t in range(11):
        dlas1.record_layer_output("EURUSD", "sdil", float(t),
                                   {"signal": 1 if t % 2 == 0 else -1})
        dlas1.record_layer_output("EURUSD", "mrsrl", float(t),
                                   {"signal": 1})

    for t in range(0, 11, 3):
        dlas1.record_layer_output("EURUSD", "csfr", float(t),
                                   {"signal": 1, "evidence": t})

    for t in range(0, 11, 5):
        dlas1.record_layer_output("EURUSD", "saal", float(t),
                                   {"signal": -1, "authority": "high"})

    align1 = dlas1.align_for_decision("EURUSD", 10.0)
    print(f"  SDIL  timestamp: {align1['timestamps']['sdil']}")
    print(f"  CSRF  timestamp: {align1['timestamps']['csfr']}")
    print(f"  SAAL  timestamp: {align1['timestamps']['saal']}")
    print(f"  MRSRL timestamp: {align1['timestamps']['mrsrl']}")
    print(f"  max_lag          = {align1['max_lag']}")
    print(f"  decision_timestamp = {align1['decision_timestamp']}")

    _check(align1["timestamps"]["sdil"] == 10.0,
           f"SDIL closest to 10 → 10, got {align1['timestamps']['sdil']}")
    _check(align1["timestamps"]["csfr"] == 9.0,
           f"CSRF closest to 10 → 9, got {align1['timestamps']['csfr']}")
    _check(align1["timestamps"]["saal"] == 10.0,
           f"SAAL closest to 10 → 10, got {align1['timestamps']['saal']}")
    _check(align1["timestamps"]["mrsrl"] == 10.0,
           f"MRSRL closest to 10 → 10, got {align1['timestamps']['mrsrl']}")
    _check(align1["max_lag"] == 1.0,
           f"max_lag = 1, got {align1['max_lag']}")
    _check(align1["decision_timestamp"] == 9.0,
           f"decision_timestamp = 9, got {align1['decision_timestamp']}")

    # ==================================================================
    # Scenario 2: Alignment at tick 6
    #
    #   SDIL  → tick 6, lag=0
    #   CSRF  → tick 6, lag=0
    #   SAAL  → tick 5, lag=1  (5 is closest; 10 is further)
    #   MRSRL → tick 6, lag=0
    #   max_lag = 1
    #   decision_timestamp = 5
    # ==================================================================
    print("\n--- SCE2: Alignment at tick 6 ---")
    align2 = dlas1.align_for_decision("EURUSD", 6.0)
    print(f"  SDIL  timestamp: {align2['timestamps']['sdil']}")
    print(f"  CSRF  timestamp: {align2['timestamps']['csfr']}")
    print(f"  SAAL  timestamp: {align2['timestamps']['saal']}")
    print(f"  MRSRL timestamp: {align2['timestamps']['mrsrl']}")
    print(f"  max_lag          = {align2['max_lag']}")
    print(f"  decision_timestamp = {align2['decision_timestamp']}")

    _check(align2["timestamps"]["sdil"] == 6.0,
           f"SDIL closest to 6 → 6, got {align2['timestamps']['sdil']}")
    _check(align2["timestamps"]["csfr"] == 6.0,
           f"CSRF closest to 6 → 6, got {align2['timestamps']['csfr']}")
    _check(align2["timestamps"]["saal"] == 5.0,
           f"SAAL closest to 6 → 5, got {align2['timestamps']['saal']}")
    _check(align2["timestamps"]["mrsrl"] == 6.0,
           f"MRSRL closest to 6 → 6, got {align2['timestamps']['mrsrl']}")
    _check(align2["max_lag"] == 1.0,
           f"max_lag = 1, got {align2['max_lag']}")
    _check(align2["decision_timestamp"] == 5.0,
           f"decision_timestamp = 5, got {align2['decision_timestamp']}")

    # ==================================================================
    # Scenario 3: Alignment at tick 11 (beyond recorded data)
    #
    #   SDIL  → tick 10, lag=1
    #   CSRF  → tick  9, lag=2
    #   SAAL  → tick 10, lag=1
    #   MRSRL → tick 10, lag=1
    #   max_lag = 2
    #   decision_timestamp = 9
    # ==================================================================
    print("\n--- SCE3: Alignment at tick 11 (beyond recorded) ---")
    align3 = dlas1.align_for_decision("EURUSD", 11.0)
    print(f"  SDIL  timestamp: {align3['timestamps']['sdil']}")
    print(f"  CSRF  timestamp: {align3['timestamps']['csfr']}")
    print(f"  SAAL  timestamp: {align3['timestamps']['saal']}")
    print(f"  MRSRL timestamp: {align3['timestamps']['mrsrl']}")
    print(f"  max_lag          = {align3['max_lag']}")
    print(f"  decision_timestamp = {align3['decision_timestamp']}")

    _check(align3["timestamps"]["sdil"] == 10.0,
           f"SDIL closest to 11 → 10, got {align3['timestamps']['sdil']}")
    _check(align3["timestamps"]["csfr"] == 9.0,
           f"CSRF closest to 11 → 9, got {align3['timestamps']['csfr']}")
    _check(align3["timestamps"]["saal"] == 10.0,
           f"SAAL closest to 11 → 10, got {align3['timestamps']['saal']}")
    _check(align3["timestamps"]["mrsrl"] == 10.0,
           f"MRSRL closest to 11 → 10, got {align3['timestamps']['mrsrl']}")
    _check(align3["max_lag"] == 2.0,
           f"max_lag = 2, got {align3['max_lag']}")
    _check(align3["decision_timestamp"] == 9.0,
           f"decision_timestamp = 9, got {align3['decision_timestamp']}")

    # ==================================================================
    # Scenario 4: No data for a symbol → all None
    # ==================================================================
    print("\n--- SCE4: No data for symbol ---")
    align4 = dlas1.align_for_decision("NONEXISTENT", 5.0)
    _check(align4["aligned_sdil"] is None,
           "No symbol → aligned_sdil is None")
    _check(align4["aligned_csfr"] is None,
           "No symbol → aligned_csfr is None")
    _check(align4["aligned_saal"] is None,
           "No symbol → aligned_saal is None")
    _check(align4["aligned_mrsrl"] is None,
           "No symbol → aligned_mrsrl is None")
    _check(align4["max_lag"] == 0,
           "No symbol → max_lag = 0")
    _check(align4["decision_timestamp"] == 5.0,
           "No symbol → decision_timestamp = current_timestamp")

    # ==================================================================
    # Scenario 5: Layer missing (only SDIL recorded)
    # ==================================================================
    print("\n--- SCE5: Only SDIL has data ---")
    dlas5 = DecisionLatencyAlignmentSystem("sce5")
    dlas5.record_layer_output("EURUSD", "sdil", 1.0, {"signal": 1})
    dlas5.record_layer_output("EURUSD", "sdil", 2.0, {"signal": -1})
    dlas5.record_layer_output("EURUSD", "sdil", 3.0, {"signal": 1})
    align5 = dlas5.align_for_decision("EURUSD", 2.5)
    _check(align5["aligned_sdil"] is not None,
           "SDIL has data → aligned_sdil is not None")
    _check(align5["aligned_csfr"] is None,
           "No CSRF data → aligned_csfr is None")
    _check(align5["aligned_saal"] is None,
           "No SAAL data → aligned_saal is None")
    _check(align5["aligned_mrsrl"] is None,
           "No MRSRL data → aligned_mrsrl is None")
    _check(align5["max_lag"] == 0.5,
           f"Only SDIL → max_lag = |2.5-2.0| = 0.5, got {align5['max_lag']}")

    # ==================================================================
    # Scenario 6: Latency report
    # ==================================================================
    print("\n--- SCE6: Latency report ---")
    report1 = dlas1.get_latency_report("EURUSD")
    print(f"  sdil  report: {report1.get('sdil', {})}")
    print(f"  csfr  report: {report1.get('csfr', {})}")
    print(f"  saal  report: {report1.get('saal', {})}")
    print(f"  mrsrl report: {report1.get('mrsrl', {})}")
    _check(report1["sdil"]["count"] == 11,
           f"SDIL count = 11, got {report1['sdil']['count']}")
    _check(report1["csfr"]["count"] == 4,
           f"CSRF count = 4, got {report1['csfr']['count']}")
    _check(report1["saal"]["count"] == 3,
           f"SAAL count = 3, got {report1['saal']['count']}")
    _check(report1["mrsrl"]["count"] == 11,
           f"MRSRL count = 11, got {report1['mrsrl']['count']}")
    _check(abs(report1["sdil"]["avg_lag"] - 1.0) < 0.01,
           f"SDIL avg gap ≈ 1.0, got {report1['sdil']['avg_lag']}")
    _check(abs(report1["csfr"]["avg_lag"] - 3.0) < 0.01,
           f"CSRF avg gap ≈ 3.0, got {report1['csfr']['avg_lag']}")
    _check(abs(report1["saal"]["avg_lag"] - 5.0) < 0.01,
           f"SAAL avg gap ≈ 5.0, got {report1['saal']['avg_lag']}")

    # ==================================================================
    # Scenario 7: Latency estimation
    # ==================================================================
    print("\n--- SCE7: estimate_latency() ---")
    est_sdil = dlas1.estimate_latency("sdil")
    est_csfr = dlas1.estimate_latency("csfr")
    est_saal = dlas1.estimate_latency("saal")
    est_mrsrl = dlas1.estimate_latency("mrsrl")
    print(f"  sdil  estimated: {est_sdil}")
    print(f"  csfr  estimated: {est_csfr}")
    print(f"  saal  estimated: {est_saal}")
    print(f"  mrsrl estimated: {est_mrsrl}")
    _check(abs(est_sdil - 1.0) < 0.01,
           f"SDIL estimated latency ≈ 1.0, got {est_sdil}")
    _check(abs(est_csfr - 3.0) < 0.01,
           f"CSRF estimated latency ≈ 3.0, got {est_csfr}")
    _check(abs(est_saal - 5.0) < 0.01,
           f"SAAL estimated latency ≈ 5.0, got {est_saal}")
    _check(abs(est_mrsrl - 1.0) < 0.01,
           f"MRSRL estimated latency ≈ 1.0, got {est_mrsrl}")

    # ==================================================================
    # Scenario 8: estimate_latency with insufficient data (falls back)
    # ==================================================================
    print("\n--- SCE8: estimate_latency fallback ---")
    dlas8 = DecisionLatencyAlignmentSystem("sce8")
    est8 = dlas8.estimate_latency("sdil")
    _check(est8 == 0.0,
           f"Fallback SDIL = 0.0, got {est8}")

    est8_csfr = dlas8.estimate_latency("csfr")
    _check(est8_csfr == 2.0,
           f"Fallback CSFR = 2.0, got {est8_csfr}")

    est8_saal = dlas8.estimate_latency("saal")
    _check(est8_saal == 5.0,
           f"Fallback SAAL = 5.0, got {est8_saal}")

    # ==================================================================
    # Scenario 9: Reset
    # ==================================================================
    print("\n--- SCE9: Reset ---")
    dlas1.reset()
    align_reset = dlas1.align_for_decision("EURUSD", 10.0)
    _check(align_reset["aligned_sdil"] is None,
           "After reset → aligned_sdil is None")
    _check(align_reset["max_lag"] == 0,
           "After reset → max_lag = 0")
    report_reset = dlas1.get_latency_report("EURUSD")
    _check(report_reset == {},
           "After reset → latency report is empty dict")

    # ==================================================================
    # Scenario 10: Singleton identity
    # ==================================================================
    print("\n--- SCE10: Singleton identity ---")
    a = DecisionLatencyAlignmentSystem("sce1")
    b = DecisionLatencyAlignmentSystem("sce1")
    c = DecisionLatencyAlignmentSystem("other")
    d = DecisionLatencyAlignmentSystem()
    e = DecisionLatencyAlignmentSystem("default")
    _check(a is b, "Same instance_id returns same object")
    _check(a is not c, "Different instance_id returns different object")
    _check(d is e, "Default singleton identity")

    # ==================================================================
    # Scenario 11: Buffer size enforcement
    # ==================================================================
    print("\n--- SCE11: Buffer size enforcement ---")
    dlas11 = DecisionLatencyAlignmentSystem("sce11")
    for t in range(50):
        dlas11.record_layer_output("EURUSD", "sdil", float(t), {"t": t})
    report11 = dlas11.get_latency_report("EURUSD")
    _check(report11["sdil"]["count"] == 20,
           f"Buffer trimmed to 20, got {report11['sdil']['count']}")
    _check(report11["sdil"]["max_lag"] == 1.0,
           "Buffer of 20 entries at 1-tick intervals → max gap = 1.0")

    # ==================================================================
    # Scenario 12: Unknown layer warning (graceful handling)
    # ==================================================================
    print("\n--- SCE12: Unknown layer ---")
    dlas12 = DecisionLatencyAlignmentSystem("sce12")
    dlas12.record_layer_output("EURUSD", "unknown_layer", 1.0, {})
    # Should not crash; no entry in buffer
    align12 = dlas12.align_for_decision("EURUSD", 1.0)
    _check(align12["aligned_sdil"] is None,
           "Unknown layer record → no effect on alignment")

    # ==================================================================
    # Scenario 13: Multiple symbols independent
    # ==================================================================
    print("\n--- SCE13: Multiple symbols independent ---")
    dlas13 = DecisionLatencyAlignmentSystem("sce13")
    dlas13.record_layer_output("EURUSD", "sdil", 5.0, {"sym": "EURUSD"})
    dlas13.record_layer_output("GBPUSD", "sdil", 3.0, {"sym": "GBPUSD"})
    align_eur = dlas13.align_for_decision("EURUSD", 5.0)
    align_gbp = dlas13.align_for_decision("GBPUSD", 5.0)
    _check(align_eur["aligned_sdil"]["sym"] == "EURUSD",
           "EURUSD buffer independent")
    _check(align_gbp["aligned_sdil"]["sym"] == "GBPUSD",
           "GBPUSD buffer independent")

    # ==================================================================
    # Scenario 14: Exact tie — two outputs equidistant (choose either)
    # ==================================================================
    print("\n--- SCE14: Equidistant tie-breaking ---")
    dlas14 = DecisionLatencyAlignmentSystem("sce14")
    dlas14.record_layer_output("EURUSD", "sdil", 0.0, {"v": "early"})
    dlas14.record_layer_output("EURUSD", "sdil", 10.0, {"v": "late"})
    # At timestamp=5, both 0 and 10 are distance 5 → min chooses 0 (first)
    align14 = dlas14.align_for_decision("EURUSD", 5.0)
    _check(align14["timestamps"]["sdil"] == 0.0,
           "Equidistant: min by insertion order → 0.0")

    # ==================================================================
    # Final result
    # ==================================================================
    print()
    print("=" * 60)
    if _state["passed"]:
        print("ALL SELF-TESTS PASSED ✓")
    else:
        print("SOME SELF-TESTS FAILED ✗")
    print("=" * 60)

    import sys
    sys.exit(0 if _state["passed"] else 1)
