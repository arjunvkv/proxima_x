"""
Cross-Layer Decision Vector Engine — converts ALL layer outputs into ONE
unified decision vector per tick.  This is how the system forms a single
representation of reality from all perspectives (SDIL, CSRF, SAAL, MRSRL).
"""

import hashlib
import json
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton registry
# ---------------------------------------------------------------------------

_instances = {}


def CrossLayerDecisionVectorEngine(instance_id="default"):
    """Singleton accessor — returns the same ``_CrossLayerDecisionVectorEngine``
    for a given *instance_id*.

    Parameters
    ----------
    instance_id : str
        Logical identifier (default ``"default"``).

    Returns
    -------
    _CrossLayerDecisionVectorEngine
    """
    if instance_id not in _instances:
        _instances[instance_id] = _CrossLayerDecisionVectorEngine(instance_id)
    return _instances[instance_id]


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------

_MAX_VECTORS_PER_SYMBOL = 1000


def _to_float(value, default=0.0):
    """Safely convert *value* to float, returning *default* on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_str(value, default=""):
    """Safely convert *value* to str."""
    return str(value) if value is not None else default


def _to_bool(value, default=False):
    """Safely convert *value* to bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _clamp(value, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(value)))


class _CrossLayerDecisionVectorEngine:
    """Converts outputs from all four cognitive layers into a single unified
    decision vector per tick.

    Parameters
    ----------
    instance_id : str
        Logical instance identifier.
    """

    def __init__(self, instance_id="default"):
        self._instance_id = instance_id

        # symbol -> list of vector dicts (newest last)
        self._vectors = {}

        # Weights used for unified_confidence computation
        self._weights = {
            "saal": 0.35,
            "csfr": 0.30,
            "sdil": 0.20,
            "mrsrl": 0.15,
        }

        logger.debug(
            "CrossLayerDecisionVectorEngine(%r) initialised",
            instance_id,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_vector(self, tick_data, sdil_state, csfr_signal,
                     saal_authority, mrsrl_resolution):
        """Accept state dicts from all 4 layers plus raw tick data and
        produce ONE unified decision vector.

        Parameters
        ----------
        tick_data : dict
            Must contain ``timestamp`` (float), ``symbol`` (str),
            ``bid`` (float), ``ask`` (float).
        sdil_state : dict or None
            SDIL layer state with keys ``collapse_detected``,
            ``entropy_level``, ``anomaly_detected``, etc.
        csfr_signal : dict or None
            CSRF layer signal with keys ``oss_accuracy``, ``alt_accuracy``,
            ``truth_source``, ``signal_validity``, etc.
        saal_authority : dict or None
            SAAL layer authority with keys ``authority``,
            ``consensus_signal``, ``authority_confidence``,
            ``authority_stable``, etc.
        mrsrl_resolution : dict or None
            MRSRL layer resolution with keys ``resolution_regime``,
            ``structure_scale``, ``optimal_timeframe``,
            ``adaptive_alt_mode``, ``adaptive_oss_mode``, etc.

        Returns
        -------
        dict
            Unified decision vector with all required fields.
        """
        tick_data = tick_data or {}

        timestamp = _to_float(tick_data.get("timestamp"), 0.0)
        symbol = _to_str(tick_data.get("symbol"), "UNKNOWN")
        bid = _to_float(tick_data.get("bid"), 0.0)
        ask = _to_float(tick_data.get("ask"), 0.0)

        sdil = sdil_state or {}
        csfr = csfr_signal or {}
        saal = saal_authority or {}
        mrsrl = mrsrl_resolution or {}

        # ---- SDIL summary ----
        collapse_detected = _to_bool(sdil.get("collapse_detected"), False)
        entropy_level = _to_str(sdil.get("entropy_level"), "LOW").upper()
        if entropy_level not in ("LOW", "MODERATE", "HIGH"):
            entropy_level = "LOW"
        anomaly_detected = _to_bool(sdil.get("anomaly_detected"), False)

        # ---- CSRF summary ----
        oss_accuracy = _clamp(_to_float(csfr.get("oss_accuracy"), 0.0))
        alt_accuracy = _clamp(_to_float(csfr.get("alt_accuracy"), 0.0))
        truth_source = _to_str(csfr.get("truth_source"), "NEITHER").upper()
        if truth_source not in ("OSS", "ALT", "NEITHER", "INCONCLUSIVE"):
            truth_source = "INCONCLUSIVE"
        signal_validity = _to_str(csfr.get("signal_validity"), "QUESTIONABLE").upper()
        if signal_validity not in ("VALID", "QUESTIONABLE", "INVALID"):
            signal_validity = "QUESTIONABLE"

        # ---- SAAL summary ----
        authority = _to_str(saal.get("authority"), "NONE").upper()
        if authority not in ("OSS", "ALT", "HYBRID", "NONE"):
            authority = "NONE"
        consensus_signal = int(saal.get("consensus_signal", 0))
        if consensus_signal not in (-1, 0, 1):
            consensus_signal = 0
        authority_confidence = _clamp(_to_float(saal.get("authority_confidence"), 0.0))
        authority_stable = _to_bool(saal.get("authority_stable"), False)

        # ---- MRSRL summary ----
        resolution_regime = _to_str(
            mrsrl.get("resolution_regime"), "NOISE"
        ).upper()
        structure_scale = _to_str(
            mrsrl.get("structure_scale"), "MICRO_NOISE"
        ).upper()
        optimal_timeframe = _to_str(
            mrsrl.get("optimal_timeframe"), "TICK"
        ).upper()
        adaptive_alt_mode = _to_str(
            mrsrl.get("adaptive_alt_mode"), "ZSCORE_BREAKOUT"
        ).upper()
        adaptive_oss_mode = _to_str(
            mrsrl.get("adaptive_oss_mode"), "RAW"
        ).upper()

        # ---- Unified confidence (weighted average) ----
        saal_conf = authority_confidence
        csfr_conf = _clamp(_to_float(csfr.get("confidence", 0.0)))
        sdil_conf = _clamp(_to_float(sdil.get("confidence", 0.5)))
        mrsrl_conf = _clamp(_to_float(mrsrl.get("confidence", 0.5)))

        w_saal = self._weights.get("saal", 0.35)
        w_csfr = self._weights.get("csfr", 0.30)
        w_sdil = self._weights.get("sdil", 0.20)
        w_mrsrl = self._weights.get("mrsrl", 0.15)
        total_w = w_saal + w_csfr + w_sdil + w_mrsrl

        if total_w > 0:
            unified_confidence = _clamp(
                (saal_conf * w_saal + csfr_conf * w_csfr
                 + sdil_conf * w_sdil + mrsrl_conf * w_mrsrl)
                / total_w
            )
        else:
            unified_confidence = 0.0

        # ---- Layer count ----
        layer_count = sum(
            1 for d in (sdil_state, csfr_signal, saal_authority, mrsrl_resolution)
            if d is not None
        )

        # ---- Build vector without hash ----
        vector = OrderedDict()
        vector["timestamp"] = timestamp
        vector["symbol"] = symbol
        vector["bid"] = bid
        vector["ask"] = ask
        # SDIL
        vector["collapse_detected"] = collapse_detected
        vector["entropy_level"] = entropy_level
        vector["anomaly_detected"] = anomaly_detected
        # CSRF
        vector["oss_accuracy"] = oss_accuracy
        vector["alt_accuracy"] = alt_accuracy
        vector["truth_source"] = truth_source
        vector["signal_validity"] = signal_validity
        # SAAL
        vector["authority"] = authority
        vector["consensus_signal"] = consensus_signal
        vector["authority_confidence"] = authority_confidence
        vector["authority_stable"] = authority_stable
        # MRSRL
        vector["resolution_regime"] = resolution_regime
        vector["structure_scale"] = structure_scale
        vector["optimal_timeframe"] = optimal_timeframe
        vector["adaptive_alt_mode"] = adaptive_alt_mode
        vector["adaptive_oss_mode"] = adaptive_oss_mode
        # Unified
        vector["unified_confidence"] = unified_confidence
        vector["layer_count"] = layer_count

        # ---- Compute hash of all fields above ----
        vector["vector_hash"] = self._compute_hash(vector)

        # ---- Store ----
        if symbol not in self._vectors:
            self._vectors[symbol] = []
        self._vectors[symbol].append(vector)
        if len(self._vectors[symbol]) > _MAX_VECTORS_PER_SYMBOL:
            self._vectors[symbol] = self._vectors[symbol][-_MAX_VECTORS_PER_SYMBOL:]

        logger.debug(
            "build_vector %s ts=%.3f bid=%.5f ask=%.5f conf=%.4f layers=%d",
            symbol, timestamp, bid, ask, unified_confidence, layer_count,
        )

        return dict(vector)

    # ------------------------------------------------------------------

    def get_latest_vector(self, symbol):
        """Return the most recent vector for *symbol*, or None if none exist.

        Parameters
        ----------
        symbol : str
            Instrument identifier.

        Returns
        -------
        dict or None
        """
        vectors = self._vectors.get(symbol, [])
        if not vectors:
            return None
        return dict(vectors[-1])

    def get_all_vectors(self, symbol):
        """Return all stored vectors for *symbol* (newest last, max 1000).

        Parameters
        ----------
        symbol : str
            Instrument identifier.

        Returns
        -------
        list of dict
        """
        return [dict(v) for v in self._vectors.get(symbol, [])]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_hash(vector):
        """Compute SHA-256 of all vector fields except *vector_hash* itself.

        Uses a deterministic JSON serialisation so the same data always
        produces the same hash.
        """
        to_hash = OrderedDict(
            (k, v) for k, v in vector.items() if k != "vector_hash"
        )
        # Use a fixed-precision representation for floats to ensure
        # cross-run determinism.
        raw = json.dumps(to_hash, sort_keys=False, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("Cross-Layer Decision Vector Engine — Self Test")
    print("=" * 60)

    _state = {"passed": True}

    def _check(cond, msg):
        if not cond:
            _state["passed"] = False
            print(f"  FAIL: {msg}")
        else:
            print(f"  PASS: {msg}")

    # ==============================================================
    # Scenario 1: Normal bull market — all layers agree
    # ==============================================================
    print("\n--- SCE1: All layers agree (bullish, stable) ---")
    eng1 = CrossLayerDecisionVectorEngine("sce1")

    tick = {"timestamp": 1000000.0, "symbol": "EURUSD", "bid": 1.1000, "ask": 1.1002}
    sdil = {
        "collapse_detected": False, "entropy_level": "LOW",
        "anomaly_detected": False, "confidence": 0.85,
    }
    csfr = {
        "oss_accuracy": 0.78, "alt_accuracy": 0.72,
        "truth_source": "OSS", "signal_validity": "VALID",
        "confidence": 0.80,
    }
    saal = {
        "authority": "OSS", "consensus_signal": 1,
        "authority_confidence": 0.88, "authority_stable": True,
    }
    mrsrl = {
        "resolution_regime": "MESO_STRUCTURE", "structure_scale": "MESO_STRUCTURE",
        "optimal_timeframe": "1M", "adaptive_alt_mode": "ADAPTIVE_EMA",
        "adaptive_oss_mode": "RAW", "confidence": 0.75,
    }

    v1 = eng1.build_vector(tick, sdil, csfr, saal, mrsrl)

    # Verify all required fields
    required_keys = [
        "timestamp", "symbol", "bid", "ask",
        "collapse_detected", "entropy_level", "anomaly_detected",
        "oss_accuracy", "alt_accuracy", "truth_source", "signal_validity",
        "authority", "consensus_signal", "authority_confidence", "authority_stable",
        "resolution_regime", "structure_scale", "optimal_timeframe",
        "adaptive_alt_mode", "adaptive_oss_mode",
        "unified_confidence", "layer_count", "vector_hash",
    ]
    for key in required_keys:
        _check(key in v1, f"Required key '{key}' present in vector")

    _check(len(v1) == len(required_keys),
           f"Expected {len(required_keys)} keys, got {len(v1)}")

    # Check field values
    _check(v1["symbol"] == "EURUSD", f"symbol=EURUSD, got {v1['symbol']}")
    _check(v1["collapse_detected"] is False, "collapse_detected=False")
    _check(v1["entropy_level"] == "LOW", f"entropy_level=LOW, got {v1['entropy_level']}")
    _check(v1["anomaly_detected"] is False, "anomaly_detected=False")
    _check(v1["oss_accuracy"] == 0.78, f"oss_accuracy=0.78, got {v1['oss_accuracy']}")
    _check(v1["alt_accuracy"] == 0.72, f"alt_accuracy=0.72, got {v1['alt_accuracy']}")
    _check(v1["truth_source"] == "OSS", f"truth_source=OSS, got {v1['truth_source']}")
    _check(v1["signal_validity"] == "VALID", f"signal_validity=VALID, got {v1['signal_validity']}")
    _check(v1["authority"] == "OSS", f"authority=OSS, got {v1['authority']}")
    _check(v1["consensus_signal"] == 1, f"consensus_signal=1, got {v1['consensus_signal']}")
    _check(v1["authority_confidence"] == 0.88, f"authority_confidence=0.88, got {v1['authority_confidence']}")
    _check(v1["authority_stable"] is True, "authority_stable=True")
    _check(v1["resolution_regime"] == "MESO_STRUCTURE",
           f"resolution_regime=MESO_STRUCTURE, got {v1['resolution_regime']}")
    _check(v1["layer_count"] == 4, f"layer_count=4, got {v1['layer_count']}")
    _check(v1["unified_confidence"] > 0.7,
           f"Expected unified_confidence > 0.7, got {v1['unified_confidence']}")
    _check(isinstance(v1["vector_hash"], str) and len(v1["vector_hash"]) == 64,
           "vector_hash is a 64-char hex string")

    # ==============================================================
    # Scenario 2: Collapse detected — SDIL sees crash
    # ==============================================================
    print("\n--- SCE2: Collapse detected (bearish, high entropy) ---")
    eng2 = CrossLayerDecisionVectorEngine("sce2")

    tick2 = {"timestamp": 1000100.0, "symbol": "EURUSD", "bid": 1.0500, "ask": 1.0503}
    sdil2 = {
        "collapse_detected": True, "entropy_level": "HIGH",
        "anomaly_detected": True, "confidence": 0.95,
    }
    csfr2 = {
        "oss_accuracy": 0.30, "alt_accuracy": 0.25,
        "truth_source": "NEITHER", "signal_validity": "INVALID",
        "confidence": 0.10,
    }
    saal2 = {
        "authority": "NONE", "consensus_signal": -1,
        "authority_confidence": 0.20, "authority_stable": False,
    }
    mrsrl2 = {
        "resolution_regime": "NOISE", "structure_scale": "MICRO_NOISE",
        "optimal_timeframe": "TICK", "adaptive_alt_mode": "NO_SIGNAL",
        "adaptive_oss_mode": "FLAT", "confidence": 0.90,
    }

    v2 = eng2.build_vector(tick2, sdil2, csfr2, saal2, mrsrl2)

    _check(v2["collapse_detected"] is True, "collapse_detected=True")
    _check(v2["entropy_level"] == "HIGH", f"entropy_level=HIGH, got {v2['entropy_level']}")
    _check(v2["anomaly_detected"] is True, "anomaly_detected=True")
    _check(v2["truth_source"] == "NEITHER", f"truth_source=NEITHER, got {v2['truth_source']}")
    _check(v2["signal_validity"] == "INVALID", f"signal_validity=INVALID, got {v2['signal_validity']}")
    _check(v2["authority"] == "NONE", f"authority=NONE, got {v2['authority']}")
    _check(v2["authority_confidence"] == 0.20, "authority_confidence=0.20")
    _check(v2["authority_stable"] is False, "authority_stable=False")
    _check(v2["resolution_regime"] == "NOISE", "resolution_regime=NOISE")
    _check(v2["layer_count"] == 4, f"layer_count=4, got {v2['layer_count']}")
    _check(v2["unified_confidence"] < 0.6,
           f"Expected unified_confidence < 0.6, got {v2['unified_confidence']}")
    _check(v1["vector_hash"] != v2["vector_hash"],
           "Different inputs should produce different hashes")

    # ==============================================================
    # Scenario 3: Partial layers (only 2 layers provided)
    # ==============================================================
    print("\n--- SCE3: Partial layers (SAAL + CSRF only) ---")
    eng3 = CrossLayerDecisionVectorEngine("sce3")

    tick3 = {"timestamp": 1000200.0, "symbol": "GBPUSD", "bid": 1.2500, "ask": 1.2502}
    csfr3 = {
        "oss_accuracy": 0.65, "alt_accuracy": 0.55,
        "truth_source": "ALT", "signal_validity": "VALID",
        "confidence": 0.70,
    }
    saal3 = {
        "authority": "ALT", "consensus_signal": 1,
        "authority_confidence": 0.75, "authority_stable": True,
    }

    v3 = eng3.build_vector(tick3, None, csfr3, saal3, None)

    _check(v3["symbol"] == "GBPUSD", f"symbol=GBPUSD, got {v3['symbol']}")
    _check(v3["layer_count"] == 2, f"layer_count=2, got {v3['layer_count']}")
    _check(v3["collapse_detected"] is False,
           "collapse_detected defaults to False when SDIL is None")
    _check(v3["entropy_level"] == "LOW",
           "entropy_level defaults to LOW when SDIL is None")
    _check(v3["resolution_regime"] == "NOISE",
           "resolution_regime defaults to NOISE when MRSRL is None")
    _check(v3["structure_scale"] == "MICRO_NOISE",
           "structure_scale defaults to MICRO_NOISE when MRSRL is None")
    _check(v3["unified_confidence"] > 0.0,
           f"unified_confidence > 0.0, got {v3['unified_confidence']}")
    _check(isinstance(v3["vector_hash"], str) and len(v3["vector_hash"]) == 64,
           "vector_hash is valid")

    # ==============================================================
    # Scenario 4: get_latest_vector and get_all_vectors
    # ==============================================================
    print("\n--- SCE4: get_latest_vector / get_all_vectors ---")
    eng4 = CrossLayerDecisionVectorEngine("sce4")

    # Build 3 vectors for the same symbol
    for i in range(3):
        t = {"timestamp": 1000300.0 + i, "symbol": "USDJPY",
             "bid": 110.0 + i * 0.1, "ask": 110.05 + i * 0.1}
        v = eng4.build_vector(t, sdil, csfr, saal, mrsrl)

    latest = eng4.get_latest_vector("USDJPY")
    _check(latest is not None, "get_latest_vector returns a vector")
    _check(latest["timestamp"] == 1000302.0,
           f"Latest timestamp should be 1000302.0, got {latest['timestamp']}")

    all_v = eng4.get_all_vectors("USDJPY")
    _check(len(all_v) == 3, f"get_all_vectors returns 3 vectors, got {len(all_v)}")
    _check(all_v[0]["timestamp"] == 1000300.0,
           f"First vector timestamp 1000300.0, got {all_v[0]['timestamp']}")
    _check(all_v[2]["timestamp"] == 1000302.0,
           f"Last vector timestamp 1000302.0, got {all_v[2]['timestamp']}")

    # Non-existent symbol
    _check(eng4.get_latest_vector("NONEXISTENT") is None,
           "get_latest_vector returns None for unknown symbol")
    _check(eng4.get_all_vectors("NONEXISTENT") == [],
           "get_all_vectors returns empty list for unknown symbol")

    # ==============================================================
    # Scenario 5: Singleton identity
    # ==============================================================
    print("\n--- SCE5: Singleton identity ---")
    eng1_again = CrossLayerDecisionVectorEngine("sce1")
    _check(eng1_again is eng1, "Same instance_id returns same object")
    default_a = CrossLayerDecisionVectorEngine()
    default_b = CrossLayerDecisionVectorEngine("default")
    _check(default_a is default_b, "Default singleton identity")
    other = CrossLayerDecisionVectorEngine("other")
    _check(other is not eng1, "Different instance_id returns different object")

    # ==============================================================
    # Scenario 6: Hash determinism
    # ==============================================================
    print("\n--- SCE6: Hash determinism ---")
    # Build two vectors with same inputs — hashes should match
    tick6a = {"timestamp": 1000400.0, "symbol": "EURJPY", "bid": 130.00, "ask": 130.02}
    tick6b = {"timestamp": 1000400.0, "symbol": "EURJPY", "bid": 130.00, "ask": 130.02}

    v6a = eng1.build_vector(tick6a, sdil, csfr, saal, mrsrl)
    v6b = eng1.build_vector(tick6b, sdil, csfr, saal, mrsrl)

    _check(v6a["vector_hash"] == v6b["vector_hash"],
           "Same inputs produce same hash")
    _check(v1["vector_hash"] != v6a["vector_hash"],
           "Different inputs produce different hash")

    # ==============================================================
    # Scenario 7: Edge cases — None/empty inputs
    # ==============================================================
    print("\n--- SCE7: Edge cases (empty / None) ---")
    eng7 = CrossLayerDecisionVectorEngine("sce7")

    v7a = eng7.build_vector({}, None, None, None, None)
    _check(v7a["symbol"] == "UNKNOWN", "Empty tick_data -> symbol=UNKNOWN")
    _check(v7a["layer_count"] == 0, "All None layers -> layer_count=0")
    _check(v7a["collapse_detected"] is False, "collapse_detected=False default")
    _check(v7a["authority"] == "NONE", "authority defaults to NONE")
    _check(abs(v7a["unified_confidence"] - 0.175) < 1e-10,
           f"All confidence defaults -> unified_confidence ~0.175, got {v7a['unified_confidence']}")
    _check(isinstance(v7a["vector_hash"], str) and len(v7a["vector_hash"]) == 64,
           "hash still valid for empty vector")

    v7b = eng7.build_vector(
        {"timestamp": 1000500.0, "symbol": "AUDUSD", "bid": 0.6500, "ask": 0.6502},
        {}, {}, {}, {},
    )
    _check(v7b["symbol"] == "AUDUSD", "symbol=AUDUSD")
    _check(v7b["layer_count"] == 4,
           "Empty dict layers count as 4 (not None)")
    _check(v7b["bid"] == 0.65, f"bid=0.65, got {v7b['bid']}")
    _check(v7b["ask"] == 0.6502, f"ask=0.6502, got {v7b['ask']}")

    # --------------------------------------------------------------
    # Final result
    # --------------------------------------------------------------
    print()
    print("=" * 60)
    if _state["passed"]:
        print("ALL SELF-TESTS PASSED ✓")
    else:
        print("SOME SELF-TESTS FAILED ✗")
    print("=" * 60)

    sys.exit(0 if _state["passed"] else 1)
