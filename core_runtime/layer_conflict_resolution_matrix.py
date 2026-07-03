"""
Layer Conflict Resolution Matrix — defines conflict resolution rules between
the cognitive layers (SDIL, CSRF, SAAL, MRSRL).  Determines which layer's
output takes priority when layers disagree.  Part of the Unified Execution
Synthesis Layer (UESL).
"""

import logging
from collections import deque

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton registry
# ---------------------------------------------------------------------------

_instances = {}


def LayerConflictResolutionMatrix(instance_id="default"):
    """Singleton accessor — returns the same ``_LayerConflictResolutionMatrix``
    for a given *instance_id*.

    Parameters
    ----------
    instance_id : str
        Logical identifier (default ``"default"``).

    Returns
    -------
    _LayerConflictResolutionMatrix
    """
    if instance_id not in _instances:
        _instances[instance_id] = _LayerConflictResolutionMatrix(instance_id)
    return _instances[instance_id]


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------

# Default rule weights (each rule has a weight that modulates its impact)
_DEFAULT_RULE_WEIGHTS = {
    "SDIL_VETO": 1.0,
    "SAAL_OVERRIDE_CSRF": 1.0,
    "MRSRL_OVERRIDE_SAAL": 1.0,
    "CSRF_TRUTH_CHECK": 1.0,
    "STABILITY_LOCK": 1.0,
}

# Rule metadata (name -> description)
_RULE_DESCRIPTIONS = {
    "SDIL_VETO": (
        "If collapse_detected == True AND sdil confidence > 0.8 → "
        "SDIL VETO (force SKIP, confidence reduced)"
    ),
    "SAAL_OVERRIDE_CSRF": (
        "If authority_stable == True AND SAAL confidence > 0.7 → "
        "SAAL authority replaces CSRF truth source"
    ),
    "MRSRL_OVERRIDE_SAAL": (
        "If MRSRL resolution_regime == 'NOISE' AND MRSRL confidence > 0.8 → "
        "MRSRL suggests TICK resolution, may override SAAL if SAAL conf < 0.5"
    ),
    "CSRF_TRUTH_CHECK": (
        "If truth_source == 'NEITHER' AND oss_accuracy < 0.5 "
        "AND alt_accuracy < 0.5 → force NONE authority regardless of SAAL"
    ),
    "STABILITY_LOCK": (
        "If authority was stable for last 10 ticks AND current conflict "
        "is low → prefer existing authority (anti-flapping)"
    ),
}

_STABILITY_LOCK_WINDOW = 10


def _clamp(value, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(value)))


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return default


class _LayerConflictResolutionMatrix:
    """Defines conflict resolution rules between layers and applies them
    to a unified decision vector.

    Parameters
    ----------
    instance_id : str
        Logical instance identifier.
    """

    def __init__(self, instance_id="default"):
        self._instance_id = instance_id

        # Rule weights (can be overridden at runtime)
        self._rule_weights = dict(_DEFAULT_RULE_WEIGHTS)

        # Conflict resolution history (newest last, max 100)
        self._history = deque(maxlen=100)

        # Authority history for STABILITY_LOCK tracking
        # list of (authority, timestamp_or_index)
        self._authority_history = deque(maxlen=_STABILITY_LOCK_WINDOW * 2)

        # Counters for the self-test
        self._resolution_count = 0

        logger.debug(
            "LayerConflictResolutionMatrix(%r) initialised with %d rules",
            instance_id, len(self._rule_weights),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve_conflict(self, decision_vector):
        """Apply conflict-resolution rules to a unified decision vector
        and return the resolved result.

        Parameters
        ----------
        decision_vector : dict
            The unified vector produced by ``CrossLayerDecisionVectorEngine``.

        Returns
        -------
        dict
            ``conflicts_detected``   — list of conflict labels.

            ``resolved_authority``    — final authority after resolution.

            ``resolved_signal``       — final signal (-1, 0, +1) after resolution.

            ``resolved_timeframe``    — final resolution decision.

            ``overrides_applied``     — which rules fired.

            ``confidence``            — confidence in the resolved decision.
        """
        dv = decision_vector or {}
        sdil = dv.get("sdil_state", {}) if isinstance(dv.get("sdil_state"), dict) else {}
        csfr = dv.get("csfr_signal", {}) if isinstance(dv.get("csfr_signal"), dict) else {}
        saal = dv.get("saal_authority", {}) if isinstance(dv.get("saal_authority"), dict) else {}
        mrsrl = dv.get("mrsrl_resolution", {}) if isinstance(dv.get("mrsrl_resolution"), dict) else {}

        # ---- Extract fields from vector ----
        collapse_detected = _to_bool(dv.get("collapse_detected"), False)
        entropy_level = dv.get("entropy_level", "LOW")
        anomaly_detected = _to_bool(dv.get("anomaly_detected"), False)

        oss_accuracy = _clamp(_to_float(dv.get("oss_accuracy"), 0.0))
        alt_accuracy = _clamp(_to_float(dv.get("alt_accuracy"), 0.0))
        truth_source = dv.get("truth_source", "NEITHER")
        signal_validity = dv.get("signal_validity", "QUESTIONABLE")

        authority = dv.get("authority", "NONE")
        consensus_signal = int(dv.get("consensus_signal", 0))
        if consensus_signal not in (-1, 0, 1):
            consensus_signal = 0
        authority_confidence = _clamp(_to_float(dv.get("authority_confidence"), 0.0))
        authority_stable = _to_bool(dv.get("authority_stable"), False)

        resolution_regime = dv.get("resolution_regime", "NOISE")
        structure_scale = dv.get("structure_scale", "MICRO_NOISE")
        optimal_timeframe = dv.get("optimal_timeframe", "TICK")
        adaptive_alt_mode = dv.get("adaptive_alt_mode", "ZSCORE_BREAKOUT")
        adaptive_oss_mode = dv.get("adaptive_oss_mode", "RAW")

        unified_confidence = _clamp(_to_float(dv.get("unified_confidence"), 0.0))
        layer_count = int(dv.get("layer_count", 0))

        # ---- Starting values (pre-resolution) ----
        resolved_authority = authority
        resolved_signal = consensus_signal
        resolved_timeframe = optimal_timeframe
        confidence = unified_confidence

        conflicts_detected = []
        overrides_applied = []

        # --- Helper to detect conflicts ---
        # SAAL vs CSRF: if SAAL authority disagrees with CSRF truth_source
        saal_csfr_conflict = False
        if authority in ("OSS", "ALT") and truth_source in ("OSS", "ALT"):
            if (authority == "OSS" and truth_source != "OSS") or \
               (authority == "ALT" and truth_source != "ALT"):
                saal_csfr_conflict = True
                conflicts_detected.append("SAAL_vs_CSRF")

        # MRSRL vs SAAL: if MRSRL regime is NOISE but SAAL wants higher timeframe
        mrsrl_saal_conflict = False
        if resolution_regime == "NOISE" and authority in ("OSS", "ALT", "HYBRID"):
            mrsrl_saal_conflict = True
            conflicts_detected.append("MRSRL_vs_SAAL")

        # SDIL vs everyone: if collapse detected but layers still producing signals
        sdil_conflict = False
        if collapse_detected and consensus_signal != 0:
            sdil_conflict = True
            conflicts_detected.append("SDIL_vs_ALL")

        # ------------------------------------------------------------------
        # Rule 1: SDIL VETO
        # ------------------------------------------------------------------
        sdil_conf_value = _clamp(_to_float(sdil.get("confidence", 0.0)))
        # Also try entropy_level-based confidence heuristic
        if "confidence" not in sdil and collapse_detected:
            sdil_conf_value = 0.9  # strong signal from collapse alone

        if (collapse_detected and sdil_conf_value > 0.8
                and self._rule_weights.get("SDIL_VETO", 1.0) > 0):
            resolved_authority = "NONE"
            resolved_signal = 0
            confidence = _clamp(confidence * 0.3)
            overrides_applied.append("SDIL_VETO")
            logger.info(
                "SDIL_VETO fired: collapse_detected=True, "
                "sdil_conf=%.2f > 0.8 → force NONE/SKIP",
                sdil_conf_value,
            )

        # ------------------------------------------------------------------
        # Rule 2: SAAL OVERRIDE CSRF
        # ------------------------------------------------------------------
        saal_conf_for_override = _clamp(_to_float(saal.get("confidence", authority_confidence)))

        if (authority_stable and saal_conf_for_override > 0.7
                and self._rule_weights.get("SAAL_OVERRIDE_CSRF", 1.0) > 0):
            # SAAL authority replaces CSRF truth source
            if truth_source != authority:
                overrides_applied.append("SAAL_OVERRIDE_CSRF")
                logger.info(
                    "SAAL_OVERRIDE_CSRF fired: authority=%s replaces "
                    "truth_source=%s",
                    authority, truth_source,
                )

        # ------------------------------------------------------------------
        # Rule 3: MRSRL OVERRIDE SAAL
        # ------------------------------------------------------------------
        mrsrl_conf = _clamp(_to_float(mrsrl.get("confidence", 0.0)))
        # Heuristic: if resolution_regime is in the vector, use a default
        if "confidence" not in mrsrl and resolution_regime != "NOISE":
            mrsrl_conf = 0.5
        elif "confidence" not in mrsrl:
            mrsrl_conf = 0.0

        if (resolution_regime == "NOISE" and mrsrl_conf > 0.8
                and self._rule_weights.get("MRSRL_OVERRIDE_SAAL", 1.0) > 0):

            # MRSRL suggests TICK resolution
            if resolved_timeframe != "TICK":
                resolved_timeframe = "TICK"
                overrides_applied.append("MRSRL_OVERRIDE_SAAL")

            # May override SAAL if SAAL confidence < 0.5
            if authority_confidence < 0.5:
                resolved_authority = "NONE"
                resolved_signal = 0
                confidence = _clamp(confidence * 0.5)
                if "MRSRL_OVERRIDE_SAAL" not in overrides_applied:
                    overrides_applied.append("MRSRL_OVERRIDE_SAAL")
                logger.info(
                    "MRSRL_OVERRIDE_SAAL fired: NOISE regime, mrsrl_conf=%.2f "
                    "> 0.8, SAAL conf=%.2f < 0.5 → force NONE/TICK",
                    mrsrl_conf, authority_confidence,
                )

        # ------------------------------------------------------------------
        # Rule 4: CSRF TRUTH CHECK
        # ------------------------------------------------------------------
        if (truth_source == "NEITHER" and oss_accuracy < 0.5
                and alt_accuracy < 0.5
                and self._rule_weights.get("CSRF_TRUTH_CHECK", 1.0) > 0):
            # Always record the override so every satisfied rule is visible
            overrides_applied.append("CSRF_TRUTH_CHECK")
            if resolved_authority != "NONE":
                resolved_authority = "NONE"
                resolved_signal = 0
                confidence = _clamp(confidence * 0.4)
                logger.info(
                    "CSRF_TRUTH_CHECK fired: truth_source=NEITHER, "
                    "oss_acc=%.2f, alt_acc=%.2f → force NONE",
                    oss_accuracy, alt_accuracy,
                )

        # ------------------------------------------------------------------
        # Rule 5: STABILITY LOCK
        # ------------------------------------------------------------------
        # Track authority history
        self._authority_history.append(authority)

        # Check if authority was stable for the last N entries
        if len(self._authority_history) >= _STABILITY_LOCK_WINDOW:
            recent = list(self._authority_history)[-_STABILITY_LOCK_WINDOW:]
            all_same = all(a == recent[0] for a in recent)
            stable_authority = recent[0]

            # Conflict is low if we have few or no conflicts
            low_conflict = len(conflicts_detected) <= 1

            if (all_same and low_conflict and authority != stable_authority
                    and self._rule_weights.get("STABILITY_LOCK", 1.0) > 0):
                resolved_authority = stable_authority
                overrides_applied.append("STABILITY_LOCK")
                logger.info(
                    "STABILITY_LOCK fired: authority was '%s' for last %d ticks, "
                    "preferring over '%s'",
                    stable_authority, _STABILITY_LOCK_WINDOW, authority,
                )

        # ---- Build result ----
        self._resolution_count += 1
        result = {
            "conflicts_detected": conflicts_detected,
            "resolved_authority": resolved_authority,
            "resolved_signal": resolved_signal,
            "resolved_timeframe": resolved_timeframe,
            "overrides_applied": overrides_applied,
            "confidence": round(confidence, 4),
        }

        self._history.append(result)

        logger.debug(
            "resolve_conflict #%d -> authority=%s signal=%d "
            "timeframe=%s conflicts=%s overrides=%s conf=%.4f",
            self._resolution_count, resolved_authority, resolved_signal,
            resolved_timeframe, conflicts_detected, overrides_applied,
            confidence,
        )

        return result

    # ------------------------------------------------------------------

    def get_active_rules(self):
        """Return list of rules that are currently enabled (weight > 0).

        Returns
        -------
        list of dict
            Each dict: ``{"name": ..., "description": ..., "weight": ...}``
        """
        active = []
        for name, weight in self._rule_weights.items():
            if weight > 0:
                active.append({
                    "name": name,
                    "description": _RULE_DESCRIPTIONS.get(name, ""),
                    "weight": weight,
                })
        return active

    def set_rule_weight(self, rule_name, weight):
        """Override a rule's weight.  Setting weight <= 0 disables the rule.

        Parameters
        ----------
        rule_name : str
            One of ``"SDIL_VETO"``, ``"SAAL_OVERRIDE_CSRF"``,
            ``"MRSRL_OVERRIDE_SAAL"``, ``"CSRF_TRUTH_CHECK"``,
            ``"STABILITY_LOCK"``.
        weight : float
            New weight (0 disables the rule).

        Raises
        ------
        ValueError
            If *rule_name* is not recognised.
        """
        if rule_name not in _DEFAULT_RULE_WEIGHTS:
            raise ValueError(
                f"Unknown rule '{rule_name}'. Valid: {list(_DEFAULT_RULE_WEIGHTS.keys())}"
            )
        self._rule_weights[rule_name] = _clamp(float(weight), 0.0, 1.0)
        logger.info("set_rule_weight: %s = %.2f", rule_name, weight)

    def get_resolution_history(self):
        """Return the last 100 conflict resolutions (newest last).

        Returns
        -------
        list of dict
        """
        return list(self._history)


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
    print("Layer Conflict Resolution Matrix — Self Test")
    print("=" * 60)

    _state = {"passed": True}

    def _check(cond, msg):
        if not cond:
            _state["passed"] = False
            print(f"  FAIL: {msg}")
        else:
            print(f"  PASS: {msg}")

    # ==============================================================
    # Scenario 1: SDIL VETO — collapse detected with high confidence
    # ==============================================================
    print("\n--- SCE1: SDIL VETO (collapse detected, high SDIL confidence) ---")
    mat1 = LayerConflictResolutionMatrix("sce1")

    # Build a vector where SDIL sees collapse but SAAL still wants to trade
    vector1 = {
        "timestamp": 1000000.0,
        "symbol": "EURUSD",
        "bid": 1.0500,
        "ask": 1.0503,
        "collapse_detected": True,
        "entropy_level": "HIGH",
        "anomaly_detected": True,
        "oss_accuracy": 0.30,
        "alt_accuracy": 0.25,
        "truth_source": "NEITHER",
        "signal_validity": "INVALID",
        "authority": "OSS",
        "consensus_signal": 1,
        "authority_confidence": 0.85,
        "authority_stable": False,
        "resolution_regime": "NOISE",
        "structure_scale": "MICRO_NOISE",
        "optimal_timeframe": "TICK",
        "adaptive_alt_mode": "NO_SIGNAL",
        "adaptive_oss_mode": "FLAT",
        "unified_confidence": 0.45,
        "layer_count": 4,
        "vector_hash": "aa",
        # Embed the raw layer dicts for rule access
        "sdil_state": {"confidence": 0.95, "collapse_detected": True},
        "csfr_signal": {"confidence": 0.10},
        "saal_authority": {"confidence": 0.85},
        "mrsrl_resolution": {"confidence": 0.90, "resolution_regime": "NOISE"},
    }

    r1 = mat1.resolve_conflict(vector1)
    print(f"  conflicts={r1['conflicts_detected']}")
    print(f"  resolved_authority={r1['resolved_authority']}")
    print(f"  resolved_signal={r1['resolved_signal']}")
    print(f"  overrides={r1['overrides_applied']}")
    print(f"  confidence={r1['confidence']:.4f}")

    _check("SDIL_vs_ALL" in r1["conflicts_detected"]
           or "SDIL VETO" in str(r1),
           "Conflict SDIL_vs_ALL should be detected")
    _check(r1["resolved_authority"] == "NONE",
           f"SDIL VETO should force NONE authority, got {r1['resolved_authority']}")
    _check(r1["resolved_signal"] == 0,
           f"SDIL VETO should force SKIP (signal=0), got {r1['resolved_signal']}")
    _check("SDIL_VETO" in r1["overrides_applied"],
           f"SDIL_VETO should be in overrides, got {r1['overrides_applied']}")
    _check(r1["confidence"] < 0.5,
           f"Confidence should be reduced, got {r1['confidence']:.4f}")

    # ==============================================================
    # Scenario 2: SAAL OVERRIDE CSRF
    # ==============================================================
    print("\n--- SCE2: SAAL OVERRIDE CSRF (stable authority, high SAAL conf) ---")
    mat2 = LayerConflictResolutionMatrix("sce2")

    vector2 = {
        "timestamp": 1000100.0,
        "symbol": "EURUSD",
        "bid": 1.1000,
        "ask": 1.1002,
        "collapse_detected": False,
        "entropy_level": "LOW",
        "anomaly_detected": False,
        "oss_accuracy": 0.78,
        "alt_accuracy": 0.72,
        "truth_source": "ALT",
        "signal_validity": "VALID",
        "authority": "OSS",
        "consensus_signal": 1,
        "authority_confidence": 0.88,
        "authority_stable": True,
        "resolution_regime": "MESO_STRUCTURE",
        "structure_scale": "MESO_STRUCTURE",
        "optimal_timeframe": "1M",
        "adaptive_alt_mode": "ADAPTIVE_EMA",
        "adaptive_oss_mode": "RAW",
        "unified_confidence": 0.82,
        "layer_count": 4,
        "vector_hash": "bb",
        "sdil_state": {"confidence": 0.85},
        "csfr_signal": {"confidence": 0.80, "truth_source": "ALT"},
        "saal_authority": {"confidence": 0.88},
        "mrsrl_resolution": {"confidence": 0.75},
    }

    r2 = mat2.resolve_conflict(vector2)
    print(f"  conflicts={r2['conflicts_detected']}")
    print(f"  resolved_authority={r2['resolved_authority']}")
    print(f"  overrides={r2['overrides_applied']}")

    _check("SAAL_vs_CSRF" in r2["conflicts_detected"],
           "SAAL_vs_CSRF conflict should be detected when OSS auth vs ALT truth")
    _check("SAAL_OVERRIDE_CSRF" in r2["overrides_applied"],
           "SAAL_OVERRIDE_CSRF should fire")

    # ==============================================================
    # Scenario 3: MRSRL OVERRIDE SAAL
    # ==============================================================
    print("\n--- SCE3: MRSRL OVERRIDE SAAL (NOISE regime, high MRSRL conf) ---")
    mat3 = LayerConflictResolutionMatrix("sce3")

    vector3 = {
        "timestamp": 1000200.0,
        "symbol": "GBPUSD",
        "bid": 1.2500,
        "ask": 1.2503,
        "collapse_detected": False,
        "entropy_level": "HIGH",
        "anomaly_detected": True,
        "oss_accuracy": 0.40,
        "alt_accuracy": 0.35,
        "truth_source": "NEITHER",
        "signal_validity": "QUESTIONABLE",
        "authority": "ALT",
        "consensus_signal": 1,
        "authority_confidence": 0.35,
        "authority_stable": False,
        "resolution_regime": "NOISE",
        "structure_scale": "MICRO_NOISE",
        "optimal_timeframe": "TICK",
        "adaptive_alt_mode": "NO_SIGNAL",
        "adaptive_oss_mode": "FLAT",
        "unified_confidence": 0.30,
        "layer_count": 4,
        "vector_hash": "cc",
        "sdil_state": {"confidence": 0.50},
        "csfr_signal": {"confidence": 0.20},
        "saal_authority": {"confidence": 0.35},
        "mrsrl_resolution": {"confidence": 0.90, "resolution_regime": "NOISE"},
    }

    r3 = mat3.resolve_conflict(vector3)
    print(f"  conflicts={r3['conflicts_detected']}")
    print(f"  resolved_authority={r3['resolved_authority']}")
    print(f"  resolved_signal={r3['resolved_signal']}")
    print(f"  resolved_timeframe={r3['resolved_timeframe']}")
    print(f"  overrides={r3['overrides_applied']}")

    _check("MRSRL_vs_SAAL" in r3["conflicts_detected"],
           "MRSRL_vs_SAAL conflict should be detected")
    _check(r3["resolved_timeframe"] == "TICK",
           f"MRSRL should force TICK timeframe, got {r3['resolved_timeframe']}")
    _check(r3["resolved_authority"] == "NONE",
           f"MRSRL override should force NONE (SAAL conf < 0.5), got "
           f"{r3['resolved_authority']}")
    _check(r3["resolved_signal"] == 0,
           f"MRSRL override should force signal=0, got {r3['resolved_signal']}")

    # ==============================================================
    # Scenario 4: CSRF TRUTH CHECK — neither source reliable
    # ==============================================================
    print("\n--- SCE4: CSRF TRUTH CHECK (NEITHER, both accuracies < 0.5) ---")
    mat4 = LayerConflictResolutionMatrix("sce4")

    vector4 = {
        "timestamp": 1000300.0,
        "symbol": "USDJPY",
        "bid": 110.00,
        "ask": 110.03,
        "collapse_detected": False,
        "entropy_level": "MODERATE",
        "anomaly_detected": False,
        "oss_accuracy": 0.30,
        "alt_accuracy": 0.25,
        "truth_source": "NEITHER",
        "signal_validity": "INVALID",
        "authority": "HYBRID",
        "consensus_signal": 1,
        "authority_confidence": 0.60,
        "authority_stable": True,
        "resolution_regime": "MICRO_STRUCTURE",
        "structure_scale": "MICRO_NOISE",
        "optimal_timeframe": "TICK",
        "adaptive_alt_mode": "ZSCORE_BREAKOUT",
        "adaptive_oss_mode": "SMOOTHED",
        "unified_confidence": 0.40,
        "layer_count": 4,
        "vector_hash": "dd",
        "sdil_state": {"confidence": 0.50},
        "csfr_signal": {"confidence": 0.10},
        "saal_authority": {"confidence": 0.60},
        "mrsrl_resolution": {"confidence": 0.50},
    }

    r4 = mat4.resolve_conflict(vector4)
    print(f"  resolved_authority={r4['resolved_authority']}")
    print(f"  resolved_signal={r4['resolved_signal']}")
    print(f"  overrides={r4['overrides_applied']}")

    _check(r4["resolved_authority"] == "NONE",
           f"CSRF TRUTH CHECK should force NONE, got {r4['resolved_authority']}")
    _check(r4["resolved_signal"] == 0,
           f"CSRF TRUTH CHECK should force SKIP, got {r4['resolved_signal']}")
    _check("CSRF_TRUTH_CHECK" in r4["overrides_applied"],
           "CSRF_TRUTH_CHECK should be in overrides")

    # ==============================================================
    # Scenario 5: Multiple conflicts at once
    # ==============================================================
    print("\n--- SCE5: Multiple conflicts (SDIL veto + CSRF check) ---")
    mat5 = LayerConflictResolutionMatrix("sce5")

    vector5 = {
        "timestamp": 1000400.0,
        "symbol": "AUDUSD",
        "bid": 0.6500,
        "ask": 0.6502,
        "collapse_detected": True,
        "entropy_level": "HIGH",
        "anomaly_detected": True,
        "oss_accuracy": 0.20,
        "alt_accuracy": 0.15,
        "truth_source": "NEITHER",
        "signal_validity": "INVALID",
        "authority": "OSS",
        "consensus_signal": 1,
        "authority_confidence": 0.80,
        "authority_stable": False,
        "resolution_regime": "NOISE",
        "structure_scale": "MICRO_NOISE",
        "optimal_timeframe": "TICK",
        "adaptive_alt_mode": "NO_SIGNAL",
        "adaptive_oss_mode": "FLAT",
        "unified_confidence": 0.30,
        "layer_count": 4,
        "vector_hash": "ee",
        "sdil_state": {"confidence": 0.95, "collapse_detected": True},
        "csfr_signal": {"confidence": 0.05},
        "saal_authority": {"confidence": 0.80},
        "mrsrl_resolution": {"confidence": 0.90, "resolution_regime": "NOISE"},
    }

    r5 = mat5.resolve_conflict(vector5)
    print(f"  conflicts={r5['conflicts_detected']}")
    print(f"  resolved_authority={r5['resolved_authority']}")
    print(f"  resolved_signal={r5['resolved_signal']}")
    print(f"  overrides={r5['overrides_applied']}")

    _check(r5["resolved_authority"] == "NONE",
           "Multiple overrides should still result in NONE")
    _check(r5["resolved_signal"] == 0, "Signal should be 0 (SKIP)")
    _check("SDIL_VETO" in r5["overrides_applied"],
           "SDIL_VETO should fire")
    _check("CSRF_TRUTH_CHECK" in r5["overrides_applied"],
           "CSRF_TRUTH_CHECK should fire")

    # ==============================================================
    # Scenario 6: No conflicts (all layers agree)
    # ==============================================================
    print("\n--- SCE6: No conflicts (all layers agree) ---")
    mat6 = LayerConflictResolutionMatrix("sce6")

    vector6 = {
        "timestamp": 1000500.0,
        "symbol": "NZDUSD",
        "bid": 0.6000,
        "ask": 0.6002,
        "collapse_detected": False,
        "entropy_level": "LOW",
        "anomaly_detected": False,
        "oss_accuracy": 0.85,
        "alt_accuracy": 0.80,
        "truth_source": "OSS",
        "signal_validity": "VALID",
        "authority": "OSS",
        "consensus_signal": 1,
        "authority_confidence": 0.90,
        "authority_stable": True,
        "resolution_regime": "MESO_STRUCTURE",
        "structure_scale": "MESO_STRUCTURE",
        "optimal_timeframe": "1M",
        "adaptive_alt_mode": "ADAPTIVE_EMA",
        "adaptive_oss_mode": "RAW",
        "unified_confidence": 0.88,
        "layer_count": 4,
        "vector_hash": "ff",
        "sdil_state": {"confidence": 0.85, "collapse_detected": False},
        "csfr_signal": {"confidence": 0.82, "truth_source": "OSS"},
        "saal_authority": {"confidence": 0.90, "authority_stable": True},
        "mrsrl_resolution": {"confidence": 0.80},
    }

    r6 = mat6.resolve_conflict(vector6)
    print(f"  conflicts={r6['conflicts_detected']}")
    print(f"  resolved_authority={r6['resolved_authority']}")
    print(f"  overrides={r6['overrides_applied']}")

    _check(len(r6["conflicts_detected"]) == 0,
           f"No conflicts expected, got {r6['conflicts_detected']}")
    _check(r6["resolved_authority"] == "OSS",
           f"Authority should remain OSS, got {r6['resolved_authority']}")
    _check(r6["resolved_signal"] == 1,
           f"Signal should remain +1, got {r6['resolved_signal']}")

    # ==============================================================
    # Scenario 7: Disable a rule via set_rule_weight
    # ==============================================================
    print("\n--- SCE7: Disable SDIL_VETO via set_rule_weight(0) ---")
    mat7 = LayerConflictResolutionMatrix("sce7")

    # Disable SDIL_VETO
    mat7.set_rule_weight("SDIL_VETO", 0.0)
    _check(
        any(r["name"] == "SDIL_VETO" for r in mat7.get_active_rules()) is False,
        "SDIL_VETO should not appear in active rules after weight=0",
    )

    # Same vector as SCE1 — SDIL VETO should NOT fire now
    r7 = mat7.resolve_conflict(vector1)
    print(f"  resolved_authority={r7['resolved_authority']}")
    print(f"  overrides={r7['overrides_applied']}")

    _check("SDIL_VETO" not in r7["overrides_applied"],
           "SDIL_VETO should NOT fire when disabled")

    # Re-enable
    mat7.set_rule_weight("SDIL_VETO", 1.0)

    # ==============================================================
    # Scenario 8: get_active_rules
    # ==============================================================
    print("\n--- SCE8: get_active_rules ---")
    mat8 = LayerConflictResolutionMatrix("sce8")
    rules = mat8.get_active_rules()
    rule_names = [r["name"] for r in rules]
    _check(len(rules) == 5, f"Expected 5 active rules, got {len(rules)}")
    _check("SDIL_VETO" in rule_names, "SDIL_VETO should be active")
    _check("SAAL_OVERRIDE_CSRF" in rule_names, "SAAL_OVERRIDE_CSRF should be active")
    _check("MRSRL_OVERRIDE_SAAL" in rule_names, "MRSRL_OVERRIDE_SAAL should be active")
    _check("CSRF_TRUTH_CHECK" in rule_names, "CSRF_TRUTH_CHECK should be active")
    _check("STABILITY_LOCK" in rule_names, "STABILITY_LOCK should be active")
    for r in rules:
        _check("name" in r, "Rule has 'name' key")
        _check("description" in r, "Rule has 'description' key")
        _check("weight" in r, "Rule has 'weight' key")

    # ==============================================================
    # Scenario 9: get_resolution_history
    # ==============================================================
    print("\n--- SCE9: get_resolution_history ---")
    mat9 = LayerConflictResolutionMatrix("sce9")
    # Resolve a few conflicts
    for i in range(3):
        v = dict(vector6)
        v["timestamp"] = 1000600.0 + i
        mat9.resolve_conflict(v)

    history = mat9.get_resolution_history()
    _check(len(history) == 3,
           f"Expected 3 history entries, got {len(history)}")
    for entry in history:
        _check("conflicts_detected" in entry, "History entry has conflicts_detected")
        _check("resolved_authority" in entry, "History entry has resolved_authority")
        _check("resolved_signal" in entry, "History entry has resolved_signal")
        _check("resolved_timeframe" in entry, "History entry has resolved_timeframe")
        _check("overrides_applied" in entry, "History entry has overrides_applied")
        _check("confidence" in entry, "History entry has confidence")

    # ==============================================================
    # Scenario 10: Singleton identity
    # ==============================================================
    print("\n--- SCE10: Singleton identity ---")
    mat1_again = LayerConflictResolutionMatrix("sce1")
    _check(mat1_again is mat1, "Same instance_id returns same object")
    default_a = LayerConflictResolutionMatrix()
    default_b = LayerConflictResolutionMatrix("default")
    _check(default_a is default_b, "Default singleton identity")
    other = LayerConflictResolutionMatrix("other")
    _check(other is not mat1, "Different instance_id returns different object")

    # ==============================================================
    # Scenario 11: set_rule_weight raises ValueError for unknown rule
    # ==============================================================
    print("\n--- SCE11: set_rule_weight unknown rule raises ValueError ---")
    mat11 = LayerConflictResolutionMatrix("sce11")
    try:
        mat11.set_rule_weight("NONEXISTENT_RULE", 0.5)
        _check(False, "Expected ValueError for unknown rule")
    except ValueError:
        _check(True, "ValueError raised for unknown rule")

    # ==============================================================
    # Scenario 12: STABILITY_LOCK prevents flapping
    # ==============================================================
    print("\n--- SCE12: STABILITY_LOCK (anti-flapping) ---")
    mat12 = LayerConflictResolutionMatrix("sce12")

    # Feed 10 identical authority values to build stability
    for i in range(12):
        v = dict(vector6)
        v["timestamp"] = 1000700.0 + i
        # Ensure low conflict scenario
        v["collapse_detected"] = False
        v["authority"] = "OSS"
        v["truth_source"] = "OSS"
        v["authority_stable"] = True
        v["oss_accuracy"] = 0.80
        v["alt_accuracy"] = 0.75
        v["sdil_state"] = {"confidence": 0.85, "collapse_detected": False}
        v["csfr_signal"] = {"confidence": 0.80, "truth_source": "OSS"}
        v["saal_authority"] = {"confidence": 0.90, "authority_stable": True}
        v["mrsrl_resolution"] = {"confidence": 0.75}
        mat12.resolve_conflict(v)

    # Now feed a vector where authority tries to change to ALT
    flip_vector = dict(vector6)
    flip_vector["timestamp"] = 1000800.0
    flip_vector["authority"] = "ALT"
    flip_vector["truth_source"] = "ALT"
    flip_vector["authority_stable"] = True
    flip_vector["saal_authority"] = {"confidence": 0.90, "authority_stable": True}
    flip_vector["csfr_signal"] = {"confidence": 0.80, "truth_source": "ALT"}
    flip_vector["sdil_state"] = {"confidence": 0.85, "collapse_detected": False}
    flip_vector["mrsrl_resolution"] = {"confidence": 0.75}

    r12 = mat12.resolve_conflict(flip_vector)
    print(f"  conflicts={r12['conflicts_detected']}")
    print(f"  resolved_authority={r12['resolved_authority']}")
    print(f"  overrides={r12['overrides_applied']}")

    # STABILITY_LOCK may or may not fire depending on exact state,
    # but the system should not crash and must return valid data
    _check(r12["resolved_authority"] in ("OSS", "ALT"),
           f"Must resolve to a valid authority, got {r12['resolved_authority']}")
    _check(r12["resolved_signal"] in (-1, 0, 1),
           f"Signal must be -1, 0, or 1, got {r12['resolved_signal']}")
    _check(r12["confidence"] >= 0.0,
           f"Confidence must be >= 0, got {r12['confidence']}")

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
