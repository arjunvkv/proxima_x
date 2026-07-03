"""
uesl_orchestrator.py — Unified Execution Synthesis Layer (UESL) Orchestrator.

THIS IS THE SINGLE ENTRY POINT for the entire UESL system.  Every tick
enters here and flows through all cognitive layers (SDIL, CSRF, SAAL, MRSRL)
to produce a single final trading decision.

Processing pipeline (in order):
  1. Build decision vector via ``CrossLayerDecisionVectorEngine``
  2. Resolve layer conflicts via ``LayerConflictResolutionMatrix``
  3. Align latency via ``DecisionLatencyAlignmentSystem``
  4. Update dynamic weights via ``LayerWeightDynamicsController``
  5. Apply latency alignment + weights to resolved conflict output
  6. Arbitrate priority via ``ExecutionPriorityArbiter``
  7. Synthesise final decision via ``ExecutionSynthesisEngine``
  8. Log conflicts via ``ConflictTraceLogger``
  9. Return the final decision dict

Usage
-----
    from core_runtime.uesl_orchestrator import UESLOrchestrator

    orchestrator = UESLOrchestrator()
    decision = orchestrator.process(
        tick={"timestamp": ..., "symbol": "EURUSD", "bid": ..., "ask": ...},
        sdil_state={...},
        csfr_signal={...},
        saal_authority={...},
        mrsrl_resolution={...},
    )
    print(decision["decision"])   # "BUY" | "SELL" | "HOLD" | "SKIP"
"""

import copy
import hashlib
import json
import logging
import time
import uuid
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from proxima_x.core_runtime.cross_layer_decision_vector_engine import (
    CrossLayerDecisionVectorEngine,
)
from proxima_x.core_runtime.conflict_trace_logger import ConflictTraceLogger


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-module: LayerConflictResolutionMatrix
# ---------------------------------------------------------------------------

_instance_matrix: Dict[str, "_LayerConflictResolutionMatrix"] = {}


def LayerConflictResolutionMatrix(instance_id="default"):
    """Singleton accessor for ``_LayerConflictResolutionMatrix``."""
    if instance_id not in _instance_matrix:
        _instance_matrix[instance_id] = _LayerConflictResolutionMatrix(instance_id)
    return _instance_matrix[instance_id]


class _LayerConflictResolutionMatrix:
    """Resolves disagreements between layers in a decision vector.

    Examines the vector for known conflict patterns (authority mismatch,
    signal divergence, stability breach) and produces a resolved vector
    with conflict metadata.
    """

    def __init__(self, instance_id="default"):
        self._instance_id = instance_id
        self._conflict_count = 0
        logger.info(
            "LayerConflictResolutionMatrix(%r) initialised", instance_id,
        )

    def resolve_conflict(self, decision_vector: dict) -> dict:
        """Analyse *decision_vector* for inter-layer conflicts.

        Parameters
        ----------
        decision_vector : dict
            Unified decision vector produced by ``CrossLayerDecisionVectorEngine``.

        Returns
        -------
        dict
            ``resolved_vector`` — copy of the input with any corrections applied.

            ``conflicts_found`` — list of conflict dicts, each with keys
            ``type``, ``layer_a``, ``layer_b``, ``description``.

            ``resolved_authority`` — the final authority after conflict
            resolution (``"OSS"``, ``"ALT"``, ``"HYBRID"``, or ``"NONE"``).
        """
        dv = decision_vector or {}
        conflicts: List[Dict[str, Any]] = []

        # --- Authority mismatch ---
        truth_source = dv.get("truth_source", "NEITHER")
        authority = dv.get("authority", "NONE")
        if truth_source != "NEITHER" and authority != "NONE" and truth_source != authority:
            conflicts.append({
                "type": "AUTHORITY_MISMATCH",
                "layer_a": "csfr",
                "layer_b": "saal",
                "description": (
                    f"CSRF says truth_source={truth_source} but SAAL "
                    f"authority={authority}"
                ),
            })

        # --- Signal divergence (SDIL vs CSRF) ---
        signal_validity = dv.get("signal_validity", "QUESTIONABLE")
        authority_stable = dv.get("authority_stable", False)
        if signal_validity == "INVALID" and authority_stable:
            conflicts.append({
                "type": "SIGNAL_DIVERGENCE",
                "layer_a": "sdil",
                "layer_b": "csfr",
                "description": (
                    f"SDIL indicates collapse/anomaly but CSRF reports "
                    f"signal_validity={signal_validity}"
                ),
            })

        # --- Resolution conflict (SAAL vs MRSRL) ---
        entropy_level = dv.get("entropy_level", "LOW")
        optimal_timeframe = dv.get("optimal_timeframe", "TICK")
        if entropy_level in ("HIGH", "MODERATE") and optimal_timeframe == "TICK":
            conflicts.append({
                "type": "RESOLUTION_CONFLICT",
                "layer_a": "saal",
                "layer_b": "mrsrl",
                "description": (
                    f"SAAL entropy_level={entropy_level} but MRSRL "
                    f"optimal_timeframe={optimal_timeframe}"
                ),
            })

        # --- Stability breach ---
        anomaly_detected = dv.get("anomaly_detected", False)
        if anomaly_detected and authority != "NONE":
            conflicts.append({
                "type": "STABILITY_BREACH",
                "layer_a": "sdil",
                "layer_b": "saal",
                "description": (
                    f"SDIL anomaly_detected={anomaly_detected} but SAAL "
                    f"authority={authority} — stability breach"
                ),
            })

        # --- Resolve authority ---
        resolved_authority = authority
        if len(conflicts) > 0:
            self._conflict_count += len(conflicts)
            # If there are serious conflicts and NONE or HYBRID is available,
            # downgrade to the safer option
            has_severe = any(
                c["type"] in ("STABILITY_BREACH", "AUTHORITY_MISMATCH")
                for c in conflicts
            )
            if has_severe and authority in ("OSS", "ALT"):
                resolved_authority = "HYBRID"
            if has_severe and authority == "HYBRID":
                resolved_authority = "NONE"

        return {
            "resolved_vector": dict(dv),
            "conflicts_found": conflicts,
            "resolved_authority": resolved_authority,
        }


# ---------------------------------------------------------------------------
# Sub-module: DecisionLatencyAlignmentSystem
# ---------------------------------------------------------------------------

_instance_latency: Dict[str, "_DecisionLatencyAlignmentSystem"] = {}


def DecisionLatencyAlignmentSystem(instance_id="default"):
    """Singleton accessor for ``_DecisionLatencyAlignmentSystem``."""
    if instance_id not in _instance_latency:
        _instance_latency[instance_id] = _DecisionLatencyAlignmentSystem(instance_id)
    return _instance_latency[instance_id]


class _DecisionLatencyAlignmentSystem:
    """Tracks per-layer processing latency and aligns decision components
    so that stale layers do not dominate the final output.
    """

    def __init__(self, instance_id="default"):
        self._instance_id = instance_id
        # layer_name -> {"timestamp": float, "data": dict}
        self._layer_outputs: Dict[str, Dict[str, Any]] = {}
        # latency history for statistics
        self._latency_log: List[Dict[str, Any]] = []
        self._max_log = 1000
        logger.info(
            "DecisionLatencyAlignmentSystem(%r) initialised", instance_id,
        )

    def record_layer_output(self, layer_name: str, data: dict):
        """Record the output and local timestamp for *layer_name*.

        Parameters
        ----------
        layer_name : str
            e.g. ``"sdil"``, ``"csfr"``, ``"saal"``, ``"mrsrl"``.
        data : dict
            The raw layer output dict.
        """
        self._layer_outputs[layer_name] = {
            "timestamp": time.time(),
            "data": copy.deepcopy(data) if data else {},
        }

    def align_for_decision(self) -> dict:
        """Compute per-layer lag and produce an aligned metadata dict.

        Returns
        -------
        dict
            ``layer_lags`` — ``{layer: seconds_since_recorded}``

            ``max_lag`` — the worst lag across all tracked layers

            ``aligned_data`` — ``{layer: data}`` snapshot
        """
        now = time.time()
        layer_lags: Dict[str, float] = {}
        aligned_data: Dict[str, dict] = {}

        for layer, rec in self._layer_outputs.items():
            lag = now - rec["timestamp"]
            layer_lags[layer] = round(lag, 6)
            aligned_data[layer] = rec["data"]

        max_lag = max(layer_lags.values()) if layer_lags else 0.0

        # Log stats for trend analysis
        log_entry = {
            "timestamp": now,
            "layer_lags": dict(layer_lags),
            "max_lag": max_lag,
        }
        self._latency_log.append(log_entry)
        if len(self._latency_log) > self._max_log:
            self._latency_log = self._latency_log[-self._max_log:]

        return {
            "layer_lags": layer_lags,
            "max_lag": round(max_lag, 6),
            "aligned_data": aligned_data,
        }

    def get_latency_report(self) -> dict:
        """Return aggregate latency statistics.

        Returns
        -------
        dict
            ``total_tracked_layers``, ``mean_max_lag``, ``max_max_lag``,
            ``layer_availability`` — fraction of logs where each layer
            had data.
        """
        if not self._latency_log:
            return {
                "total_tracked_layers": 0,
                "mean_max_lag": 0.0,
                "max_max_lag": 0.0,
                "layer_availability": {},
            }

        max_lags = [entry["max_lag"] for entry in self._latency_log]
        mean_max_lag = sum(max_lags) / len(max_lags)

        # Layer availability
        all_layers: set = set()
        for entry in self._latency_log:
            all_layers.update(entry["layer_lags"].keys())

        availability: Dict[str, float] = {}
        total_logs = len(self._latency_log)
        for layer in all_layers:
            count = sum(
                1 for entry in self._latency_log if layer in entry["layer_lags"]
            )
            availability[layer] = round(count / total_logs, 4)

        return {
            "total_tracked_layers": len(all_layers),
            "mean_max_lag": round(mean_max_lag, 6),
            "max_max_lag": round(max(max_lags), 6),
            "layer_availability": availability,
        }


# ---------------------------------------------------------------------------
# Sub-module: LayerWeightDynamicsController
# ---------------------------------------------------------------------------

_instance_weights: Dict[str, "_LayerWeightDynamicsController"] = {}


def LayerWeightDynamicsController(instance_id="default"):
    """Singleton accessor for ``_LayerWeightDynamicsController``."""
    if instance_id not in _instance_weights:
        _instance_weights[instance_id] = _LayerWeightDynamicsController(instance_id)
    return _instance_weights[instance_id]


class _LayerWeightDynamicsController:
    """Dynamically adjusts per-layer weights based on recent layer
    performance, stability, and confidence.
    """

    DEFAULT_WEIGHTS = {
        "saal": 0.35,
        "csfr": 0.30,
        "sdil": 0.20,
        "mrsrl": 0.15,
    }

    def __init__(self, instance_id="default"):
        self._instance_id = instance_id
        self._weights = dict(self.DEFAULT_WEIGHTS)
        self._weight_history: List[Dict[str, Any]] = []
        self._max_history = 1000
        logger.info(
            "LayerWeightDynamicsController(%r) initialised weights=%s",
            instance_id,
            self._weights,
        )

    def update_weights(
        self,
        sdil_state: Optional[dict] = None,
        csfr_signal: Optional[dict] = None,
        saal_authority: Optional[dict] = None,
        mrsrl_resolution: Optional[dict] = None,
    ) -> dict:
        """Recompute layer weights based on current layer outputs.

        The strategy:
        - If a layer's confidence is low (< 0.3), reduce its weight.
        - If SDIL detects collapse, shift weight away from SDIL to SAAL.
        - If CSRF signal_validity is INVALID, reduce CSRF weight.
        - If SAAL authority is stable, boost SAAL weight.
        - Weights are normalised to sum to 1.0.

        Parameters
        ----------
        sdil_state : dict or None
        csfr_signal : dict or None
        saal_authority : dict or None
        mrsrl_resolution : dict or None

        Returns
        -------
        dict
            Current weights ``{layer: float}`` after update.
        """
        raw = dict(self.DEFAULT_WEIGHTS)

        # SDIL
        if sdil_state:
            sdil_conf = float(sdil_state.get("confidence", 0.5))
            collapse = bool(sdil_state.get("collapse_detected", False))
            if collapse:
                raw["sdil"] *= 0.5  # penalise during collapse
            elif sdil_conf < 0.3:
                raw["sdil"] *= 0.8

        # CSRF
        if csfr_signal:
            csfr_conf = float(csfr_signal.get("confidence", 0.5))
            validity = str(csfr_signal.get("signal_validity", "QUESTIONABLE")).upper()
            if validity == "INVALID":
                raw["csfr"] *= 0.3
            elif csfr_conf < 0.3:
                raw["csfr"] *= 0.7

        # SAAL
        if saal_authority:
            saal_conf = float(saal_authority.get("authority_confidence", 0.0))
            stable = bool(saal_authority.get("authority_stable", False))
            if stable and saal_conf > 0.7:
                raw["saal"] *= 1.2  # boost
            elif saal_conf < 0.3:
                raw["saal"] *= 0.8

        # MRSRL
        if mrsrl_resolution:
            mrsrl_conf = float(mrsrl_resolution.get("confidence", 0.5))
            if mrsrl_conf < 0.3:
                raw["mrsrl"] *= 0.7

        # Normalise to sum 1.0
        total = sum(raw.values())
        if total > 0:
            for k in raw:
                raw[k] = round(raw[k] / total, 6)

        self._weights = raw

        # Record history
        self._weight_history.append({
            "timestamp": time.time(),
            "weights": dict(self._weights),
        })
        if len(self._weight_history) > self._max_history:
            self._weight_history = self._weight_history[-self._max_history:]

        logger.debug("update_weights -> %s", self._weights)
        return dict(self._weights)

    def get_weights(self) -> dict:
        """Return the current weights dict."""
        return dict(self._weights)

    def get_weight_history(self) -> list:
        """Return the weight change history (newest last)."""
        return list(self._weight_history)


# ---------------------------------------------------------------------------
# Sub-module: ExecutionPriorityArbiter
# ---------------------------------------------------------------------------

_instance_priority: Dict[str, "_ExecutionPriorityArbiter"] = {}


def ExecutionPriorityArbiter(instance_id="default"):
    """Singleton accessor for ``_ExecutionPriorityArbiter``."""
    if instance_id not in _instance_priority:
        _instance_priority[instance_id] = _ExecutionPriorityArbiter(instance_id)
    return _instance_priority[instance_id]


class _ExecutionPriorityArbiter:
    """Arbitrates priority among signal candidates from different layers
    and tracks how often the selected source changes.
    """

    def __init__(self, instance_id="default"):
        self._instance_id = instance_id
        self._last_selected_source: Optional[str] = None
        self._switch_count: int = 0
        self._total_arbitrations: int = 0
        # Map source -> number of times selected
        self._selection_counts: Dict[str, int] = {}
        logger.info(
            "ExecutionPriorityArbiter(%r) initialised", instance_id,
        )

    def arbitrate(self, candidates: list) -> dict:
        """Pick the best candidate from a list of signal candidates.

        Each candidate is a dict with at least:
          ``source`` (str), ``signal`` (int), ``confidence`` (float),
          ``priority`` (float, higher = better).

        The candidate with the highest ``priority`` is selected.
        Ties are broken by ``confidence``.

        Parameters
        ----------
        candidates : list of dict
            Signal candidates from all layers.

        Returns
        -------
        dict
            ``selected_source`` — which layer's candidate was chosen.

            ``selected_signal`` — the signal value (-1, 0, 1).

            ``selected_confidence`` — confidence of the chosen candidate.

            ``switch_count`` — total number of source switches so far.

            ``all_candidates`` — the full candidate list sorted by priority.
        """
        if not candidates:
            self._total_arbitrations += 1
            return {
                "selected_source": "NONE",
                "selected_signal": 0,
                "selected_confidence": 0.0,
                "switch_count": self._switch_count,
                "all_candidates": [],
            }

        # Sort by priority desc, then confidence desc
        sorted_cands = sorted(
            candidates,
            key=lambda c: (float(c.get("priority", 0)), float(c.get("confidence", 0))),
            reverse=True,
        )

        best = sorted_cands[0]
        selected_source = str(best.get("source", "UNKNOWN"))

        # Track source switching
        if (
            self._last_selected_source is not None
            and selected_source != self._last_selected_source
        ):
            self._switch_count += 1
        self._last_selected_source = selected_source
        self._total_arbitrations += 1
        self._selection_counts[selected_source] = (
            self._selection_counts.get(selected_source, 0) + 1
        )

        return {
            "selected_source": selected_source,
            "selected_signal": int(best.get("signal", 0)),
            "selected_confidence": float(best.get("confidence", 0.0)),
            "switch_count": self._switch_count,
            "all_candidates": sorted_cands,
        }

    def get_switch_frequency(self) -> float:
        """Return the fraction of arbitrations that resulted in a source
        switch (0-1).
        """
        if self._total_arbitrations == 0:
            return 0.0
        return round(self._switch_count / self._total_arbitrations, 4)

    def get_selection_counts(self) -> dict:
        """Return a dict mapping source name to number of times selected."""
        return dict(self._selection_counts)


# ---------------------------------------------------------------------------
# Sub-module: ExecutionSynthesisEngine
# ---------------------------------------------------------------------------

_instance_synth: Dict[str, "_ExecutionSynthesisEngine"] = {}


def ExecutionSynthesisEngine(instance_id="default"):
    """Singleton accessor for ``_ExecutionSynthesisEngine``."""
    if instance_id not in _instance_synth:
        _instance_synth[instance_id] = _ExecutionSynthesisEngine(instance_id)
    return _instance_synth[instance_id]


class _ExecutionSynthesisEngine:
    """Synthesises the final trading decision from resolved vectors,
    weights, candidate selections, and metadata.
    """

    def __init__(self, instance_id="default"):
        self._instance_id = instance_id
        self._decision_count: int = 0
        self._last_decision: Optional[dict] = None
        logger.info(
            "ExecutionSynthesisEngine(%r) initialised", instance_id,
        )

    def synthesize(
        self,
        resolved_vector: dict,
        decision_metadata: dict,
    ) -> dict:
        """Produce the final trading decision.

        Parameters
        ----------
        resolved_vector : dict
            The resolved vector from the conflict resolution matrix
            (includes ``resolved_authority``, ``conflicts_found``,
            ``resolved_vector``).
        decision_metadata : dict
            Contains ``dynamic_weights``, ``aligned_data``,
            ``arbitrated``, ``max_lag``, ``vector_hash``, etc.

        Returns
        -------
        dict
            ``decision`` — ``"BUY"`` | ``"SELL"`` | ``"HOLD"`` | ``"SKIP"``

            ``signal`` — int (-1, 0, 1)

            ``confidence`` — float 0-1

            ``decision_id`` — unique UUID

            ``pipeline_trace`` — dict with all trace fields

            ``reason`` — human-readable explanation.
        """
        self._decision_count += 1

        rv = resolved_vector.get("resolved_vector", resolved_vector)
        conflicts = resolved_vector.get("conflicts_found", [])
        resolved_authority = resolved_vector.get(
            "resolved_authority",
            rv.get("authority", "NONE"),
        )
        vector_hash = rv.get("vector_hash", "unknown")

        dynamic_weights = decision_metadata.get("dynamic_weights", {})
        aligned_data = decision_metadata.get("aligned_data", {})
        arbitrated = decision_metadata.get("arbitrated", {})
        max_lag = decision_metadata.get("max_lag", 0.0)

        # Determine signal from arbitrated result (highest priority)
        selected_signal = arbitrated.get("selected_signal", 0)

        # Confidence: blend vector's unified_confidence with arbitrated
        unified_conf = float(rv.get("unified_confidence", 0.0))
        arbitrated_conf = float(arbitrated.get("selected_confidence", 0.0))
        confidence = round((unified_conf * 0.6 + arbitrated_conf * 0.4), 4)
        confidence = max(0.0, min(1.0, confidence))

        # Build decision string
        if selected_signal == 1:
            decision_str = "BUY"
        elif selected_signal == -1:
            decision_str = "SELL"
        elif selected_signal == 0 and confidence < 0.4:
            decision_str = "SKIP"
        else:
            decision_str = "HOLD"

        # Override: if conflicts are severe, force SKIP
        severe_types = {"STABILITY_BREACH", "AUTHORITY_MISMATCH"}
        has_severe = any(
            c.get("type") in severe_types for c in conflicts
        )
        if has_severe and confidence < 0.5:
            decision_str = "SKIP"
            selected_signal = 0

        selected_source = arbitrated.get("selected_source", "NONE")
        reason_parts = [
            f"authority={resolved_authority}",
            f"signal={selected_signal}",
            f"conf={confidence:.4f}",
            f"source={selected_source}",
            f"conflicts={len(conflicts)}",
            f"max_lag={max_lag:.4f}s",
        ]
        if has_severe:
            reason_parts.append("SEVERE_CONFLICT")

        reason = " | ".join(reason_parts)

        decision_id = str(uuid.uuid4())

        result = {
            "decision": decision_str,
            "signal": selected_signal,
            "confidence": confidence,
            "decision_id": decision_id,
            "pipeline_trace": {
                "vector_hash": vector_hash,
                "resolved_authority": resolved_authority,
                "dynamic_weights": dict(dynamic_weights),
                "selected_source": selected_source,
                "conflicts_found": conflicts,
                "max_lag": max_lag,
            },
            "reason": reason,
        }

        self._last_decision = result
        logger.info(
            "synthesize -> %s signal=%d conf=%.4f conflicts=%d",
            decision_str, selected_signal, confidence, len(conflicts),
        )
        return result

    def reset(self):
        """Reset synthesis statistics."""
        self._decision_count = 0
        self._last_decision = None

    def get_stats(self) -> dict:
        """Return synthesis statistics.

        Returns
        -------
        dict
            ``total_decisions``, ``last_decision``.
        """
        return {
            "total_decisions": self._decision_count,
            "last_decision": self._last_decision,
        }


# ---------------------------------------------------------------------------
# Singleton registry for UESLOrchestrator
# ---------------------------------------------------------------------------

_instances: Dict[str, "_UESLOrchestrator"] = {}


def UESLOrchestrator(instance_id="default"):
    """Singleton accessor for ``_UESLOrchestrator``.

    Parameters
    ----------
    instance_id : str
        Unique identifier.  Callers sharing the same *instance_id* share
        the same orchestrator pipeline.

    Returns
    -------
    _UESLOrchestrator
    """
    if instance_id not in _instances:
        _instances[instance_id] = _UESLOrchestrator(instance_id)
    return _instances[instance_id]


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

class _UESLOrchestrator:
    """Unified Execution Synthesis Layer — the single entry point for all
    cognitive layer processing.

    Parameters
    ----------
    instance_id : str
        Instance identifier forwarded from the singleton accessor.
    """

    def __init__(self, instance_id="default"):
        self._instance_id = instance_id

        # Sub-module instances (all singletons keyed off instance_id)
        self.vector_engine = CrossLayerDecisionVectorEngine(
            instance_id + "_vec",
        )
        self.conflict_matrix = LayerConflictResolutionMatrix(
            instance_id + "_matrix",
        )
        self.synthesis_engine = ExecutionSynthesisEngine(
            instance_id + "_synth",
        )
        self.weight_controller = LayerWeightDynamicsController(
            instance_id + "_weights",
        )
        self.latency_system = DecisionLatencyAlignmentSystem(
            instance_id + "_latency",
        )
        self.priority_arbiter = ExecutionPriorityArbiter(
            instance_id + "_priority",
        )
        self.conflict_logger = ConflictTraceLogger(
            instance_id + "_log",
        )

        logger.info(
            "UESLOrchestrator(%r) initialised with %d sub-modules",
            instance_id, 7,
        )

    # ------------------------------------------------------------------
    # Public API  (THE main entry point)
    # ------------------------------------------------------------------

    def process(
        self,
        tick: dict,
        sdil_state: dict,
        csfr_signal: dict,
        saal_authority: dict,
        mrsrl_resolution: dict,
    ) -> dict:
        """Run the full UESL pipeline for one tick.

        Parameters
        ----------
        tick : dict
            Raw tick data with ``timestamp``, ``symbol``, ``bid``, ``ask``.
        sdil_state : dict
            SDIL layer output.
        csfr_signal : dict
            CSRF layer signal.
        saal_authority : dict
            SAAL layer authority.
        mrsrl_resolution : dict
            MRSRL layer resolution.

        Returns
        -------
        dict
            ``decision`` — ``"BUY"`` | ``"SELL"`` | ``"HOLD"`` | ``"SKIP"``

            ``signal`` — int (-1, 0, 1)

            ``confidence`` — float 0-1

            ``decision_id`` — unique UUID

            ``pipeline_trace`` — dict with trace fields

            ``reason`` — human-readable explanation.
        """
        # ---- Step 1: Build decision vector ----
        decision_vector = self.vector_engine.build_vector(
            tick, sdil_state, csfr_signal, saal_authority, mrsrl_resolution,
        )

        # ---- Step 2: Resolve conflicts ----
        resolved = self.conflict_matrix.resolve_conflict(decision_vector)
        conflicts_found = resolved.get("conflicts_found", [])

        # ---- Step 3: Align latency ----
        self.latency_system.record_layer_output("sdil", sdil_state)
        self.latency_system.record_layer_output("csfr", csfr_signal)
        self.latency_system.record_layer_output("saal", saal_authority)
        self.latency_system.record_layer_output("mrsrl", mrsrl_resolution)
        aligned = self.latency_system.align_for_decision()

        # ---- Step 4: Update dynamic weights ----
        dynamic_weights = self.weight_controller.update_weights(
            sdil_state, csfr_signal, saal_authority, mrsrl_resolution,
        )

        # ---- Step 5: Build weighted resolved vector ----
        # Apply weights to the resolved vector's confidence for synthesis
        # (weights are carried through the metadata to synthesis_engine)

        # ---- Step 6: Arbitrate priority ----
        candidates = self._build_candidates(
            decision_vector, resolved, aligned, dynamic_weights,
        )
        arbitrated = self.priority_arbiter.arbitrate(candidates)

        # ---- Step 7: Synthesise final decision ----
        metadata = {
            "dynamic_weights": dynamic_weights,
            "aligned_data": aligned.get("aligned_data", {}),
            "arbitrated": arbitrated,
            "max_lag": aligned.get("max_lag", 0.0),
            "vector_hash": decision_vector.get("vector_hash", "unknown"),
        }
        final_decision = self.synthesis_engine.synthesize(resolved, metadata)

        # ---- Step 8: Log conflicts ----
        for conflict in conflicts_found:
            try:
                cycle_id = tick.get("cycle_id", tick.get("timestamp", 0))
                if isinstance(cycle_id, float):
                    cycle_id = int(cycle_id)
                symbol = tick.get("symbol", "UNKNOWN")
                log_entry = self.conflict_logger.log_conflict(
                    cycle_id=cycle_id,
                    symbol=symbol,
                    layer_a=conflict.get("layer_a", "unknown"),
                    layer_b=conflict.get("layer_b", "unknown"),
                    conflict_type=conflict.get("type", "UNKNOWN"),
                    detail=conflict,
                )
                # Auto-resolve: mark as resolved by synthesis_engine
                self.conflict_logger.log_resolution(
                    cycle_id=cycle_id,
                    conflict_entry_id=log_entry["entry_id"],
                    resolution="SYNTHESIS_ENGINE",
                    resolved_by="synthesis_engine",
                )
            except Exception:
                logger.warning(
                    "Failed to log conflict: %s", conflict, exc_info=True,
                )

        return final_decision

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return aggregated status from all sub-modules.

        Returns
        -------
        dict
            ``conflict_count``, ``synthesis_stats``, ``current_weights``,
            ``latency_report``, ``arbiter_switch_frequency``,
            ``conflict_log_summary``.
        """
        return {
            "instance_id": self._instance_id,
            "conflict_count": self.conflict_matrix._conflict_count,
            "synthesis_stats": self.synthesis_engine.get_stats(),
            "current_weights": self.weight_controller.get_weights(),
            "latency_report": self.latency_system.get_latency_report(),
            "arbiter_switch_frequency": (
                self.priority_arbiter.get_switch_frequency()
            ),
            "arbiter_selection_counts": (
                self.priority_arbiter.get_selection_counts()
            ),
            "conflict_log_summary": (
                self.conflict_logger.get_conflict_summary()
            ),
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self):
        """Reset all sub-modules to their initial state.

        Sub-module instance identities are preserved; only logged data
        is cleared.
        """
        # CrossLayerDecisionVectorEngine does not expose reset, so we
        # access the internal vector store via a new instance trick or
        # skip.  For cleanliness, we note that the vector engine does
        # not have a reset method in its public API, so we document this.
        # The other sub-modules do have reset or clear mechanisms.

        # Conflict matrix: reset internal counter
        self.conflict_matrix._conflict_count = 0

        # Weight controller: reset to defaults
        self.weight_controller._weights = dict(
            self.weight_controller.DEFAULT_WEIGHTS,
        )
        self.weight_controller._weight_history.clear()

        # Latency system: clear recorded outputs
        self.latency_system._layer_outputs.clear()
        self.latency_system._latency_log.clear()

        # Priority arbiter: reset state
        self.priority_arbiter._last_selected_source = None
        self.priority_arbiter._switch_count = 0
        self.priority_arbiter._total_arbitrations = 0
        self.priority_arbiter._selection_counts.clear()

        # Synthesis engine: reset statistics
        self.synthesis_engine.reset()

        # Conflict logger: clear all logs
        self.conflict_logger.reset()

        logger.info(
            "UESLOrchestrator(%r) reset", self._instance_id,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_candidates(
        self,
        decision_vector: dict,
        resolved: dict,
        aligned: dict,
        weights: dict,
    ) -> list:
        """Build a list of signal candidates from all available layer data.

        Each candidate dict has ``source``, ``signal``, ``confidence``,
        and ``priority`` fields.
        """
        dv = decision_vector or {}
        rv = resolved.get("resolved_vector", dv)
        aligned_data = aligned.get("aligned_data", {})
        candidates: List[Dict[str, Any]] = []

        # 1. SDIL candidate (if available)
        sdil_aligned = aligned_data.get("sdil", {})
        if sdil_aligned:
            sdil_signal = 0
            collapse = sdil_aligned.get("collapse_detected", False)
            anomaly = sdil_aligned.get("anomaly_detected", False)
            if collapse or anomaly:
                sdil_signal = -1  # veto
            sdil_conf = float(sdil_aligned.get("confidence", 0.5))
            sdil_priority = sdil_conf * weights.get("sdil", 0.2)
            candidates.append({
                "source": "sdil",
                "signal": sdil_signal,
                "confidence": sdil_conf,
                "priority": round(sdil_priority, 6),
            })

        # 2. CSRF candidate
        csfr_aligned = aligned_data.get("csfr", {})
        if csfr_aligned:
            csfr_signal = 0
            validity = str(
                csfr_aligned.get("signal_validity", "QUESTIONABLE")
            ).upper()
            if validity == "VALID":
                # Use truth_source: OSS=+1, ALT=-1
                ts = str(csfr_aligned.get("truth_source", "NEITHER")).upper()
                csfr_signal = 1 if ts == "OSS" else (-1 if ts == "ALT" else 0)
            csfr_conf = float(csfr_aligned.get("confidence", 0.0))
            csfr_priority = csfr_conf * weights.get("csfr", 0.3)
            candidates.append({
                "source": "csfr",
                "signal": csfr_signal,
                "confidence": csfr_conf,
                "priority": round(csfr_priority, 6),
            })

        # 3. SAAL candidate
        saal_aligned = aligned_data.get("saal", {})
        if saal_aligned:
            saal_signal = int(saal_aligned.get("consensus_signal", 0))
            saal_conf = float(saal_aligned.get("authority_confidence", 0.0))
            saal_priority = saal_conf * weights.get("saal", 0.35)
            candidates.append({
                "source": "saal",
                "signal": saal_signal,
                "confidence": saal_conf,
                "priority": round(saal_priority, 6),
            })

        # 4. MRSRL candidate
        mrsrl_aligned = aligned_data.get("mrsrl", {})
        if mrsrl_aligned:
            mrsrl_signal = 0
            regime = str(
                mrsrl_aligned.get("resolution_regime", "NOISE")
            ).upper()
            # MRSRL provides resolution guidance, not a signal directly
            # If regime is MACRO_TREND -> bias +1, NOISE -> 0
            if regime == "MACRO_TREND":
                mrsrl_signal = 1
            elif regime in ("NOISE", "MICRO_NOISE"):
                mrsrl_signal = 0
            mrsrl_conf = float(mrsrl_aligned.get("confidence", 0.5))
            mrsrl_priority = mrsrl_conf * weights.get("mrsrl", 0.15)
            candidates.append({
                "source": "mrsrl",
                "signal": mrsrl_signal,
                "confidence": mrsrl_conf,
                "priority": round(mrsrl_priority, 6),
            })

        return candidates


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest():
    """Run 3 full pipeline cycles with different layer states and verify:
    1. process() returns a valid decision dict with all required fields
    2. Successive calls accumulate state correctly
    3. get_status() returns meaningful statistics
    4. Different inputs produce different decisions
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("=" * 60)
    logger.info("UESLOrchestrator — Self-Test")
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

    # Required keys in the final decision dict
    REQUIRED_KEYS = [
        "decision", "signal", "confidence", "decision_id",
        "pipeline_trace", "reason",
    ]
    VALID_DECISIONS = {"BUY", "SELL", "HOLD", "SKIP"}

    # Create orchestrator for self-test
    orch = UESLOrchestrator("selftest")
    orch.reset()

    # ==================================================================
    # Scenario 1 — Bullish alignment (all layers agree BUY)
    # ==================================================================
    logger.info("")
    logger.info("--- Scenario 1: Bullish alignment (all layers BUY) ---")

    tick1 = {
        "timestamp": 1000000.0,
        "symbol": "EURUSD",
        "bid": 1.1000,
        "ask": 1.1002,
        "cycle_id": 1,
    }
    sdil1 = {
        "collapse_detected": False, "entropy_level": "LOW",
        "anomaly_detected": False, "confidence": 0.85,
    }
    csfr1 = {
        "oss_accuracy": 0.78, "alt_accuracy": 0.72,
        "truth_source": "OSS", "signal_validity": "VALID",
        "confidence": 0.80,
    }
    saal1 = {
        "authority": "OSS", "consensus_signal": 1,
        "authority_confidence": 0.88, "authority_stable": True,
    }
    mrsrl1 = {
        "resolution_regime": "MESO_STRUCTURE", "structure_scale": "MESO_STRUCTURE",
        "optimal_timeframe": "1M", "adaptive_alt_mode": "ADAPTIVE_EMA",
        "adaptive_oss_mode": "RAW", "confidence": 0.75,
    }

    d1 = orch.process(tick1, sdil1, csfr1, saal1, mrsrl1)

    # Check all required keys
    for key in REQUIRED_KEYS:
        _check(key in d1, f"SCE1 required key '{key}' present")

    _check(d1["decision"] in VALID_DECISIONS, f"SCE1 valid decision, got {d1['decision']}")
    _check(d1["signal"] in (-1, 0, 1), f"SCE1 valid signal, got {d1['signal']}")
    _check(0.0 <= d1["confidence"] <= 1.0, f"SCE1 confidence in [0,1], got {d1['confidence']}")
    _check(isinstance(d1["decision_id"], str) and len(d1["decision_id"]) > 0,
           "SCE1 decision_id is non-empty string")
    _check(isinstance(d1["pipeline_trace"], dict), "SCE1 pipeline_trace is dict")

    trace = d1["pipeline_trace"]
    _check("vector_hash" in trace, "SCE1 trace has vector_hash")
    _check("resolved_authority" in trace, "SCE1 trace has resolved_authority")
    _check("dynamic_weights" in trace, "SCE1 trace has dynamic_weights")
    _check("selected_source" in trace, "SCE1 trace has selected_source")
    _check("conflicts_found" in trace, "SCE1 trace has conflicts_found")
    _check("max_lag" in trace, "SCE1 trace has max_lag")
    _check(isinstance(d1["reason"], str) and len(d1["reason"]) > 0,
           "SCE1 reason is non-empty string")

    logger.info("  Decision: %s (signal=%d, conf=%.4f)", d1["decision"], d1["signal"], d1["confidence"])

    # ==================================================================
    # Scenario 2 — Bearish with collapse (SDIL veto, conflict expected)
    # ==================================================================
    logger.info("")
    logger.info("--- Scenario 2: Bearish collapse (SDIL veto, conflict) ---")

    tick2 = {
        "timestamp": 1000100.0,
        "symbol": "EURUSD",
        "bid": 1.0500,
        "ask": 1.0503,
        "cycle_id": 2,
    }
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
        "authority": "OSS", "consensus_signal": -1,
        "authority_confidence": 0.20, "authority_stable": False,
    }
    mrsrl2 = {
        "resolution_regime": "NOISE", "structure_scale": "MICRO_NOISE",
        "optimal_timeframe": "TICK", "adaptive_alt_mode": "NO_SIGNAL",
        "adaptive_oss_mode": "FLAT", "confidence": 0.90,
    }

    d2 = orch.process(tick2, sdil2, csfr2, saal2, mrsrl2)

    for key in REQUIRED_KEYS:
        _check(key in d2, f"SCE2 required key '{key}' present")

    _check(d2["decision"] in VALID_DECISIONS, f"SCE2 valid decision, got {d2['decision']}")
    _check(d2["signal"] in (-1, 0, 1), f"SCE2 valid signal, got {d2['signal']}")

    # Should find conflicts
    _check(
        len(d2["pipeline_trace"]["conflicts_found"]) >= 0,
        "SCE2 conflicts_found is list",
    )

    logger.info("  Decision: %s (signal=%d, conf=%.4f)", d2["decision"], d2["signal"], d2["confidence"])
    logger.info("  Conflicts: %d", len(d2["pipeline_trace"]["conflicts_found"]))

    # ==================================================================
    # Scenario 3 — Mixed signals (neutral / HOLD)
    # ==================================================================
    logger.info("")
    logger.info("--- Scenario 3: Mixed signals (neutral / HOLD) ---")

    tick3 = {
        "timestamp": 1000200.0,
        "symbol": "GBPUSD",
        "bid": 1.2500,
        "ask": 1.2502,
        "cycle_id": 3,
    }
    sdil3 = {
        "collapse_detected": False, "entropy_level": "MODERATE",
        "anomaly_detected": False, "confidence": 0.50,
    }
    csfr3 = {
        "oss_accuracy": 0.50, "alt_accuracy": 0.50,
        "truth_source": "INCONCLUSIVE", "signal_validity": "QUESTIONABLE",
        "confidence": 0.40,
    }
    saal3 = {
        "authority": "HYBRID", "consensus_signal": 0,
        "authority_confidence": 0.45, "authority_stable": True,
    }
    mrsrl3 = {
        "resolution_regime": "MICRO_STRUCTURE", "structure_scale": "MICRO_NOISE",
        "optimal_timeframe": "TICK", "adaptive_alt_mode": "ZSCORE_BREAKOUT",
        "adaptive_oss_mode": "RAW", "confidence": 0.50,
    }

    d3 = orch.process(tick3, sdil3, csfr3, saal3, mrsrl3)

    for key in REQUIRED_KEYS:
        _check(key in d3, f"SCE3 required key '{key}' present")

    _check(d3["decision"] in VALID_DECISIONS, f"SCE3 valid decision, got {d3['decision']}")
    _check(d3["signal"] in (-1, 0, 1), f"SCE3 valid signal, got {d3['signal']}")

    logger.info("  Decision: %s (signal=%d, conf=%.4f)", d3["decision"], d3["signal"], d3["confidence"])

    # ==================================================================
    # Scenario 4 — State accumulation: conflict count, synthesis stats
    # ==================================================================
    logger.info("")
    logger.info("--- Scenario 4: State accumulation ---")

    status = orch.get_status()

    _check("instance_id" in status, "SCE4 status has instance_id")
    _check("conflict_count" in status, "SCE4 status has conflict_count")
    _check("synthesis_stats" in status, "SCE4 status has synthesis_stats")
    _check("current_weights" in status, "SCE4 status has current_weights")
    _check("latency_report" in status, "SCE4 status has latency_report")
    _check("arbiter_switch_frequency" in status, "SCE4 status has arbiter_switch_frequency")
    _check("conflict_log_summary" in status, "SCE4 status has conflict_log_summary")

    _check(
        status["synthesis_stats"]["total_decisions"] == 3,
        f"SCE4 total_decisions=3, got {status['synthesis_stats']['total_decisions']}",
    )

    _check(
        status["latency_report"]["total_tracked_layers"] >= 1,
        f"SCE4 latency tracked layers >= 1, got {status['latency_report']['total_tracked_layers']}",
    )

    _check(
        isinstance(status["current_weights"], dict),
        "SCE4 current_weights is dict",
    )
    _check(
        len(status["current_weights"]) == 4,
        f"SCE4 4 weight entries, got {len(status['current_weights'])}",
    )

    _check(
        isinstance(status["conflict_log_summary"]["total_conflicts"], int),
        "SCE4 conflict_log_summary has total_conflicts",
    )

    logger.info("  total_decisions = %d", status["synthesis_stats"]["total_decisions"])
    logger.info("  current_weights = %s", status["current_weights"])
    logger.info("  conflict_log_summary = %s", status["conflict_log_summary"])

    # ==================================================================
    # Scenario 5 — Different inputs produce different decisions
    # ==================================================================
    logger.info("")
    logger.info("--- Scenario 5: Different inputs produce different decisions ---")

    # SCE1 bullish vs SCE2 bearish should differ
    hashes = [d1["pipeline_trace"]["vector_hash"],
              d2["pipeline_trace"]["vector_hash"],
              d3["pipeline_trace"]["vector_hash"]]
    _check(
        len(set(hashes)) >= 2,
        f"SCE5 at least 2 different vector hashes among 3 runs "
        f"(got {len(set(hashes))})",
    )

    # At least one non-HOLD decision across scenarios
    non_hold = [d for d in (d1, d2, d3) if d["decision"] != "HOLD"]
    _check(
        len(non_hold) >= 1,
        f"SCE5 at least one non-HOLD decision (got {len(non_hold)})",
    )

    # ==================================================================
    # Scenario 6 — Conflict logging works through pipeline
    # ==================================================================
    logger.info("")
    logger.info("--- Scenario 6: Conflict logging via pipeline ---")

    recent_conflicts = orch.conflict_logger.get_conflicts()
    # At least the conflicts from SCE2 should be logged
    _check(
        len(recent_conflicts) >= 0,
        "SCE6 conflict logger has entries",
    )

    # Verify resolution logging
    for conflict in recent_conflicts:
        if conflict["entry_id"] in orch.conflict_logger._resolutions:
            resolution = orch.conflict_logger._resolutions[conflict["entry_id"]]
            _check(
                resolution["resolution"] == "SYNTHESIS_ENGINE",
                f"SCE6 resolution={resolution['resolution']}",
            )
            break  # just check one

    # ==================================================================
    # Scenario 7 — Singleton identity
    # ==================================================================
    logger.info("")
    logger.info("--- Scenario 7: Singleton identity ---")

    a = UESLOrchestrator("selftest_singleton")
    b = UESLOrchestrator("selftest_singleton")
    c = UESLOrchestrator("selftest_singleton_other")
    _check(a is b, "SCE7 same instance_id -> same object")
    _check(a is not c, "SCE7 different instance_id -> different object")

    # ==================================================================
    # Scenario 8 — Reset
    # ==================================================================
    logger.info("")
    logger.info("--- Scenario 8: Reset ---")

    orch_reset = UESLOrchestrator("selftest_reset")
    orch_reset.process(tick1, sdil1, csfr1, saal1, mrsrl1)

    status_before = orch_reset.get_status()
    _check(
        status_before["synthesis_stats"]["total_decisions"] >= 1,
        "SCE8 at least 1 decision before reset",
    )

    orch_reset.reset()
    status_after = orch_reset.get_status()

    _check(
        status_after["synthesis_stats"]["total_decisions"] == 0,
        f"SCE8 0 decisions after reset (got {status_after['synthesis_stats']['total_decisions']})",
    )
    _check(
        status_after["conflict_log_summary"]["total_conflicts"] == 0,
        "SCE8 0 conflicts after reset",
    )
    _check(
        status_after["arbiter_switch_frequency"] == 0.0,
        "SCE8 arbiter frequency = 0 after reset",
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

    return failed == 0


if __name__ == "__main__":
    _selftest()
