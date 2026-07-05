"""
engine_harvester.py — Lightweight hot-path engine state harvester.

Extracts ONLY scalar values from all Proxima engine subsystems for the
HOT PATH (the 32D cognition vector + 12 scalars written to shared memory
every cycle). This is the ONLY module that touches engine internals in
the hot path.

Hot-path rules:
  - NO string generation
  - NO dict deep copies
  - NO allocations beyond the return value
  - Every method handles missing attributes gracefully via getattr
  - harvest_engine_vector() returns exactly 32 floats
  - harvest_scalars() returns exactly 12 key: float pairs
"""

from __future__ import annotations

import math
from typing import Optional

from ..schema.telemetry_schema import (
    ExecutionTopologySnapshot,
    InformationThermodynamicsSnapshot,
    RegimeSnapshot,
)


class EngineHarvester:
    """
    Lightweight hot-path engine state harvester.
    Extracts ONLY scalar values — NO string generation, NO dict deep copies.

    This is the ONLY module that reads engine internals in the hot path.
    All methods aim for zero allocations beyond the return value.
    """

    # ── Regime encoding map ──────────────────────────────────────────────
    _REGIME_MAP = {
        "BULL": 1.0,
        "BEAR": -1.0,
        "CHAOTIC": 0.75,
        "COMPRESSED": -0.5,
        "NORMAL": 0.0,
        "WIDE": -0.75,
    }

    def __init__(self, demo) -> None:
        self._demo = demo

    # ─────────────────────────────────────────────────────────────────────
    # HOT PATH — harvest_engine_vector (32 floats)
    # ─────────────────────────────────────────────────────────────────────

    def harvest_engine_vector(self) -> list[float]:
        """
        Extract 32 floats representing the full system state manifold.

        GROUP 1 (0-3): Regime Geometry
        GROUP 2 (4-9): Information Thermodynamics
        GROUP 3 (10-15): TPI / Flow Dynamics
        GROUP 4 (16-21): Execution Topology
        GROUP 5 (22-27): Shadow / Edge Space
        GROUP 6 (28-31): System Health
        """
        vector = [0.0] * 32

        # ── GROUP 1: Regime Geometry (indices 0-3) ──────────────────────
        regime_memory = getattr(self._demo, "_regime_memory", None)
        if regime_memory is not None:
            prev_regimes = getattr(regime_memory, "_prev_regime", {})
            regimes = list(prev_regimes.values()) if prev_regimes else []
            if regimes:
                # regime_state — encode last known regime as float
                vector[0] = self._REGIME_MAP.get(regimes[-1], 0.0)
                # transition_pressure — absolute difference between last
                # two regime encodings
                if len(regimes) >= 2:
                    current = self._REGIME_MAP.get(regimes[-1], 0.0)
                    previous = self._REGIME_MAP.get(regimes[-2], 0.0)
                    vector[1] = abs(current - previous)
                # entropy_gradient — placeholder (no real-time gradient
                # available in hot path)
                vector[2] = 0.5
                # stability_velocity — placeholder
                vector[3] = 0.5

        # ── GROUP 2: Information Thermodynamics (indices 4-9) ───────────
        ec = getattr(self._demo, "_entropy_compression", None)
        if ec is not None:
            eval_data = getattr(self._demo, "_current_eval_data", {}) or {}
            syms = list(eval_data.keys())
            if syms:
                try:
                    entropy_state = ec.compute_state(syms[0])
                    # 4: entropy_level
                    vector[4] = entropy_state.get("normalized_entropy", 0.5)
                    # 6: compression_ratio
                    vector[6] = entropy_state.get("compression_ratio", 0.0)
                except Exception:
                    pass

        # Average entropy across all active symbols
        eval_data = getattr(self._demo, "_current_eval_data", {}) or {}
        if eval_data:
            entropies = [
                d.get("entropy", 0.5)
                for d in eval_data.values()
                if d.get("entropy") is not None
            ]
            if entropies:
                avg_entropy = sum(entropies) / len(entropies)
                # 5: signal_entropy (average symbol entropy)
                vector[5] = avg_entropy
                # 7: predictability_index
                vector[7] = 1.0 - avg_entropy

        # 8: noise_floor_estimate — placeholder
        vector[8] = 0.0
        # 9: entropy_derivative — placeholder
        vector[9] = 0.0

        # ── GROUP 3: TPI / Flow Dynamics (indices 10-15) ────────────────
        # 10: tpi_confidence
        tpi_tracker = getattr(self._demo, "_tpi_tracker", None)
        if tpi_tracker is not None:
            vector[10] = float(getattr(tpi_tracker, "tpi_confidence", 0.5))

        # 11: tpi_alignment — from TPI tracker's alignment score
        if tpi_tracker is not None:
            vector[11] = float(getattr(tpi_tracker, "alignment", 0.0))

        # 12: tpi_persistence — from persistence tracker
        pers = getattr(self._demo, "_tpi_persistence", None)
        if pers is not None:
            vector[12] = float(getattr(pers, "persistence", 0.0))

        # 13: tpi_curvature — from curvature tracker
        curv = getattr(self._demo, "_tpi_curvature", None)
        if curv is not None:
            vector[13] = float(getattr(curv, "curvature", 0.0))

        # 14: flow_momentum — from propagation engine
        prop = getattr(self._demo, "_tpi_propagation", None)
        if prop is not None and hasattr(prop, "compute"):
            try:
                result = prop.compute()
                if isinstance(result, dict):
                    vector[14] = float(result.get("momentum", 0.0))
            except Exception:
                pass

        # 15: flow_divergence — placeholder
        vector[15] = 0.0

        # ── GROUP 4: Execution Topology (indices 16-21) ─────────────────
        # 16: signal_density — from fusion kernel
        fusion = getattr(self._demo, "_fusion", None)
        if fusion is not None:
            vector[16] = float(getattr(fusion, "signal_density", 0.0))

        # 17: execution_rate — from rotation engine
        rotation = getattr(self._demo, "_rotation", None)
        if rotation is not None:
            vector[17] = float(getattr(rotation, "execution_rate", 0.0))

        # 18: fill_ratio — from execution bridge
        bridge = getattr(self._demo, "_execution_bridge", None)
        if bridge is not None:
            vector[18] = float(getattr(bridge, "fill_ratio", 0.0))

        # 19: slippage_proxy — placeholder
        vector[19] = 0.0

        # 20: win_rate_proxy — from ranking engine
        ranking = getattr(self._demo, "_ranking", None)
        if ranking is not None:
            vector[20] = float(getattr(ranking, "win_rate", 0.0))

        # 21: risk_exposure — from H20 engine
        h20 = getattr(self._demo, "_h20", None)
        if h20 is not None and hasattr(h20, "current_exposure"):
            try:
                vector[21] = float(h20.current_exposure())
            except Exception:
                pass

        # ── GROUP 5: Shadow / Edge Space (indices 22-27) ────────────────
        stre = getattr(self._demo, "_last_stre_result", None)
        if stre:
            # 22: shadow_alignment
            vector[22] = float(stre.get("gt_corr", 0.0))
            # 23: sof_score
            vector[23] = float(stre.get("SOF", 0.0))
            # 24: edge_preservation
            vector[24] = float(stre.get("edge_preservation", 0.0))
            # 25: execution_efficiency
            vector[25] = float(stre.get("execution_efficiency", 0.0))
            # 26: stas (shadow temporal alignment score)
            vector[26] = float(stre.get("stas", 0.0))
            # 27: sy_corr (shadow-yield correlation)
            vector[27] = float(stre.get("sy_corr", 0.0))

        # ── GROUP 6: System Health (indices 28-31) ──────────────────────
        # 28: stability_score — from stability subsystems
        mso = getattr(self._demo, "_mso", None)
        if mso is not None:
            vector[28] = float(getattr(mso, "stability_score", 0.0))

        # 29: kill_switch_pressure — from budget block TTL
        budget_ttl = getattr(self._demo, "_budget_block_ttl", {}) or {}
        if budget_ttl:
            max_ttl = max(budget_ttl.values()) if budget_ttl else 0
            vector[29] = min(1.0, max_ttl / 100.0)
        else:
            vector[29] = 0.0

        # 30: rollout_progress — from warmup ticks
        warmup = getattr(self._demo, "_warmup_ticks", 0)
        if warmup and hasattr(self._demo, "_warmup_target"):
            target = getattr(self._demo, "_warmup_target", 100)
            vector[30] = min(1.0, warmup / max(target, 1))
        else:
            vector[30] = 0.0

        # 31: system_integrity — always 1.0 unless quarantined
        quar = getattr(self._demo, "_quarantine_cycles_remaining", 0)
        paused = getattr(self._demo, "_paused", False)
        if paused or quar > 0:
            vector[31] = 0.0  # system is degraded
        else:
            vector[31] = 1.0

        return [0.0 if (v is None or not math.isfinite(v)) else v for v in vector]

    # ─────────────────────────────────────────────────────────────────────
    # HOT PATH — harvest_scalars (12 key: float pairs)
    # ─────────────────────────────────────────────────────────────────────

    def harvest_scalars(self) -> dict[str, float]:
        """
        Extract 12 key scalar values for the hot path frame.

        Returns dict matching the 12 scalar fields of the SHM frame.
        """
        result = {
            "alignment": 0.0,
            "stability": 0.0,
            "entropy": 0.5,
            "regime_state": 0.0,
            "tpi_confidence": 0.0,
            "shadow_alignment": 0.0,
            "sof_score": 0.0,
            "kill_switch_pressure": 0.0,
            "rollout_progress": 0.0,
            "execution_intensity": 0.0,
            "risk_exposure": 0.0,
            "system_integrity": 1.0,
        }

        # ── Alignment — from shadow mirror summary ──────────────────────
        shadow = getattr(self._demo, "_shadow_mirror", None)
        if shadow is not None and hasattr(shadow, "summary"):
            try:
                s = shadow.summary()
                result["alignment"] = float(s.get("agreement_rate", 0.0))
            except Exception:
                pass

        # ── Stability — from MetaStabilityOptimizer ─────────────────────
        mso = getattr(self._demo, "_mso", None)
        if mso is not None:
            result["stability"] = float(getattr(mso, "stability_score", 0.0))

        # ── Entropy — from current eval_data ────────────────────────────
        eval_data = getattr(self._demo, "_current_eval_data", {}) or {}
        if eval_data:
            entropies = [
                d.get("entropy", 0.5)
                for d in eval_data.values()
                if d.get("entropy") is not None
            ]
            if entropies:
                result["entropy"] = sum(entropies) / len(entropies)

        # ── Regime state — from regime memory ───────────────────────────
        regime_memory = getattr(self._demo, "_regime_memory", None)
        if regime_memory is not None:
            prev_regimes = getattr(regime_memory, "_prev_regime", {})
            regimes = list(prev_regimes.values()) if prev_regimes else []
            if regimes:
                result["regime_state"] = self._REGIME_MAP.get(
                    regimes[-1], 0.0
                )

        # ── TPI confidence ──────────────────────────────────────────────
        tracker = getattr(self._demo, "_tpi_tracker", None)
        if tracker is not None:
            result["tpi_confidence"] = float(
                getattr(tracker, "tpi_confidence", 0.0)
            )

        # ── Shadow system ───────────────────────────────────────────────
        stre = getattr(self._demo, "_last_stre_result", None)
        if stre:
            result["shadow_alignment"] = float(stre.get("gt_corr", 0.0))
            result["sof_score"] = float(stre.get("SOF", 0.0))

        # ── Kill switch pressure ────────────────────────────────────────
        budget_ttl = getattr(self._demo, "_budget_block_ttl", {}) or {}
        if budget_ttl:
            max_ttl = max(budget_ttl.values()) if budget_ttl else 0
            result["kill_switch_pressure"] = min(1.0, max_ttl / 100.0)

        # ── Rollout progress ────────────────────────────────────────────
        warmup = getattr(self._demo, "_warmup_ticks", 0)
        if warmup and hasattr(self._demo, "_warmup_target"):
            target = getattr(self._demo, "_warmup_target", 100)
            result["rollout_progress"] = min(1.0, warmup / max(target, 1))

        # ── Execution intensity ─────────────────────────────────────────
        rotation = getattr(self._demo, "_rotation", None)
        if rotation is not None:
            intensity = getattr(rotation, "execution_rate", 0.0)
            result["execution_intensity"] = float(intensity)

        # ── Risk exposure — via H20 engine ──────────────────────────────
        h20 = getattr(self._demo, "_h20", None)
        if h20 is not None and hasattr(h20, "current_exposure"):
            try:
                result["risk_exposure"] = float(h20.current_exposure())
            except Exception:
                pass

        # ── System integrity ────────────────────────────────────────────
        quar = getattr(self._demo, "_quarantine_cycles_remaining", 0)
        paused = getattr(self._demo, "_paused", False)
        result["system_integrity"] = 0.0 if (paused or quar > 0) else 1.0

        return {k: (0.0 if not math.isfinite(v) else v) for k, v in result.items()}

    # ─────────────────────────────────────────────────────────────────────
    # COLD PATH — Full snapshot extractors (more work allowed)
    # ─────────────────────────────────────────────────────────────────────

    def extract_regime(self) -> Optional[RegimeSnapshot]:
        """Extract regime geometry (for COLD path full snapshot)."""
        regime_memory = getattr(self._demo, "_regime_memory", None)
        prev_regimes = (
            getattr(regime_memory, "_prev_regime", {})
            if regime_memory is not None
            else {}
        )

        regimes = list(prev_regimes.values()) if prev_regimes else []

        regime_state = 0.0
        transition_pressure = 0.0
        if regimes:
            regime_state = self._REGIME_MAP.get(regimes[-1], 0.0)
            if len(regimes) >= 2:
                current = self._REGIME_MAP.get(regimes[-1], 0.0)
                previous = self._REGIME_MAP.get(regimes[-2], 0.0)
                transition_pressure = abs(current - previous)

        eval_data = getattr(self._demo, "_current_eval_data", {}) or {}
        per_symbol = {}
        for sym, data in eval_data.items():
            per_symbol[sym] = str(data.get("regime", "N/A"))

        return RegimeSnapshot(
            regime_state=regime_state,
            regime_transition_pressure=transition_pressure,
            regime_entropy_gradient=0.5,
            regime_stability_velocity=0.5,
            per_symbol_regime=per_symbol,
        )

    def extract_thermodynamics(
        self,
    ) -> Optional[InformationThermodynamicsSnapshot]:
        """Extract information thermodynamics (for COLD path)."""
        ec = getattr(self._demo, "_entropy_compression", None)
        eval_data = getattr(self._demo, "_current_eval_data", {}) or {}

        entropies = [
            d.get("entropy", 0.5)
            for d in eval_data.values()
            if d.get("entropy") is not None
        ]
        avg_entropy = sum(entropies) / len(entropies) if entropies else 0.5

        compression_ratio = 0.0
        if ec is not None and eval_data:
            try:
                sym = list(eval_data.keys())[0]
                state = ec.compute_state(sym)
                compression_ratio = state.get("compression_ratio", 0.0)
            except Exception:
                pass

        return InformationThermodynamicsSnapshot(
            entropy_level=avg_entropy,
            entropy_derivative=0.0,
            compression_ratio=compression_ratio,
            signal_entropy=avg_entropy,
            noise_floor_estimate=0.0,
            predictability_index=1.0 - avg_entropy,
        )

    def extract_execution_topology(self) -> Optional[ExecutionTopologySnapshot]:
        """Extract execution topology (for COLD path)."""
        demo = self._demo

        return ExecutionTopologySnapshot(
            signal_density=0.0,
            execution_rate=0.0,
            fill_ratio=0.0,
            slippage_proxy=0.0,
            win_rate_proxy=0.0,
            risk_exposure=0.0,
            rotation_events=getattr(demo, "_rotation_event_count", 0),
            lock_events=getattr(demo, "_lock_event_count", 0),
            migration_events=getattr(demo, "_migration_event_count", 0),
        )
