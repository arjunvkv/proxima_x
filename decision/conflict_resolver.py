"""
conflict_resolver.py — Contradiction resolution for intelligence sub-systems.

Resolves contradictions between anomaly detector, regime transition detector,
system health, and causal graph signals. When sub-systems disagree, this
module decides what to trust and produces a DecisionContext with adjusted
weights and resolved tensions.

Rules
-----
1. Regime vs Health contradiction — transition predicted but system is healthy.
2. Anomaly vs Regime contradiction — anomaly fired without regime change.
3. Health vs Anomaly contradiction — health critical without discrete anomaly.
4. Causal graph evidence           — coupled engines diverge in latest frame.
5. Multiple simultaneous transitions — regime oscillation detected.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "DecisionContext",
    "ConflictResolver",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Number of engine dimensions in the 32D telemetry vector
_N_ENGINE_DIMS = 32

# Scalar metric labels (matches causal_graph_builder.SCALAR_NAMES)
_SCALAR_METRICS: List[str] = [
    "alignment",
    "stability",
    "entropy",
    "regime_state",
    "tpi_confidence",
    "shadow_alignment",
    "sof_score",
    "kill_switch_pressure",
    "rollout_progress",
    "execution_intensity",
    "risk_exposure",
    "system_integrity",
]

# Default number of frames to look back for Rule 5 (oscillation detection)
_DEFAULT_WINDOW_SIZE = 10

# ---------------------------------------------------------------------------
# DecisionContext
# ---------------------------------------------------------------------------


@dataclass
class DecisionContext:
    """Resolved decision context after conflict resolution.

    Attributes
    ----------
    regime_confidence : float
        How much to trust the regime detector (0.0 — 1.0).
    anomaly_weight : float
        How much to weight anomaly signals (0.0 — 1.0).
    stability_bias : float
        Bias toward stability interpretation (-1.0 to +1.0).
    causal_priority_map : dict[str, float]
        Per-node priority weights (engines + metrics), normalised to sum 1.0.
    resolved_tensions : list[str]
        Human-readable descriptions of the contradictions that were resolved.
    timestamp : float
        Unix timestamp of the resolution.
    """
    regime_confidence: float = 0.5
    anomaly_weight: float = 0.5
    stability_bias: float = 0.0
    causal_priority_map: Dict[str, float] = field(default_factory=dict)
    resolved_tensions: List[str] = field(default_factory=list)
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# ConflictResolver
# ---------------------------------------------------------------------------


class ConflictResolver:
    """Resolve contradictions between intelligence sub-systems.

    The resolver ingests :class:`~proxima_x.intelligence.intelligence_bus.IntelligenceFrame`
    objects via :meth:`feed` and analyses the latest frame on :meth:`resolve`
    by applying five conflict-resolution rules.

    Internal state
    --------------
    _history : list[dict]
        Lightweight serialisation of past frames (frame_id, timestamp, regime
        probability, anomalies, health state, etc.) for trend analysis and
        Rule 5 oscillation detection.
    """

    def __init__(self, window_size: int = _DEFAULT_WINDOW_SIZE) -> None:
        self._window_size = window_size
        self._history: List[dict] = []
        self._latest_frame: Optional[Any] = None

    # ── Public API ──────────────────────────────────────────────────────────

    def feed(self, intelligence_frame: Any) -> None:
        """Feed an ``IntelligenceFrame`` from the intelligence bus.

        The frame is duck-typed and must provide attributes:
        ``.regime``, ``.anomalies``, ``.causal_graph``, ``.compressed_state``,
        ``.health``, ``.priority``, ``.summary``.

        Parameters
        ----------
        intelligence_frame : IntelligenceFrame
            The latest intelligence snapshot.
        """
        self._latest_frame = intelligence_frame
        snapshot = self._snapshot(intelligence_frame)
        self._history.append(snapshot)

        # Keep history bounded
        if len(self._history) > self._window_size * 5:
            self._history = self._history[-self._window_size * 5:]

    def resolve(self) -> DecisionContext:
        """Analyse the latest frame, resolve contradictions, return a decision.

        Returns
        -------
        DecisionContext
            Resolved context with adjusted weights, stability bias, priority
            map, and a list of tension descriptions.
        """
        ctx = DecisionContext(
            regime_confidence=0.5,
            anomaly_weight=0.5,
            stability_bias=0.0,
            causal_priority_map={},
            resolved_tensions=[],
            timestamp=self._latest_frame.timestamp if self._latest_frame is not None else time.time(),
        )

        frame = self._latest_frame
        if frame is None:
            return ctx

        # ── Gather signals ──────────────────────────────────────────────────
        regime = self._get_regime(frame)
        anomalies = self._get_anomalies(frame)
        health = self._get_health(frame)
        causal_graph = self._get_causal_graph(frame)
        snapshot = self._snapshot(frame) if self._history else {}

        # ── Rule 1: Regime vs Health contradiction ──────────────────────────
        ctx = self._rule_regime_vs_health(ctx, regime, health)

        # ── Rule 2: Anomaly vs Regime contradiction ─────────────────────────
        ctx = self._rule_anomaly_vs_regime(ctx, regime, anomalies)

        # ── Rule 3: Health vs Anomaly contradiction ─────────────────────────
        ctx = self._rule_health_vs_anomaly(ctx, health, anomalies)

        # ── Rule 4: Causal graph evidence ───────────────────────────────────
        ctx = self._rule_causal_divergence(ctx, causal_graph, anomalies)

        # ── Rule 5: Multiple simultaneous transitions ───────────────────────
        ctx = self._rule_oscillation(ctx)

        # ── Build priority map ──────────────────────────────────────────────
        ctx.causal_priority_map = self._build_priority_map(
            anomalies=anomalies,
            causal_graph=causal_graph,
        )

        # Clamp all scalar fields
        ctx.regime_confidence = max(0.0, min(1.0, ctx.regime_confidence))
        ctx.anomaly_weight = max(0.0, min(1.0, ctx.anomaly_weight))
        ctx.stability_bias = max(-1.0, min(1.0, ctx.stability_bias))

        return ctx

    def get_tensions(self) -> List[dict]:
        """Return current unresolved tensions for transparency.

        Analyses the latest frame and returns a list of dicts describing
        each contradiction that *would* be resolved, before applying
        adjustments.  Useful for debugging and logging.

        Returns
        -------
        list[dict]
            Each entry contains keys ``"rule"``, ``"description"``,
            ``"severity"``.
        """
        tensions: List[dict] = []
        frame = self._latest_frame
        if frame is None:
            return tensions

        regime = self._get_regime(frame)
        anomalies = self._get_anomalies(frame)
        health = self._get_health(frame)
        causal_graph = self._get_causal_graph(frame)

        # Rule 1
        if self._rule1_active(regime, health):
            tensions.append({
                "rule": 1,
                "description": (
                    "Regime transition predicted but system is healthy — "
                    "transition likely opportunistic, not defensive"
                ),
                "severity": "MEDIUM",
            })

        # Rule 2
        if self._rule2_active(regime, anomalies):
            tensions.append({
                "rule": 2,
                "description": (
                    "Anomaly detected without regime change signal — "
                    "could be sensor noise or leading indicator"
                ),
                "severity": "HIGH",
            })

        # Rule 3
        if self._rule3_active(health, anomalies):
            tensions.append({
                "rule": 3,
                "description": (
                    "Health declining without discrete anomaly — "
                    "systemic drift, not event-driven"
                ),
                "severity": "CRITICAL",
            })

        # Rule 4
        rule4_pairs = self._find_diverged_coupled_pairs(causal_graph, anomalies)
        for src, tgt, weight in rule4_pairs:
            tensions.append({
                "rule": 4,
                "description": (
                    f"Engine coupling break: {src} and {tgt} diverged "
                    f"despite historical correlation (weight={weight:.2f})"
                ),
                "severity": "HIGH",
            })

        # Rule 5
        if self._count_recent_transitions() >= 2:
            tensions.append({
                "rule": 5,
                "description": (
                    "Regime oscillation detected — transition signals unreliable"
                ),
                "severity": "CRITICAL",
            })

        return tensions

    def reset(self) -> None:
        """Clear all internal history and state."""
        self._history.clear()
        self._latest_frame = None

    # ── Rule implementations ────────────────────────────────────────────────

    def _rule_regime_vs_health(
        self,
        ctx: DecisionContext,
        regime: Optional[Any],
        health: Optional[Any],
    ) -> DecisionContext:
        """Rule 1: Regime vs Health contradiction.

        If regime transition with probability > 0.7 BUT health says HEALTHY
        with score > 0.5 → the system is stable enough to handle the transition.
        """
        if not self._rule1_active(regime, health):
            return ctx

        ctx.regime_confidence *= 0.8
        ctx.stability_bias = +0.3
        ctx.resolved_tensions.append(
            "Regime transition predicted but system is healthy — "
            "transition likely opportunistic, not defensive"
        )
        return ctx

    def _rule_anomaly_vs_regime(
        self,
        ctx: DecisionContext,
        regime: Optional[Any],
        anomalies: List[Any],
    ) -> DecisionContext:
        """Rule 2: Anomaly vs Regime contradiction.

        If HIGH/CRITICAL anomaly fired BUT regime detector says no transition
        → anomaly may be a false positive or regime detector is lagging.
        """
        if not self._rule2_active(regime, anomalies):
            return ctx

        ctx.anomaly_weight *= 0.7
        ctx.regime_confidence *= 0.9
        ctx.resolved_tensions.append(
            "Anomaly detected without regime change signal — "
            "could be sensor noise or leading indicator"
        )
        return ctx

    def _rule_health_vs_anomaly(
        self,
        ctx: DecisionContext,
        health: Optional[Any],
        anomalies: List[Any],
    ) -> DecisionContext:
        """Rule 3: Health vs Anomaly contradiction.

        If health says CRITICAL (score < -0.3) BUT no anomalies fired
        → health degradation is gradual/structural, not event-driven.
        """
        if not self._rule3_active(health, anomalies):
            return ctx

        ctx.regime_confidence *= 0.85
        ctx.stability_bias = -0.4
        ctx.resolved_tensions.append(
            "Health declining without discrete anomaly — "
            "systemic drift, not event-driven"
        )
        return ctx

    def _rule_causal_divergence(
        self,
        ctx: DecisionContext,
        causal_graph: Optional[Any],
        anomalies: List[Any],
    ) -> DecisionContext:
        """Rule 4: Causal graph evidence.

        If causal graph shows strong coupling (>0.7) between two engines,
        but they diverge in the latest frame → divergence from normally
        coupled pair is significant.
        """
        diverged_pairs = self._find_diverged_coupled_pairs(causal_graph, anomalies)
        if not diverged_pairs:
            return ctx

        ctx.anomaly_weight *= 1.3

        for src, tgt, weight in diverged_pairs:
            # Boost these engines in the priority map
            ctx.causal_priority_map[src] = max(
                ctx.causal_priority_map.get(src, 0.0),
                weight,
            )
            ctx.causal_priority_map[tgt] = max(
                ctx.causal_priority_map.get(tgt, 0.0),
                weight,
            )
            ctx.resolved_tensions.append(
                f"Engine coupling break: {src} and {tgt} diverged "
                f"despite historical correlation (weight={weight:.2f})"
            )

        return ctx

    def _rule_oscillation(self, ctx: DecisionContext) -> DecisionContext:
        """Rule 5: Multiple simultaneous transitions.

        If regime detector fired 2+ transitions within window_size frames
        → system is oscillating.
        """
        count = self._count_recent_transitions()
        if count < 2:
            return ctx

        ctx.stability_bias = -0.6
        ctx.regime_confidence *= 0.5
        ctx.resolved_tensions.append(
            "Regime oscillation detected — transition signals unreliable"
        )
        return ctx

    # ── Guard helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _rule1_active(
        regime: Optional[Any],
        health: Optional[Any],
    ) -> bool:
        """Check whether Rule 1 preconditions are met."""
        if regime is None or health is None:
            return False

        prob = getattr(regime, 'probability', 0.0)
        if prob <= 0.7:
            return False

        health_state = getattr(health, 'state', None)
        if health_state is None:
            return False

        # Resolve HealthState enum to string
        state_str = health_state.value if hasattr(health_state, 'value') else str(health_state)
        if state_str != "HEALTHY":
            return False

        score = getattr(health, 'score', 0.0)
        if score <= 0.5:
            return False

        return True

    @staticmethod
    def _rule2_active(
        regime: Optional[Any],
        anomalies: List[Any],
    ) -> bool:
        """Check whether Rule 2 preconditions are met."""
        if regime is not None:
            return False  # regime IS present → no contradiction

        if not anomalies:
            return False

        # Check for HIGH or CRITICAL severity
        for a in anomalies:
            sev = getattr(a, 'severity', 'LOW')
            if sev in ('HIGH', 'CRITICAL'):
                return True

        return False

    @staticmethod
    def _rule3_active(
        health: Optional[Any],
        anomalies: List[Any],
    ) -> bool:
        """Check whether Rule 3 preconditions are met."""
        if health is None:
            return False

        # Health must be CRITICAL
        health_state = getattr(health, 'state', None)
        if health_state is None:
            return False
        state_str = health_state.value if hasattr(health_state, 'value') else str(health_state)
        if state_str != "CRITICAL":
            return False

        # Score must be < -0.3
        score = getattr(health, 'score', 0.0)
        if score >= -0.3:
            return False

        # No anomalies fired
        if anomalies:
            return False

        return True

    # ── Causal divergence detection (Rule 4 helper) ─────────────────────────

    def _find_diverged_coupled_pairs(
        self,
        causal_graph: Optional[Any],
        anomalies: List[Any],
    ) -> List[Tuple[str, str, float]]:
        """Find engine pairs that are strongly coupled but have diverged.

        Scans the causal graph for edges between engine nodes with weight > 0.7,
        then checks anomaly descriptions for indications that those specific
        engines have diverged (contradictory outputs).

        Returns
        -------
        list[tuple[str, str, float]]
            ``(source_id, target_id, edge_weight)`` for each diverged pair.
        """
        if causal_graph is None:
            return []

        strong_edges: List[Tuple[str, str, float]] = []
        edges = getattr(causal_graph, 'edges', [])
        if not edges:
            return []

        for edge in edges:
            src = getattr(edge, 'source', '')
            tgt = getattr(edge, 'target', '')
            weight = getattr(edge, 'weight', 0.0)
            # Only consider engine↔engine edges with strong coupling
            if weight > 0.7 and src.startswith("engine_") and tgt.startswith("engine_"):
                strong_edges.append((src, tgt, weight))

        if not strong_edges:
            return []

        # Extract engine indices mentioned in anomaly descriptions
        diverged_indices: set[int] = set()
        for anomaly in anomalies:
            desc = getattr(anomaly, 'description', '')
            subsystem = getattr(anomaly, 'subsystem', '')
            if subsystem == 'engine_vector' or 'engine' in desc.lower():
                # Parse engine indices from description like "engine[7]=..." or "engine_7..."
                self._extract_engine_indices(desc, diverged_indices)

        # Also check compressed_state divergence (if available)
        if self._latest_frame is not None:
            cs = getattr(self._latest_frame, 'compressed_state', None)
            if cs is not None and hasattr(cs, 'components') and hasattr(cs, 'explained_variance'):
                # High explained variance with large component values can indicate divergence
                ev = getattr(cs, 'explained_variance', 0.0)
                if ev > 0.8:
                    components = getattr(cs, 'components', [])
                    if components and max(abs(c) for c in components) > 3.0:
                        # Significant deviation in latent space — mark all
                        # engines 0-7 as potentially diverged (first latent dim
                        # captures the most variance)
                        for i in range(min(8, _N_ENGINE_DIMS)):
                            diverged_indices.add(i)

        if not diverged_indices:
            return []

        # Match strong edges against diverged engine indices
        result: List[Tuple[str, str, float]] = []
        for src, tgt, weight in strong_edges:
            src_idx = self._parse_engine_index(src)
            tgt_idx = self._parse_engine_index(tgt)
            if src_idx in diverged_indices or tgt_idx in diverged_indices:
                result.append((src, tgt, weight))

        return result

    @staticmethod
    def _extract_engine_indices(desc: str, accum: set[int]) -> None:
        """Parse engine index numbers from an anomaly description string.

        Handles patterns like:
        - ``"engine[7]=..."``
        - ``"engine_7..."``
        - ``"engine 7..."``
        """
        # Pattern: engine[<digits>]
        import re
        for m in re.finditer(r'engine\[(\d+)\]', desc):
            accum.add(int(m.group(1)))
        # Pattern: engine_<digits>
        for m in re.finditer(r'engine_(\d+)', desc):
            accum.add(int(m.group(1)))
        # Pattern: "engine <digits>"
        for m in re.finditer(r'engine\s+(\d+)', desc):
            accum.add(int(m.group(1)))

    @staticmethod
    def _parse_engine_index(node_id: str) -> int:
        """Extract the integer index from a node ID like ``"engine_7"``."""
        # node_id is like "engine_7"
        try:
            return int(node_id.split("_", 1)[1])
        except (IndexError, ValueError):
            return -1

    # ── Priority map construction ──────────────────────────────────────────

    @staticmethod
    def _build_priority_map(
        anomalies: List[Any],
        causal_graph: Optional[Any],
    ) -> Dict[str, float]:
        """Construct a normalised priority map over engines and metrics.

        1. Start with uniform weights for engines 0-31 and scalar metrics.
        2. Boost (×3) engines flagged by anomaly detector vector_signature
           positions.
        3. Boost (×2) engines with strong causal edges (weight > 0.5) in the
           causal graph.
        4. Normalise to sum to 1.0.

        Returns
        -------
        dict[str, float]
            Node ID → normalised priority weight.
        """
        weights: Dict[str, float] = {}

        # ── Step 1: uniform base weights ────────────────────────────────────
        # Engine nodes
        for i in range(_N_ENGINE_DIMS):
            weights[f"engine_{i}"] = 1.0
        # Metric nodes
        for name in _SCALAR_METRICS:
            weights[f"metric_{name}"] = 1.0

        # ── Step 2: boost engines flagged by anomaly detector ──────────────
        boosted_indices: set[int] = set()
        for anomaly in anomalies:
            sig = getattr(anomaly, 'vector_signature', [])
            desc = getattr(anomaly, 'description', '')
            severity = getattr(anomaly, 'severity', 'LOW')
            score = getattr(anomaly, 'score', 0.0)

            # If the anomaly has a vector_signature, the length hints at how
            # many engine dimensions are involved
            if sig and len(sig) <= _N_ENGINE_DIMS:
                # The signature may be a subset of engine dims; treat each
                # non-zero position as a flagged engine
                for idx, val in enumerate(sig):
                    if idx < _N_ENGINE_DIMS and abs(val) > 1e-8:
                        boosted_indices.add(idx)

            # Also parse engine indices from the description text
            ConflictResolver._extract_engine_indices(desc, boosted_indices)

        for idx in boosted_indices:
            node = f"engine_{idx}"
            if node in weights:
                # Triple weight for flagged engines (more if HIGH/CRITICAL)
                multiplier = 3.0
                weights[node] *= multiplier

        # ── Step 3: boost engines with strong causal edges ──────────────────
        if causal_graph is not None:
            edges = getattr(causal_graph, 'edges', [])
            edge_targets: Dict[str, float] = {}
            for edge in edges:
                src = getattr(edge, 'source', '')
                tgt = getattr(edge, 'target', '')
                w = getattr(edge, 'weight', 0.0)
                if w > 0.5:
                    for node_id in (src, tgt):
                        if node_id in weights:
                            cur = edge_targets.get(node_id, 0.0)
                            edge_targets[node_id] = max(cur, w)

            for node_id, w in edge_targets.items():
                # Scale boost by edge weight (2× at w=0.5, up to 3× at w=1.0)
                boost = 1.0 + w  # gives range [1.5, 2.0]
                weights[node_id] *= boost

        # ── Step 4: normalise to sum 1.0 ────────────────────────────────────
        total = sum(weights.values())
        if total > 0:
            inv_total = 1.0 / total
            for key in weights:
                weights[key] *= inv_total

        return weights

    # ── History helpers ─────────────────────────────────────────────────────

    def _count_recent_transitions(self) -> int:
        """Count the number of regime transitions in the last window_size frames.

        Uses the lightweight history snapshot, not the full IntelligenceFrame
        history, to avoid unbounded memory growth.
        """
        if not self._history:
            return 0

        recent = self._history[-self._window_size:]
        count = 0
        for snap in recent:
            if snap.get("regime_present", False):
                prob = snap.get("regime_probability", 0.0)
                if prob > 0.3:  # minimum threshold to count as a transition
                    count += 1
        return count

    # ── Frame introspection helpers ─────────────────────────────────────────

    @staticmethod
    def _snapshot(frame: Any) -> dict:
        """Extract a lightweight serialisable dict from an IntelligenceFrame."""
        regime = getattr(frame, 'regime', None)
        health = getattr(frame, 'health', None)
        anomalies = getattr(frame, 'anomalies', [])

        return {
            "frame_id": getattr(frame, 'frame_id', 0),
            "timestamp": getattr(frame, 'timestamp', 0.0),
            "regime_present": regime is not None,
            "regime_probability": getattr(regime, 'probability', 0.0) if regime is not None else 0.0,
            "regime_from": getattr(regime, 'from_regime', '') if regime is not None else '',
            "regime_to": getattr(regime, 'to_regime', '') if regime is not None else '',
            "anomaly_count": len(anomalies),
            "anomaly_max_severity": max(
                (getattr(a, 'severity', 'LOW') for a in anomalies),
                default='LOW',
            ),
            "health_score": getattr(health, 'score', 0.0) if health is not None else 0.0,
            "health_state": (
                getattr(health, 'state', None).value
                if health is not None and hasattr(getattr(health, 'state', None), 'value')
                else str(getattr(health, 'state', 'N/A'))
                if health is not None
                else 'N/A'
            ),
            "causal_graph_present": getattr(frame, 'causal_graph', None) is not None,
        }

    @staticmethod
    def _get_regime(frame: Any) -> Optional[Any]:
        """Safely extract the regime signal from a frame."""
        return getattr(frame, 'regime', None)

    @staticmethod
    def _get_anomalies(frame: Any) -> List[Any]:
        """Safely extract anomaly events from a frame."""
        return getattr(frame, 'anomalies', []) or []

    @staticmethod
    def _get_health(frame: Any) -> Optional[Any]:
        """Safely extract the health score from a frame."""
        return getattr(frame, 'health', None)

    @staticmethod
    def _get_causal_graph(frame: Any) -> Optional[Any]:
        """Safely extract the causal graph from a frame."""
        return getattr(frame, 'causal_graph', None)
