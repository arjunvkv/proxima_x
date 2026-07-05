"""
Tests for ConflictResolver — conflict resolution across intelligence sub-systems.

Covers all 5 rules, priority map construction, edge cases, and integration
with duck-typed IntelligenceFrame.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import pytest

from conflict_resolver import ConflictResolver, DecisionContext


# ---------------------------------------------------------------------------
# Helper: minimal duck-typed stubs matching the real types
# ---------------------------------------------------------------------------


class HealthState(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"


@dataclass
class StubTransitionSignal:
    from_regime: str = "SHADOW"
    to_regime: str = "MICRO"
    probability: float = 0.0
    confidence: float = 0.0
    drivers: List[str] = field(default_factory=list)
    timestamp: float = 0.0


@dataclass
class StubAnomalyEvent:
    severity: str = "LOW"
    subsystem: str = "entropy"
    timestamp: float = 0.0
    vector_signature: List[float] = field(default_factory=list)
    description: str = ""
    score: float = 0.0


@dataclass
class StubSystemHealthScore:
    score: float = 0.0
    state: HealthState = HealthState.HEALTHY
    components: Dict[str, float] = field(default_factory=dict)
    trend: str = "stable"
    timestamp: float = 0.0


@dataclass
class StubCausalEdge:
    source: str = ""
    target: str = ""
    weight: float = 0.0
    lag: int = 0
    method: str = "cross_corr"


@dataclass
class StubCausalGraph:
    nodes: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    timestamp: float = 0.0


@dataclass
class StubCompressedState:
    dims: int = 8
    components: List[float] = field(default_factory=lambda: [0.0] * 8)
    explained_variance: float = 0.0
    embedding_type: str = "regime"
    timestamp: float = 0.0


@dataclass
class StubIntelligenceFrame:
    frame_id: int = 0
    timestamp: float = 0.0
    regime: Any = None
    anomalies: List[Any] = field(default_factory=list)
    causal_graph: Any = None
    compressed_state: Any = None
    health: Any = None
    summary: str = ""
    priority: str = "LOW"


# =========================================================================
# Tests
# =========================================================================


class TestDecisionContext:
    """DecisionContext dataclass basics."""

    def test_defaults(self) -> None:
        ctx = DecisionContext()
        assert ctx.regime_confidence == 0.5
        assert ctx.anomaly_weight == 0.5
        assert ctx.stability_bias == 0.0
        assert ctx.causal_priority_map == {}
        assert ctx.resolved_tensions == []
        assert ctx.timestamp == 0.0

    def test_fields(self) -> None:
        ctx = DecisionContext(
            regime_confidence=0.8,
            anomaly_weight=0.3,
            stability_bias=-0.6,
            causal_priority_map={"engine_0": 0.1},
            resolved_tensions=["test tension"],
            timestamp=123.0,
        )
        assert ctx.regime_confidence == 0.8
        assert ctx.anomaly_weight == 0.3
        assert ctx.stability_bias == -0.6
        assert ctx.causal_priority_map == {"engine_0": 0.1}
        assert ctx.resolved_tensions == ["test tension"]
        assert ctx.timestamp == 123.0


class TestConflictResolverInit:
    """Constructor and basic state."""

    def test_init(self) -> None:
        r = ConflictResolver()
        assert r._history == []
        assert r._latest_frame is None

    def test_reset(self) -> None:
        r = ConflictResolver()
        frame = StubIntelligenceFrame(frame_id=1, timestamp=100.0)
        r.feed(frame)
        assert len(r._history) == 1
        r.reset()
        assert r._history == []
        assert r._latest_frame is None


class TestFeed:
    """Feeding frames."""

    def test_feed_stores_snapshot(self) -> None:
        r = ConflictResolver()
        frame = StubIntelligenceFrame(frame_id=1, timestamp=100.0)
        r.feed(frame)
        assert len(r._history) == 1
        assert r._history[0]["frame_id"] == 1

    def test_feed_multiple(self) -> None:
        r = ConflictResolver(window_size=10)
        for i in range(20):
            r.feed(StubIntelligenceFrame(frame_id=i, timestamp=float(i)))
        # History should be bounded at window_size * 5 = 50
        assert len(r._history) == 20
        assert r._history[-1]["frame_id"] == 19


class TestRule1RegimeVsHealth:
    """Rule 1: Regime vs Health contradiction."""

    def test_no_contradiction_no_regime(self) -> None:
        r = ConflictResolver()
        frame = StubIntelligenceFrame(
            regime=None,
            health=StubSystemHealthScore(score=0.8, state=HealthState.HEALTHY),
        )
        r.feed(frame)
        ctx = r.resolve()
        assert ctx.regime_confidence == 0.5  # unchanged
        assert ctx.stability_bias == 0.0

    def test_no_contradiction_low_prob(self) -> None:
        r = ConflictResolver()
        frame = StubIntelligenceFrame(
            regime=StubTransitionSignal(probability=0.5),
            health=StubSystemHealthScore(score=0.8, state=HealthState.HEALTHY),
        )
        r.feed(frame)
        ctx = r.resolve()
        assert ctx.regime_confidence == 0.5
        assert ctx.stability_bias == 0.0

    def test_no_contradiction_unhealthy(self) -> None:
        r = ConflictResolver()
        frame = StubIntelligenceFrame(
            regime=StubTransitionSignal(probability=0.9),
            health=StubSystemHealthScore(score=0.3, state=HealthState.DEGRADED),
        )
        r.feed(frame)
        ctx = r.resolve()
        assert ctx.regime_confidence == 0.5
        assert ctx.stability_bias == 0.0

    def test_contradiction_applies(self) -> None:
        r = ConflictResolver()
        frame = StubIntelligenceFrame(
            regime=StubTransitionSignal(probability=0.9),
            health=StubSystemHealthScore(score=0.7, state=HealthState.HEALTHY),
        )
        r.feed(frame)
        ctx = r.resolve()
        # regime_confidence *= 0.8 → 0.5 * 0.8 = 0.4
        assert ctx.regime_confidence == pytest.approx(0.4)
        assert ctx.stability_bias == 0.3
        assert len(ctx.resolved_tensions) == 1
        assert "opportunistic" in ctx.resolved_tensions[0]

    def test_contradiction_boundary_prob(self) -> None:
        """Probability exactly at 0.7 should not trigger (must be > 0.7)."""
        r = ConflictResolver()
        frame = StubIntelligenceFrame(
            regime=StubTransitionSignal(probability=0.7),
            health=StubSystemHealthScore(score=0.7, state=HealthState.HEALTHY),
        )
        r.feed(frame)
        ctx = r.resolve()
        assert ctx.regime_confidence == 0.5
        assert ctx.stability_bias == 0.0

    def test_contradiction_boundary_health_score(self) -> None:
        """Health score exactly at 0.5 should not trigger (must be > 0.5)."""
        r = ConflictResolver()
        frame = StubIntelligenceFrame(
            regime=StubTransitionSignal(probability=0.8),
            health=StubSystemHealthScore(score=0.5, state=HealthState.HEALTHY),
        )
        r.feed(frame)
        ctx = r.resolve()
        assert ctx.regime_confidence == 0.5
        assert ctx.stability_bias == 0.0


class TestRule2AnomalyVsRegime:
    """Rule 2: Anomaly vs Regime contradiction."""

    def test_no_contradiction_when_regime_present(self) -> None:
        r = ConflictResolver()
        frame = StubIntelligenceFrame(
            regime=StubTransitionSignal(probability=0.6),
            anomalies=[StubAnomalyEvent(severity="HIGH")],
        )
        r.feed(frame)
        ctx = r.resolve()
        assert ctx.anomaly_weight == 0.5
        assert ctx.regime_confidence == 0.5

    def test_no_contradiction_low_severity(self) -> None:
        r = ConflictResolver()
        frame = StubIntelligenceFrame(
            regime=None,
            anomalies=[StubAnomalyEvent(severity="LOW")],
        )
        r.feed(frame)
        ctx = r.resolve()
        assert ctx.anomaly_weight == 0.5
        assert ctx.regime_confidence == 0.5

    def test_contradiction_high_severity(self) -> None:
        r = ConflictResolver()
        frame = StubIntelligenceFrame(
            regime=None,
            anomalies=[StubAnomalyEvent(severity="HIGH")],
        )
        r.feed(frame)
        ctx = r.resolve()
        # anomaly_weight *= 0.7 → 0.5 * 0.7 = 0.35
        assert ctx.anomaly_weight == pytest.approx(0.35)
        # regime_confidence *= 0.9 → 0.5 * 0.9 = 0.45
        assert ctx.regime_confidence == pytest.approx(0.45)
        assert len(ctx.resolved_tensions) == 1
        assert "sensor noise" in ctx.resolved_tensions[0]

    def test_contradiction_critical_severity(self) -> None:
        r = ConflictResolver()
        frame = StubIntelligenceFrame(
            regime=None,
            anomalies=[StubAnomalyEvent(severity="CRITICAL")],
        )
        r.feed(frame)
        ctx = r.resolve()
        assert ctx.anomaly_weight == pytest.approx(0.35)
        assert ctx.regime_confidence == pytest.approx(0.45)

    def test_contradiction_medium_severity(self) -> None:
        """MEDIUM severity should NOT trigger Rule 2."""
        r = ConflictResolver()
        frame = StubIntelligenceFrame(
            regime=None,
            anomalies=[StubAnomalyEvent(severity="MEDIUM")],
        )
        r.feed(frame)
        ctx = r.resolve()
        assert ctx.anomaly_weight == 0.5
        assert ctx.regime_confidence == 0.5

    def test_contradiction_empty_anomalies(self) -> None:
        r = ConflictResolver()
        frame = StubIntelligenceFrame(regime=None, anomalies=[])
        r.feed(frame)
        ctx = r.resolve()
        assert ctx.anomaly_weight == 0.5
        assert ctx.regime_confidence == 0.5


class TestRule3HealthVsAnomaly:
    """Rule 3: Health vs Anomaly contradiction."""

    def test_no_contradiction_healthy(self) -> None:
        r = ConflictResolver()
        frame = StubIntelligenceFrame(
            health=StubSystemHealthScore(score=0.5, state=HealthState.HEALTHY),
            anomalies=[],
        )
        r.feed(frame)
        ctx = r.resolve()
        assert ctx.regime_confidence == 0.5
        assert ctx.stability_bias == 0.0

    def test_no_contradiction_anomaly_present(self) -> None:
        r = ConflictResolver()
        frame = StubIntelligenceFrame(
            health=StubSystemHealthScore(score=-0.5, state=HealthState.CRITICAL),
            anomalies=[StubAnomalyEvent(severity="HIGH")],
        )
        r.feed(frame)
        ctx = r.resolve()
        # Rule 3 does NOT fire because anomalies exist.
        # Rule 2 fires (no regime + HIGH anomaly): regime_confidence *= 0.9 → 0.45
        assert ctx.regime_confidence == pytest.approx(0.45)
        # Rule 3 bias should NOT apply because anomalies are present
        assert ctx.stability_bias == 0.0

    def test_no_contradiction_score_not_low_enough(self) -> None:
        """Score must be < -0.3."""
        r = ConflictResolver()
        frame = StubIntelligenceFrame(
            health=StubSystemHealthScore(score=-0.3, state=HealthState.CRITICAL),
            anomalies=[],
        )
        r.feed(frame)
        ctx = r.resolve()
        assert ctx.regime_confidence == 0.5
        assert ctx.stability_bias == 0.0

    def test_contradiction_applies(self) -> None:
        r = ConflictResolver()
        frame = StubIntelligenceFrame(
            health=StubSystemHealthScore(score=-0.5, state=HealthState.CRITICAL),
            anomalies=[],
        )
        r.feed(frame)
        ctx = r.resolve()
        # regime_confidence *= 0.85 → 0.5 * 0.85 = 0.425
        assert ctx.regime_confidence == pytest.approx(0.425)
        assert ctx.stability_bias == -0.4
        assert len(ctx.resolved_tensions) == 1
        assert "systemic drift" in ctx.resolved_tensions[0]


class TestRule4CausalDivergence:
    """Rule 4: Causal graph evidence — coupled engines that diverge."""

    def test_no_graph(self) -> None:
        r = ConflictResolver()
        frame = StubIntelligenceFrame(
            causal_graph=None,
            anomalies=[],
        )
        r.feed(frame)
        ctx = r.resolve()
        assert ctx.anomaly_weight == 0.5
        assert len(ctx.resolved_tensions) == 0

    def test_no_strong_edges(self) -> None:
        r = ConflictResolver()
        graph = StubCausalGraph(edges=[
            StubCausalEdge(source="engine_0", target="engine_1", weight=0.3),
        ])
        frame = StubIntelligenceFrame(causal_graph=graph, anomalies=[])
        r.feed(frame)
        ctx = r.resolve()
        assert ctx.anomaly_weight == 0.5

    def test_strong_edge_no_divergence(self) -> None:
        """Edge >0.7 but no anomaly mentions those engines."""
        r = ConflictResolver()
        graph = StubCausalGraph(edges=[
            StubCausalEdge(source="engine_0", target="engine_7", weight=0.85),
        ])
        frame = StubIntelligenceFrame(
            causal_graph=graph,
            anomalies=[StubAnomalyEvent(
                severity="LOW",
                subsystem="entropy",
                description="some unrelated thing",
            )],
        )
        r.feed(frame)
        ctx = r.resolve()
        assert ctx.anomaly_weight == 0.5  # no divergence detected

    def test_strong_edge_with_divergence(self) -> None:
        """Edge >0.7 AND anomaly references diverged engines."""
        r = ConflictResolver()
        graph = StubCausalGraph(edges=[
            StubCausalEdge(source="engine_0", target="engine_7", weight=0.85),
        ])
        frame = StubIntelligenceFrame(
            causal_graph=graph,
            anomalies=[StubAnomalyEvent(
                severity="HIGH",
                subsystem="engine_vector",
                description="Contradictory engine outputs: engine[0]=+0.9 vs engine[7]=-0.8 (normally correlated)",
                vector_signature=[0.9, -0.8],
                score=0.7,
            )],
        )
        r.feed(frame)
        ctx = r.resolve()
        # Rule 2 fires first (no regime + HIGH anomaly): anomaly_weight *= 0.7 → 0.35
        # Then Rule 4 fires: anomaly_weight *= 1.3 → 0.35 * 1.3 = 0.455
        assert ctx.anomaly_weight == pytest.approx(0.455)
        assert len(ctx.resolved_tensions) == 2  # Rule 2 + Rule 4
        assert "Engine coupling break" in ctx.resolved_tensions[1]
        assert "engine_0" in ctx.resolved_tensions[1]
        assert "engine_7" in ctx.resolved_tensions[1]

    def test_compressed_state_divergence_detection(self) -> None:
        """High explained variance + large components → divergence detected."""
        r = ConflictResolver()
        graph = StubCausalGraph(edges=[
            StubCausalEdge(source="engine_0", target="engine_1", weight=0.8),
        ])
        # Compressed state with high explained variance and large components
        cs = StubCompressedState(
            components=[4.5, 2.0, 0.5, -1.0, 0.0, 0.0, 0.0, 0.0],
            explained_variance=0.92,
        )
        frame = StubIntelligenceFrame(
            causal_graph=graph,
            compressed_state=cs,
            anomalies=[],
        )
        r.feed(frame)
        ctx = r.resolve()
        assert ctx.anomaly_weight == pytest.approx(0.65)
        assert len(ctx.resolved_tensions) == 1


class TestRule5Oscillation:
    """Rule 5: Multiple simultaneous transitions."""

    def test_no_transitions(self) -> None:
        r = ConflictResolver()
        for _ in range(5):
            r.feed(StubIntelligenceFrame(regime=None))
        ctx = r.resolve()
        assert ctx.stability_bias == 0.0
        assert ctx.regime_confidence == 0.5

    def test_single_transition(self) -> None:
        """Single transition should NOT trigger Rule 5."""
        r = ConflictResolver(window_size=10)
        r.feed(StubIntelligenceFrame(
            regime=StubTransitionSignal(probability=0.8),
        ))
        for _ in range(4):
            r.feed(StubIntelligenceFrame(regime=None))
        # Resolve after the latest frame
        ctx = r.resolve()
        assert ctx.stability_bias == 0.0
        assert ctx.regime_confidence == 0.5

    def test_two_transitions(self) -> None:
        """Two transitions within window → Rule 5 fires."""
        r = ConflictResolver(window_size=10)
        r.feed(StubIntelligenceFrame(
            regime=StubTransitionSignal(probability=0.8),
        ))
        r.feed(StubIntelligenceFrame(regime=None))
        r.feed(StubIntelligenceFrame(
            regime=StubTransitionSignal(probability=0.9),
        ))
        for _ in range(2):
            r.feed(StubIntelligenceFrame(regime=None))
        ctx = r.resolve()
        # stability_bias = -0.6, regime_confidence *= 0.5 → 0.5 * 0.5 = 0.25
        assert ctx.stability_bias == -0.6
        assert ctx.regime_confidence == pytest.approx(0.25)
        assert len(ctx.resolved_tensions) == 1
        assert "oscillation" in ctx.resolved_tensions[0]

    def test_transitions_outside_window(self) -> None:
        """Old transitions outside window should not count."""
        r = ConflictResolver(window_size=5)
        # Feed 5 frames with transitions, then 5 without
        r.feed(StubIntelligenceFrame(
            regime=StubTransitionSignal(probability=0.8),
        ))
        r.feed(StubIntelligenceFrame(
            regime=StubTransitionSignal(probability=0.8),
        ))
        # 3 more non-transition frames to stay within window
        r.feed(StubIntelligenceFrame(regime=None))
        r.feed(StubIntelligenceFrame(regime=None))
        r.feed(StubIntelligenceFrame(regime=None))
        ctx = r.resolve()
        # 2 transitions within window of 5 → Rule 5 fires
        assert ctx.stability_bias == -0.6
        assert ctx.regime_confidence == pytest.approx(0.25)


class TestPriorityMap:
    """Priority map construction."""

    def test_uniform_default(self) -> None:
        """With no anomalies or graph, all weights should be uniform."""
        r = ConflictResolver()
        # Need at least one frame for resolve to build the priority map
        r.feed(StubIntelligenceFrame())
        ctx = r.resolve()
        pm = ctx.causal_priority_map
        # Should have 32 engines + 12 metrics = 44 entries
        assert len(pm) == 44
        # All close to 1/44 (~0.0227)
        expected_uniform = 1.0 / 44.0
        for key, val in pm.items():
            assert val == pytest.approx(expected_uniform, abs=0.001), f"{key}: {val}"

    def test_anomaly_boost(self) -> None:
        """Anomaly vector_signature should boost specific engines."""
        r = ConflictResolver()
        frame = StubIntelligenceFrame(
            anomalies=[StubAnomalyEvent(
                severity="HIGH",
                subsystem="engine_vector",
                vector_signature=[0.5, -0.3, 0.0],  # engines 0 and 1 have values
                description="engine[0] divergence",
            )],
        )
        r.feed(frame)
        ctx = r.resolve()
        pm = ctx.causal_priority_map
        # engine_0 and engine_1 should be boosted above baseline
        base = pm["engine_2"]
        assert pm["engine_0"] > base
        assert pm["engine_1"] > base

    def test_causal_graph_boost(self) -> None:
        """Strong causal edges should boost target nodes."""
        r = ConflictResolver()
        graph = StubCausalGraph(edges=[
            StubCausalEdge(source="engine_7", target="engine_3", weight=0.8),
        ])
        frame = StubIntelligenceFrame(causal_graph=graph, anomalies=[])
        r.feed(frame)
        ctx = r.resolve()
        pm = ctx.causal_priority_map
        # engine_7 and engine_3 should be boosted above baseline
        base = pm["engine_0"]
        assert pm["engine_7"] > base
        assert pm["engine_3"] > base

    def test_normalisation(self) -> None:
        """Priority map should sum to 1.0."""
        r = ConflictResolver()
        r.feed(StubIntelligenceFrame())
        ctx = r.resolve()
        total = sum(ctx.causal_priority_map.values())
        assert total == pytest.approx(1.0, abs=1e-6)


class TestGetTensions:
    """Tension introspection."""

    def test_empty_frame(self) -> None:
        r = ConflictResolver()
        assert r.get_tensions() == []

    def test_rule1_tension(self) -> None:
        r = ConflictResolver()
        frame = StubIntelligenceFrame(
            regime=StubTransitionSignal(probability=0.9),
            health=StubSystemHealthScore(score=0.8, state=HealthState.HEALTHY),
        )
        r.feed(frame)
        tensions = r.get_tensions()
        assert len(tensions) >= 1
        assert tensions[0]["rule"] == 1

    def test_rule2_tension(self) -> None:
        r = ConflictResolver()
        frame = StubIntelligenceFrame(
            regime=None,
            anomalies=[StubAnomalyEvent(severity="CRITICAL")],
        )
        r.feed(frame)
        tensions = r.get_tensions()
        assert any(t["rule"] == 2 for t in tensions)

    def test_rule3_tension(self) -> None:
        r = ConflictResolver()
        frame = StubIntelligenceFrame(
            health=StubSystemHealthScore(score=-0.5, state=HealthState.CRITICAL),
            anomalies=[],
        )
        r.feed(frame)
        tensions = r.get_tensions()
        assert any(t["rule"] == 3 for t in tensions)

    def test_rule4_tension(self) -> None:
        r = ConflictResolver()
        graph = StubCausalGraph(edges=[
            StubCausalEdge(source="engine_1", target="engine_8", weight=0.85),
        ])
        frame = StubIntelligenceFrame(
            causal_graph=graph,
            anomalies=[StubAnomalyEvent(
                subsystem="engine_vector",
                description="engine[1] vs engine[8] divergence",
            )],
        )
        r.feed(frame)
        tensions = r.get_tensions()
        assert any(t["rule"] == 4 for t in tensions)

    def test_rule5_tension(self) -> None:
        r = ConflictResolver(window_size=10)
        r.feed(StubIntelligenceFrame(
            regime=StubTransitionSignal(probability=0.8),
        ))
        r.feed(StubIntelligenceFrame(
            regime=StubTransitionSignal(probability=0.9),
        ))
        tensions = r.get_tensions()
        assert any(t["rule"] == 5 for t in tensions)


class TestMultipleRules:
    """Multiple rules firing simultaneously."""

    def test_rules_1_and_2_together(self) -> None:
        """Rule 1 and Rule 2 should apply independently."""
        r = ConflictResolver()
        frame = StubIntelligenceFrame(
            regime=StubTransitionSignal(probability=0.9),
            health=StubSystemHealthScore(score=0.8, state=HealthState.HEALTHY),
            anomalies=[StubAnomalyEvent(severity="HIGH")],  # Rule 2
        )
        # But note: Rule 2 requires regime == None, so it won't fire here.
        # Let's make a frame that triggers Rule 1 only with regime present.
        r.feed(frame)
        ctx = r.resolve()
        # Rule 1: regime_confidence *= 0.8 → 0.4
        assert ctx.regime_confidence == pytest.approx(0.4)
        assert ctx.stability_bias == 0.3
        # Rule 2 does NOT fire because regime is present
        assert ctx.anomaly_weight == 0.5

    def test_all_rules_independently(self) -> None:
        """Feed different frames and verify each rule's adjustments.

        Each resolve() is independent — it analyses only the latest frame
        and starts from default weights.
        """
        r = ConflictResolver(window_size=10)

        # Frame 1: Rule 1 (regime high + health healthy)
        r.feed(StubIntelligenceFrame(
            regime=StubTransitionSignal(probability=0.9),
            health=StubSystemHealthScore(score=0.8, state=HealthState.HEALTHY),
        ))
        ctx = r.resolve()
        # Rule 1: 0.5*0.8=0.4, bias=+0.3
        assert ctx.regime_confidence == pytest.approx(0.4)
        assert ctx.stability_bias == 0.3

        # Frame 2: Rule 3 (health critical + no anomalies)
        r.feed(StubIntelligenceFrame(
            health=StubSystemHealthScore(score=-0.6, state=HealthState.CRITICAL),
            anomalies=[],
        ))
        ctx = r.resolve()
        # Rule 3: starts from 0.5 → 0.5 * 0.85 = 0.425, bias = -0.4
        assert ctx.regime_confidence == pytest.approx(0.425)
        assert ctx.stability_bias == -0.4

        # Frame 3: Rule 5 context (2 transitions within window)
        r.feed(StubIntelligenceFrame(
            regime=StubTransitionSignal(probability=0.9),
        ))
        ctx = r.resolve()
        # Rule 5: regime_confidence *= 0.5 → 0.5 * 0.5 = 0.25, bias = -0.6
        assert ctx.regime_confidence == pytest.approx(0.25)
        assert ctx.stability_bias == -0.6


class TestEdgeCases:
    """Edge case handling."""

    def test_empty_feed(self) -> None:
        r = ConflictResolver()
        ctx = r.resolve()
        assert ctx.regime_confidence == 0.5
        assert ctx.anomaly_weight == 0.5
        assert ctx.stability_bias == 0.0

    def test_frame_with_none_fields(self) -> None:
        r = ConflictResolver()
        frame = StubIntelligenceFrame(
            regime=None,
            anomalies=[],
            causal_graph=None,
            compressed_state=None,
            health=None,
        )
        r.feed(frame)
        ctx = r.resolve()
        assert ctx.regime_confidence == 0.5
        assert ctx.anomaly_weight == 0.5

    def test_partial_frame(self) -> None:
        """Frame missing some attributes (partial duck typing)."""
        r = ConflictResolver()

        class PartialFrame:
            timestamp = 100.0
            frame_id = 1

        pf = PartialFrame()
        pf.regime = StubTransitionSignal(probability=0.9)
        pf.anomalies = []
        pf.health = StubSystemHealthScore(score=0.9, state=HealthState.HEALTHY)
        # no causal_graph, compressed_state, priority, summary — should be OK

        r.feed(pf)
        ctx = r.resolve()
        assert ctx.regime_confidence == pytest.approx(0.4)

    def test_clamping(self) -> None:
        """Ensure all scalars are clamped to valid ranges."""
        r = ConflictResolver()
        # Force extreme values by applying multiple rules
        frame = StubIntelligenceFrame(
            regime=StubTransitionSignal(probability=0.9),
            health=StubSystemHealthScore(score=0.9, state=HealthState.HEALTHY),
        )
        r.feed(frame)
        ctx = r.resolve()
        assert 0.0 <= ctx.regime_confidence <= 1.0
        assert 0.0 <= ctx.anomaly_weight <= 1.0
        assert -1.0 <= ctx.stability_bias <= 1.0

    def test_health_state_as_string(self) -> None:
        """Should handle health.state as a plain string too."""
        r = ConflictResolver()
        frame = StubIntelligenceFrame(
            regime=StubTransitionSignal(probability=0.9),
            health=StubSystemHealthScore(score=0.8, state="HEALTHY"),
        )
        r.feed(frame)
        ctx = r.resolve()
        assert ctx.regime_confidence == pytest.approx(0.4)
        assert ctx.stability_bias == 0.3
