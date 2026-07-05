"""
Tests for DecisionSynthesizer — the central synthesis layer that converts
intelligence + policy into actionable trading signals.

Covers: signal extraction, policy weighting, action tendency mapping, regime
action signals, risk bias calculation, reasoning chain generation, history
retention, and edge cases with duck-typed inputs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from decision_synthesizer import (
    ActionTendency,
    DecisionSynthesizer,
    RegimeActionSignal,
    SystemDecision,
)


# ---------------------------------------------------------------------------
# Helper: minimal duck-typed stubs matching the real types
# ---------------------------------------------------------------------------


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
    description: str = ""
    score: float = 0.0
    vector_signature: List[float] = field(default_factory=list)


@dataclass
class StubSystemHealthScore:
    score: float = 0.0
    state: str = "HEALTHY"
    components: Dict[str, float] = field(default_factory=dict)
    trend: str = "stable"
    timestamp: float = 0.0


@dataclass
class StubIntelligenceFrame:
    """Matches IntelligenceFrame protocol."""
    frame_id: int = 0
    timestamp: float = 0.0
    regime: Optional[Any] = None
    anomalies: List[Any] = field(default_factory=list)
    causal_graph: Optional[Any] = None
    compressed_state: Optional[Any] = None
    health: Optional[Any] = None
    summary: str = ""
    priority: str = "LOW"


@dataclass
class StubPolicyVector:
    """Matches PolicyVector protocol."""
    anomaly_weight: float = 0.25
    regime_weight: float = 0.25
    stability_weight: float = 0.25
    causality_weight: float = 0.25
    sensitivity: float = 0.5
    dominant_concern: str = "balanced"
    timestamp: float = 0.0


@dataclass
class StubDecisionContext:
    """Matches DecisionContext protocol."""
    regime_confidence: float = 0.5
    anomaly_weight: float = 0.5
    stability_bias: float = 0.0
    causal_priority_map: Dict[str, float] = field(default_factory=dict)
    resolved_tensions: List[str] = field(default_factory=list)
    timestamp: float = 0.0


# =========================================================================
# Tests: SystemDecision dataclass
# =========================================================================


class TestSystemDecision:
    """SystemDecision dataclass basics."""

    def test_defaults(self) -> None:
        sd = SystemDecision(
            action_tendency=ActionTendency.HOLD,
            risk_bias=0.0,
            regime_action_signal=RegimeActionSignal.MAINTAIN,
            confidence=0.5,
        )
        assert sd.action_tendency == ActionTendency.HOLD
        assert sd.risk_bias == 0.0
        assert sd.regime_action_signal == RegimeActionSignal.MAINTAIN
        assert sd.confidence == 0.5
        assert sd.components == {}
        assert sd.reasoning == []
        assert sd.timestamp == 0.0

    def test_fields(self) -> None:
        sd = SystemDecision(
            action_tendency=ActionTendency.STRONG_BUY,
            risk_bias=0.8,
            regime_action_signal=RegimeActionSignal.ESCALATE,
            confidence=0.9,
            components={"buy_score": 0.8, "net": 0.7},
            reasoning=["Regime strong", "Health good"],
            timestamp=100.0,
        )
        assert sd.action_tendency == ActionTendency.STRONG_BUY
        assert sd.risk_bias == 0.8
        assert sd.regime_action_signal == RegimeActionSignal.ESCALATE
        assert sd.confidence == 0.9
        assert sd.components["buy_score"] == 0.8
        assert len(sd.reasoning) == 2


# =========================================================================
# Tests: ActionTendency and RegimeActionSignal enums
# =========================================================================


class TestEnums:
    """Enum values match the spec."""

    def test_action_tendency_values(self) -> None:
        assert ActionTendency.STRONG_BUY.value == "STRONG_BUY"
        assert ActionTendency.BUY.value == "BUY"
        assert ActionTendency.HOLD.value == "HOLD"
        assert ActionTendency.REDUCE.value == "REDUCE"
        assert ActionTendency.EXIT.value == "EXIT"
        assert ActionTendency.STRONG_SELL.value == "STRONG_SELL"

    def test_regime_action_signal_values(self) -> None:
        assert RegimeActionSignal.ESCALATE.value == "ESCALATE"
        assert RegimeActionSignal.MAINTAIN.value == "MAINTAIN"
        assert RegimeActionSignal.DE_ESCALATE.value == "DE_ESCALATE"
        assert RegimeActionSignal.PREPARE_TRANSITION.value == "PREPARE_TRANSITION"
        assert RegimeActionSignal.EMERGENCY_STOP.value == "EMERGENCY_STOP"


# =========================================================================
# Tests: DecisionSynthesizer basics
# =========================================================================


class TestDecisionSynthesizerBasics:
    """Instantiation, feeding, history."""

    def test_instantiate(self) -> None:
        synth = DecisionSynthesizer()
        assert synth is not None
        assert synth.get_history() == []
        assert synth.get_history(n=5) == []

    def test_feed_intelligence(self) -> None:
        synth = DecisionSynthesizer()
        frame = StubIntelligenceFrame(priority="LOW")
        synth.feed_intelligence(frame)
        # No direct state access, but shouldn't crash

    def test_feed_policy(self) -> None:
        synth = DecisionSynthesizer()
        policy = StubPolicyVector()
        synth.feed_policy(policy)

    def test_feed_context(self) -> None:
        synth = DecisionSynthesizer()
        ctx = StubDecisionContext()
        synth.feed_context(ctx)

    def test_synthesize_without_inputs(self) -> None:
        """Synthesize with no inputs should produce a HOLD/MAINTAIN decision."""
        synth = DecisionSynthesizer()
        decision = synth.synthesize()
        assert isinstance(decision, SystemDecision)
        assert decision.action_tendency == ActionTendency.HOLD
        assert decision.regime_action_signal == RegimeActionSignal.MAINTAIN
        assert decision.timestamp > 0

    def test_history_after_synthesize(self) -> None:
        synth = DecisionSynthesizer()
        decision = synth.synthesize()
        history = synth.get_history()
        assert len(history) == 1
        assert history[0] is decision

    def test_history_limit(self) -> None:
        synth = DecisionSynthesizer()
        for _ in range(20):
            synth.synthesize()
        assert len(synth.get_history()) == 10  # default n=10
        assert len(synth.get_history(n=5)) == 5
        assert len(synth.get_history(n=30)) == 20  # capped at available


# =========================================================================
# Tests: Full synthesis scenarios
# =========================================================================


class TestSynthesisScenarios:
    """End-to-end synthesis with various input configurations."""

    # ── Scenario: Strong Buy ────────────────────────────────────────────────

    def test_strong_buy_scenario(self) -> None:
        """FULL regime with high probability, healthy, no anomalies → STRONG_BUY."""
        synth = DecisionSynthesizer()
        frame = StubIntelligenceFrame(
            regime=StubTransitionSignal(from_regime="MICRO", to_regime="FULL", probability=0.85),
            health=StubSystemHealthScore(score=0.7),
            anomalies=[],
            priority="LOW",
        )
        policy = StubPolicyVector()
        ctx = StubDecisionContext()

        synth.feed_intelligence(frame)
        synth.feed_policy(policy)
        synth.feed_context(ctx)
        decision = synth.synthesize()

        assert decision.action_tendency == ActionTendency.STRONG_BUY, (
            f"Expected STRONG_BUY got {decision.action_tendency}"
        )
        assert decision.regime_action_signal == RegimeActionSignal.ESCALATE
        assert decision.confidence > 0.6
        assert decision.risk_bias > 0.0
        assert len(decision.reasoning) >= 3

    # ── Scenario: Strong Sell ───────────────────────────────────────────────

    def test_strong_sell_scenario(self) -> None:
        """SHADOW regime, bad health, critical anomalies → STRONG_SELL."""
        synth = DecisionSynthesizer()
        frame = StubIntelligenceFrame(
            regime=StubTransitionSignal(from_regime="FULL", to_regime="SHADOW", probability=0.9),
            health=StubSystemHealthScore(score=-0.8),
            anomalies=[
                StubAnomalyEvent(severity="CRITICAL", description="System failure imminent"),
                StubAnomalyEvent(severity="HIGH", description="Engine vector divergence"),
            ],
            priority="HIGH",
        )
        policy = StubPolicyVector()
        ctx = StubDecisionContext()

        synth.feed_intelligence(frame)
        synth.feed_policy(policy)
        synth.feed_context(ctx)
        decision = synth.synthesize()

        assert decision.action_tendency == ActionTendency.STRONG_SELL, (
            f"Expected STRONG_SELL got {decision.action_tendency}"
        )
        assert decision.regime_action_signal == RegimeActionSignal.DE_ESCALATE
        assert decision.risk_bias < -0.3
        assert decision.components["sell_score"] >= decision.components["buy_score"]

    # ── Scenario: Emergency Stop ────────────────────────────────────────────

    def test_emergency_stop(self) -> None:
        """CRITICAL priority → EMERGENCY_STOP regardless of regime."""
        synth = DecisionSynthesizer()
        frame = StubIntelligenceFrame(
            regime=StubTransitionSignal(from_regime="MICRO", to_regime="FULL", probability=0.9),
            health=StubSystemHealthScore(score=0.8),
            anomalies=[],
            priority="CRITICAL",
        )
        policy = StubPolicyVector()
        ctx = StubDecisionContext()

        synth.feed_intelligence(frame)
        synth.feed_policy(policy)
        synth.feed_context(ctx)
        decision = synth.synthesize()

        assert decision.regime_action_signal == RegimeActionSignal.EMERGENCY_STOP, (
            f"Expected EMERGENCY_STOP got {decision.regime_action_signal}"
        )
        assert "EMERGENCY_STOP" in " ".join(decision.reasoning)

    # ── Scenario: Prepare Transition ────────────────────────────────────────

    def test_prepare_transition(self) -> None:
        """MICRO regime with sufficient probability → PREPARE_TRANSITION."""
        synth = DecisionSynthesizer()
        frame = StubIntelligenceFrame(
            regime=StubTransitionSignal(from_regime="SHADOW", to_regime="MICRO", probability=0.7),
            health=StubSystemHealthScore(score=0.0),
            anomalies=[],
            priority="LOW",
        )
        policy = StubPolicyVector()
        ctx = StubDecisionContext()

        synth.feed_intelligence(frame)
        synth.feed_policy(policy)
        synth.feed_context(ctx)
        decision = synth.synthesize()

        assert decision.regime_action_signal == RegimeActionSignal.PREPARE_TRANSITION
        assert "PREPARE_TRANSITION" in " ".join(decision.reasoning)

    # ── Scenario: De-escalate ───────────────────────────────────────────────

    def test_de_escalate(self) -> None:
        """SHADOW regime with sufficient probability → DE_ESCALATE."""
        synth = DecisionSynthesizer()
        frame = StubIntelligenceFrame(
            regime=StubTransitionSignal(from_regime="MICRO", to_regime="SHADOW", probability=0.65),
            health=StubSystemHealthScore(score=0.1),
            anomalies=[],
            priority="LOW",
        )
        policy = StubPolicyVector()
        ctx = StubDecisionContext()

        synth.feed_intelligence(frame)
        synth.feed_policy(policy)
        synth.feed_context(ctx)
        decision = synth.synthesize()

        assert decision.regime_action_signal == RegimeActionSignal.DE_ESCALATE

    # ── Scenario: Maintain ──────────────────────────────────────────────────

    def test_maintain_no_regime(self) -> None:
        """No regime signal → MAINTAIN."""
        synth = DecisionSynthesizer()
        frame = StubIntelligenceFrame(
            regime=None,
            health=StubSystemHealthScore(score=0.0),
            anomalies=[],
            priority="LOW",
        )
        policy = StubPolicyVector()
        ctx = StubDecisionContext()

        synth.feed_intelligence(frame)
        synth.feed_policy(policy)
        synth.feed_context(ctx)
        decision = synth.synthesize()

        assert decision.regime_action_signal == RegimeActionSignal.MAINTAIN

    # ── Scenario: Hold ──────────────────────────────────────────────────────

    def test_hold_neutral(self) -> None:
        """Neutral signals → HOLD."""
        synth = DecisionSynthesizer()
        frame = StubIntelligenceFrame(
            regime=StubTransitionSignal(from_regime="SHADOW", to_regime="MICRO", probability=0.3),
            health=StubSystemHealthScore(score=0.0),
            anomalies=[],
            priority="LOW",
        )
        policy = StubPolicyVector()
        ctx = StubDecisionContext()

        synth.feed_intelligence(frame)
        synth.feed_policy(policy)
        synth.feed_context(ctx)
        decision = synth.synthesize()

        assert decision.action_tendency == ActionTendency.HOLD

    # ── Scenario: Reduce ────────────────────────────────────────────────────

    def test_reduce(self) -> None:
        """Moderately negative signals → REDUCE.

        A HIGH anomaly (sell +0.3) with no other signals gives:
          buy_score = 0 (no +0.2 from 'no HIGH/CRITICAL'),
          sell_score = 0.3, net = -0.3 → REDUCE.
        """
        synth = DecisionSynthesizer()
        frame = StubIntelligenceFrame(
            regime=None,
            health=StubSystemHealthScore(score=0.0),  # no sell from health
            anomalies=[
                StubAnomalyEvent(severity="HIGH", description="Moderate anomaly"),
            ],
            priority="LOW",
        )
        policy = StubPolicyVector()
        ctx = StubDecisionContext()

        synth.feed_intelligence(frame)
        synth.feed_policy(policy)
        synth.feed_context(ctx)
        decision = synth.synthesize()

        assert decision.action_tendency == ActionTendency.REDUCE, (
            f"Expected REDUCE got {decision.action_tendency}"
        )

    # ── Scenario: Exit ──────────────────────────────────────────────────────

    def test_exit(self) -> None:
        """Strongly negative signals → EXIT.

        SHADOW regime (sell +0.4) + HIGH anomaly (sell +0.3, removes +0.2 buy):
          buy = 0.0, sell = 0.7, net = -0.7 → EXIT.
        """
        synth = DecisionSynthesizer()
        frame = StubIntelligenceFrame(
            regime=StubTransitionSignal(from_regime="FULL", to_regime="SHADOW", probability=0.7),
            health=StubSystemHealthScore(score=0.0),
            anomalies=[
                StubAnomalyEvent(severity="HIGH", description="Engine anomaly"),
            ],
            priority="HIGH",
        )
        policy = StubPolicyVector()
        ctx = StubDecisionContext()

        synth.feed_intelligence(frame)
        synth.feed_policy(policy)
        synth.feed_context(ctx)
        decision = synth.synthesize()

        assert decision.action_tendency == ActionTendency.EXIT, (
            f"Expected EXIT got {decision.action_tendency}"
        )


# =========================================================================
# Tests: Policy weighting
# =========================================================================


class TestPolicyWeighting:
    """PolicyVector weights modulate signal importance."""

    def test_weights_modulate_action(self) -> None:
        """Higher anomaly_weight should make the same anomaly more impactful."""
        synth = DecisionSynthesizer()

        # Same frame with mild anomalies
        frame = StubIntelligenceFrame(
            regime=StubTransitionSignal(from_regime="FULL", to_regime="SHADOW", probability=0.5),
            health=StubSystemHealthScore(score=-0.2),
            anomalies=[
                StubAnomalyEvent(severity="MEDIUM", description="Minor anomaly"),
            ],
            priority="LOW",
        )

        # Policy with high anomaly weight
        policy_high_anomaly = StubPolicyVector(
            anomaly_weight=1.0,
            regime_weight=0.0,
            stability_weight=0.0,
        )

        synth.feed_intelligence(frame)
        synth.feed_policy(policy_high_anomaly)
        synth.feed_context(StubDecisionContext())
        decision_high = synth.synthesize()

        # Reset and try with low anomaly weight
        synth2 = DecisionSynthesizer()
        policy_low_anomaly = StubPolicyVector(
            anomaly_weight=0.0,
            regime_weight=0.0,
            stability_weight=0.0,
        )
        synth2.feed_intelligence(frame)
        synth2.feed_policy(policy_low_anomaly)
        synth2.feed_context(StubDecisionContext())
        decision_low = synth2.synthesize()

        # High anomaly weight should produce different (more negative) result
        assert decision_high.components["weighted_anomaly"] > decision_low.components["weighted_anomaly"]


# =========================================================================
# Tests: Risk bias calculation
# =========================================================================


class TestRiskBias:
    """Risk bias combines health, anomaly, and regime contributions."""

    def test_risk_bias_healthy_full(self) -> None:
        """Healthy system trending to FULL → positive risk bias."""
        synth = DecisionSynthesizer()
        frame = StubIntelligenceFrame(
            regime=StubTransitionSignal(from_regime="MICRO", to_regime="FULL", probability=0.8),
            health=StubSystemHealthScore(score=0.8),
            anomalies=[],
            priority="LOW",
        )
        synth.feed_intelligence(frame)
        synth.feed_policy(StubPolicyVector())
        synth.feed_context(StubDecisionContext())
        decision = synth.synthesize()

        # health_contrib = 0.8 * 0.4 = 0.32, regime_contrib = 0.3
        # total ≈ 0.62 (clamped OK)
        assert decision.risk_bias > 0.3

    def test_risk_bias_unhealthy_shadow(self) -> None:
        """Unhealthy system trending to SHADOW → negative risk bias."""
        synth = DecisionSynthesizer()
        frame = StubIntelligenceFrame(
            regime=StubTransitionSignal(from_regime="FULL", to_regime="SHADOW", probability=0.8),
            health=StubSystemHealthScore(score=-0.6),
            anomalies=[],
            priority="LOW",
        )
        synth.feed_intelligence(frame)
        synth.feed_policy(StubPolicyVector())
        synth.feed_context(StubDecisionContext())
        decision = synth.synthesize()

        # health_contrib = -0.6 * 0.4 = -0.24, regime_contrib = -0.3
        # total ≈ -0.54
        assert decision.risk_bias < -0.3

    def test_risk_bias_critical_anomalies(self) -> None:
        """Critical anomalies heavily penalise risk bias."""
        synth = DecisionSynthesizer()
        frame = StubIntelligenceFrame(
            regime=StubTransitionSignal(from_regime="SHADOW", to_regime="MICRO", probability=0.5),
            health=StubSystemHealthScore(score=0.3),
            anomalies=[
                StubAnomalyEvent(severity="CRITICAL", description="Critical failure"),
                StubAnomalyEvent(severity="HIGH", description="High anomaly"),
            ],
            priority="HIGH",
        )
        synth.feed_intelligence(frame)
        synth.feed_policy(StubPolicyVector())
        synth.feed_context(StubDecisionContext())
        decision = synth.synthesize()

        # anomaly_contrib = -0.5 (CRITICAL) + -0.3 (HIGH) = -0.8
        assert decision.risk_bias < -0.3

    def test_risk_bias_clamping(self) -> None:
        """Risk bias is clamped to [-1.0, 1.0]."""
        synth = DecisionSynthesizer()
        # Extreme values should clamp
        frame = StubIntelligenceFrame(
            regime=StubTransitionSignal(from_regime="SHADOW", to_regime="FULL", probability=0.9),
            health=StubSystemHealthScore(score=5.0),  # Unrealistically high
            anomalies=[],
            priority="LOW",
        )
        synth.feed_intelligence(frame)
        synth.feed_policy(StubPolicyVector())
        synth.feed_context(StubDecisionContext())
        decision = synth.synthesize()

        assert -1.0 <= decision.risk_bias <= 1.0

        # Also test very negative
        synth2 = DecisionSynthesizer()
        frame2 = StubIntelligenceFrame(
            regime=StubTransitionSignal(from_regime="FULL", to_regime="SHADOW", probability=0.9),
            health=StubSystemHealthScore(score=-5.0),
            anomalies=[
                StubAnomalyEvent(severity="CRITICAL", description="Fail"),
            ],
            priority="CRITICAL",
        )
        synth2.feed_intelligence(frame2)
        synth2.feed_policy(StubPolicyVector())
        synth2.feed_context(StubDecisionContext())
        decision2 = synth2.synthesize()

        assert -1.0 <= decision2.risk_bias <= 1.0


# =========================================================================
# Tests: Reasoning chain
# =========================================================================


class TestReasoning:
    """Decision synthesizer produces human-readable reasoning."""

    def test_reasoning_contains_signal_descriptions(self) -> None:
        """Reasoning chain should describe the key signals."""
        synth = DecisionSynthesizer()
        frame = StubIntelligenceFrame(
            regime=StubTransitionSignal(from_regime="MICRO", to_regime="FULL", probability=0.85),
            health=StubSystemHealthScore(score=0.7),
            anomalies=[],
            priority="LOW",
        )
        synth.feed_intelligence(frame)
        synth.feed_policy(StubPolicyVector())
        synth.feed_context(StubDecisionContext())
        decision = synth.synthesize()

        full_reasoning = " ".join(decision.reasoning)
        # Should mention regime transition
        assert "FULL" in full_reasoning
        # Should mention health
        assert "health" in full_reasoning.lower() or "Health" in full_reasoning
        # Should mention ESCALATE
        assert "ESCALATE" in full_reasoning

    def test_reasoning_empty_with_no_inputs(self) -> None:
        """Even with no inputs, reasoning should be present."""
        synth = DecisionSynthesizer()
        decision = synth.synthesize()
        assert len(decision.reasoning) > 0


# =========================================================================
# Tests: Component transparency
# =========================================================================


class TestComponents:
    """Decision components provide sub-score transparency."""

    def test_components_contains_all_keys(self) -> None:
        synth = DecisionSynthesizer()
        frame = StubIntelligenceFrame(
            regime=StubTransitionSignal(from_regime="SHADOW", to_regime="FULL", probability=0.8),
            health=StubSystemHealthScore(score=0.5),
            anomalies=[],
            priority="LOW",
        )
        synth.feed_intelligence(frame)
        synth.feed_policy(StubPolicyVector())
        synth.feed_context(StubDecisionContext())
        decision = synth.synthesize()

        # Check for expected component keys
        expected = {"buy_score", "sell_score", "net_score", "health_score",
                     "regime_probability", "anomaly_severity_value",
                     "weighted_anomaly", "weighted_regime", "weighted_stability"}
        for key in expected:
            assert key in decision.components, f"Missing component: {key}"

    def test_buy_sell_score_mutual_exclusivity(self) -> None:
        """Buy and sell scores should reflect opposite scenarios."""
        # Buy scenario
        synth_buy = DecisionSynthesizer()
        frame_buy = StubIntelligenceFrame(
            regime=StubTransitionSignal(from_regime="SHADOW", to_regime="FULL", probability=0.9),
            health=StubSystemHealthScore(score=0.8),
            anomalies=[],
            priority="LOW",
        )
        synth_buy.feed_intelligence(frame_buy)
        synth_buy.feed_policy(StubPolicyVector())
        synth_buy.feed_context(StubDecisionContext())
        dec_buy = synth_buy.synthesize()

        # Sell scenario
        synth_sell = DecisionSynthesizer()
        frame_sell = StubIntelligenceFrame(
            regime=StubTransitionSignal(from_regime="FULL", to_regime="SHADOW", probability=0.9),
            health=StubSystemHealthScore(score=-0.8),
            anomalies=[StubAnomalyEvent(severity="CRITICAL", description="Fail")],
            priority="HIGH",
        )
        synth_sell.feed_intelligence(frame_sell)
        synth_sell.feed_policy(StubPolicyVector())
        synth_sell.feed_context(StubDecisionContext())
        dec_sell = synth_sell.synthesize()

        assert dec_buy.components["buy_score"] > dec_buy.components["sell_score"]
        assert dec_sell.components["sell_score"] > dec_sell.components["buy_score"]


# =========================================================================
# Tests: Duck-typed edge cases
# =========================================================================


class TestDuckTyping:
    """Handles partially-initialised or unusual inputs gracefully."""

    def test_frame_with_only_priority(self) -> None:
        """Frame with only priority set should not crash."""
        synth = DecisionSynthesizer()
        frame = StubIntelligenceFrame(priority="HIGH")
        synth.feed_intelligence(frame)
        decision = synth.synthesize()
        assert isinstance(decision, SystemDecision)

    def test_frame_with_only_health(self) -> None:
        """Frame with only health set should not crash."""
        synth = DecisionSynthesizer()
        frame = StubIntelligenceFrame(
            health=StubSystemHealthScore(score=-0.5),
        )
        synth.feed_intelligence(frame)
        decision = synth.synthesize()
        assert isinstance(decision, SystemDecision)

    def test_frame_with_only_anomalies(self) -> None:
        """Frame with only anomalies set should not crash."""
        synth = DecisionSynthesizer()
        frame = StubIntelligenceFrame(
            anomalies=[StubAnomalyEvent(severity="HIGH")],
        )
        synth.feed_intelligence(frame)
        decision = synth.synthesize()
        assert isinstance(decision, SystemDecision)

    def test_empty_anomalies_list(self) -> None:
        """Empty anomalies list should be handled."""
        synth = DecisionSynthesizer()
        frame = StubIntelligenceFrame(
            regime=StubTransitionSignal(to_regime="FULL", probability=0.8),
            health=StubSystemHealthScore(score=0.5),
            anomalies=[],
            priority="LOW",
        )
        synth.feed_intelligence(frame)
        decision = synth.synthesize()
        assert decision.action_tendency in (
            ActionTendency.STRONG_BUY, ActionTendency.BUY, ActionTendency.HOLD
        )

    def test_policy_is_none(self) -> None:
        """None policy should fall back to defaults."""
        synth = DecisionSynthesizer()
        frame = StubIntelligenceFrame(
            health=StubSystemHealthScore(score=0.5),
        )
        synth.feed_intelligence(frame)
        # Don't feed policy
        decision = synth.synthesize()
        assert isinstance(decision, SystemDecision)

    def test_context_is_none(self) -> None:
        """None context should fall back to defaults."""
        synth = DecisionSynthesizer()
        frame = StubIntelligenceFrame(
            health=StubSystemHealthScore(score=0.5),
        )
        synth.feed_intelligence(frame)
        synth.feed_policy(StubPolicyVector())
        # Don't feed context
        decision = synth.synthesize()
        assert isinstance(decision, SystemDecision)


# =========================================================================
# Tests: Consecutive synthesis (regime tracking)
# =========================================================================


class TestConsecutiveSynthesis:
    """Multiple synthesize() calls accumulate history and regime tracking."""

    def test_history_accumulates(self) -> None:
        synth = DecisionSynthesizer()
        for _ in range(5):
            synth.synthesize()
        assert len(synth.get_history()) == 5
        assert len(synth.get_history(n=3)) == 3

    def test_regime_transitions_tracked(self) -> None:
        """Consecutive regime signals are tracked for oscillation detection."""
        synth = DecisionSynthesizer()

        # Feed a series of regime transitions
        frames = [
            StubIntelligenceFrame(
                regime=StubTransitionSignal(from_regime="SHADOW", to_regime="MICRO", probability=0.8),
            ),
            StubIntelligenceFrame(
                regime=StubTransitionSignal(from_regime="MICRO", to_regime="FULL", probability=0.8),
            ),
            StubIntelligenceFrame(
                regime=StubTransitionSignal(from_regime="FULL", to_regime="SHADOW", probability=0.8),
            ),
        ]

        for frame in frames:
            synth.feed_intelligence(frame)
            synth.feed_policy(StubPolicyVector())
            synth.feed_context(StubDecisionContext())
            synth.synthesize()

        # Should have 3 decisions in history
        assert len(synth.get_history()) == 3


# =========================================================================
# Tests: Confidence calculation
# =========================================================================


class TestConfidence:
    """Confidence reflects the strongest signal."""

    def test_confidence_high_with_strong_signals(self) -> None:
        """Strong buy signals → high confidence."""
        synth = DecisionSynthesizer()
        frame = StubIntelligenceFrame(
            regime=StubTransitionSignal(from_regime="SHADOW", to_regime="FULL", probability=0.95),
            health=StubSystemHealthScore(score=0.9),
            anomalies=[],
            priority="LOW",
        )
        synth.feed_intelligence(frame)
        synth.feed_policy(StubPolicyVector())
        synth.feed_context(StubDecisionContext())
        decision = synth.synthesize()

        assert decision.confidence >= 0.6

    def test_confidence_low_with_weak_signals(self) -> None:
        """Weak neutral signals → low confidence.

        Confidence = max(buy_score, sell_score). With no inputs,
        buy_score = 0.2 (no HIGH/CRITICAL anomalies) → confidence = 0.2.
        """
        synth = DecisionSynthesizer()
        frame = StubIntelligenceFrame(
            regime=None,
            health=StubSystemHealthScore(score=0.0),
            anomalies=[],
            priority="LOW",
        )
        synth.feed_intelligence(frame)
        synth.feed_policy(StubPolicyVector())
        synth.feed_context(StubDecisionContext())
        decision = synth.synthesize()

        # No strong buy or sell signals → low confidence (0.2 from default buy boost)
        assert decision.confidence == 0.2
