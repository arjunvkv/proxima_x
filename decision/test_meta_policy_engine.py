"""
Tests for MetaPolicyEngine — dynamic policy weighting for intelligence subsystems.

Covers default policy, all 4 adjustment rules, normalisation, decay smoothing,
sensitivity clamping, and edge cases with duck-typed inputs.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from meta_policy_engine import MetaPolicyEngine, PolicyVector


# ---------------------------------------------------------------------------
# Helper: minimal duck-typed stubs matching IntelligenceFrame & DecisionContext
# ---------------------------------------------------------------------------


@dataclass
class StubTransitionSignal:
    probability: float = 0.0
    confidence: float = 0.0
    from_regime: str = "SHADOW"
    to_regime: str = "MICRO"


@dataclass
class StubSystemHealthScore:
    score: float = 0.0
    state: str = "HEALTHY"
    components: Dict[str, float] = field(default_factory=dict)


@dataclass
class StubIntelligenceFrame:
    priority: str = "LOW"
    regime: Any = None
    health: Any = None
    timestamp: float = 0.0


@dataclass
class StubDecisionContext:
    stability_bias: float = 0.0
    timestamp: float = 0.0


# =========================================================================
# Tests
# =========================================================================


class TestPolicyVector:
    """PolicyVector dataclass basics."""

    def test_defaults(self) -> None:
        pv = PolicyVector()
        assert pv.anomaly_weight == 0.25
        assert pv.regime_weight == 0.25
        assert pv.stability_weight == 0.25
        assert pv.causality_weight == 0.25
        assert pv.sensitivity == 0.5
        assert pv.dominant_concern == "balanced"
        assert pv.timestamp == 0.0

    def test_fields(self) -> None:
        pv = PolicyVector(
            anomaly_weight=0.4,
            regime_weight=0.3,
            stability_weight=0.2,
            causality_weight=0.1,
            sensitivity=0.8,
            dominant_concern="anomaly",
            timestamp=123.0,
        )
        assert pv.anomaly_weight == 0.4
        assert pv.regime_weight == 0.3
        assert pv.stability_weight == 0.2
        assert pv.causality_weight == 0.1
        assert pv.sensitivity == 0.8
        assert pv.dominant_concern == "anomaly"
        assert pv.timestamp == 123.0


class TestMetaPolicyEngineInit:
    """Constructor and basic state."""

    def test_default_decay(self) -> None:
        mpe = MetaPolicyEngine()
        assert mpe._decay_factor == 0.95
        assert mpe._latest_frame is None
        assert mpe._latest_context is None
        assert mpe._previous is None
        assert mpe._history == []

    def test_custom_decay(self) -> None:
        mpe = MetaPolicyEngine(decay_factor=0.5)
        assert mpe._decay_factor == 0.5


class TestDefaultPolicy:
    """Policy when no signals have been fed."""

    def test_no_inputs(self) -> None:
        mpe = MetaPolicyEngine()
        pv = mpe.compute_policy()
        # Default weights should be equal
        assert pv.anomaly_weight == pytest.approx(0.25)
        assert pv.regime_weight == pytest.approx(0.25)
        assert pv.stability_weight == pytest.approx(0.25)
        assert pv.causality_weight == pytest.approx(0.25)
        assert pv.sensitivity == 0.5
        assert pv.dominant_concern == "balanced"

    def test_no_inputs_sum_to_one(self) -> None:
        mpe = MetaPolicyEngine()
        pv = mpe.compute_policy()
        total = pv.anomaly_weight + pv.regime_weight + pv.stability_weight + pv.causality_weight
        assert total == pytest.approx(1.0)

    def test_no_inputs_sensitivity_clamped(self) -> None:
        mpe = MetaPolicyEngine()
        pv = mpe.compute_policy()
        assert 0.1 <= pv.sensitivity <= 1.0


class TestRule1AnomalySensitivity:
    """Rule 1: Anomaly sensitivity based on frame priority."""

    def test_low_priority_default(self) -> None:
        mpe = MetaPolicyEngine()
        frame = StubIntelligenceFrame(priority="LOW")
        mpe.feed(frame)
        pv = mpe.compute_policy()
        assert pv.dominant_concern == "balanced"
        assert pv.anomaly_weight == pytest.approx(0.25, abs=0.01)

    def test_high_priority(self) -> None:
        mpe = MetaPolicyEngine()
        frame = StubIntelligenceFrame(priority="HIGH")
        mpe.feed(frame)
        pv = mpe.compute_policy()
        # Raw target: anomaly=0.35, others=0.25 → total=1.10
        # Normalised anomaly = 0.35 / 1.10 ≈ 0.318
        assert pv.anomaly_weight == pytest.approx(0.318, abs=0.005)
        assert pv.sensitivity == 0.7
        assert pv.dominant_concern == "anomaly"

    def test_critical_priority(self) -> None:
        mpe = MetaPolicyEngine()
        frame = StubIntelligenceFrame(priority="CRITICAL")
        mpe.feed(frame)
        pv = mpe.compute_policy()
        # Raw target: anomaly=0.45, others=0.25 → total=1.20
        # Normalised anomaly = 0.45 / 1.20 = 0.375
        assert pv.anomaly_weight == pytest.approx(0.375)
        assert pv.sensitivity == 0.9
        assert pv.dominant_concern == "anomaly"

    def test_high_priority_normalised(self) -> None:
        """Weights should still sum to 1.0 after adjustment."""
        mpe = MetaPolicyEngine()
        frame = StubIntelligenceFrame(priority="HIGH")
        mpe.feed(frame)
        pv = mpe.compute_policy()
        total = pv.anomaly_weight + pv.regime_weight + pv.stability_weight + pv.causality_weight
        assert total == pytest.approx(1.0)

    def test_critical_priority_normalised(self) -> None:
        mpe = MetaPolicyEngine()
        frame = StubIntelligenceFrame(priority="CRITICAL")
        mpe.feed(frame)
        pv = mpe.compute_policy()
        total = pv.anomaly_weight + pv.regime_weight + pv.stability_weight + pv.causality_weight
        assert total == pytest.approx(1.0)


class TestRule2RegimeTransition:
    """Rule 2: Regime transition focus."""

    def test_regime_probability_high(self) -> None:
        """Probability > 0.6 should boost regime_weight."""
        mpe = MetaPolicyEngine()
        regime = StubTransitionSignal(probability=0.7)
        frame = StubIntelligenceFrame(regime=regime)
        mpe.feed(frame)
        pv = mpe.compute_policy()
        # Raw target: regime=0.40, stability=0.30, others=0.25 → total=1.20
        # Normalised regime = 0.40 / 1.20 ≈ 0.333, stability = 0.30 / 1.20 = 0.25
        assert pv.regime_weight == pytest.approx(0.333, abs=0.005)
        assert pv.stability_weight == pytest.approx(0.25, abs=0.005)
        assert pv.dominant_concern == "regime"

    def test_regime_probability_very_high(self) -> None:
        """Probability > 0.8 should boost regime_weight further."""
        mpe = MetaPolicyEngine()
        regime = StubTransitionSignal(probability=0.9)
        frame = StubIntelligenceFrame(regime=regime)
        mpe.feed(frame)
        pv = mpe.compute_policy()
        # Raw target: regime=0.50, others=0.25 each → total=1.25
        # Normalised regime = 0.50 / 1.25 = 0.4
        assert pv.regime_weight == pytest.approx(0.4)
        assert pv.sensitivity == 0.8
        assert pv.dominant_concern == "regime"

    def test_regime_probability_boundary_06(self) -> None:
        """Probability exactly 0.6 should NOT trigger (must be > 0.6)."""
        mpe = MetaPolicyEngine()
        regime = StubTransitionSignal(probability=0.6)
        frame = StubIntelligenceFrame(regime=regime)
        mpe.feed(frame)
        pv = mpe.compute_policy()
        assert pv.dominant_concern == "balanced"
        # Weights should remain default (~0.25 each after normalisation)
        for w in (pv.anomaly_weight, pv.regime_weight, pv.stability_weight, pv.causality_weight):
            assert w == pytest.approx(0.25, abs=0.02)

    def test_regime_probability_boundary_08(self) -> None:
        """Probability exactly 0.8 should NOT trigger >0.8 branch (must be > 0.8)."""
        mpe = MetaPolicyEngine()
        regime = StubTransitionSignal(probability=0.8)
        frame = StubIntelligenceFrame(regime=regime)
        mpe.feed(frame)
        pv = mpe.compute_policy()
        # Should use the >0.6 branch
        # Raw target: regime=0.40, stability=0.30, others=0.25 → total=1.20
        # Normalised regime = 0.40 / 1.20 ≈ 0.333
        assert pv.regime_weight == pytest.approx(0.333, abs=0.005)
        assert pv.dominant_concern == "regime"

    def test_no_regime(self) -> None:
        mpe = MetaPolicyEngine()
        frame = StubIntelligenceFrame(regime=None)
        mpe.feed(frame)
        pv = mpe.compute_policy()
        assert pv.regime_weight == pytest.approx(0.25, abs=0.02)


class TestRule3HealthDriven:
    """Rule 3: Health-driven adjustments."""

    def test_health_low(self) -> None:
        """Health score < -0.5 should boost stability."""
        mpe = MetaPolicyEngine()
        health = StubSystemHealthScore(score=-0.7)
        frame = StubIntelligenceFrame(health=health)
        mpe.feed(frame)
        pv = mpe.compute_policy()
        # Raw target: stability=0.40, others=0.25 each → total=1.15
        # Normalised stability = 0.40 / 1.15 ≈ 0.348
        assert pv.stability_weight == pytest.approx(0.348, abs=0.005)
        assert pv.sensitivity == 0.85
        assert pv.dominant_concern == "stability"

    def test_health_very_healthy(self) -> None:
        """Health score > 0.7 should boost causality (exploratory mode)."""
        mpe = MetaPolicyEngine()
        health = StubSystemHealthScore(score=0.9)
        frame = StubIntelligenceFrame(health=health)
        mpe.feed(frame)
        pv = mpe.compute_policy()
        # Raw target: causality=0.40, others=0.25 each → total=1.15
        # Normalised causality = 0.40 / 1.15 ≈ 0.348
        assert pv.causality_weight == pytest.approx(0.348, abs=0.005)
        assert pv.dominant_concern == "causality"

    def test_health_boundary_minus_05(self) -> None:
        """Score exactly -0.5 should NOT trigger (must be < -0.5)."""
        mpe = MetaPolicyEngine()
        health = StubSystemHealthScore(score=-0.5)
        frame = StubIntelligenceFrame(health=health)
        mpe.feed(frame)
        pv = mpe.compute_policy()
        assert pv.dominant_concern == "balanced"

    def test_health_boundary_07(self) -> None:
        """Score exactly 0.7 should NOT trigger (must be > 0.7)."""
        mpe = MetaPolicyEngine()
        health = StubSystemHealthScore(score=0.7)
        frame = StubIntelligenceFrame(health=health)
        mpe.feed(frame)
        pv = mpe.compute_policy()
        assert pv.dominant_concern == "balanced"

    def test_health_normal(self) -> None:
        """Health score in normal range should leave defaults."""
        mpe = MetaPolicyEngine()
        health = StubSystemHealthScore(score=0.0)
        frame = StubIntelligenceFrame(health=health)
        mpe.feed(frame)
        pv = mpe.compute_policy()
        assert pv.dominant_concern == "balanced"
        for w in (pv.anomaly_weight, pv.regime_weight, pv.stability_weight, pv.causality_weight):
            assert w == pytest.approx(0.25, abs=0.02)

    def test_no_health(self) -> None:
        mpe = MetaPolicyEngine()
        frame = StubIntelligenceFrame(health=None)
        mpe.feed(frame)
        pv = mpe.compute_policy()
        assert pv.dominant_concern == "balanced"


class TestRule4ConflictAware:
    """Rule 4: Conflict-aware adjustment from DecisionContext."""

    def test_oscillation_detected(self) -> None:
        """stability_bias < -0.3 should boost stability and discount regime."""
        mpe = MetaPolicyEngine()
        context = StubDecisionContext(stability_bias=-0.5)
        mpe.feed_decision_context(context)
        pv = mpe.compute_policy()
        # stability_weight += 0.15 → 0.25 + 0.15 = 0.40 (before norm)
        # regime_weight *= 0.8 → 0.25 * 0.8 = 0.20 (before norm)
        # After normalisation: stability should be higher than regime
        assert pv.stability_weight > pv.regime_weight

    def test_stable(self) -> None:
        """stability_bias > 0.3 should boost causality."""
        mpe = MetaPolicyEngine()
        context = StubDecisionContext(stability_bias=0.5)
        mpe.feed_decision_context(context)
        pv = mpe.compute_policy()
        # causality_weight += 0.10 → 0.25 + 0.10 = 0.35 (before norm)
        # After normalisation causality should still be highest
        max_weight = max(
            pv.anomaly_weight, pv.regime_weight, pv.stability_weight, pv.causality_weight
        )
        assert max_weight == pv.causality_weight

    def test_neutral_bias(self) -> None:
        """stability_bias between -0.3 and 0.3 should leave defaults."""
        mpe = MetaPolicyEngine()
        context = StubDecisionContext(stability_bias=0.0)
        mpe.feed_decision_context(context)
        pv = mpe.compute_policy()
        for w in (pv.anomaly_weight, pv.regime_weight, pv.stability_weight, pv.causality_weight):
            assert w == pytest.approx(0.25, abs=0.02)
        assert pv.dominant_concern == "balanced"

    def test_boundary_minus_03(self) -> None:
        """stability_bias exactly -0.3 should NOT trigger (must be < -0.3)."""
        mpe = MetaPolicyEngine()
        context = StubDecisionContext(stability_bias=-0.3)
        mpe.feed_decision_context(context)
        pv = mpe.compute_policy()
        assert pv.dominant_concern == "balanced"

    def test_boundary_03(self) -> None:
        """stability_bias exactly 0.3 should NOT trigger (must be > 0.3)."""
        mpe = MetaPolicyEngine()
        context = StubDecisionContext(stability_bias=0.3)
        mpe.feed_decision_context(context)
        pv = mpe.compute_policy()
        assert pv.dominant_concern == "balanced"


class TestDecaySmoothing:
    """Exponential decay smoothing across consecutive compute_policy calls."""

    def test_first_call_no_decay(self) -> None:
        """First call should return the raw target (no previous policy)."""
        mpe = MetaPolicyEngine()
        frame = StubIntelligenceFrame(priority="CRITICAL")
        mpe.feed(frame)
        pv = mpe.compute_policy()
        # Raw target: anomaly=0.45, others=0.25 each → total=1.20
        # Normalised anomaly = 0.45 / 1.20 = 0.375
        assert pv.anomaly_weight == pytest.approx(0.375)
        assert pv.sensitivity == 0.9

    def test_decay_approaches_target(self) -> None:
        """Repeated calls should converge toward target."""
        mpe = MetaPolicyEngine(decay_factor=0.5)
        frame = StubIntelligenceFrame(priority="HIGH")
        mpe.feed(frame)
        # First call: raw target normalised
        pv1 = mpe.compute_policy()
        # Raw target: anomaly=0.35, others=0.25 → total=1.10
        # Normalised anomaly = 0.35/1.10 ≈ 0.318
        expected_norm = 0.35 / 1.10
        assert pv1.anomaly_weight == pytest.approx(expected_norm)
        # Second call — decays halfway toward target again
        pv2 = mpe.compute_policy()
        # decay: prev * 0.5 + target * 0.5
        assert pv2.anomaly_weight == pytest.approx(expected_norm)
        # Third call — should converge toward the normalised target
        pv3 = mpe.compute_policy()
        # With decay=0.5, after 2nd call: decay toward same target, stays at target
        assert pv3.anomaly_weight == pytest.approx(expected_norm)

    def test_decay_from_default_to_target(self) -> None:
        """After a change in inputs, decay should smoothly transition."""
        mpe = MetaPolicyEngine(decay_factor=0.8)

        # First: normal policy (defaults)
        pv_default = mpe.compute_policy()
        assert pv_default.sensitivity == 0.5

        # Now feed CRITICAL priority
        frame = StubIntelligenceFrame(priority="CRITICAL")
        mpe.feed(frame)
        pv_critical = mpe.compute_policy()
        # target sensitivity = 0.9, prev = 0.5
        # smoothed = 0.5 * 0.8 + 0.9 * 0.2 = 0.4 + 0.18 = 0.58
        expected = 0.5 * 0.8 + 0.9 * 0.2
        assert pv_critical.sensitivity == pytest.approx(expected)

        # Third call: decay further toward 0.9
        pv_critical2 = mpe.compute_policy()
        expected2 = expected * 0.8 + 0.9 * 0.2
        assert pv_critical2.sensitivity == pytest.approx(expected2)


class TestSensitivityClamping:
    """Sensitivity clamping to [0.1, 1.0]."""

    def test_clamp_low(self) -> None:
        """Sensitivity should not drop below 0.1."""
        mpe = MetaPolicyEngine(decay_factor=0.0)  # no smoothing
        # Create a scenario where sensitivity would be very low
        # (no signals = 0.5 default, which is already within range)
        # The minimum possible from rules is 0.5, but after decay from 0.1 it could go lower
        # Let's force by feeding multiple contexts that don't touch sensitivity
        pv = mpe.compute_policy()
        assert pv.sensitivity >= 0.1

    def test_clamp_high(self) -> None:
        """Sensitivity should not exceed 1.0."""
        mpe = MetaPolicyEngine(decay_factor=0.0)
        frame = StubIntelligenceFrame(priority="CRITICAL")
        mpe.feed(frame)
        pv = mpe.compute_policy()
        # CRITICAL sets sensitivity to 0.9, well within range
        assert pv.sensitivity <= 1.0


class TestNormalisation:
    """Weight normalisation to sum 1.0."""

    def test_default_normalised(self) -> None:
        mpe = MetaPolicyEngine()
        pv = mpe.compute_policy()
        total = pv.anomaly_weight + pv.regime_weight + pv.stability_weight + pv.causality_weight
        assert total == pytest.approx(1.0)

    def test_multiple_rules_normalised(self) -> None:
        """Multiple rules together should still sum to 1.0."""
        mpe = MetaPolicyEngine()
        regime = StubTransitionSignal(probability=0.9)
        health = StubSystemHealthScore(score=-0.7)
        frame = StubIntelligenceFrame(priority="HIGH", regime=regime, health=health)
        mpe.feed(frame)
        context = StubDecisionContext(stability_bias=-0.5)
        mpe.feed_decision_context(context)
        pv = mpe.compute_policy()
        total = pv.anomaly_weight + pv.regime_weight + pv.stability_weight + pv.causality_weight
        assert total == pytest.approx(1.0)

    def test_all_weights_non_negative(self) -> None:
        """All weights should be >= 0 after normalisation."""
        mpe = MetaPolicyEngine()
        for _ in range(10):
            pv = mpe.compute_policy()
            assert pv.anomaly_weight >= 0
            assert pv.regime_weight >= 0
            assert pv.stability_weight >= 0
            assert pv.causality_weight >= 0


class TestWeightHistory:
    """Weight history tracking."""

    def test_history_empty_initially(self) -> None:
        mpe = MetaPolicyEngine()
        assert mpe.get_weight_history() == []

    def test_history_after_compute(self) -> None:
        mpe = MetaPolicyEngine()
        mpe.compute_policy()
        hist = mpe.get_weight_history()
        assert len(hist) == 1

    def test_history_multiple_entries(self) -> None:
        mpe = MetaPolicyEngine()
        for _ in range(5):
            mpe.compute_policy()
        hist = mpe.get_weight_history()
        assert len(hist) == 5

    def test_history_n_parameter(self) -> None:
        mpe = MetaPolicyEngine()
        for _ in range(20):
            mpe.compute_policy()
        hist = mpe.get_weight_history(n=5)
        assert len(hist) == 5

    def test_history_fields(self) -> None:
        mpe = MetaPolicyEngine()
        mpe.compute_policy()
        hist = mpe.get_weight_history()
        entry = hist[0]
        assert "anomaly_weight" in entry
        assert "regime_weight" in entry
        assert "stability_weight" in entry
        assert "causality_weight" in entry
        assert "sensitivity" in entry
        assert "dominant_concern" in entry
        assert "timestamp" in entry


class TestFeedOrderIndependence:
    """feed() and feed_decision_context() can be called in any order."""

    def test_feed_context_before_frame(self) -> None:
        mpe = MetaPolicyEngine()
        context = StubDecisionContext(stability_bias=-0.5)
        mpe.feed_decision_context(context)
        frame = StubIntelligenceFrame(priority="HIGH")
        mpe.feed(frame)
        pv = mpe.compute_policy()
        total = pv.anomaly_weight + pv.regime_weight + pv.stability_weight + pv.causality_weight
        assert total == pytest.approx(1.0)

    def test_feed_only_frame(self) -> None:
        mpe = MetaPolicyEngine()
        frame = StubIntelligenceFrame(priority="HIGH")
        mpe.feed(frame)
        pv = mpe.compute_policy()
        # Raw target: anomaly=0.35, others=0.25 each → total=1.10
        # Normalised anomaly = 0.35 / 1.10 ≈ 0.318
        assert pv.anomaly_weight == pytest.approx(0.318, abs=0.005)

    def test_feed_only_context(self) -> None:
        mpe = MetaPolicyEngine()
        context = StubDecisionContext(stability_bias=-0.5)
        mpe.feed_decision_context(context)
        pv = mpe.compute_policy()
        # stability should be higher than regime
        assert pv.stability_weight > pv.regime_weight


class TestMultipleRulesInteraction:
    """Interaction between multiple rules firing simultaneously."""

    def test_anomaly_and_regime_together(self) -> None:
        """Rule 1 (HIGH) and Rule 2 (>0.8) should both apply."""
        mpe = MetaPolicyEngine()
        regime = StubTransitionSignal(probability=0.9)
        frame = StubIntelligenceFrame(priority="HIGH", regime=regime)
        mpe.feed(frame)
        pv = mpe.compute_policy()
        # Rule 1: anomaly=0.35, sensitivity=0.7
        # Rule 2: regime=0.50, sensitivity=0.8 (overwrites Rule 1's sensitivity)
        # Raw target: anomaly=0.35, regime=0.50, stability=0.25, causality=0.25
        # Total=1.35 → normalised anomaly=0.35/1.35≈0.259, regime=0.50/1.35≈0.370
        assert pv.anomaly_weight == pytest.approx(0.259, abs=0.005)
        assert pv.regime_weight == pytest.approx(0.370, abs=0.005)
        assert pv.dominant_concern == "regime"
        total = pv.anomaly_weight + pv.regime_weight + pv.stability_weight + pv.causality_weight
        assert total == pytest.approx(1.0)

    def test_health_critical_and_context_oscillation(self) -> None:
        """Rule 3 (health < -0.5) and Rule 4 (oscillation) stack."""
        mpe = MetaPolicyEngine()
        health = StubSystemHealthScore(score=-0.9)
        frame = StubIntelligenceFrame(health=health)
        mpe.feed(frame)
        context = StubDecisionContext(stability_bias=-0.6)
        mpe.feed_decision_context(context)
        pv = mpe.compute_policy()
        # Rule 3: stability_weight=0.40, sensitivity=0.85, dominant="stability"
        # Rule 4: stability_weight += 0.15 → 0.55, regime_weight *= 0.8 → 0.20
        # After norm: stability should dominate
        assert pv.dominant_concern == "stability"
        assert pv.stability_weight > 0.4
        # sensitivity from Rule 3 = 0.85
        assert pv.sensitivity == 0.85


class TestEdgeCases:
    """Edge case handling."""

    def test_partial_frame(self) -> None:
        """Frame with missing attributes (partial duck typing)."""
        mpe = MetaPolicyEngine()

        class PartialFrame:
            timestamp = 100.0

        pf = PartialFrame()
        # No priority, regime, health — should use defaults
        mpe.feed(pf)
        pv = mpe.compute_policy()
        assert pv.dominant_concern == "balanced"

    def test_partial_context(self) -> None:
        """Context with missing attributes."""
        mpe = MetaPolicyEngine()

        class PartialContext:
            pass  # no stability_bias

        mpe.feed_decision_context(PartialContext())
        pv = mpe.compute_policy()
        # Should fall back to default stability_bias = 0.0
        assert pv.dominant_concern == "balanced"

    def test_regime_without_probability(self) -> None:
        """Regime signal missing probability attribute."""
        mpe = MetaPolicyEngine()

        class PartialRegime:
            pass

        frame = StubIntelligenceFrame(regime=PartialRegime())
        mpe.feed(frame)
        pv = mpe.compute_policy()
        assert pv.dominant_concern == "balanced"

    def test_health_without_score(self) -> None:
        """Health missing score attribute."""
        mpe = MetaPolicyEngine()

        class PartialHealth:
            state = "HEALTHY"

        frame = StubIntelligenceFrame(health=PartialHealth())
        mpe.feed(frame)
        pv = mpe.compute_policy()
        assert pv.dominant_concern == "balanced"

    def test_dominant_concern_overwritten_by_later_rule(self) -> None:
        """Rules later in the sequence overwrite dominant_concern."""
        mpe = MetaPolicyEngine()
        # Rule 1 sets "anomaly" (HIGH priority)
        # Rule 2 sets "regime" (prob > 0.6) — should win since it's later
        regime = StubTransitionSignal(probability=0.7)
        frame = StubIntelligenceFrame(priority="HIGH", regime=regime)
        mpe.feed(frame)
        pv = mpe.compute_policy()
        assert pv.dominant_concern == "regime"

    def test_high_priority_with_healthy_health(self) -> None:
        """Rule 1 (HIGH) + Rule 3 (healthy > 0.7): causality should be explored."""
        mpe = MetaPolicyEngine()
        health = StubSystemHealthScore(score=0.9)
        frame = StubIntelligenceFrame(priority="HIGH", health=health)
        mpe.feed(frame)
        pv = mpe.compute_policy()
        # Rule 1: anomaly=0.35, sensitivity=0.7, dominant="anomaly"
        # Rule 3: causality=0.40, dominant="causality" (overwrites)
        # Raw target: anomaly=0.35, causality=0.40, others=0.25 each → total=1.25
        # Normalised causality = 0.40/1.25 = 0.32
        assert pv.dominant_concern == "causality"
        assert pv.causality_weight == pytest.approx(0.32)
        assert pv.anomaly_weight == pytest.approx(0.28)


class TestIntegration:
    """Full integration scenarios."""

    def test_crisis_scenario(self) -> None:
        """Multiple critical signals: HIGH priority + regime transition + low health."""
        mpe = MetaPolicyEngine()

        regime = StubTransitionSignal(probability=0.85)
        health = StubSystemHealthScore(score=-0.8)
        frame = StubIntelligenceFrame(priority="HIGH", regime=regime, health=health)
        mpe.feed(frame)

        context = StubDecisionContext(stability_bias=-0.5)
        mpe.feed_decision_context(context)

        pv = mpe.compute_policy()

        # Rules applied:
        # Rule 1 (HIGH): anomaly=0.35, sensitivity=0.7, dominant="anomaly"
        # Rule 2 (>0.8): regime=0.50, sensitivity=0.8, dominant="regime"
        # Rule 3 (< -0.5): stability=0.40, sensitivity=0.85, dominant="stability"
        # Rule 4 (< -0.3): stability+=0.15, regime*=0.8
        #   → target: anomaly=0.35, regime=0.40, stability=0.55, causality=0.25
        #   → normalised: ...

        # Just check invariants:
        total = pv.anomaly_weight + pv.regime_weight + pv.stability_weight + pv.causality_weight
        assert total == pytest.approx(1.0)
        assert 0.1 <= pv.sensitivity <= 1.0
        # Health < -0.5 should make stability dominant
        assert pv.dominant_concern == "stability"

    def test_placid_scenario(self) -> None:
        """All signals normal/none — should produce default balanced policy."""
        mpe = MetaPolicyEngine()

        frame = StubIntelligenceFrame(priority="LOW")
        mpe.feed(frame)

        context = StubDecisionContext(stability_bias=0.0)
        mpe.feed_decision_context(context)

        pv = mpe.compute_policy()

        assert pv.dominant_concern == "balanced"
        assert pv.sensitivity == 0.5
        for w in (pv.anomaly_weight, pv.regime_weight, pv.stability_weight, pv.causality_weight):
            assert w == pytest.approx(0.25, abs=0.02)
