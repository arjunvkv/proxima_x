"""
Tests for ExecutionIntentTranslator.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pytest

from execution_intent import (
    ExecutionIntent,
    ExecutionIntentType,
    ExecutionIntentTranslator,
)


# ---------------------------------------------------------------------------
# Duck-typed stub — mirrors SystemDecision interface
# ---------------------------------------------------------------------------


@dataclass
class StubDecision:
    action_tendency: Any
    risk_bias: float = 0.0
    regime_action_signal: Any = None
    confidence: float = 0.5
    components: Dict[str, float] = field(default_factory=dict)
    reasoning: List[str] = field(default_factory=list)
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def translator() -> ExecutionIntentTranslator:
    return ExecutionIntentTranslator()


# ---------------------------------------------------------------------------
# Intent type mapping
# ---------------------------------------------------------------------------


class TestIntentMapping:
    def test_strong_buy_maps_to_buy_strong(self, translator: ExecutionIntentTranslator):
        d = StubDecision(
            action_tendency="STRONG_BUY",
            confidence=0.82,
        )
        translator.feed_decision(d)
        intent = translator.translate()
        assert intent.intent_type == ExecutionIntentType.BUY_STRONG

    def test_buy_maps_to_buy_moderate(self, translator: ExecutionIntentTranslator):
        d = StubDecision(action_tendency="BUY", confidence=0.6)
        translator.feed_decision(d)
        intent = translator.translate()
        assert intent.intent_type == ExecutionIntentType.BUY_MODERATE

    def test_hold_maps_to_hold(self, translator: ExecutionIntentTranslator):
        d = StubDecision(action_tendency="HOLD")
        translator.feed_decision(d)
        intent = translator.translate()
        assert intent.intent_type == ExecutionIntentType.HOLD

    def test_hold_with_prepare_transition(self, translator: ExecutionIntentTranslator):
        d = StubDecision(
            action_tendency="HOLD",
            regime_action_signal="PREPARE_TRANSITION",
        )
        translator.feed_decision(d)
        intent = translator.translate()
        assert intent.intent_type == ExecutionIntentType.TRANSITION_PREP

    def test_reduce_with_low_health_is_strong(self, translator: ExecutionIntentTranslator):
        d = StubDecision(
            action_tendency="REDUCE",
            components={"health_score": -0.7},
        )
        translator.feed_decision(d)
        intent = translator.translate()
        assert intent.intent_type == ExecutionIntentType.REDUCE_STRONG

    def test_reduce_with_healthy_is_moderate(self, translator: ExecutionIntentTranslator):
        d = StubDecision(
            action_tendency="REDUCE",
            components={"health_score": 0.2},
        )
        translator.feed_decision(d)
        intent = translator.translate()
        assert intent.intent_type == ExecutionIntentType.REDUCE_MODERATE

    def test_exit_with_low_risk_bias_is_exit_all(self, translator: ExecutionIntentTranslator):
        d = StubDecision(
            action_tendency="EXIT",
            risk_bias=-0.6,
        )
        translator.feed_decision(d)
        intent = translator.translate()
        assert intent.intent_type == ExecutionIntentType.EXIT_ALL

    def test_exit_with_high_risk_bias_is_reduce_strong(self, translator: ExecutionIntentTranslator):
        d = StubDecision(
            action_tendency="EXIT",
            risk_bias=0.0,
        )
        translator.feed_decision(d)
        intent = translator.translate()
        assert intent.intent_type == ExecutionIntentType.REDUCE_STRONG

    def test_strong_sell_maps_to_sell_short(self, translator: ExecutionIntentTranslator):
        d = StubDecision(action_tendency="STRONG_SELL")
        translator.feed_decision(d)
        intent = translator.translate()
        assert intent.intent_type == ExecutionIntentType.SELL_SHORT

    def test_emergency_stop_overrides_all(self, translator: ExecutionIntentTranslator):
        d = StubDecision(
            action_tendency="STRONG_BUY",
            regime_action_signal="EMERGENCY_STOP",
            confidence=0.9,
        )
        translator.feed_decision(d)
        intent = translator.translate()
        assert intent.intent_type == ExecutionIntentType.EMERGENCY_STOP

    def test_emergency_stop_overrides_exit(self, translator: ExecutionIntentTranslator):
        d = StubDecision(
            action_tendency="EXIT",
            regime_action_signal="EMERGENCY_STOP",
            risk_bias=-0.7,
        )
        translator.feed_decision(d)
        intent = translator.translate()
        assert intent.intent_type == ExecutionIntentType.EMERGENCY_STOP


# ---------------------------------------------------------------------------
# Magnitude calculation
# ---------------------------------------------------------------------------


class TestMagnitude:
    def test_magnitude_equals_confidence_when_no_risk_penalty(self, translator: ExecutionIntentTranslator):
        d = StubDecision(action_tendency="BUY", confidence=0.8, risk_bias=0.0)
        translator.feed_decision(d)
        intent = translator.translate()
        assert intent.magnitude == pytest.approx(0.8)

    def test_magnitude_reduced_by_risk_aversion(self, translator: ExecutionIntentTranslator):
        d = StubDecision(action_tendency="BUY", confidence=0.8, risk_bias=-0.5)
        translator.feed_decision(d)
        intent = translator.translate()
        # risk_penalty = max(0, 0.5) = 0.5
        # magnitude = 0.8 * (1.0 - 0.5 * 0.5) = 0.8 * 0.75 = 0.6
        assert intent.magnitude == pytest.approx(0.6)

    def test_magnitude_clamped_above(self, translator: ExecutionIntentTranslator):
        d = StubDecision(action_tendency="BUY", confidence=1.5, risk_bias=0.0)
        translator.feed_decision(d)
        intent = translator.translate()
        assert intent.magnitude == pytest.approx(1.0)

    def test_magnitude_clamped_below(self, translator: ExecutionIntentTranslator):
        d = StubDecision(action_tendency="BUY", confidence=-0.5, risk_bias=2.0)
        translator.feed_decision(d)
        intent = translator.translate()
        assert intent.magnitude == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Risk limit calculation
# ---------------------------------------------------------------------------


class TestRiskLimit:
    def test_default_risk_limit(self, translator: ExecutionIntentTranslator):
        d = StubDecision(action_tendency="HOLD", risk_bias=0.0)
        translator.feed_decision(d)
        intent = translator.translate()
        # base_risk=0.5, penalty=0, boost=0
        assert intent.risk_limit == pytest.approx(0.5)

    def test_risk_averse_reduces_limit(self, translator: ExecutionIntentTranslator):
        d = StubDecision(action_tendency="REDUCE", risk_bias=-0.5)
        translator.feed_decision(d)
        intent = translator.translate()
        # base=0.5, penalty=0.5*0.3=0.15, boost=0 → 0.35
        assert intent.risk_limit == pytest.approx(0.35)

    def test_risk_seeking_boosts_limit(self, translator: ExecutionIntentTranslator):
        d = StubDecision(action_tendency="STRONG_BUY", risk_bias=0.5)
        translator.feed_decision(d)
        intent = translator.translate()
        # base=0.5, penalty=0, boost=0.5*0.2=0.1 → 0.6
        assert intent.risk_limit == pytest.approx(0.6)

    def test_risk_limit_clamped(self, translator: ExecutionIntentTranslator):
        d = StubDecision(action_tendency="STRONG_BUY", risk_bias=10.0)
        translator.feed_decision(d)
        intent = translator.translate()
        assert intent.risk_limit == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Max slippage
# ---------------------------------------------------------------------------


class TestMaxSlippage:
    def test_emergency_stop_slippage(self, translator: ExecutionIntentTranslator):
        d = StubDecision(action_tendency="HOLD", regime_action_signal="EMERGENCY_STOP")
        translator.feed_decision(d)
        intent = translator.translate()
        assert intent.max_slippage == 50.0

    def test_buy_strong_slippage(self, translator: ExecutionIntentTranslator):
        d = StubDecision(action_tendency="STRONG_BUY")
        translator.feed_decision(d)
        intent = translator.translate()
        assert intent.max_slippage == 10.0

    def test_sell_short_slippage(self, translator: ExecutionIntentTranslator):
        d = StubDecision(action_tendency="STRONG_SELL")
        translator.feed_decision(d)
        intent = translator.translate()
        assert intent.max_slippage == 10.0

    def test_buy_moderate_slippage(self, translator: ExecutionIntentTranslator):
        d = StubDecision(action_tendency="BUY")
        translator.feed_decision(d)
        intent = translator.translate()
        assert intent.max_slippage == 5.0

    def test_reduce_moderate_slippage(self, translator: ExecutionIntentTranslator):
        d = StubDecision(action_tendency="REDUCE")
        translator.feed_decision(d)
        intent = translator.translate()
        assert intent.max_slippage == 5.0

    def test_hold_slippage(self, translator: ExecutionIntentTranslator):
        d = StubDecision(action_tendency="HOLD")
        translator.feed_decision(d)
        intent = translator.translate()
        assert intent.max_slippage == 1.0

    def test_transition_prep_slippage(self, translator: ExecutionIntentTranslator):
        d = StubDecision(
            action_tendency="HOLD",
            regime_action_signal="PREPARE_TRANSITION",
        )
        translator.feed_decision(d)
        intent = translator.translate()
        assert intent.max_slippage == 1.0

    def test_exit_all_default_slippage(self, translator: ExecutionIntentTranslator):
        d = StubDecision(action_tendency="EXIT", risk_bias=-0.6)
        translator.feed_decision(d)
        intent = translator.translate()
        # EXIT_ALL is not in the explicit map → falls to 3.0
        assert intent.max_slippage == 3.0

    def test_reduce_strong_default_slippage(self, translator: ExecutionIntentTranslator):
        d = StubDecision(action_tendency="EXIT", risk_bias=0.0)
        translator.feed_decision(d)
        intent = translator.translate()
        # REDUCE_STRONG is not in the explicit map → falls to 3.0
        assert intent.max_slippage == 3.0


# ---------------------------------------------------------------------------
# Time preference
# ---------------------------------------------------------------------------


class TestTimePreference:
    def test_emergency_stop_immediate(self, translator: ExecutionIntentTranslator):
        d = StubDecision(action_tendency="HOLD", regime_action_signal="EMERGENCY_STOP")
        translator.feed_decision(d)
        intent = translator.translate()
        assert intent.time_preference == "IMMEDIATE"

    def test_buy_strong_fast(self, translator: ExecutionIntentTranslator):
        d = StubDecision(action_tendency="STRONG_BUY")
        translator.feed_decision(d)
        intent = translator.translate()
        assert intent.time_preference == "FAST"

    def test_sell_short_fast(self, translator: ExecutionIntentTranslator):
        d = StubDecision(action_tendency="STRONG_SELL")
        translator.feed_decision(d)
        intent = translator.translate()
        assert intent.time_preference == "FAST"

    def test_buy_moderate_normal(self, translator: ExecutionIntentTranslator):
        d = StubDecision(action_tendency="BUY")
        translator.feed_decision(d)
        intent = translator.translate()
        assert intent.time_preference == "NORMAL"

    def test_reduce_moderate_normal(self, translator: ExecutionIntentTranslator):
        d = StubDecision(action_tendency="REDUCE")
        translator.feed_decision(d)
        intent = translator.translate()
        assert intent.time_preference == "NORMAL"

    def test_hold_slow(self, translator: ExecutionIntentTranslator):
        d = StubDecision(action_tendency="HOLD")
        translator.feed_decision(d)
        intent = translator.translate()
        assert intent.time_preference == "SLOW"

    def test_transition_prep_slow(self, translator: ExecutionIntentTranslator):
        d = StubDecision(
            action_tendency="HOLD",
            regime_action_signal="PREPARE_TRANSITION",
        )
        translator.feed_decision(d)
        intent = translator.translate()
        assert intent.time_preference == "SLOW"

    def test_exit_all_default_time_pref(self, translator: ExecutionIntentTranslator):
        d = StubDecision(action_tendency="EXIT", risk_bias=-0.6)
        translator.feed_decision(d)
        intent = translator.translate()
        assert intent.time_preference == "NORMAL"


# ---------------------------------------------------------------------------
# History / get_intent
# ---------------------------------------------------------------------------


class TestHistory:
    def test_get_intent_returns_none_when_empty(self, translator: ExecutionIntentTranslator):
        assert translator.get_intent() is None

    def test_get_intent_returns_latest(self, translator: ExecutionIntentTranslator):
        d1 = StubDecision(action_tendency="HOLD")
        d2 = StubDecision(action_tendency="STRONG_BUY", confidence=0.9)

        translator.feed_decision(d1)
        intent1 = translator.translate()

        translator.feed_decision(d2)
        intent2 = translator.translate()

        assert translator.get_intent() is intent2

    def test_get_history_returns_last_n(self, translator: ExecutionIntentTranslator):
        for i in range(5):
            translator.feed_decision(StubDecision(action_tendency="HOLD"))
            translator.translate()

        history = translator.get_history(n=3)
        assert len(history) == 3

    def test_get_history_returns_all_when_fewer(self, translator: ExecutionIntentTranslator):
        for i in range(3):
            translator.feed_decision(StubDecision(action_tendency="HOLD"))
            translator.translate()

        history = translator.get_history(n=10)
        assert len(history) == 3

    def test_translate_appends_to_history(self, translator: ExecutionIntentTranslator):
        d = StubDecision(action_tendency="BUY")
        translator.feed_decision(d)
        translator.translate()
        assert len(translator.get_history()) == 1


# ---------------------------------------------------------------------------
# Reasoning output
# ---------------------------------------------------------------------------


class TestReasoning:
    def test_reasoning_contains_intent_mapping(self, translator: ExecutionIntentTranslator):
        d = StubDecision(action_tendency="STRONG_BUY", confidence=0.82)
        translator.feed_decision(d)
        intent = translator.translate()
        combined = " ".join(intent.reasoning)
        assert "STRONG_BUY" in combined
        assert "BUY_STRONG" in combined
        assert "0.82" in combined

    def test_reasoning_contains_emergency_stop(self, translator: ExecutionIntentTranslator):
        d = StubDecision(
            action_tendency="STRONG_BUY",
            regime_action_signal="EMERGENCY_STOP",
        )
        translator.feed_decision(d)
        intent = translator.translate()
        combined = " ".join(intent.reasoning)
        assert "EMERGENCY_STOP" in combined
        assert "50bps" in combined
        assert "IMMEDIATE" in combined

    def test_reasoning_contains_risk_limit_info(self, translator: ExecutionIntentTranslator):
        d = StubDecision(action_tendency="BUY", risk_bias=-0.3)
        translator.feed_decision(d)
        intent = translator.translate()
        combined = " ".join(intent.reasoning)
        assert "risk_limit" in combined
        assert "risk_bias" in combined

    def test_reasoning_contains_magnitude(self, translator: ExecutionIntentTranslator):
        d = StubDecision(action_tendency="BUY", confidence=0.7, risk_bias=0.0)
        translator.feed_decision(d)
        intent = translator.translate()
        combined = " ".join(intent.reasoning)
        assert "magnitude" in combined

    def test_reasoning_contains_time_preference(self, translator: ExecutionIntentTranslator):
        d = StubDecision(action_tendency="HOLD")
        translator.feed_decision(d)
        intent = translator.translate()
        combined = " ".join(intent.reasoning)
        assert "time_preference=SLOW" in combined


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_translate_without_decision_raises(self, translator: ExecutionIntentTranslator):
        with pytest.raises(ValueError, match="No decision available"):
            translator.translate()


# ---------------------------------------------------------------------------
# Real enum input (SystemDecision style)
# ---------------------------------------------------------------------------

from enum import Enum


class _ActionTendency(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    EXIT = "EXIT"
    STRONG_SELL = "STRONG_SELL"


class _RegimeSignal(Enum):
    ESCALATE = "ESCALATE"
    MAINTAIN = "MAINTAIN"
    DE_ESCALATE = "DE_ESCALATE"
    PREPARE_TRANSITION = "PREPARE_TRANSITION"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class TestEnumInput:
    def test_with_real_enums(self, translator: ExecutionIntentTranslator):
        d = StubDecision(
            action_tendency=_ActionTendency.STRONG_BUY,
            regime_action_signal=_RegimeSignal.MAINTAIN,
            confidence=0.75,
            risk_bias=0.2,
        )
        translator.feed_decision(d)
        intent = translator.translate()
        assert intent.intent_type == ExecutionIntentType.BUY_STRONG
        assert intent.magnitude == pytest.approx(0.75)  # risk_bias=+0.2 → penalty=0
        # risk_limit = 0.5 + 0.2*0.2 = 0.54
        assert intent.risk_limit == pytest.approx(0.54)

    def test_enum_emergency_stop(self, translator: ExecutionIntentTranslator):
        d = StubDecision(
            action_tendency=_ActionTendency.HOLD,
            regime_action_signal=_RegimeSignal.EMERGENCY_STOP,
        )
        translator.feed_decision(d)
        intent = translator.translate()
        assert intent.intent_type == ExecutionIntentType.EMERGENCY_STOP
        assert intent.max_slippage == 50.0
        assert intent.time_preference == "IMMEDIATE"

    def test_enum_prepare_transition(self, translator: ExecutionIntentTranslator):
        d = StubDecision(
            action_tendency=_ActionTendency.HOLD,
            regime_action_signal=_RegimeSignal.PREPARE_TRANSITION,
        )
        translator.feed_decision(d)
        intent = translator.translate()
        assert intent.intent_type == ExecutionIntentType.TRANSITION_PREP
        assert intent.time_preference == "SLOW"
        assert intent.max_slippage == 1.0


# ---------------------------------------------------------------------------
# Timestamp
# ---------------------------------------------------------------------------


class TestTimestamp:
    def test_timestamp_is_set(self, translator: ExecutionIntentTranslator):
        before = time.time()
        d = StubDecision(action_tendency="HOLD")
        translator.feed_decision(d)
        intent = translator.translate()
        after = time.time()
        assert before <= intent.timestamp <= after
