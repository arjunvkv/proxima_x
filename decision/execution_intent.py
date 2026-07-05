"""
execution_intent.py — Translate SystemDecision into abstract execution intents.

This module converts a SystemDecision (from decision_synthesizer) into an
ExecutionIntent — a high-level intent signal that bridges decision and
eventual execution. No real trades are produced. The intent captures:

  - What to do (intent_type)
  - How much (magnitude)
  - With what risk ceiling (risk_limit)
  - How urgently (time_preference)
  - At what price tolerance (max_slippage)
  - Why (reasoning chain)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional

__all__ = [
    "ExecutionIntentType",
    "ExecutionIntent",
    "ExecutionIntentTranslator",
]

# ---------------------------------------------------------------------------
# ExecutionIntentType
# ---------------------------------------------------------------------------


class ExecutionIntentType(Enum):
    """High-level execution intent signals.

    These represent abstract *intentions* — not real trades. The eventual
    execution layer may further refine or reject them based on live market
    conditions.
    """

    BUY_STRONG = "BUY_STRONG"
    BUY_MODERATE = "BUY_MODERATE"
    BUY_LIGHT = "BUY_LIGHT"
    HOLD = "HOLD"
    REDUCE_LIGHT = "REDUCE_LIGHT"
    REDUCE_MODERATE = "REDUCE_MODERATE"
    REDUCE_STRONG = "REDUCE_STRONG"
    EXIT_ALL = "EXIT_ALL"
    SELL_SHORT = "SELL_SHORT"
    TRANSITION_PREP = "TRANSITION_PREP"
    EMERGENCY_STOP = "EMERGENCY_STOP"


# ---------------------------------------------------------------------------
# ExecutionIntent
# ---------------------------------------------------------------------------


@dataclass
class ExecutionIntent:
    """A single execution intent produced by the translator.

    Attributes
    ----------
    intent_type : ExecutionIntentType
        The high-level intent signal.
    magnitude : float
        Position size fraction in [0.0, 1.0].
    risk_limit : float
        Maximum allowed risk in [0.0, 1.0].
    max_slippage : float
        Maximum acceptable slippage in basis points.
    time_preference : str
        Urgency: one of "IMMEDIATE", "FAST", "NORMAL", "SLOW".
    reasoning : list[str]
        Human-readable chain of reasoning that produced this intent.
    timestamp : float
        Unix timestamp when this intent was produced.
    """

    intent_type: ExecutionIntentType
    magnitude: float
    risk_limit: float
    max_slippage: float
    time_preference: str
    reasoning: List[str]
    timestamp: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SLIPPAGE_MAP: dict[ExecutionIntentType, float] = {
    ExecutionIntentType.EMERGENCY_STOP: 50.0,
    ExecutionIntentType.BUY_STRONG: 10.0,
    ExecutionIntentType.SELL_SHORT: 10.0,
    ExecutionIntentType.BUY_MODERATE: 5.0,
    ExecutionIntentType.REDUCE_MODERATE: 5.0,
    ExecutionIntentType.HOLD: 1.0,
    ExecutionIntentType.TRANSITION_PREP: 1.0,
}

_TIME_PREF_MAP: dict[ExecutionIntentType, str] = {
    ExecutionIntentType.EMERGENCY_STOP: "IMMEDIATE",
    ExecutionIntentType.BUY_STRONG: "FAST",
    ExecutionIntentType.SELL_SHORT: "FAST",
    ExecutionIntentType.BUY_MODERATE: "NORMAL",
    ExecutionIntentType.REDUCE_MODERATE: "NORMAL",
    ExecutionIntentType.HOLD: "SLOW",
    ExecutionIntentType.TRANSITION_PREP: "SLOW",
}


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to the inclusive [*lo*, *hi*] range."""
    return max(lo, min(hi, value))


def _str_val(obj: Any) -> str:
    """Safely extract a string value from an enum member or plain string."""
    if obj is None:
        return ""
    if isinstance(obj, Enum):
        return obj.value
    return str(obj)


# ---------------------------------------------------------------------------
# ExecutionIntentTranslator
# ---------------------------------------------------------------------------


class ExecutionIntentTranslator:
    """Translate a SystemDecision into an ExecutionIntent.

    Usage::

        translator = ExecutionIntentTranslator()
        translator.feed_decision(decision)
        intent = translator.translate()

    The translator maintains an internal history of produced intents.
    """

    def __init__(self) -> None:
        self._latest_decision: Any = None
        self._history: List[ExecutionIntent] = []

    # ── Public API ──────────────────────────────────────────────────────────

    def feed_decision(self, system_decision: Any) -> None:
        """Feed a ``SystemDecision`` from the decision synthesizer.

        The decision object is duck-typed; it must provide attributes:
        ``.action_tendency``, ``.risk_bias``, ``.regime_action_signal``,
        ``.confidence``, ``.components``, ``.reasoning``, ``.timestamp``.

        Parameters
        ----------
        system_decision : SystemDecision
            The latest synthesized system decision.
        """
        self._latest_decision = system_decision

    def translate(self) -> ExecutionIntent:
        """Convert the latest SystemDecision into an ExecutionIntent.

        Returns
        -------
        ExecutionIntent
            The translated high-level execution intent.

        Raises
        ------
        ValueError
            If no decision has been fed yet.
        """
        if self._latest_decision is None:
            raise ValueError("No decision available to translate — call feed_decision first.")

        intent = self._translate_decision(self._latest_decision)
        self._history.append(intent)
        return intent

    def get_intent(self) -> ExecutionIntent | None:
        """Return the latest translated intent without re-translating.

        Returns
        -------
        ExecutionIntent or None
            The most recent intent, or ``None`` if no intents exist.
        """
        if not self._history:
            return None
        return self._history[-1]

    def get_history(self, n: int = 10) -> List[ExecutionIntent]:
        """Return the last *n* translated intents.

        Parameters
        ----------
        n : int
            Number of recent intents to return (default 10).

        Returns
        -------
        list[ExecutionIntent]
            The most recent *n* intents (or all if fewer exist).
        """
        return list(self._history[-n:])

    # ── Translation core ────────────────────────────────────────────────────

    def _translate_decision(self, decision: Any) -> ExecutionIntent:
        reasoning: List[str] = []

        # ── Extract attributes (duck-typed) ─────────────────────────────────
        confidence: float = getattr(decision, "confidence", 0.0)
        risk_bias: float = getattr(decision, "risk_bias", 0.0)
        components: Any = getattr(decision, "components", {}) or {}

        health_score: float = 0.0
        if isinstance(components, dict):
            health_score = components.get("health_score", 0.0)

        # ── Determine intent type ───────────────────────────────────────────
        intent_type = self._map_intent_type(decision, health_score)
        reasoning.append(self._format_intent_reason(decision, intent_type))

        # ── Magnitude calculation ───────────────────────────────────────────
        base_magnitude = confidence
        risk_penalty = max(0.0, -risk_bias)
        magnitude = base_magnitude * (1.0 - risk_penalty * 0.5)
        magnitude = _clamp(magnitude)

        reasoning.append(
            f"magnitude = confidence({confidence:.2f}) × "
            f"(1 − risk_penalty({risk_penalty:.2f}) × 0.5) = {magnitude:.2f}"
        )

        # ── Risk limit calculation ──────────────────────────────────────────
        base_risk = 0.5
        risk_penalty = max(0.0, -risk_bias) * 0.3
        risk_boost = max(0.0, risk_bias) * 0.2
        risk_limit = _clamp(base_risk - risk_penalty + risk_boost)

        reasoning.append(
            f"risk_limit = base({base_risk}) − penalty({risk_penalty:.2f}) + "
            f"boost({risk_boost:.2f}) = {risk_limit:.2f}  "
            f"(risk_bias={risk_bias:+.2f})"
        )

        # ── Max slippage (based on intent type) ─────────────────────────────
        max_slippage = _SLIPPAGE_MAP.get(intent_type, 3.0)

        if intent_type == ExecutionIntentType.EMERGENCY_STOP:
            reasoning.append(
                f"EMERGENCY_STOP triggered → {max_slippage:.0f}bps slippage, "
                f"IMMEDIATE execution"
            )
        else:
            reasoning.append(
                f"slippage={max_slippage:.0f}bps for {intent_type.value}"
            )

        # ── Time preference (based on intent type) ──────────────────────────
        time_preference = _TIME_PREF_MAP.get(intent_type, "NORMAL")

        reasoning.append(f"time_preference={time_preference}")

        # ── Timestamp ───────────────────────────────────────────────────────
        now = time.time()

        return ExecutionIntent(
            intent_type=intent_type,
            magnitude=magnitude,
            risk_limit=risk_limit,
            max_slippage=max_slippage,
            time_preference=time_preference,
            reasoning=reasoning,
            timestamp=now,
        )

    # ── Mapping helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _map_intent_type(decision: Any, health_score: float) -> ExecutionIntentType:
        """Map a SystemDecision's action tendency + regime signal to an intent.

        Parameters
        ----------
        decision : Any
            Duck-typed SystemDecision.
        health_score : float
            Health score extracted from decision.components.

        Returns
        -------
        ExecutionIntentType
        """
        action = getattr(decision, "action_tendency", None)
        regime_signal = getattr(decision, "regime_action_signal", None)
        risk_bias: float = getattr(decision, "risk_bias", 0.0)

        action_str = _str_val(action)
        regime_str = _str_val(regime_signal)

        # ── EMERGENCY_STOP overrides everything ─────────────────────────────
        if regime_str == "EMERGENCY_STOP":
            return ExecutionIntentType.EMERGENCY_STOP

        # ── Action-tendency mapping ─────────────────────────────────────────
        if action_str == "STRONG_BUY":
            return ExecutionIntentType.BUY_STRONG

        if action_str == "BUY":
            return ExecutionIntentType.BUY_MODERATE

        if action_str == "HOLD":
            if regime_str == "PREPARE_TRANSITION":
                return ExecutionIntentType.TRANSITION_PREP
            return ExecutionIntentType.HOLD

        if action_str == "REDUCE":
            if health_score < -0.5:
                return ExecutionIntentType.REDUCE_STRONG
            return ExecutionIntentType.REDUCE_MODERATE

        if action_str == "EXIT":
            if risk_bias < -0.5:
                return ExecutionIntentType.EXIT_ALL
            return ExecutionIntentType.REDUCE_STRONG

        if action_str == "STRONG_SELL":
            return ExecutionIntentType.SELL_SHORT

        # Fallback
        return ExecutionIntentType.HOLD

    @staticmethod
    def _format_intent_reason(decision: Any, intent_type: ExecutionIntentType) -> str:
        """Build the first reasoning line describing the type mapping."""
        action = getattr(decision, "action_tendency", None)
        regime_signal = getattr(decision, "regime_action_signal", None)
        confidence: float = getattr(decision, "confidence", 0.0)

        action_str = _str_val(action)
        regime_str = _str_val(regime_signal)

        if intent_type == ExecutionIntentType.EMERGENCY_STOP:
            return (
                f"EMERGENCY_STOP triggered (regime_action_signal={regime_str})"
            )

        # Standard mapping line
        extra = ""
        if action_str == "HOLD" and intent_type == ExecutionIntentType.TRANSITION_PREP:
            extra = f" regime_action_signal={regime_str}"
        elif action_str == "REDUCE" and intent_type == ExecutionIntentType.REDUCE_STRONG:
            extra = " health_score < -0.5"
        elif action_str == "EXIT" and intent_type == ExecutionIntentType.EXIT_ALL:
            extra = " risk_bias < -0.5"

        return (
            f"{action_str} confidence={confidence:.2f}{extra}"
            f" → {intent_type.value}"
        )
