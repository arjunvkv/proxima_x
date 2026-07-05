"""
decision_synthesizer.py — Central synthesis layer that converts all intelligence
+ policy into a single decision state with actionable trading signals.

The synthesizer consumes:
  - IntelligenceFrame (from intelligence_bus)
  - PolicyVector (from meta_policy_engine)
  - DecisionContext (from conflict_resolver)

and produces a SystemDecision with an action tendency, risk bias, regime action
signal, confidence score, component breakdown, and a human-readable reasoning
chain.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

__all__ = [
    "ActionTendency",
    "RegimeActionSignal",
    "SystemDecision",
    "DecisionSynthesizer",
]

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ActionTendency(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    EXIT = "EXIT"
    STRONG_SELL = "STRONG_SELL"


class RegimeActionSignal(Enum):
    ESCALATE = "ESCALATE"
    MAINTAIN = "MAINTAIN"
    DE_ESCALATE = "DE_ESCALATE"
    PREPARE_TRANSITION = "PREPARE_TRANSITION"
    EMERGENCY_STOP = "EMERGENCY_STOP"


# ---------------------------------------------------------------------------
# SystemDecision
# ---------------------------------------------------------------------------


@dataclass
class SystemDecision:
    """A single synthesized decision produced by :class:`DecisionSynthesizer`.

    Attributes
    ----------
    action_tendency : ActionTendency
        The recommended trading action.
    risk_bias : float
        Bias from -1.0 (max risk-averse) to +1.0 (max risk-seeking).
    regime_action_signal : RegimeActionSignal
        What the system should do regarding regime transitions.
    confidence : float
        Overall confidence in the decision, 0.0 – 1.0.
    components : dict[str, float]
        Sub-scores for transparency (e.g. ``buy_score``, ``sell_score``,
        ``net_score``, ``weighted_anomaly``, etc.).
    reasoning : list[str]
        Human-readable reasoning steps that led to the decision.
    timestamp : float
        Unix timestamp when the decision was made.
    """
    action_tendency: ActionTendency
    risk_bias: float
    regime_action_signal: RegimeActionSignal
    confidence: float
    components: Dict[str, float] = field(default_factory=dict)
    reasoning: List[str] = field(default_factory=list)
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Severity mapping helpers
# ---------------------------------------------------------------------------

# Numeric value assigned to each anomaly severity level
_SEVERITY_VALUE: Dict[str, float] = {
    "LOW": 0.1,
    "MEDIUM": 0.3,
    "HIGH": 0.7,
    "CRITICAL": 1.0,
}

# History window for oscillation detection (frames)
_OSCILLATION_WINDOW = 10


# ---------------------------------------------------------------------------
# DecisionSynthesizer
# ---------------------------------------------------------------------------


class DecisionSynthesizer:
    """Central synthesis layer that produces actionable trading signals.

    The synthesizer fuses intelligence signals, policy weights, and conflict-
    resolved context into a single :class:`SystemDecision` per call to
    :meth:`synthesize`.

    Usage::

        synth = DecisionSynthesizer()
        synth.feed_intelligence(frame)
        synth.feed_policy(policy_vector)
        synth.feed_context(decision_context)
        decision = synth.synthesize()

    Duck-typed inputs — any object with the required attributes will be
    accepted, following the protocols established by ``IntelligenceFrame``,
    ``PolicyVector``, and ``DecisionContext``.
    """

    def __init__(self) -> None:
        # Latest inputs (duck-typed)
        self._latest_frame: Optional[Any] = None
        self._latest_policy: Optional[Any] = None
        self._latest_context: Optional[Any] = None

        # History of regime to/from transitions for oscillation detection
        self._regime_transitions: deque = deque(maxlen=_OSCILLATION_WINDOW * 2)

        # Decision history
        self._history: List[SystemDecision] = []

    # ── Public API ──────────────────────────────────────────────────────────

    def feed_intelligence(self, frame: Any) -> None:
        """Feed an ``IntelligenceFrame`` from the intelligence bus.

        The frame is duck-typed and must provide attributes:
        ``.regime``, ``.anomalies``, ``.health``, ``.priority``, ``.timestamp``.

        Parameters
        ----------
        frame : IntelligenceFrame
            The latest intelligence snapshot.
        """
        self._latest_frame = frame

    def feed_policy(self, policy_vector: Any) -> None:
        """Feed a ``PolicyVector`` from the meta-policy engine.

        The vector is duck-typed and must provide attributes:
        ``.anomaly_weight``, ``.regime_weight``, ``.stability_weight``.

        Parameters
        ----------
        policy_vector : PolicyVector
            Current policy weights.
        """
        self._latest_policy = policy_vector

    def feed_context(self, decision_context: Any) -> None:
        """Feed a ``DecisionContext`` from the conflict resolver.

        The context is duck-typed and must provide attributes:
        ``.regime_confidence``, ``.anomaly_weight``, ``.stability_bias``,
        ``.resolved_tensions``.

        Parameters
        ----------
        decision_context : DecisionContext
            Resolved context after conflict resolution.
        """
        self._latest_context = decision_context

    def synthesize(self) -> SystemDecision:
        """Combine all inputs into a single final decision.

        Returns
        -------
        SystemDecision
            The synthesized decision with action tendency, risk bias, regime
            action signal, confidence, component scores, and reasoning chain.
        """
        now = time.time()

        # ── Step 1: Extract signals from inputs ─────────────────────────────
        frame = self._latest_frame
        policy = self._latest_policy
        ctx = self._latest_context

        signals = self._extract_signals(frame)
        weights = self._extract_weights(policy)
        context_signals = self._extract_context_signals(ctx)

        # Track regime transitions for oscillation detection
        self._track_regime_transition(signals)

        reasoning: List[str] = []
        components: Dict[str, float] = {}

        # ── Step 2: Weight intelligence signals ─────────────────────────────
        anomaly_severity_value = signals["anomaly_severity_value"]
        regime_prob = signals["regime_probability"]
        health_score = signals["health_score"]

        weighted_anomaly = anomaly_severity_value * weights["anomaly_weight"]
        weighted_regime = regime_prob * weights["regime_weight"]
        weighted_stability = abs(health_score) * weights["stability_weight"]

        components["anomaly_severity_value"] = anomaly_severity_value
        components["regime_probability"] = regime_prob
        components["health_score"] = health_score
        components["weighted_anomaly"] = weighted_anomaly
        components["weighted_regime"] = weighted_regime
        components["weighted_stability"] = weighted_stability

        # ── Step 3: Determine action tendency ───────────────────────────────
        buy_score, sell_score, tendency_reasoning = self._compute_tendency_scores(
            signals=signals,
        )
        reasoning.extend(tendency_reasoning)

        net = buy_score - sell_score
        confidence = max(buy_score, sell_score)

        components["buy_score"] = buy_score
        components["sell_score"] = sell_score
        components["net_score"] = net

        action_tendency = self._map_net_to_tendency(net)

        # ── Step 4: Determine regime action signal ──────────────────────────
        regime_signal, regime_reasoning = self._determine_regime_action(
            signals=signals,
        )
        reasoning.extend(regime_reasoning)

        # ── Step 5: Calculate risk bias ─────────────────────────────────────
        risk_bias, risk_reasoning = self._calculate_risk_bias(
            signals=signals,
        )
        reasoning.extend(risk_reasoning)

        # ── Step 6: Build final decision ────────────────────────────────────
        decision = SystemDecision(
            action_tendency=action_tendency,
            risk_bias=risk_bias,
            regime_action_signal=regime_signal,
            confidence=confidence,
            components=components,
            reasoning=reasoning,
            timestamp=now,
        )

        # ── Step 7: Store in history ────────────────────────────────────────
        self._history.append(decision)

        return decision

    def get_history(self, n: int = 10) -> List[SystemDecision]:
        """Return the last *n* decisions.

        Parameters
        ----------
        n : int
            Number of recent decisions to return (default 10).

        Returns
        -------
        list[SystemDecision]
            The most recent *n* decisions (or all if fewer exist).
        """
        return list(self._history[-n:])

    # ── Signal extraction helpers ───────────────────────────────────────────

    @staticmethod
    def _extract_signals(frame: Any) -> Dict[str, Any]:
        """Extract and normalise all signals from an IntelligenceFrame.

        Returns a dict with keys:
            regime_from, regime_to, regime_probability, regime_present,
            anomaly_max_severity, anomaly_severity_value, anomaly_count,
            has_high_critical_anomalies, health_score, priority.
        """
        signals: Dict[str, Any] = {}

        # Regime signal
        regime = getattr(frame, 'regime', None) if frame is not None else None

        signals["regime_present"] = regime is not None
        signals["regime_from"] = (
            getattr(regime, 'from_regime', '')
            if regime is not None else ''
        )
        signals["regime_to"] = (
            getattr(regime, 'to_regime', '')
            if regime is not None else ''
        )
        signals["regime_probability"] = (
            getattr(regime, 'probability', 0.0)
            if regime is not None else 0.0
        )

        # Anomalies
        anomalies = (
            getattr(frame, 'anomalies', [])
            if frame is not None else []
        )
        signals["anomaly_count"] = len(anomalies)

        max_severity: Optional[str] = None
        has_high_critical = False
        for a in anomalies:
            sev = getattr(a, 'severity', 'LOW')
            if sev in ('HIGH', 'CRITICAL'):
                has_high_critical = True
            if max_severity is None or _SEVERITY_VALUE.get(sev, 0.0) > _SEVERITY_VALUE.get(max_severity, 0.0):
                max_severity = sev

        signals["anomaly_max_severity"] = max_severity
        signals["anomaly_severity_value"] = _SEVERITY_VALUE.get(max_severity, 0.0) if max_severity else 0.0
        signals["has_high_critical_anomalies"] = has_high_critical

        # Health
        health = getattr(frame, 'health', None) if frame is not None else None
        signals["health_score"] = (
            getattr(health, 'score', 0.0)
            if health is not None else 0.0
        )

        # Priority
        signals["priority"] = (
            getattr(frame, 'priority', 'LOW')
            if frame is not None else 'LOW'
        )

        return signals

    @staticmethod
    def _extract_weights(policy: Any) -> Dict[str, float]:
        """Extract weight values from a PolicyVector.

        Returns a dict with keys: anomaly_weight, regime_weight, stability_weight.
        """
        if policy is None:
            return {
                "anomaly_weight": 0.25,
                "regime_weight": 0.25,
                "stability_weight": 0.25,
            }

        return {
            "anomaly_weight": getattr(policy, 'anomaly_weight', 0.25),
            "regime_weight": getattr(policy, 'regime_weight', 0.25),
            "stability_weight": getattr(policy, 'stability_weight', 0.25),
        }

    @staticmethod
    def _extract_context_signals(ctx: Any) -> Dict[str, Any]:
        """Extract relevant signals from a DecisionContext.

        Returns a dict with keys: regime_confidence, stability_bias,
        resolved_tensions.
        """
        if ctx is None:
            return {
                "regime_confidence": 0.5,
                "stability_bias": 0.0,
                "resolved_tensions": [],
            }

        return {
            "regime_confidence": getattr(ctx, 'regime_confidence', 0.5),
            "stability_bias": getattr(ctx, 'stability_bias', 0.0),
            "resolved_tensions": getattr(ctx, 'resolved_tensions', []),
        }

    # ── Regime transition tracking ──────────────────────────────────────────

    def _track_regime_transition(self, signals: Dict[str, Any]) -> None:
        """Record a regime transition event when the regime signal is present.

        Stores a tuple ``(from_regime, to_regime, probability)`` in the
        transition history for oscillation detection.
        """
        if signals["regime_present"] and signals["regime_probability"] > 0.3:
            self._regime_transitions.append((
                signals["regime_from"],
                signals["regime_to"],
                signals["regime_probability"],
            ))

    def _count_recent_oscillations(self) -> int:
        """Count the number of regime transitions within the oscillation window.

        A transition is counted when the direction (from→to) differs from the
        previous recorded transition, indicating oscillation rather than a
        monotonic progression.
        """
        if len(self._regime_transitions) < 2:
            return 0

        # Look at the last N transitions within the oscillation window
        recent = list(self._regime_transitions)[-_OSCILLATION_WINDOW:]
        if not recent:
            return 0

        # Count direction changes (oscillations)
        oscillations = 0
        prev_to = recent[0][1]
        for i in range(1, len(recent)):
            curr_from = recent[i][0]
            # If we keep switching back and forth, it's an oscillation
            if curr_from != prev_to and recent[i][1] != prev_to:
                oscillations += 1
            prev_to = recent[i][1]

        return oscillations

    # ── Core synthesis logic ────────────────────────────────────────────────

    @staticmethod
    def _compute_tendency_scores(
        signals: Dict[str, Any],
    ) -> tuple[float, float, List[str]]:
        """Compute buy and sell scores from extracted signals.

        Returns
        -------
        tuple[float, float, list[str]]
            ``(buy_score, sell_score, reasoning_lines)``
        """
        reasoning: List[str] = []
        regime_to = signals["regime_to"]
        regime_prob = signals["regime_probability"]
        health_score = signals["health_score"]
        has_high_critical = signals["has_high_critical_anomalies"]

        # ── Positive signals (buy) ──────────────────────────────────────────
        buy_score = 0.0

        if regime_to == "FULL" and regime_prob > 0.6:
            buy_score += 0.4
            reasoning.append(
                f"Regime trending to FULL (p={regime_prob:.2f}) → +0.4 buy"
            )

        if health_score > 0.5:
            buy_score += 0.3
            reasoning.append(
                f"Health score={health_score:+.2f} > 0.5 → +0.3 buy"
            )

        if not has_high_critical:
            buy_score += 0.2
            reasoning.append(
                "No HIGH/CRITICAL anomalies → +0.2 buy"
            )

        if regime_prob > 0.8:
            buy_score += 0.1
            reasoning.append(
                f"Regime probability={regime_prob:.2f} > 0.8 → +0.1 buy"
            )

        # ── Negative signals (sell/exit) ────────────────────────────────────
        sell_score = 0.0

        if regime_to == "SHADOW" and regime_prob > 0.6:
            sell_score += 0.4
            reasoning.append(
                f"Regime trending to SHADOW (p={regime_prob:.2f}) → +0.4 sell"
            )

        if health_score < -0.3:
            sell_score += 0.3
            reasoning.append(
                f"Health score={health_score:+.2f} < -0.3 → +0.3 sell"
            )

        if has_high_critical:
            sell_score += 0.3
            reasoning.append(
                "HIGH/CRITICAL anomalies detected → +0.3 sell"
            )

        return buy_score, sell_score, reasoning

    @staticmethod
    def _map_net_to_tendency(net: float) -> ActionTendency:
        """Map the composite net score to an ActionTendency.

        Thresholds:
            net > 0.50  → STRONG_BUY
            net > 0.20  → BUY
            net > -0.20 → HOLD
            net > -0.50 → REDUCE
            net > -0.80 → EXIT
            else        → STRONG_SELL
        """
        if net > 0.5:
            return ActionTendency.STRONG_BUY
        if net > 0.2:
            return ActionTendency.BUY
        if net > -0.2:
            return ActionTendency.HOLD
        if net > -0.5:
            return ActionTendency.REDUCE
        if net > -0.8:
            return ActionTendency.EXIT
        return ActionTendency.STRONG_SELL

    def _determine_regime_action(
        self,
        signals: Dict[str, Any],
    ) -> tuple[RegimeActionSignal, List[str]]:
        """Determine the regime action signal from the current intelligence.

        Based on regime_from → regime_to prediction and priority:
        - regime_to == "FULL" and probability > 0.7  → ESCALATE
        - regime_to == "MICRO" and probability > 0.6 → PREPARE_TRANSITION
        - regime_to == "SHADOW" and probability > 0.6 → DE_ESCALATE
        - CRITICAL priority                          → EMERGENCY_STOP
        - No regime signal                           → MAINTAIN
        """
        reasoning: List[str] = []
        regime_to = signals["regime_to"]
        regime_prob = signals["regime_probability"]
        priority = signals["priority"]

        # EMERGENCY_STOP takes precedence over everything
        if priority == "CRITICAL":
            reasoning.append(
                "CRITICAL priority detected → EMERGENCY_STOP"
            )
            return RegimeActionSignal.EMERGENCY_STOP, reasoning

        if regime_to == "FULL" and regime_prob > 0.7:
            reasoning.append(
                f"Regime transitioning to FULL (p={regime_prob:.2f}) → ESCALATE"
            )
            return RegimeActionSignal.ESCALATE, reasoning

        if regime_to == "MICRO" and regime_prob > 0.6:
            reasoning.append(
                f"Regime transitioning to MICRO (p={regime_prob:.2f}) → PREPARE_TRANSITION"
            )
            return RegimeActionSignal.PREPARE_TRANSITION, reasoning

        if regime_to == "SHADOW" and regime_prob > 0.6:
            reasoning.append(
                f"Regime transitioning to SHADOW (p={regime_prob:.2f}) → DE_ESCALATE"
            )
            return RegimeActionSignal.DE_ESCALATE, reasoning

        if not signals["regime_present"]:
            reasoning.append(
                "No regime signal present → MAINTAIN"
            )
        else:
            reasoning.append(
                f"Regime signal insufficient ({regime_to}, p={regime_prob:.2f}) → MAINTAIN"
            )

        return RegimeActionSignal.MAINTAIN, reasoning

    def _calculate_risk_bias(
        self,
        signals: Dict[str, Any],
    ) -> tuple[float, List[str]]:
        """Calculate the risk bias from system health, anomalies, and regime.

        Formula:
            health_contrib    = health_score * 0.4
            anomaly_contrib   = -0.3 per HIGH anomaly, -0.5 per CRITICAL anomaly
            regime_contrib    = +0.3 if trending to FULL, -0.3 if trending to SHADOW

            total = health_contrib + anomaly_contrib + regime_contrib
            clamped to [-1.0, 1.0]
        """
        reasoning: List[str] = []
        health_score = signals["health_score"]
        regime_to = signals["regime_to"]

        # Health contribution
        health_contrib = health_score * 0.4
        if abs(health_contrib) > 0.01:
            reasoning.append(
                f"Health contribution: {health_score:+.2f} × 0.4 = {health_contrib:+.2f}"
            )

        # Anomaly contribution
        anomaly_contrib = 0.0
        frame = self._latest_frame
        if frame is not None:
            anomalies = getattr(frame, 'anomalies', [])
            high_count = 0
            critical_count = 0
            for a in anomalies:
                sev = getattr(a, 'severity', 'LOW')
                if sev == 'HIGH':
                    high_count += 1
                elif sev == 'CRITICAL':
                    critical_count += 1

            anomaly_contrib = (-0.3 * high_count) + (-0.5 * critical_count)
            if high_count > 0 or critical_count > 0:
                reasoning.append(
                    f"Anomaly contribution: {high_count} HIGH(-0.3) + "
                    f"{critical_count} CRITICAL(-0.5) = {anomaly_contrib:+.2f}"
                )

        # Regime contribution
        regime_contrib = 0.0
        if regime_to == "FULL":
            regime_contrib = 0.3
            reasoning.append(
                "Regime contribution: trending to FULL → +0.3"
            )
        elif regime_to == "SHADOW":
            regime_contrib = -0.3
            reasoning.append(
                "Regime contribution: trending to SHADOW → -0.3"
            )

        total = health_contrib + anomaly_contrib + regime_contrib
        total = max(-1.0, min(1.0, total))

        reasoning.append(
            f"Total risk_bias = {health_contrib:+.2f} + {anomaly_contrib:+.2f} + "
            f"{regime_contrib:+.2f} = {total:+.2f}"
        )

        return total, reasoning
