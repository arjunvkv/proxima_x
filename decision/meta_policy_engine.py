"""
meta_policy_engine.py — Dynamically weight intelligence subsystems based on
current system state.

The engine consumes IntelligenceFrame objects (from the intelligence bus) and
DecisionContext objects (from the conflict resolver) and produces a PolicyVector
that answers "what matters more right now?".

Rules
-----
1. Anomaly sensitivity — escalate anomaly_weight when priority is HIGH/CRITICAL
2. Regime transition focus — boost regime_weight when regime detector fires
   with high probability
3. Health-driven adjustment — shift toward stability (or causality) based on
   system health
4. Conflict-aware adjustment — react to stability_bias from DecisionContext
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    "PolicyVector",
    "MetaPolicyEngine",
]

# ---------------------------------------------------------------------------
# PolicyVector
# ---------------------------------------------------------------------------


@dataclass
class PolicyVector:
    """Current policy weights for the intelligence sub-system ensemble.

    Attributes
    ----------
    anomaly_weight : float
        How much to weight anomaly signals (0 — 1).
    regime_weight : float
        How much to weight regime transition signals (0 — 1).
    stability_weight : float
        How much to weight stability metrics (0 — 1).
    causality_weight : float
        How much to weight causal graph signals (0 — 1).
    sensitivity : float
        Overall sensitivity multiplier (0 — 1).
    dominant_concern : str
        Which subsystem is currently dominant.
        One of "anomaly", "regime", "stability", "causality", "balanced".
    timestamp : float
        Unix timestamp when this policy was computed.
    """
    anomaly_weight: float = 0.25
    regime_weight: float = 0.25
    stability_weight: float = 0.25
    causality_weight: float = 0.25
    sensitivity: float = 0.5
    dominant_concern: str = "balanced"
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# MetaPolicyEngine
# ---------------------------------------------------------------------------


class MetaPolicyEngine:
    """Compute a dynamic policy vector from intelligence + decision context.

    The engine applies rule-based adjustments to the default equal-weight
    policy based on:
    - Frame priority (anomaly detection criticality)
    - Regime transition probability
    - System health score
    - Conflict resolver stability bias

    All weight changes are smoothed via exponential decay and normalised
    so that anomaly + regime + stability + causality = 1.0.
    """

    def __init__(self, decay_factor: float = 0.95) -> None:
        self._decay_factor = decay_factor

        # Latest inputs
        self._latest_frame: Optional[Any] = None
        self._latest_context: Optional[Any] = None

        # Previous policy (for decay smoothing)
        self._previous: Optional[PolicyVector] = None

        # Weight history — list of dicts for introspection
        self._history: List[Dict[str, Any]] = []

    # ── Public API ──────────────────────────────────────────────────────────

    def feed(self, intelligence_frame: Any) -> None:
        """Feed an ``IntelligenceFrame`` from the intelligence bus.

        The frame is duck-typed and must provide attributes:
        ``.priority``, ``.regime``, ``.health``, ``.timestamp`` (optional).

        Parameters
        ----------
        intelligence_frame : IntelligenceFrame
            The latest intelligence snapshot.
        """
        self._latest_frame = intelligence_frame

    def feed_decision_context(self, context: Any) -> None:
        """Feed a ``DecisionContext`` from the conflict resolver.

        The context is duck-typed and must provide attributes:
        ``.stability_bias``, ``.timestamp`` (optional).

        Parameters
        ----------
        context : DecisionContext
            Resolved decision context after conflict resolution.
        """
        self._latest_context = context

    def compute_policy(self) -> PolicyVector:
        """Compute the current policy vector from all inputs.

        Applies dynamic adjustment rules, normalises weights, applies
        exponential decay smoothing, and clamps sensitivity.

        Returns
        -------
        PolicyVector
            The current policy with smoothed and normalised weights.
        """
        target = self._compute_target()
        smoothed = self._smooth(target)
        self._previous = smoothed
        self._record_history(smoothed)
        return smoothed

    def get_weight_history(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return the last *n* weight snapshots.

        Parameters
        ----------
        n : int
            Number of recent snapshots to return (default 10).

        Returns
        -------
        list[dict]
            Each dict contains the fields of a ``PolicyVector``.
        """
        return list(self._history)[-n:]

    # ── Internal helpers ───────────────────────────────────────────────────

    def _compute_target(self) -> PolicyVector:
        """Build a raw (unsmoothed) policy vector from current signals."""
        now = time.time()

        frame = self._latest_frame
        context = self._latest_context

        # ── Start with defaults ─────────────────────────────────────────────
        p = PolicyVector(timestamp=now)

        # ── Rule 1: Anomaly sensitivity ─────────────────────────────────────
        if frame is not None:
            priority = self._safe_attr(frame, 'priority', 'LOW')

            if priority == "CRITICAL":
                p.anomaly_weight = 0.45
                p.sensitivity = 0.9
                p.dominant_concern = "anomaly"
            elif priority == "HIGH":
                p.anomaly_weight = 0.35
                p.sensitivity = 0.7
                p.dominant_concern = "anomaly"

        # ── Rule 2: Regime transition focus ─────────────────────────────────
        if frame is not None:
            regime = self._safe_attr(frame, 'regime', None)
            if regime is not None:
                prob = self._safe_attr(regime, 'probability', 0.0)

                if prob > 0.8:
                    p.regime_weight = 0.50
                    p.sensitivity = 0.8
                    p.dominant_concern = "regime"
                elif prob > 0.6:
                    p.regime_weight = 0.40
                    p.stability_weight = 0.30
                    p.dominant_concern = "regime"

        # ── Rule 3: Health-driven ───────────────────────────────────────────
        if frame is not None:
            health = self._safe_attr(frame, 'health', None)
            if health is not None:
                score = self._safe_attr(health, 'score', 0.0)

                if score < -0.5:
                    p.stability_weight = 0.40
                    p.sensitivity = 0.85
                    p.dominant_concern = "stability"
                elif score > 0.7:
                    p.causality_weight = 0.40
                    p.dominant_concern = "causality"

        # ── Rule 4: Conflict-aware adjustment ───────────────────────────────
        if context is not None:
            stability_bias = self._safe_attr(context, 'stability_bias', 0.0)

            if stability_bias < -0.3:
                # Oscillation detected — boost stability, discount regime
                p.stability_weight += 0.15
                p.regime_weight *= 0.8
            elif stability_bias > 0.3:
                # Stable — can afford exploratory causal analysis
                p.causality_weight += 0.10

        # ── Post-processing ─────────────────────────────────────────────────
        self._normalise(p)
        p.sensitivity = max(0.1, min(1.0, p.sensitivity))

        return p

    def _smooth(self, target: PolicyVector) -> PolicyVector:
        """Apply exponential decay to smooth weight transitions."""
        if self._previous is None:
            return target

        prev = self._previous
        d = self._decay_factor

        return PolicyVector(
            anomaly_weight=prev.anomaly_weight * d + target.anomaly_weight * (1 - d),
            regime_weight=prev.regime_weight * d + target.regime_weight * (1 - d),
            stability_weight=prev.stability_weight * d + target.stability_weight * (1 - d),
            causality_weight=prev.causality_weight * d + target.causality_weight * (1 - d),
            sensitivity=prev.sensitivity * d + target.sensitivity * (1 - d),
            dominant_concern=target.dominant_concern,
            timestamp=target.timestamp,
        )

    @staticmethod
    def _normalise(p: PolicyVector) -> None:
        """Normalise the four weights so they sum to 1.0.

        Mutates *p* in place.
        """
        total = p.anomaly_weight + p.regime_weight + p.stability_weight + p.causality_weight
        if total > 0:
            inv = 1.0 / total
            p.anomaly_weight *= inv
            p.regime_weight *= inv
            p.stability_weight *= inv
            p.causality_weight *= inv
        else:
            # Fallback to uniform (should never happen)
            p.anomaly_weight = 0.25
            p.regime_weight = 0.25
            p.stability_weight = 0.25
            p.causality_weight = 0.25

    def _record_history(self, p: PolicyVector) -> None:
        """Append a snapshot of *p* to the weight history."""
        self._history.append({
            "anomaly_weight": p.anomaly_weight,
            "regime_weight": p.regime_weight,
            "stability_weight": p.stability_weight,
            "causality_weight": p.causality_weight,
            "sensitivity": p.sensitivity,
            "dominant_concern": p.dominant_concern,
            "timestamp": p.timestamp,
        })

    # ── Duck-typing helpers ─────────────────────────────────────────────────

    @staticmethod
    def _safe_attr(obj: Any, name: str, default: Any = None) -> Any:
        """Safely extract an attribute from a duck-typed object."""
        return getattr(obj, name, default)
