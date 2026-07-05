"""
intent_validator.py — Checks consistency of ExecutionIntent vs system state.

Validates anomaly conflicts, regime mismatches, risk violations, and causal
contradictions before an intent reaches the governor pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Outcome of a single validation pass.

    Attributes
    ----------
    passed : bool
        True if no violations were found.
    rejection_reason : str | None
        The first violation message, or None if passed.
    warnings : list[str]
        Non-blocking advisory messages.
    violations : list[str]
        Blocking violation messages.
    score : float
        0.0 (failed) to 1.0 (perfect).
    timestamp : float
        Unix timestamp of the validation.
    """
    passed: bool
    rejection_reason: str | None
    warnings: list[str]
    violations: list[str]
    score: float
    timestamp: float


# ---------------------------------------------------------------------------
# Constants — intent categories
# ---------------------------------------------------------------------------

_DIRECTIONAL_INTENTS: frozenset[str] = frozenset({
    "BUY_STRONG",
    "BUY_MODERATE",
    "BUY_LIGHT",
    "SELL_SHORT",
})

_DEFENSIVE_INTENTS: frozenset[str] = frozenset({
    "EMERGENCY_STOP",
    "REDUCE_STRONG",
    "REDUCE_MODERATE",
    "REDUCE_LIGHT",
    "EXIT_ALL",
})


# ---------------------------------------------------------------------------
# IntentValidator
# ---------------------------------------------------------------------------


class IntentValidator:
    """Validates an ExecutionIntent against the current system state.

    Duck-typed inputs:

    *execution_intent* — expects attributes:
        ``.intent_type`` (Enum or str), ``.magnitude`` (float),
        ``.risk_limit`` (float).

    *intelligence_frame* (optional) — expects attributes:
        ``.regime``, ``.anomalies`` (list), ``.causal_graph``,
        ``.health``.
    """

    def __init__(self) -> None:
        self._validation_history: list[ValidationResult] = []

    # ── Public API ──────────────────────────────────────────────────────────

    def validate(
        self,
        execution_intent: Any,
        intelligence_frame: Any = None,
    ) -> ValidationResult:
        """Validate ExecutionIntent against system state.

        Checks performed:

        1. **Anomaly conflicts** — HIGH/CRITICAL anomalies block directional
           actions; too many anomalies block non-defensive actions.
        2. **Regime mismatches** — SHADOW blocks directional actions; MICRO
           blocks strong directional actions.
        3. **Risk violations** — high magnitude without proportional limits;
           near-maximum magnitude warnings; contradictory risk_limit.
        4. **Causal contradictions** — diverged coupled engines that may lead
           to unexpected effects.
        5. **Emergency override** — EMERGENCY_STOP bypasses all validation.

        Parameters
        ----------
        execution_intent : Any
            Duck-typed ExecutionIntent with ``.intent_type``, ``.magnitude``,
            ``.risk_limit`` attributes.
        intelligence_frame : Any, optional
            Duck-typed intelligence context with ``.regime``, ``.anomalies``,
            ``.causal_graph``, ``.health`` attributes.

        Returns
        -------
        ValidationResult
            The combined validation outcome.
        """
        warnings: list[str] = []
        violations: list[str] = []

        # Extract intent attributes (duck-typed)
        intent_type = self._get_intent_type(execution_intent)
        magnitude = self._safe_float(getattr(execution_intent, "magnitude", 0.0), 0.0)
        risk_limit = self._safe_float(getattr(execution_intent, "risk_limit", 0.0), 0.0)

        # -------------------------------------------------------------------
        # Rule 5 — Emergency Override Validation (highest priority)
        # -------------------------------------------------------------------
        if intent_type == "EMERGENCY_STOP":
            result = ValidationResult(
                passed=True,
                rejection_reason=None,
                warnings=[],
                violations=[],
                score=1.0,
                timestamp=time.time(),
            )
            self._validation_history.append(result)
            return result

        # -------------------------------------------------------------------
        # Rule 1 — Anomaly Conflict Check
        # -------------------------------------------------------------------
        anomalies = self._get_anomalies(intelligence_frame)
        max_severity = self._get_max_anomaly_severity(anomalies)
        anomaly_count = len(anomalies)

        if max_severity in ("HIGH", "CRITICAL"):
            if intent_type in _DIRECTIONAL_INTENTS:
                violations.append(
                    "Cannot take directional action during active HIGH/CRITICAL anomaly"
                )
            elif intent_type in _DEFENSIVE_INTENTS:
                warnings.append("Anomaly active — defensive action appropriate")

        if anomaly_count > 5:
            if intent_type not in _DEFENSIVE_INTENTS:
                violations.append(
                    "Too many active anomalies for non-defensive action"
                )

        # -------------------------------------------------------------------
        # Rule 2 — Regime Mismatch Check
        # -------------------------------------------------------------------
        regime_name, regime_state = self._resolve_regime(intelligence_frame)

        if regime_name == "SHADOW" or regime_state == 0:
            if intent_type in _DIRECTIONAL_INTENTS:
                violations.append(
                    "SHADOW regime does not permit directional actions"
                )

        if regime_name == "MICRO" or regime_state == 1:
            if intent_type in ("BUY_STRONG", "SELL_SHORT"):
                violations.append(
                    "MICRO regime does not permit strong directional actions"
                )

        # -------------------------------------------------------------------
        # Rule 3 — Risk Violation Check
        # -------------------------------------------------------------------
        if magnitude > 0.8:
            # Check for a corresponding risk_limit (must exist and be > 0)
            if risk_limit <= 0.0:
                warnings.append("High magnitude without proportional risk limit")

        if magnitude > 0.95:
            warnings.append("Near-maximum magnitude — potential overexposure")

        if risk_limit < 0.1 and intent_type not in _DEFENSIVE_INTENTS:
            warnings.append(
                "Very low risk limit with non-defensive intent — possible contradiction"
            )

        # -------------------------------------------------------------------
        # Rule 4 — Causal Contradiction Check
        # -------------------------------------------------------------------
        if intelligence_frame is not None:
            causal_graph = getattr(intelligence_frame, "causal_graph", None)
            if causal_graph is not None:
                edges = self._get_causal_edges(causal_graph)
                if edges:
                    diverged = self._find_diverged_coupled_pairs(
                        causal_graph, anomalies
                    )
                    if diverged:
                        warnings.append(
                            "Causal coupling break detected — action may have unexpected effects"
                        )

        # -------------------------------------------------------------------
        # Score calculation
        # -------------------------------------------------------------------
        score = self._compute_score(violations, warnings)

        passed = len(violations) == 0
        rejection_reason = violations[0] if violations else None

        result = ValidationResult(
            passed=passed,
            rejection_reason=rejection_reason,
            warnings=warnings,
            violations=violations,
            score=score,
            timestamp=time.time(),
        )

        self._validation_history.append(result)
        return result

    # ── History query ───────────────────────────────────────────────────────

    def get_history(self, n: int = 10) -> list[ValidationResult]:
        """Return the last *n* validation results.

        Parameters
        ----------
        n : int
            Number of recent entries to return (default 10).

        Returns
        -------
        list[ValidationResult]
            The most recent *n* entries (or all if fewer exist).
        """
        return list(self._validation_history[-n:])

    # ── Score computation ───────────────────────────────────────────────────

    @staticmethod
    def _compute_score(violations: list[str], warnings: list[str]) -> float:
        """Compute a [0.0, 1.0] score based on violations and warnings.

        * start = 1.0
        * each violation: -0.3
        * each warning: -0.1
        * score = max(0.0, start)
        """
        score = 1.0
        score -= len(violations) * 0.3
        score -= len(warnings) * 0.1
        return max(0.0, score)

    # ── Intent type extraction ──────────────────────────────────────────────

    @staticmethod
    def _get_intent_type(intent: Any) -> str:
        """Extract intent type as a string from an enum member or plain string."""
        raw = getattr(intent, "intent_type", "HOLD")
        if isinstance(raw, Enum):
            return raw.value
        return str(raw)

    # ── Safe float conversion ───────────────────────────────────────────────

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        """Safely convert a value to float, returning *default* on failure."""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    # ── Anomaly helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _get_anomalies(frame: Any) -> list[Any]:
        """Extract the anomalies list from an intelligence frame (duck-typed)."""
        if frame is None:
            return []
        anomalies = getattr(frame, "anomalies", None)
        if anomalies is not None and isinstance(anomalies, list):
            return anomalies
        return []

    @staticmethod
    def _get_max_anomaly_severity(anomalies: list[Any]) -> str | None:
        """Return the highest severity string across all anomalies.

        Severity ranking (highest to lowest):
            CRITICAL > HIGH > MEDIUM > LOW > None
        """
        if not anomalies:
            return None

        ranking = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        max_sev: str | None = None
        max_rank = 0

        for anomaly in anomalies:
            sev = getattr(anomaly, "severity", None)
            if sev is None:
                continue
            sev_str = str(sev).upper().strip()
            rank = ranking.get(sev_str, 0)
            if rank > max_rank:
                max_rank = rank
                max_sev = sev_str

        return max_sev

    # ── Regime resolution ───────────────────────────────────────────────────

    def _resolve_regime(self, frame: Any) -> tuple[str, int]:
        """Resolve regime name and numeric state from an intelligence frame.

        Duck-typed resolution order:
        1. ``frame.regime.to_regime`` (string)
        2. ``frame.regime.regime_to`` (string)
        3. ``frame.regime`` (string directly)
        4. ``frame.regime_state`` (int)
        5. Default ``("UNKNOWN", 3)``

        Returns
        -------
        tuple[str, int]
            (regime_name, regime_state).
        """
        regime_name: str = "UNKNOWN"
        regime_state: int = 3  # 3 = UNKNOWN sentinel

        if frame is not None:
            # Try regime attribute — may be a TransitionSignal with .to_regime
            regime_attr = getattr(frame, "regime", None)
            if regime_attr is not None:
                # Duck-typed: try .to_regime or .regime_to
                to_regime = getattr(regime_attr, "to_regime", None)
                if to_regime is not None:
                    regime_name = str(to_regime)
                else:
                    regime_to = getattr(regime_attr, "regime_to", None)
                    if regime_to is not None:
                        regime_name = str(regime_to)
                    else:
                        regime_name = str(regime_attr)

            # Direct regime_state attribute
            state_attr = getattr(frame, "regime_state", None)
            if state_attr is not None:
                try:
                    regime_state = int(state_attr)
                except (ValueError, TypeError):
                    pass

            # Map numeric state to name if name is still UNKNOWN
            if regime_name == "UNKNOWN" and regime_state != 3:
                _STATE_TO_NAME: dict[int, str] = {0: "SHADOW", 1: "MICRO", 2: "FULL"}
                regime_name = _STATE_TO_NAME.get(regime_state, "UNKNOWN")

        return regime_name, regime_state

    # ── Causal graph helpers ────────────────────────────────────────────────

    @staticmethod
    def _get_causal_edges(causal_graph: Any) -> list[Any]:
        """Extract the edges list from a causal graph (duck-typed)."""
        edges = getattr(causal_graph, "edges", None)
        if edges is not None and isinstance(edges, list):
            return edges
        # Fallback: try dict-like access
        if isinstance(causal_graph, dict):
            return causal_graph.get("edges", [])
        return []

    def _find_diverged_coupled_pairs(
        self,
        causal_graph: Any,
        anomalies: list[Any],
    ) -> list[tuple[str, str]]:
        """Identify coupled engine pairs that have diverged.

        Looks for edges annotated with coupling metadata where the coupling
        state indicates divergence (e.g. ``diverged`` or ``broken``).

        Returns
        -------
        list[tuple[str, str]]
            List of (source, target) pairs that are diverged.
        """
        edges = self._get_causal_edges(causal_graph)
        diverged: list[tuple[str, str]] = []

        for edge in edges:
            if isinstance(edge, dict):
                source = str(edge.get("source", ""))
                target = str(edge.get("target", ""))
                coupling = str(edge.get("coupling", "")).lower()
                state = str(edge.get("state", "")).lower()
            else:
                source = str(getattr(edge, "source", ""))
                target = str(getattr(edge, "target", ""))
                coupling = str(getattr(edge, "coupling", "")).lower()
                state = str(getattr(edge, "state", "")).lower()

            if coupling in ("diverged", "broken") or state in ("diverged", "broken"):
                diverged.append((source, target))

        return diverged
