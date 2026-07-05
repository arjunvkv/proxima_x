"""
execution_governor.py — Final authority layer before any real action.

Validates ExecutionIntent, applies risk caps, enforces regime restrictions,
blocks unsafe transitions, overrides conflicting intents.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# GovernorVerdict
# ---------------------------------------------------------------------------


class GovernorVerdict(Enum):
    APPROVED = "APPROVED"
    ADJUSTED = "ADJUSTED"
    BLOCKED = "BLOCKED"
    OVERRIDDEN = "OVERRIDDEN"


# ---------------------------------------------------------------------------
# GovernedExecution
# ---------------------------------------------------------------------------


@dataclass
class GovernedExecution:
    approved: bool
    verdict: GovernorVerdict
    adjusted_intent_type: str  # The possibly-modified ExecutionIntentType as string
    adjusted_magnitude: float
    adjusted_risk_limit: float
    adjusted_max_slippage: float
    adjusted_time_preference: str
    risk_multiplier: float       # 0.0 to 1.0
    rejection_reason: str | None  # None if approved
    applied_rules: list[str]     # Which rules were triggered
    timestamp: float


# ---------------------------------------------------------------------------
# ExecutionGovernor
# ---------------------------------------------------------------------------


class ExecutionGovernor:
    """Final authority layer before any real action.

    Governs an ``ExecutionIntent`` through a multi-stage pipeline:

    1. Intent validation
    2. Regime permission check
    3. Risk constraint check
    4. Final verdict (APPROVED / ADJUSTED / BLOCKED / OVERRIDDEN)

    All external dependencies are duck-typed and registered via the
    ``register_*`` methods.
    """

    def __init__(self) -> None:
        self._history: list[GovernedExecution] = []
        self._risk_constraint_engine = None
        self._regime_matrix = None
        self._intent_validator = None

    # ── Registration (duck-typed dependencies) ──────────────────────────────

    def register_risk_constraint_engine(self, engine: Any) -> None:
        self._risk_constraint_engine = engine

    def register_regime_matrix(self, matrix: Any) -> None:
        self._regime_matrix = matrix

    def register_intent_validator(self, validator: Any) -> None:
        self._intent_validator = validator

    # ── Core governance pipeline ────────────────────────────────────────────

    def govern(self, execution_intent: Any, intelligence_frame: Any = None) -> GovernedExecution:
        """Process an ExecutionIntent through the full governance pipeline.

        Parameters
        ----------
        execution_intent : Any
            Duck-typed ExecutionIntent with attributes:
            ``.intent_type`` (Enum or str), ``.magnitude``, ``.risk_limit``,
            ``.max_slippage``, ``.time_preference``, ``.reasoning``,
            ``.timestamp``.
        intelligence_frame : Any, optional
            Optional intelligence context forwarded to the risk engine.

        Returns
        -------
        GovernedExecution
            The governor's decision including any adjustments applied.
        """
        applied_rules: list[str] = []
        rejection_reason: str | None = None
        verdict: GovernorVerdict = GovernorVerdict.APPROVED

        # Extract intent attributes (duck-typed)
        raw_intent_type = self._get_intent_type(execution_intent)
        raw_magnitude = self._safe_float(getattr(execution_intent, "magnitude", 0.0), 0.0)
        raw_risk_limit = self._safe_float(getattr(execution_intent, "risk_limit", 0.0), 0.0)
        raw_max_slippage = self._safe_float(getattr(execution_intent, "max_slippage", 0.0), 0.0)
        raw_time_preference = str(getattr(execution_intent, "time_preference", "NORMAL"))

        # Start with original values; pipeline steps may adjust them
        adjusted_intent_type: str = raw_intent_type
        adjusted_magnitude: float = raw_magnitude
        adjusted_risk_limit: float = raw_risk_limit
        adjusted_max_slippage: float = raw_max_slippage
        adjusted_time_preference: str = raw_time_preference
        risk_multiplier: float = 1.0

        blocked: bool = False
        override: bool = False

        # -------------------------------------------------------------------
        # Step 1 — Intent validation
        # -------------------------------------------------------------------
        if self._intent_validator is not None:
            try:
                validation_result = self._intent_validator.validate(execution_intent)
                if validation_result is not None:
                    reason = str(validation_result)
                    applied_rules.append(f"intent_validator: {reason}")
                    rejection_reason = reason
                    # Validation failure does not automatically BLOCK;
                    # other rules may still adjust or override.
            except Exception as exc:
                applied_rules.append(f"intent_validator: error during validation — {exc}")

        # -------------------------------------------------------------------
        # Step 2 — Regime permission check
        # -------------------------------------------------------------------
        if self._regime_matrix is not None:
            try:
                permissions = self._get_regime_permissions()
                allowed_actions = self._get_allowed_actions(permissions)
                if adjusted_intent_type not in allowed_actions:
                    applied_rules.append(
                        f"regime_matrix: {adjusted_intent_type} not allowed in current regime"
                    )
                    adjusted_magnitude = 0.0
                    rejection_reason = "Action not permitted in current regime"
                    blocked = True
            except Exception as exc:
                applied_rules.append(f"regime_matrix: error during permission check — {exc}")

        # -------------------------------------------------------------------
        # Step 3 — Risk constraint check
        # -------------------------------------------------------------------
        if self._risk_constraint_engine is not None:
            try:
                risk_profile = self._risk_constraint_engine.evaluate(
                    intelligence_frame, execution_intent
                )
                max_size = self._safe_float(getattr(risk_profile, "max_size", None), raw_magnitude)
                exposure_limit = self._safe_float(getattr(risk_profile, "exposure_limit", None), raw_risk_limit)
                risk_allowed_actions = self._get_risk_allowed_actions(risk_profile)

                # Apply max_size cap
                if max_size < adjusted_magnitude:
                    applied_rules.append(
                        f"risk_engine: max_size={max_size}, magnitude adjusted from {adjusted_magnitude} to {max_size}"
                    )
                    adjusted_magnitude = max_size

                # Apply exposure limit cap
                if exposure_limit < adjusted_risk_limit:
                    applied_rules.append(
                        f"risk_engine: exposure_limit={exposure_limit}, risk_limit adjusted from {adjusted_risk_limit} to {exposure_limit}"
                    )
                    adjusted_risk_limit = exposure_limit

                # Risk multiplier
                denominator = max(raw_magnitude, 0.001)
                risk_multiplier = max_size / denominator
                risk_multiplier = max(0.0, min(1.0, risk_multiplier))

                # Allowed actions check
                if risk_allowed_actions is not None and adjusted_intent_type not in risk_allowed_actions:
                    applied_rules.append(
                        f"risk_engine: {adjusted_intent_type} not in allowed_actions"
                    )
                    rejection_reason = "Action not permitted by risk engine"
                    blocked = True
                    adjusted_magnitude = 0.0

            except Exception as exc:
                applied_rules.append(f"risk_engine: error during evaluation — {exc}")

        # -------------------------------------------------------------------
        # Step 4 — Final verdict logic
        # -------------------------------------------------------------------

        # Override conditions (bypass BLOCKED)
        override = self._check_override_conditions(
            raw_intent_type, execution_intent
        )

        if override:
            verdict = GovernorVerdict.OVERRIDDEN
            rejection_reason = None
            blocked = False
            # Restore original values on override
            adjusted_intent_type = raw_intent_type
            adjusted_magnitude = raw_magnitude
            adjusted_risk_limit = raw_risk_limit
            adjusted_max_slippage = raw_max_slippage
            adjusted_time_preference = raw_time_preference
            risk_multiplier = 1.0
            applied_rules.append("OVERRIDE: EMERGENCY_STOP — all constraints bypassed")
        elif blocked:
            verdict = GovernorVerdict.BLOCKED
        elif (
            adjusted_magnitude < raw_magnitude * 0.5
            or adjusted_risk_limit < raw_risk_limit * 0.5
        ):
            verdict = GovernorVerdict.ADJUSTED
        else:
            verdict = GovernorVerdict.APPROVED

        approved = verdict in (GovernorVerdict.APPROVED, GovernorVerdict.ADJUSTED, GovernorVerdict.OVERRIDDEN)

        governed = GovernedExecution(
            approved=approved,
            verdict=verdict,
            adjusted_intent_type=adjusted_intent_type,
            adjusted_magnitude=adjusted_magnitude,
            adjusted_risk_limit=adjusted_risk_limit,
            adjusted_max_slippage=adjusted_max_slippage,
            adjusted_time_preference=adjusted_time_preference,
            risk_multiplier=risk_multiplier,
            rejection_reason=rejection_reason,
            applied_rules=applied_rules,
            timestamp=time.time(),
        )

        self._history.append(governed)
        return governed

    # ── History queries ─────────────────────────────────────────────────────

    def get_history(self, n: int = 10) -> list[GovernedExecution]:
        """Return the last *n* governed executions.

        Parameters
        ----------
        n : int
            Number of recent entries to return (default 10).

        Returns
        -------
        list[GovernedExecution]
            The most recent *n* entries (or all if fewer exist).
        """
        return list(self._history[-n:])

    def get_last_verdict(self) -> GovernedExecution | None:
        """Return the most recent GovernedExecution, or None if empty."""
        if not self._history:
            return None
        return self._history[-1]

    # ── Internal helpers ────────────────────────────────────────────────────

    @staticmethod
    def _get_intent_type(intent: Any) -> str:
        """Extract intent type as a string from an enum member or plain string."""
        raw = getattr(intent, "intent_type", "HOLD")
        if isinstance(raw, Enum):
            return raw.value
        return str(raw)

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        """Safely convert a value to float, returning *default* on failure."""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def _get_regime_permissions(self) -> Any:
        """Retrieve permissions from the registered regime matrix.

        Duck-typed interface — the regime matrix should provide either:
        - ``get_permissions_for_current_regime()``, or
        - ``get_current_permissions()``.
        """
        if self._regime_matrix is None:
            return {}
        for method_name in ("get_permissions_for_current_regime", "get_current_permissions"):
            method = getattr(self._regime_matrix, method_name, None)
            if method is not None:
                return method()
        return {}

    @staticmethod
    def _get_allowed_actions(permissions: Any) -> list[str]:
        """Extract allowed actions from regime permissions.

        Supports dict-like objects with an ``allowed_actions`` key or
        attribute, or objects with an ``allowed_actions`` attribute directly.
        Returns an empty list if the structure is unrecognised.
        """
        if permissions is None:
            return []
        # dict-like
        if isinstance(permissions, dict):
            raw = permissions.get("allowed_actions", permissions.get("actions", []))
        else:
            # object with attribute
            raw = getattr(permissions, "allowed_actions", getattr(permissions, "actions", []))
        if not isinstance(raw, list):
            return [str(raw)]
        return [str(a) for a in raw]

    @staticmethod
    def _get_risk_allowed_actions(risk_profile: Any) -> list[str] | None:
        """Extract allowed actions from a risk profile.

        Returns None if the risk profile does not have an allowed_actions
        attribute (meaning no restriction).
        """
        if risk_profile is None:
            return None
        # dict-like
        if isinstance(risk_profile, dict):
            raw = risk_profile.get("allowed_actions", None)
        else:
            raw = getattr(risk_profile, "allowed_actions", None)
        if raw is None:
            return None
        if not isinstance(raw, list):
            return [str(raw)]
        return [str(a) for a in raw]

    @staticmethod
    def _check_override_conditions(raw_intent_type: str, execution_intent: Any) -> bool:
        """Check whether override conditions are met.

        Override conditions:
        - ``intent_type`` is EMERGENCY_STOP
        - ``regime_action_signal`` attribute is EMERGENCY_STOP
        """
        if raw_intent_type == "EMERGENCY_STOP":
            return True
        regime_signal = getattr(execution_intent, "regime_action_signal", None)
        if regime_signal is not None:
            signal_str = regime_signal.value if isinstance(regime_signal, Enum) else str(regime_signal)
            if signal_str == "EMERGENCY_STOP":
                return True
        return False
