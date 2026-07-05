"""
execution_finalizer.py — Final transformation before any execution.

Produces the final actionable execution packet (FinalExecutionOrder) after all
governance stages have completed.  This is the last stop before real execution.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# FinalDirection
# ---------------------------------------------------------------------------


class FinalDirection(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    EXIT = "EXIT"
    NONE = "NONE"


# ---------------------------------------------------------------------------
# FinalExecutionOrder
# ---------------------------------------------------------------------------


@dataclass
class FinalExecutionOrder:
    """The final, validated, risk-adjusted execution packet.

    Attributes
    ----------
    symbol : str
        Trading symbol.
    direction : FinalDirection
        BUY, SELL, HOLD, REDUCE, EXIT, NONE.
    size : float
        Final risk-adjusted size (0.0 to 1.0 fraction).
    risk_adjusted_size : float
        Size after all risk scaling.
    slippage_model : str
        "IMMEDIATE", "ADAPTIVE", or "PASSIVE".
    time_constraints : str
        "IMMEDIATE", "FAST", "NORMAL", or "SLOW".
    max_slippage_bps : float
        Maximum acceptable slippage in basis points.
    execution_price : float | None
        Optional limit price.
    order_type : str
        "MARKET", "LIMIT", or "STOP".
    reason : str
        Human-readable reason for this final order.
    parent_governed : Any | None
        Reference to the :class:`GovernedExecution` that produced this order.
    timestamp : float
        Unix timestamp when this packet was created.
    """
    symbol: str
    direction: FinalDirection
    size: float
    risk_adjusted_size: float
    slippage_model: str
    time_constraints: str
    max_slippage_bps: float
    execution_price: float | None
    order_type: str
    reason: str
    parent_governed: Any | None
    timestamp: float


# ---------------------------------------------------------------------------
# Direction mapping lookup
# ---------------------------------------------------------------------------

_DIRECTION_MAP: dict[str, FinalDirection] = {
    "BUY_STRONG": FinalDirection.BUY,
    "BUY_MODERATE": FinalDirection.BUY,
    "BUY_LIGHT": FinalDirection.BUY,
    "SELL_SHORT": FinalDirection.SELL,
    "HOLD": FinalDirection.HOLD,
    "TRANSITION_PREP": FinalDirection.HOLD,
    "REDUCE_LIGHT": FinalDirection.REDUCE,
    "REDUCE_MODERATE": FinalDirection.REDUCE,
    "REDUCE_STRONG": FinalDirection.REDUCE,
    "EXIT_ALL": FinalDirection.EXIT,
    "EMERGENCY_STOP": FinalDirection.EXIT,
}

# Slippage-model / order-type lookup keyed by time_preference string.
_SLIPPAGE_MAP: dict[str, tuple[str, str]] = {
    "IMMEDIATE": ("IMMEDIATE", "MARKET"),
    "FAST": ("ADAPTIVE", "MARKET"),
    "NORMAL": ("ADAPTIVE", "LIMIT"),
    "SLOW": ("PASSIVE", "LIMIT"),
}


# ---------------------------------------------------------------------------
# ExecutionFinalizer
# ---------------------------------------------------------------------------


class ExecutionFinalizer:
    """Transforms a :class:`GovernedExecution` into a :class:`FinalExecutionOrder`.

    This is the **final** transformation before any execution system acts on
    the packet.  All governance, risk, and regime adjustments have already
    been applied by upstream pipeline stages.
    """

    def __init__(self, default_symbol: str = "DEFAULT") -> None:
        self._default_symbol = default_symbol
        self._history: list[FinalExecutionOrder] = []

    # ── Public API ──────────────────────────────────────────────────────────

    def finalize(
        self,
        governed_execution: Any,
        intelligence_frame: Any = None,
    ) -> FinalExecutionOrder:
        """Convert a ``GovernedExecution`` into a ``FinalExecutionOrder``.

        Parameters
        ----------
        governed_execution : Any
            Duck-typed ``GovernedExecution`` providing the attributes:

            - ``.verdict`` (``GovernorVerdict`` enum or string)
            - ``.approved`` (bool)
            - ``.adjusted_intent_type`` (str)
            - ``.adjusted_magnitude`` (float)
            - ``.adjusted_risk_limit`` (float)
            - ``.adjusted_max_slippage`` (float)
            - ``.adjusted_time_preference`` (str)
            - ``.risk_multiplier`` (float)
            - ``.rejection_reason`` (str | None)
            - ``.applied_rules`` (list[str])
            - ``.timestamp`` (float)

        intelligence_frame : Any, optional
            Optional intelligence context.  May carry a ``.symbol`` attribute
            for symbol resolution.

        Returns
        -------
        FinalExecutionOrder
            The final, validated, risk-adjusted execution packet.
        """
        # -------------------------------------------------------------------
        # Extract attributes (duck-typed)
        # -------------------------------------------------------------------
        intent_type: str = self._get_str_attr(
            governed_execution, "adjusted_intent_type", "HOLD"
        )
        magnitude: float = self._safe_float(
            getattr(governed_execution, "adjusted_magnitude", 0.0), 0.0
        )
        risk_multiplier: float = self._safe_float(
            getattr(governed_execution, "risk_multiplier", 1.0), 1.0
        )
        max_slippage: float = self._safe_float(
            getattr(governed_execution, "adjusted_max_slippage", 0.0), 0.0
        )
        time_preference: str = self._get_str_attr(
            governed_execution, "adjusted_time_preference", "NORMAL"
        )
        verdict_raw: Any = getattr(governed_execution, "verdict", None)
        verdict_str: str = self._verdict_to_str(verdict_raw)
        approved: bool = bool(getattr(governed_execution, "approved", False))
        rejection_reason: str | None = getattr(
            governed_execution, "rejection_reason", None
        )
        applied_rules: list[str] = getattr(
            governed_execution, "applied_rules", []
        )
        gov_timestamp: float = self._safe_float(
            getattr(governed_execution, "timestamp", 0.0), time.time()
        )

        # Detect emergency stop from intent type or verdict
        is_emergency: bool = self._is_emergency(intent_type, verdict_str)

        # Handle BLOCKED verdict
        is_blocked: bool = (
            verdict_str == "BLOCKED" or not approved
        )

        # -------------------------------------------------------------------
        # Symbol resolution
        # -------------------------------------------------------------------
        symbol: str = self._resolve_symbol(intelligence_frame)

        # -------------------------------------------------------------------
        # Direction mapping
        # -------------------------------------------------------------------
        if is_blocked:
            direction = FinalDirection.NONE
        else:
            direction = self._map_direction(intent_type)

        # -------------------------------------------------------------------
        # Size calculations
        # -------------------------------------------------------------------
        if is_blocked:
            size = 0.0
            risk_adjusted_size = 0.0
        else:
            size = magnitude
            risk_adjusted_size = magnitude * risk_multiplier

        # -------------------------------------------------------------------
        # Slippage model / order type mapping
        # -------------------------------------------------------------------
        slippage_model, order_type = self._map_slippage_and_order_type(
            time_preference, is_emergency
        )

        # -------------------------------------------------------------------
        # Time constraints
        # -------------------------------------------------------------------
        time_constraints = (
            "IMMEDIATE" if is_emergency else time_preference
        )

        # -------------------------------------------------------------------
        # Max slippage (bps)
        # -------------------------------------------------------------------
        max_slippage_bps = max_slippage

        # -------------------------------------------------------------------
        # Execution price (not set — no pricing pipeline yet)
        # -------------------------------------------------------------------
        execution_price: float | None = None

        # -------------------------------------------------------------------
        # Reason string
        # -------------------------------------------------------------------
        reason = self._build_reason(
            verdict_str=verdict_str,
            applied_rules=applied_rules,
            is_blocked=is_blocked,
            is_emergency=is_emergency,
            rejection_reason=rejection_reason,
        )

        # -------------------------------------------------------------------
        # Assemble the final packet
        # -------------------------------------------------------------------
        order = FinalExecutionOrder(
            symbol=symbol,
            direction=direction,
            size=size,
            risk_adjusted_size=risk_adjusted_size,
            slippage_model=slippage_model,
            time_constraints=time_constraints,
            max_slippage_bps=max_slippage_bps,
            execution_price=execution_price,
            order_type=order_type,
            reason=reason,
            parent_governed=governed_execution,
            timestamp=gov_timestamp,
        )

        self._history.append(order)
        return order

    # ── History queries ─────────────────────────────────────────────────────

    def get_history(self, n: int = 10) -> list[FinalExecutionOrder]:
        """Return the last *n* finalized execution orders.

        Parameters
        ----------
        n : int
            Number of recent entries to return (default 10).

        Returns
        -------
        list[FinalExecutionOrder]
            The most recent *n* entries (or all if fewer exist).
        """
        return list(self._history[-n:])

    def get_last_order(self) -> FinalExecutionOrder | None:
        """Return the most recent :class:`FinalExecutionOrder`, or None."""
        if not self._history:
            return None
        return self._history[-1]

    # ── Internal helpers ────────────────────────────────────────────────────

    def _resolve_symbol(self, intelligence_frame: Any) -> str:
        """Resolve trading symbol from intelligence frame, or return default.

        Checks (in order):
        1. ``intelligence_frame.symbol``
        2. ``intelligence_frame.ticker``
        3. Fallback to ``self._default_symbol``
        """
        if intelligence_frame is None:
            return self._default_symbol

        symbol = getattr(intelligence_frame, "symbol", None)
        if symbol is not None:
            return str(symbol)

        ticker = getattr(intelligence_frame, "ticker", None)
        if ticker is not None:
            return str(ticker)

        return self._default_symbol

    @staticmethod
    def _map_direction(intent_type: str) -> FinalDirection:
        """Map an adjusted intent type string to a FinalDirection.

        Returns ``FinalDirection.NONE`` for unrecognised intent types.
        """
        return _DIRECTION_MAP.get(intent_type, FinalDirection.NONE)

    @staticmethod
    def _map_slippage_and_order_type(
        time_preference: str,
        is_emergency: bool,
    ) -> tuple[str, str]:
        """Map time preference to (slippage_model, order_type).

        EMERGENCY_STOP always returns ("IMMEDIATE", "MARKET").
        Falls back to ("ADAPTIVE", "LIMIT") for unknown preferences.
        """
        if is_emergency:
            return "IMMEDIATE", "MARKET"

        result = _SLIPPAGE_MAP.get(time_preference)
        if result is not None:
            return result

        return "ADAPTIVE", "LIMIT"

    @staticmethod
    def _is_emergency(intent_type: str, verdict_str: str) -> bool:
        """Detect whether this order is an emergency stop.

        Triggered when the intent type is ``EMERGENCY_STOP`` or the
        verdict is ``OVERRIDDEN`` (which the governor uses for emergency
        overrides).
        """
        return intent_type == "EMERGENCY_STOP" or verdict_str == "OVERRIDDEN"

    @staticmethod
    def _build_reason(
        verdict_str: str,
        applied_rules: list[str],
        is_blocked: bool,
        is_emergency: bool,
        rejection_reason: str | None,
    ) -> str:
        """Build a human-readable reason string.

        Format::

            VERDICT | rule1, rule2, rule3 [| BLOCKED: ...] [| EMERGENCY STOP — URGENT]
        """
        parts: list[str] = [verdict_str]

        # Concatenate first 3 applied rules
        if applied_rules:
            rules_part = ", ".join(str(r) for r in applied_rules[:3])
            parts.append(rules_part)

        if is_blocked and rejection_reason:
            parts.append(f"BLOCKED: {rejection_reason}")

        if is_emergency:
            parts.append("EMERGENCY STOP \u2014 URGENT")

        return " | ".join(parts)

    @staticmethod
    def _verdict_to_str(verdict: Any) -> str:
        """Convert a verdict (enum member or string) to its string value."""
        if verdict is None:
            return "UNKNOWN"
        if isinstance(verdict, Enum):
            return str(verdict.value)
        return str(verdict)

    @staticmethod
    def _get_str_attr(obj: Any, attr: str, default: str) -> str:
        """Safely extract a string attribute from a duck-typed object."""
        val = getattr(obj, attr, None)
        if val is None:
            return default
        if isinstance(val, Enum):
            return str(val.value)
        return str(val)

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        """Safely convert a value to float, returning *default* on failure."""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
