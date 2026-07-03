"""DecisionGate — Single arbitration authority for entry decisions.

Consolidates all inline entry gating checks into a single interface.
Evaluates SignalOutput objects against portfolio/market constraints
and produces Decision objects.

This enforces the Decision System boundary:
- Accepts SignalOutput from Signal System
- Rejects or accepts based on gate checks
- Outputs Decision for Execution System

The DecisionGate does NOT modify any state — it is a pure evaluation
function. Side effects (closing positions for migration/flip, updating
state) are handled by the Execution System after the Decision is made.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .contracts import Decision, GateResult, PortfolioState, SignalOutput

logger = logging.getLogger("proxima_ops.decision.decision_gate")


class DecisionGate:
    """Single authority for entry decision arbitration.

    Evaluates signals against all gates and produces a single Decision.
    This replaces ~300 lines of inline entry gating in run_proxima_demo.py.

    Parameters
    ----------
    max_positions : int, default=6
        Maximum concurrent positions allowed.
    min_hold_ticks_flip : int, default=5
        Minimum age before a position can be flipped.
    min_hold_ticks_migration : int, default=10
        Minimum age before a position can be migrated.
    """

    def __init__(
        self,
        max_positions: int = 6,
        min_hold_ticks_flip: int = 5,
        min_hold_ticks_migration: int = 10,
    ) -> None:
        self._max_positions = max_positions
        self._min_hold_ticks_flip = min_hold_ticks_flip
        self._min_hold_ticks_migration = min_hold_ticks_migration

    def evaluate(
        self,
        signal: SignalOutput,
        portfolio: PortfolioState,
        context: Optional[Dict[str, Any]] = None,
    ) -> Decision:
        """Evaluate a signal against all gates.

        Parameters
        ----------
        signal : SignalOutput
            Immutable signal from the Signal System.
        portfolio : PortfolioState
            Current portfolio snapshot.
        context : dict, optional
            Additional market/risk context for gate evaluation.

        Returns
        -------
        Decision
            The arbitration result — entry_authorized, exit_authorized,
            and rejection_reason if applicable.
        """
        ctx = context or {}
        gates: List[GateResult] = []

        gate = self._check_system_paused(ctx)
        gates.append(gate)
        if not gate.passed:
            return Decision(
                symbol=signal.symbol,
                entry_authorized=False,
                exit_authorized=True,
                rejection_reason=gate.reason,
            )

        gate = self._check_position_lock(signal.symbol, ctx)
        gates.append(gate)
        if not gate.passed:
            return Decision(
                symbol=signal.symbol,
                entry_authorized=False,
                exit_authorized=True,
                rejection_reason=gate.reason,
            )

        already_in_position = signal.symbol in portfolio.positions_by_symbol

        if already_in_position:
            gate = self._check_in_position(
                signal, portfolio, ctx
            )
            gates.append(gate)
            if not gate.passed:
                return Decision(
                    symbol=signal.symbol,
                    entry_authorized=False,
                    exit_authorized=True,
                    rejection_reason=gate.reason,
                )

        gate = self._check_max_positions(portfolio)
        gates.append(gate)
        if not gate.passed:
            return Decision(
                symbol=signal.symbol,
                entry_authorized=False,
                exit_authorized=True,
                rejection_reason=gate.reason,
            )

        gate = self._check_spread(signal.symbol, ctx)
        gates.append(gate)
        if not gate.passed:
            return Decision(
                symbol=signal.symbol,
                entry_authorized=False,
                exit_authorized=True,
                rejection_reason=gate.reason,
            )

        gate = self._check_tick(signal.symbol, ctx)
        gates.append(gate)
        if not gate.passed:
            return Decision(
                symbol=signal.symbol,
                entry_authorized=False,
                exit_authorized=True,
                rejection_reason=gate.reason,
            )

        gate = self._check_mof(ctx)
        gates.append(gate)
        if not gate.passed:
            return Decision(
                symbol=signal.symbol,
                entry_authorized=False,
                exit_authorized=True,
                rejection_reason=gate.reason,
            )

        gate = self._check_risk(ctx)
        gates.append(gate)
        if not gate.passed:
            return Decision(
                symbol=signal.symbol,
                entry_authorized=False,
                exit_authorized=True,
                rejection_reason=gate.reason,
            )

        return Decision(
            symbol=signal.symbol,
            entry_authorized=True,
            exit_authorized=True,
        )

    def _check_system_paused(self, ctx: dict) -> GateResult:
        if ctx.get("paused", False):
            return GateResult("RISK_LIMIT", False, "System paused by risk limit")
        return GateResult("RISK_LIMIT", True)

    def _check_position_lock(self, symbol: str, ctx: dict) -> GateResult:
        locks = ctx.get("position_locks", {})
        if locks.get(symbol, 0) > 0:
            return GateResult("POSITION_LOCK", False, f"Position lock active for {symbol}")
        return GateResult("POSITION_LOCK", True)

    def _check_in_position(
        self,
        signal: SignalOutput,
        portfolio: PortfolioState,
        ctx: dict,
    ) -> GateResult:
        """Check if already in position — evaluates flip/migration eligibility."""
        held = portfolio.positions_by_symbol.get(signal.symbol, {})
        held_dir = held.get("direction", 0)
        sig_dir = signal.direction
        is_flip = held_dir != 0 and sig_dir != 0 and held_dir != sig_dir
        held_age = held.get("age", 0)

        if is_flip:
            if held_age < self._min_hold_ticks_flip:
                return GateResult(
                    "FLIP_TOO_YOUNG", False,
                    f"Flip blocked: age {held_age} < {self._min_hold_ticks_flip}",
                )
            cooldown = ctx.get("flip_cooldowns", {}).get(signal.symbol, 0)
            if cooldown > 0:
                return GateResult(
                    "FLIP_COOLDOWN", False,
                    f"Flip cooldown: {cooldown} ticks remaining",
                )
            return GateResult("FLIP", True)

        held_q = held.get("q_score", 0)
        new_q = ctx.get("q_new", 0)
        delta = new_q - held_q
        min_delta = max(0.08, abs(held_q) * 0.30)

        if held_age < self._min_hold_ticks_migration and delta <= min_delta:
            return GateResult(
                "POSITION_EXISTS", False,
                f"Position exists, migration not warranted: "
                f"delta={delta:.3f} < min={min_delta:.3f}, "
                f"age={held_age} < {self._min_hold_ticks_migration}",
            )

        if delta > min_delta:
            return GateResult("MIGRATION", True)

        return GateResult(
            "POSITION_EXISTS", False,
            f"Position exists: delta={delta:.3f} < min={min_delta:.3f}",
        )

    def _check_max_positions(self, portfolio: PortfolioState) -> GateResult:
        if portfolio.current_positions >= self._max_positions:
            return GateResult(
                "MAX_POSITIONS", False,
                f"Max positions: {portfolio.current_positions} >= {self._max_positions}",
            )
        return GateResult("MAX_POSITIONS", True)

    def _check_spread(self, symbol: str, ctx: dict) -> GateResult:
        spread_state = ctx.get("spread_states", {}).get(symbol, {})
        if spread_state.get("invalid", False):
            return GateResult("INVALID_SPREAD", False, "Spread invalid for symbol")
        if not spread_state.get("norm_passed", True):
            return GateResult("SPREAD_NORM", False, "Spread normalization failed")
        return GateResult("SPREAD", True)

    def _check_tick(self, symbol: str, ctx: dict) -> GateResult:
        if not ctx.get("has_tick", True):
            return GateResult("NO_TICK", False, "No tick available")
        return GateResult("NO_TICK", True)

    def _check_mof(self, ctx: dict) -> GateResult:
        mof_blocked = ctx.get("mof_blocked", False)
        if mof_blocked:
            return GateResult("MOF_DEGRADED", False, "Market observability degraded")
        return GateResult("MOF_DEGRADED", True)

    def _check_risk(self, ctx: dict) -> GateResult:
        risk_blocked = ctx.get("risk_blocked", False)
        if risk_blocked:
            return GateResult(
                "RHL_BLOCKED", False,
                ctx.get("risk_reason", "Risk hardening layer blocked"),
            )
        return GateResult("RHL_BLOCKED", True)
