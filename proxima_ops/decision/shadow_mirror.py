"""Shadow Decision Mirror — Non-intrusive parallel DecisionGate observer.

Runs DecisionGate in pure shadow mode alongside live execution.
NEVER writes to runtime state. NEVER influences execution.
Records divergence between shadow decisions and actual system decisions.

Safe insertion points (from execution trace linearization):
  SZ3: line 2546 — after arbitration, before audit (signal input captured)
  SZ5: line 3654 — after bridge, before per-symbol loop (portfolio input captured)
  SZ7: line 5331 — after regime snapshot, before cycle wrap (divergence logged)
"""

from __future__ import annotations

import copy
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from .contracts import Decision, PortfolioState, SignalOutput
from .decision_gate import DecisionGate

logger = logging.getLogger("proxima_ops.decision.shadow_mirror")


class ShadowDecisionMirror:
    """Non-intrusive parallel DecisionGate observer.

    Caches per-symbol signals and portfolio/context state, then evaluates
    the DecisionGate for each symbol independently at cycle end.

    Divergence is recorded when shadow and live systems disagree.
    """

    def __init__(
        self,
        max_positions: int = 6,
        min_hold_ticks_flip: int = 5,
        min_hold_ticks_migration: int = 10,
    ) -> None:
        self._gate = DecisionGate(
            max_positions=max_positions,
            min_hold_ticks_flip=min_hold_ticks_flip,
            min_hold_ticks_migration=min_hold_ticks_migration,
        )
        self._signals: Dict[str, SignalOutput] = {}
        self._portfolio: Optional[PortfolioState] = None
        self._context: Dict[str, Any] = {}
        self._shadow_history: List[dict] = []
        self._divergence_log: List[dict] = []
        self._gate_hit_counts: Dict[str, int] = defaultdict(int)
        self._total_evaluations: int = 0
        self._divergences: int = 0
        self._agreements: int = 0

    def observe_signal(
        self,
        symbol: str,
        direction: int,
        strength: float,
        ecdf_rank: float,
        confidence: float,
        source: str,
        horizon: int = 10,
    ) -> None:
        """Capture per-symbol signal output at safe zone SZ3 (after arbitration).

        Stored in a dict keyed by symbol — no live references retained.
        """
        self._signals[symbol] = SignalOutput(
            symbol=symbol,
            direction=direction,
            strength=strength,
            horizon=horizon,
            ecdf_rank=ecdf_rank,
            confidence=confidence,
            source=source,
        )

    def observe_portfolio(
        self,
        current_positions: int = 0,
        max_positions: int = 6,
        positions_by_symbol: Optional[Dict[str, Any]] = None,
        account_balance: float = 0.0,
        session_pnl: float = 0.0,
    ) -> None:
        """Capture portfolio state at safe zone SZ5 (before per-symbol loop)."""
        self._portfolio = PortfolioState(
            current_positions=current_positions,
            max_positions=max_positions,
            positions_by_symbol=copy.deepcopy(positions_by_symbol or {}),
            account_balance=account_balance,
            session_pnl=session_pnl,
        )

    def observe_context(self, context: Dict[str, Any]) -> None:
        """Capture additional market/risk context for shadow evaluation.

        All values are deep-copied to prevent reference leakage.
        """
        self._context = copy.deepcopy(context)

    def evaluate_shadow(self, symbol: str, cycle_id: int = 0) -> Optional[Decision]:
        """Run DecisionGate for a single symbol on cached inputs (pure shadow).

        Returns the Decision from shadow evaluation, or None if
        signal/portfolio not yet available for this symbol.
        """
        signal = self._signals.get(symbol)
        portfolio = self._portfolio

        if signal is None or portfolio is None:
            return None

        decision = self._gate.evaluate(signal, portfolio, self._context)
        self._total_evaluations += 1
        if not decision.entry_authorized:
            gate_name = decision.rejection_reason or "UNKNOWN"
            self._gate_hit_counts[gate_name] += 1

        self._shadow_history.append({
            "cycle": cycle_id,
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "signal": signal,
            "portfolio": portfolio,
            "decision": decision,
        })

        return decision

    def evaluate_all(self, symbols: List[str], cycle_id: int = 0) -> Dict[str, Decision]:
        """Evaluate DecisionGate for all specified symbols in one call."""
        results: Dict[str, Decision] = {}
        for sym in symbols:
            dec = self.evaluate_shadow(sym, cycle_id=cycle_id)
            if dec is not None:
                results[sym] = dec
        return results

    def record_divergence(
        self,
        symbol: str,
        shadow_decision: Decision,
        live_blocked: bool,
        live_reason: Optional[str] = None,
        cycle_id: int = 0,
    ) -> None:
        """Record a divergence between shadow and live decisions.

        A divergence occurs when:
        - Shadow says ENTER but live blocked
        - Shadow says BLOCK but live entered
        - Shadow and live blocked for different reasons
        """
        live_authorized = not live_blocked

        if shadow_decision.entry_authorized == live_authorized:
            if shadow_decision.entry_authorized or shadow_decision.rejection_reason == live_reason:
                self._agreements += 1
                return

        divergence_type = self._classify_divergence(
            shadow_decision, live_authorized, live_reason
        )
        self._divergences += 1
        self._divergence_log.append({
            "cycle": cycle_id,
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "divergence_type": divergence_type,
            "shadow_authorized": shadow_decision.entry_authorized,
            "shadow_reason": shadow_decision.rejection_reason,
            "live_authorized": live_authorized,
            "live_reason": live_reason,
        })
        logger.info(
            "[SHADOW_DIVERGENCE] %s divergence=%s shadow_ok=%s shadow_reason=%s "
            "live_ok=%s live_reason=%s",
            symbol, divergence_type,
            shadow_decision.entry_authorized, shadow_decision.rejection_reason,
            live_authorized, live_reason,
        )

    def _classify_divergence(
        self,
        shadow: Decision,
        live_authorized: bool,
        live_reason: Optional[str],
    ) -> str:
        if shadow.entry_authorized and not live_authorized:
            return "SHADOW_ACCEPTS_LIVE_BLOCKS"
        if not shadow.entry_authorized and live_authorized:
            return "SHADOW_BLOCKS_LIVE_ACCEPTS"
        return f"DIFFERENT_REASON(shadow={shadow.rejection_reason}, live={live_reason})"

    def summary(self) -> Dict[str, Any]:
        """Return divergence summary statistics."""
        total = self._total_evaluations
        div = self._divergences
        agree = self._agreements
        agree_rate = round(agree / max(total, 1), 4)

        top_gates = sorted(
            self._gate_hit_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        return {
            "total_evaluations": total,
            "agreements": agree,
            "divergences": div,
            "agreement_rate": agree_rate,
            "divergence_rate": round(div / max(total, 1), 4),
            "top_blocking_gates": top_gates,
            "dominant_blocking_gate": top_gates[0] if top_gates else ("NONE", 0),
        }

    def recent_divergences(self, n: int = 20) -> List[dict]:
        """Return the most recent N divergences."""
        return self._divergence_log[-n:]

    def clear_cycle(self) -> None:
        """Clear per-cycle signal cache (retain history and totals)."""
        self._signals.clear()
        self._portfolio = None
        self._context = {}

    def reset(self) -> None:
        """Clear all internal state (for fresh start between runs)."""
        self._shadow_history.clear()
        self._divergence_log.clear()
        self._gate_hit_counts.clear()
        self._total_evaluations = 0
        self._divergences = 0
        self._agreements = 0
        self._signals.clear()
        self._portfolio = None
        self._context = {}
