"""Shared decision kernel — the ONE causal decision path for backtest and live.

The kernel owns ONLY the strategy-invocation seam. It must not know about:
masking, execution, positions, pnl, equity, or trades. Both the backtest engine
and the LiveRunner call `process(...)` with an already-built `bars` (real or
masked) + `history` (closes strictly before the current bar). This guarantees
the validated strategy object is the single source of truth in both
environments — no duplicated decision logic can drift.
"""
from typing import Any, Dict, List

from proxima_honest_backtest.engine.types import SignalResult


class DecisionKernel:
    """Minimal seam: strategy.on_bars(...) -> List[SignalResult]."""

    def process(
        self,
        strategy: Any,
        bars: Dict[str, Dict],
        history: Dict[str, Any],
    ) -> List[SignalResult]:
        signals = strategy.on_bars(bars, history)
        return signals or []


_KERNEL = DecisionKernel()


def generate_decisions(
    strategy: Any,
    bars: Dict[str, Dict],
    history: Dict[str, Any],
) -> List[SignalResult]:
    """Module-level convenience; returns [] on None."""
    return _KERNEL.process(strategy, bars, history)
