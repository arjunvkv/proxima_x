"""proxima_ops.backtest — generalized backtest -> live engine.

Strategy-agnostic core: a declarative StrategySpec + engine (no-lookahead bar
simulation), cost-correct PnL, and the full validation battery (train/val,
walk-forward, purple shuffle, determinism, server-clock day-keying). Strategies
are added as specs — no engine changes.
"""
from .spec import StrategySpec, SignalSpec, ExitSpec
from .engine import run_strategy, simulate_exit, session_signal_indices, bar_hour
from .validation import metrics, gate, split_by_ts, walk_forward, purple_edge, determinism
from .pnl import trade_to_usd, COMMISSION_PER_LOT

__all__ = [
    "StrategySpec", "SignalSpec", "ExitSpec",
    "run_strategy", "simulate_exit", "session_signal_indices", "bar_hour",
    "metrics", "gate", "split_by_ts", "walk_forward", "purple_edge", "determinism",
    "trade_to_usd", "COMMISSION_PER_LOT",
]