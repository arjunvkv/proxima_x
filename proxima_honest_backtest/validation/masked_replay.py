"""Masked-forming-bar replay — provable anti-lookahead gate for strategies.

The honest contract enforced here:

    at the bar the strategy evaluates (timestamp ``ts``), the ONLY information it
    may legally act on is:
      * ``history`` — closes of bars STRICTLY before ``ts`` (the engine now serves
        these; the current bar's close is appended only AFTER the decision), and
      * the current bar's ``open`` (known at the start of the bar).

    The strategy must carry ``entry_price`` (the entry bar's open) in its signal
    metadata; execution fills there. Reading the current (forming) bar's
    ``close``/``high``/``low`` to make a decision is SAME-BAR LOOKAHEAD — exactly
    the bug that inflated Ultra Monster.

Probe: run the identical backtest twice —

  1. ``full``   — strategy sees real OHLC for every bar.
  2. ``masked`` — the forming bar's close/high/low are served as NaN.

If the two runs produce the IDENTICAL trade set, the strategy provably never
consumed the forming bar's close/high/low: it passes. If they diverge, the
strategy leaks the forming bar (Ultra Monster class) and fails the gate.

The gate is mandatory: a strategy cannot be marked "survives" in
``strategies/run_all.py`` without ``prove_no_lookahead(...).passed``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from proxima_honest_backtest.engine.types import Trade

_Entry = tuple


def _trade_fingerprint(t: Trade) -> tuple:
    return (
        t.timestamp,
        t.symbol,
        t.side,
        round(float(t.price), 8),
        round(float(t.pnl), 6),
        round(float(t.commission), 6),
        round(float(t.quantity), 4),
    )


def _fingerprint_set(trades: List[Trade]) -> Dict[tuple, int]:
    counts: Dict[tuple, int] = {}
    for t in trades:
        fp = _trade_fingerprint(t)
        counts[fp] = counts.get(fp, 0) + 1
    return counts


@dataclass
class MaskedReplayVerdict:
    strategy_name: str
    passed: bool
    full_n_trades: int
    masked_n_trades: int
    differing_trades: int
    full_net_pnl: float
    masked_net_pnl: float
    full_first_diff: Optional[tuple] = None
    masked_first_diff: Optional[tuple] = None
    notes: List[str] = field(default_factory=list)

    def report_line(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.strategy_name}: full={self.full_n_trades}T/${self.full_net_pnl:+.2f} "
            f"masked={self.masked_n_trades}T/${self.masked_net_pnl:+.2f} "
            f"diff={self.differing_trades}"
        )


def _compare(full_trades: List[Trade], masked_trades: List[Trade]) -> MaskedReplayVerdict:
    full_fp = _fingerprint_set(full_trades)
    masked_fp = _fingerprint_set(masked_trades)

    all_keys = sorted(set(full_fp) | set(masked_fp))
    differing = [k for k in all_keys if full_fp.get(k, 0) != masked_fp.get(k, 0)]

    full_first = None
    masked_first = None
    if differing:
        first_diff = differing[0]
        full_first = next((t for t in full_trades if _trade_fingerprint(t) == first_diff), None)
        masked_first = next((t for t in masked_trades if _trade_fingerprint(t) == first_diff), None)

    full_net = sum(t.pnl for t in full_trades)
    masked_net = sum(t.pnl for t in masked_trades)

    return MaskedReplayVerdict(
        strategy_name=full_trades[0].__class__.__name__ if full_trades else "",
        passed=not differing,
        full_n_trades=len(full_trades),
        masked_n_trades=len(masked_trades),
        differing_trades=len(differing),
        full_net_pnl=full_net,
        masked_net_pnl=masked_net,
        full_first_diff=full_first,
        masked_first_diff=masked_first,
    )


def _fresh_simulator(execution_simulator: Optional[Any]) -> Optional[Any]:
    """Return a fresh, identically-seeded simulator for a probe run.

    The probe must be bit-identical across the full and masked runs; a SHARED
    simulator would advance its RNG stream between runs and make every stochastic
    fill (spread/slippage/partial-fill) differ even for provably-honest strategies.
    A fresh instance with the same profile + seed restores determinism.
    """
    if execution_simulator is None:
        return None
    sim_cls = execution_simulator.__class__
    seed = getattr(execution_simulator, "_seed", None)
    profile = getattr(execution_simulator, "profile_name", None)
    if seed is not None and profile is not None:
        try:
            return sim_cls(profile, seed=seed)
        except TypeError:
            return sim_cls(profile)
    return execution_simulator


def prove_no_lookahead_multi(
    strategy_factory: Callable[[], Any],
    pairs_data: Dict[str, Any],
    engine_cls: Any,
    execution_simulator: Optional[Any] = None,
    pre_aligned: Optional[List[Dict[str, Any]]] = None,
) -> MaskedReplayVerdict:
    """Run the honest gate for a multi-pair strategy.

    ``strategy_factory`` must return a FRESH strategy instance (reset each run).
    ``engine_cls`` is the MultiPairBacktestEngine class. Returns the verdict.
    """
    full_engine = engine_cls(strategy_factory(), _fresh_simulator(execution_simulator))
    full_result = full_engine.run(pairs_data, pre_aligned=pre_aligned, mask_for_strategy=False)

    masked_engine = engine_cls(strategy_factory(), _fresh_simulator(execution_simulator))
    masked_result = masked_engine.run(pairs_data, pre_aligned=pre_aligned, mask_for_strategy=True)

    verdict = _compare(full_result.trades, masked_result.trades)
    verdict.strategy_name = full_result.strategy_name or "?"
    return verdict


def prove_no_lookahead_single(
    strategy_factory: Callable[[], Any],
    symbol: str,
    data: Any,
    engine_cls: Any,
    execution_simulator: Optional[Any] = None,
) -> MaskedReplayVerdict:
    """Run the honest gate for a single-pair (BaseStrategy) strategy."""
    full_engine = engine_cls(strategy_factory(), _fresh_simulator(execution_simulator))
    full_result = full_engine.run(symbol, data, mask_for_strategy=False)

    masked_engine = engine_cls(strategy_factory(), _fresh_simulator(execution_simulator))
    masked_result = masked_engine.run(symbol, data, mask_for_strategy=True)

    verdict = _compare(full_result.trades, masked_result.trades)
    verdict.strategy_name = full_result.strategy_name or "?"
    return verdict
