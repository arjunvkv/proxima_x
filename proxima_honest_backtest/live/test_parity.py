"""Phase 6 Test A — replay parity: LiveRunner vs backtest on the SAME aligned bars.

The canonical parity proof (per GPT consensus, option (a)): ship ONLY Tokyo H0 —
the single strategy whose interface path (DecisionKernel -> MultiPairStrategy.on_bars)
has a dedicated parity gate. V2z is not tested here (follow-up).

Method:
  backtest: MultiPairBacktestEngine(ship).run(data) -> trades -> Level-1 fingerprints
  live:     LiveRunner(ship, ReplayFeed(engine._align_bars(data)), LiveExecutor paper)
  compare  Level-1 fingerprints (ts, symbol, type, normalized side) — never price.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proxima_honest_backtest.execution.execution_simulator import ExecutionSimulator
from proxima_honest_backtest.live.config import get_ship
from proxima_honest_backtest.live.executor import LiveExecutor
from proxima_honest_backtest.live.feed import ReplayFeed
from proxima_honest_backtest.live.parity import (compare_level1,
                                                 extract_decision_enters)
from proxima_honest_backtest.live.runner import LiveRunner
from proxima_honest_backtest.strategies.multi_pair_engine import MultiPairBacktestEngine


def load_pairs(pairs, start_mon=1, end_mon=3, year=2026):
    out = {}
    try:
        from data.providers.mt5_provider import MT5Provider
    except Exception:
        return out
    p = MT5Provider()
    for sym in pairs:
        frames = []
        for m in range(start_mon, end_mon + 1):
            try:
                df = p.load_rates(sym, year, m, "m5")
            except Exception:
                df = pd.DataFrame()
            if not df.empty:
                frames.append(df)
        if frames:
            d = pd.concat(frames, ignore_index=True)
            d.sort_values("time", inplace=True)
            d.reset_index(drop=True, inplace=True)
            out[sym] = d
    return out


def test_tokyo_parity():
    ship = get_ship("tokyo_h0")
    data = load_pairs(ship.pairs)
    if not data:
        print("[SKIP] tokyo parity: no data available")
        return
    ex = ExecutionSimulator("exness")
    bt = MultiPairBacktestEngine(ship.factory(), ex)
    bt_result = bt.run(data)
    bt_enters = extract_decision_enters(bt_result.decisions)

    # live side: same aligned records -> ReplayFeed -> LiveRunner
    align_engine = MultiPairBacktestEngine(ship.factory(), ex)
    aligned = align_engine._align_bars(data)
    strat = ship.factory()
    live_ex = LiveExecutor(ship.pairs, magic_base=400000, base_lot=0.15,
                           mode="paper", spread_model_half=0.0)
    runner = LiveRunner(strat, ReplayFeed(aligned), live_ex, ship.pairs, persist=False)
    runner.run_replay(ReplayFeed(aligned))

    res = compare_level1(bt_enters, runner.decisions)
    assert res["passed"], f"Tokyo parity FAIL: {res}"
    assert res["n_backtest"] > 0, "expected >= 1 Tokyo ENTER in 3-mo slice"
    assert res["n_live"] == res["n_backtest"], f"count mismatch: {res}"
    print(f"[PASS] Tokyo replay parity: {res['n_backtest']} ENTERs exact match")


def test_ship_only_live_ready():
    from proxima_honest_backtest.live.config import list_live_ready
    keys = [c.key for c in list_live_ready()]
    assert "tokyo_h0" in keys
    assert "v2z_z6_long" not in keys, "v2z must not be live-ready until parity gate"
    print(f"[PASS] live-ready ships = {keys}")