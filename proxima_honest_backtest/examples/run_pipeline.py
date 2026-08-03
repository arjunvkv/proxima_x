#!/usr/bin/env python3
"""End-to-end V2+z backtest pipeline:
   Data → Strategy → Backtest → Validation → Broker Comparison → Monte Carlo."""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.backtest_engine import BacktestEngine, BacktestResult
from examples.v2z_strategy import V2zStrategy
from data.providers.mt5_provider import MT5Provider
from execution.execution_simulator import ExecutionSimulator, list_broker_profiles
from validation.gauntlet import OverfitGauntlet
from validation.linter import LookAheadLinter
from validation.walk_forward import WalkForwardValidator
from research.monte_carlo import MonteCarloSimulator
from research.broker_comparison import BrokerComparer


def step1_load_data(symbol: str) -> "pd.DataFrame":
    from datetime import datetime
    import pandas as pd

    p = MT5Provider()
    frames = []
    for year, month in [(2026, 1), (2026, 2), (2026, 3), (2026, 4), (2026, 5), (2026, 6), (2026, 7)]:
        df = p.load_rates(symbol, year, month, "m5")
        if not df.empty:
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No M5 data for {symbol}")
    data = pd.concat(frames, ignore_index=True)
    data.sort_values("time", inplace=True)
    data.reset_index(drop=True, inplace=True)
    print(f"  {symbol}: {len(data)} M5 bars ({data['time'].min()} → {data['time'].max()})")
    return data


def step2_lint_strategy():
    linter = LookAheadLinter()
    result = linter.lint_file(Path(__file__).parent / "v2z_strategy.py")
    print(f"\n  Lint {'PASS' if result.passed else 'FAIL'} "
          f"({result.error_count} errors, {result.warning_count} warnings)")
    if result.violations:
        for v in result.violations[:5]:
            print(f"    {v}")
    return result


def step3_backtest(strategy, symbol, data):
    import time
    start = time.time()
    engine = BacktestEngine(strategy, ExecutionSimulator("ftmo"))
    result = engine.run(symbol, data)
    elapsed = time.time() - start
    print(f"\n  {result.n_bars:,} bars → {result.n_trades} trades in {elapsed:.2f}s")
    print(f"  Net PnL: ${result.net_pnl:+.2f}  |  Sharpe: {result.sharpe:.2f}")
    print(f"  Win Rate: {result.win_rate*100:.1f}%  |  PF: {result.profit_factor:.2f}")
    print(f"  Max DD: {result.max_drawdown_pct:.2f}%  |  Reconciled: {result.reconciliation_pass}")
    return result


def step4_gauntlet(returns):
    gauntlet = OverfitGauntlet()
    g_result = gauntlet.run(returns, "V2zStrategy")
    print(f"\n  Gauntlet {'PASS' if g_result.passed else 'FAIL'}")
    print(f"  DSR: {g_result.deflated_sharpe}")
    print(f"  PBO: {g_result.prob_backtest_overfit}")
    print(f"  Sign-test p: {g_result.sign_test_pvalue}")
    return g_result


def step5_walk_forward(returns, timestamps):
    wf = WalkForwardValidator(n_windows=4, test_size=0.2, embargo_days=3)
    wf_result = wf.run_simple(returns, timestamps)
    print(f"\n  Walk-Forward OOS Sharpe: {wf_result.avg_oos_sharpe:.3f}")
    print(f"  Consistency: {wf_result.oos_consistency*100:.0f}%")
    print(f"  Decay: {wf_result.sharpe_decay:.2f}")
    return wf_result


def step6_monte_carlo(pnl_list):
    mc = MonteCarloSimulator(n_simulations=500, seed=42)
    mc_result = mc.run(pnl_list, initial_equity=10000.0)
    print(f"\n  MC Profit Prob: {mc_result.probability_of_profit*100:.1f}%")
    print(f"  Mean Final: ${mc_result.mean_final_equity:,.0f}")
    print(f"  5th %ile: ${mc_result.percentile_5:,.0f}  95th %ile: ${mc_result.percentile_95:,.0f}")
    print(f"  Avg DD: {mc_result.max_drawdown_stats['mean']*100:.1f}%")
    return mc_result


def step7_compare_brokers(symbol, data):
    results = []
    for bp in list_broker_profiles():
        strat = V2zStrategy()
        engine = BacktestEngine(strat, ExecutionSimulator(bp))
        r = engine.run(symbol, data)
        results.append((bp, r))
        print(f"  {bp:14s}  PnL=${r.net_pnl:>+8.2f}  WR={r.win_rate*100:>5.1f}%  "
              f"Sharpe={r.sharpe:>6.2f}  Trades={r.n_trades:>4d}")
    return results


def main():
    SYMBOL = "EURAUD"

    print("=" * 62)
    print("  V2+z Backtest Pipeline — proxima_honest_backtest")
    print("=" * 62)

    print("\n[1] Load M5 data")
    data = step1_load_data(SYMBOL)

    print("\n[2] Lint strategy (anti-lookahead)")
    lint_result = step2_lint_strategy()
    if not lint_result.passed:
        print("  WARNING: Lint found issues — review before trusting results")

    print("\n[3] Run backtest (FTMO broker)")
    strategy = V2zStrategy()
    bt_result = step3_backtest(strategy, SYMBOL, data)

    print("\n[4] Anti-overfit gauntlet")
    pnl_list = [t.pnl for t in bt_result.trades if t.pnl != 0]
    if pnl_list:
        step4_gauntlet(pnl_list)
    else:
        print("  (no trades — skipping)")

    print("\n[5] Walk-forward validation")
    timestamps = [t.timestamp for t in bt_result.trades if t.pnl != 0]
    if pnl_list and timestamps:
        step5_walk_forward(pnl_list, timestamps)
    else:
        print("  (no trades — skipping)")

    print("\n[6] Monte Carlo simulation")
    if pnl_list:
        returns = [p / 10000.0 for p in pnl_list]  # normalize $PnL to returns
        step6_monte_carlo(returns)
    else:
        print("  (no trades — skipping)")

    print("\n[7] Broker profile comparison")
    step7_compare_brokers(SYMBOL, data)

    print("\n" + "=" * 62)
    print("  Pipeline complete.")
    print("=" * 62)


if __name__ == "__main__":
    main()
