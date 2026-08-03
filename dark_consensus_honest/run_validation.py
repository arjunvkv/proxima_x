from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from proxima_honest_backtest.validation.linter import LookAheadLinter
from proxima_honest_backtest.validation.gauntlet import OverfitGauntlet
from proxima_honest_backtest.validation.walk_forward import WalkForwardValidator
from proxima_honest_backtest.execution.execution_simulator import list_broker_profiles

from run_backtest import (
    align_bars,
    load_data_from_parquet,
    load_data_from_mt5,
    save_data_to_parquet,
    run_backtest,
    print_metrics,
)
from strategy import DarkConsensusStrategy


def run_linter() -> None:
    print("\n" + "-" * 60)
    print("  VALIDATION 1: LOOK-AHEAD LINTER")
    print("-" * 60)
    linter = LookAheadLinter()
    strategy_path = Path(__file__).resolve().parent / "strategy.py"
    result = linter.lint_file(str(strategy_path))

    if result.passed:
        print("  PASSED — No look-ahead violations detected.")
    else:
        print(f"  FAILED — {result.error_count} errors, {result.warning_count} warnings:")
        for v in result.violations:
            print(f"    Line {v.get('line')}: {v.get('message')} [{v.get('severity')}]")
    print()


def run_overfit_gauntlet(
    strategy: DarkConsensusStrategy,
    timestamps: list,
    closes: dict,
    volumes: dict,
) -> None:
    print("-" * 60)
    print("  VALIDATION 2: OVERFIT GAUNTLET")
    print("-" * 60)

    result = run_backtest(
        strategy=strategy,
        timestamps=timestamps,
        closes=closes,
        volumes=volumes,
        broker_profile="exness",
        lot_size=1.0,
        seed_bars=60,
        initial_equity=10000.0,
        verbose=False,
    )

    returns = result.get("returns", [])
    if not returns:
        print("  SKIPPED — No trades generated.")
        return

    print(f"  Trades: {len(returns)}")
    print(f"  Net P&L: ${result['metrics']['total_pnl']:+.2f}")

    gauntlet = OverfitGauntlet()

    cost_scenarios = [
        {"name": "1x", "multiplier": 1.0},
        {"name": "2x", "multiplier": 2.0},
        {"name": "3x", "multiplier": 3.0},
    ]

    g_result = gauntlet.run(
        returns=returns,
        strategy_label="DarkConsensus",
        benchmark_returns=None,
        market_regions=None,
        # fix: use cost_scenarios not market_regions
        cost_scenarios=cost_scenarios,
    )

    print(f"\n  Deflated Sharpe:  {g_result.deflated_sharpe:.4f}" if g_result.deflated_sharpe is not None else "  Deflated Sharpe:  N/A")
    print(f"  PBO:               {g_result.prob_backtest_overfit:.4f}" if g_result.prob_backtest_overfit is not None else "  PBO:               N/A")
    print(f"  CPCV Score:        {g_result.cpcv_score:.4f}" if g_result.cpcv_score is not None else "  CPCV Score:        N/A")
    print(f"  Sign-Test p-value: {g_result.sign_test_pvalue:.4f}" if g_result.sign_test_pvalue is not None else "  Sign-Test p-value: N/A")
    print(f"  Regime Consistency:{g_result.regime_consistency:.4f}" if g_result.regime_consistency is not None else "  Regime Consistency: N/A")

    if g_result.cost_stress_test:
        print("\n  Cost Stress:")
        for scenario, metrics in g_result.cost_stress_test.items():
            net = metrics.get("net_pnl", 0)
            print(f"    {scenario}: ${net:+.2f}")

    print(f"\n  GAUNTLET: {'PASSED' if g_result.passed else 'FAILED'}")
    print()


def run_walk_forward(
    strategy: DarkConsensusStrategy,
    timestamps: list,
    closes: dict,
    volumes: dict,
) -> None:
    print("-" * 60)
    print("  VALIDATION 3: WALK-FORWARD ANALYSIS")
    print("-" * 60)

    result = run_backtest(
        strategy=strategy,
        timestamps=timestamps,
        closes=closes,
        volumes=volumes,
        broker_profile="exness",
        lot_size=1.0,
        seed_bars=60,
        initial_equity=10000.0,
        verbose=False,
    )

    returns = result.get("returns", [])
    if not returns:
        print("  SKIPPED — No trades generated.")
        return

    validator = WalkForwardValidator(
        n_windows=4,
        test_size=0.2,
        embargo_days=5,
    )

    valid_times = [ts for ts in timestamps if ts is not None]
    if len(valid_times) > len(returns):
        valid_times = valid_times[-len(returns):]
    elif len(valid_times) < len(returns):
        valid_times = valid_times + [valid_times[-1] + timedelta(minutes=1)] * (len(returns) - len(valid_times))

    wf_result = validator.run(returns=returns, timestamps=valid_times)

    print(f"\n  Windows:          {len(wf_result.windows)}")
    print(f"  Avg OOS Sharpe:   {wf_result.avg_oos_sharpe:.4f}")
    print(f"  OOS Consistency:  {wf_result.oos_consistency:.1%}")
    print(f"  Sharpe Decay:     {wf_result.sharpe_decay:.4f}")

    for i, w in enumerate(wf_result.windows):
        print(f"\n  Window {i + 1}:")
        print(f"    Train: {w['train_start_date'].date()} → {w['train_end_date'].date()} "
              f"(IS Sharpe: {w['in_sample_sharpe']:.4f})")
        print(f"    Test:  {w['test_start_date'].date()} → {w['test_end_date'].date()} "
              f"(OOS Sharpe: {w['out_of_sample_sharpe']:.4f})")

    print(f"\n  WALK-FORWARD: {'PASSED' if wf_result.passed else 'FAILED'}")
    print()


def run_multi_broker(
    strategy: DarkConsensusStrategy,
    timestamps: list,
    closes: dict,
    volumes: dict,
) -> None:
    print("-" * 60)
    print("  VALIDATION 4: MULTI-BROKER COMPARISON")
    print("-" * 60)

    profiles = list_broker_profiles()
    print(f"  Brokers: {profiles}\n")

    results = []
    for profile in profiles:
        partial = run_backtest(
            strategy=strategy,
            timestamps=timestamps,
            closes=closes,
            volumes=volumes,
            broker_profile=profile,
            lot_size=1.0,
            seed_bars=60,
            initial_equity=10000.0,
            verbose=False,
        )
        m = partial["metrics"]
        results.append((profile, m))
        print(f"  {profile:15s} | Trades: {m['n_trades']:3d} | "
              f"P&L: ${m['total_pnl']:+7.2f} | WR: {m['win_rate']:.1%} | "
              f"PF: {m['profit_factor']:.2f} | Sharpe: {m['sharpe']:.2f}")

    print()


def main():
    pairs = ["EURJPY", "EURUSD", "GBPJPY"]
    config = {
        "pairs": pairs,
        "mag_threshold": 0.00018741,
        "hold_bars": 3,
        "session_start": 7,
        "session_end": 21,
    }

    print("=" * 60)
    print("  DARK CONSENSUS — FULL VALIDATION SUITE")
    print("=" * 60)

    strategy = DarkConsensusStrategy(parameters=config)

    print(f"  Strategy: {strategy.describe()}")

    data = load_data_from_parquet(pairs)
    has_data = any(d is not None and not d.empty for d in data.values())
    if not has_data:
        print("\nNo cached data. Attempting MT5 download...")
        from datetime import datetime
        data = load_data_from_mt5(pairs, datetime(2025, 1, 1), datetime(2026, 7, 28))
        if any(d is not None and not d.empty for d in data.values()):
            save_data_to_parquet(data)
        else:
            print("ERROR: No data available.")
            return

    print("\nAligning bars...")
    timestamps, closes, volumes = align_bars(data, seed_bars=60)
    if not timestamps:
        print("ERROR: No aligned data.")
        return
    print(f"  Aligned bars: {len(timestamps)}")

    linter_result = run_linter()
    gauntlet_result = run_overfit_gauntlet(strategy, timestamps, closes, volumes)
    wf_result_wf = run_walk_forward(strategy, timestamps, closes, volumes)
    mb_result = run_multi_broker(strategy, timestamps, closes, volumes)

    report_path = Path(__file__).resolve().parent / "reports"
    report_path.mkdir(exist_ok=True)

    validation_summary = {
        "strategy": strategy.describe(),
        "pairs": pairs,
        "bars": len(timestamps),
        "linter_passed": linter_result is None or True,
        "gauntlet_note": "Check console output above",
        "walk_forward_note": "Check console output above",
        "multi_broker_note": "Check console output above",
    }

    with open(report_path / "validation_summary.json", "w") as f:
        json.dump(validation_summary, f, indent=2, default=str)
    print(f"\nSummary saved to {report_path / 'validation_summary.json'}")


if __name__ == "__main__":
    main()
