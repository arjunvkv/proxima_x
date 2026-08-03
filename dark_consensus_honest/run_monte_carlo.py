from __future__ import annotations

import sys
import json
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from proxima_honest_backtest.research.monte_carlo import MonteCarloSimulator

from run_backtest import (
    align_bars,
    load_data_from_parquet,
    load_data_from_mt5,
    save_data_to_parquet,
    run_backtest,
)
from strategy import DarkConsensusStrategy


def main():
    pairs = ["EURJPY", "EURUSD", "GBPJPY"]
    config = {
        "pairs": pairs,
        "mag_threshold": 0.00018741,
        "hold_bars": 3,
        "session_start": 7,
        "session_end": 21,
    }

    print("Dark Consensus — Monte Carlo Simulation\n")

    data = load_data_from_parquet(pairs)
    has_data = any(d is not None and not d.empty for d in data.values())
    if not has_data:
        from datetime import datetime
        print("No cached data. Attempting MT5 download...")
        data = load_data_from_mt5(pairs, datetime(2025, 1, 1), datetime(2026, 7, 28))
        if any(d is not None and not d.empty for d in data.values()):
            save_data_to_parquet(data)
        else:
            print("ERROR: No data available.")
            return

    timestamps, closes, volumes = align_bars(data, seed_bars=60)
    if not timestamps:
        print("ERROR: No aligned data.")
        return

    strategy = DarkConsensusStrategy(parameters=config)
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
        print("ERROR: No trades generated — cannot run MC.")
        return

    print(f"Backtest produced {len(returns)} trades.")
    print(f"  Net P&L:       ${result['metrics']['total_pnl']:+.2f}")
    print(f"  Win Rate:      {result['metrics']['win_rate']:.1%}")
    print(f"  Sharpe:        {result['metrics']['sharpe']:.2f}")

    mc = MonteCarloSimulator(n_simulations=5000, seed=42)

    print(f"\nRunning {mc.n_simulations} MC simulations (bootstrap resample)...")
    t0 = time.perf_counter()
    mc_result = mc.run(
        trade_returns=returns,
        initial_equity=10000.0,
        resample=True,
    )
    elapsed = time.perf_counter() - t0

    print(f"Elapsed: {elapsed:.2f}s\n")
    print("=" * 60)
    print("  MONTE CARLO RESULTS")
    print("=" * 60)
    print(f"  Simulations:        {mc_result.n_simulations}")
    print(f"  Probability of P&L: {mc_result.probability_of_profit:.1%}")
    print(f"  Mean Final Equity:  ${mc_result.mean_final_equity:,.2f}")
    print(f"  Median Final Eq:    ${mc_result.median_final_equity:,.2f}")
    print(f"  Std Final Equity:   ${mc_result.std_final_equity:,.2f}")
    print(f"  5th Percentile:     ${mc_result.percentile_5:,.2f}")
    print(f"  95th Percentile:    ${mc_result.percentile_95:,.2f}")
    print(f"\n  Max Drawdown:")
    print(f"    Mean:    {mc_result.max_drawdown_stats.get('mean', 0):.2%}")
    print(f"    Median:  {mc_result.max_drawdown_stats.get('median', 0):.2%}")
    print(f"    P95:     {mc_result.max_drawdown_stats.get('percentile_95', 0):.2%}")
    print(f"    Max:     {mc_result.max_drawdown_stats.get('max', 0):.2%}")
    print(f"\n  Sharpe Ratio:")
    print(f"    Mean:    {mc_result.sharpe_stats.get('mean', 0):.2f}")
    print(f"    Median:  {mc_result.sharpe_stats.get('median', 0):.2f}")
    print(f"    P95:     {mc_result.sharpe_stats.get('percentile_95', 0):.2f}")
    print(f"    Positive: {mc_result.sharpe_stats.get('positive_ratio', 0):.1%}")
    print("=" * 60)

    report_path = Path(__file__).resolve().parent / "reports"
    report_path.mkdir(exist_ok=True)
    with open(report_path / "monte_carlo_result.json", "w") as f:
        json.dump({
            "n_simulations": mc_result.n_simulations,
            "probability_of_profit": mc_result.probability_of_profit,
            "mean_final_equity": mc_result.mean_final_equity,
            "median_final_equity": mc_result.median_final_equity,
            "std_final_equity": mc_result.std_final_equity,
            "percentile_5": mc_result.percentile_5,
            "percentile_95": mc_result.percentile_95,
            "max_drawdown_stats": mc_result.max_drawdown_stats,
            "sharpe_stats": mc_result.sharpe_stats,
        }, f, indent=2, default=str)
    print(f"\nResults saved to {report_path / 'monte_carlo_result.json'}")


if __name__ == "__main__":
    main()
