from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from proxima_honest_backtest.research.sweep import ParameterSweep, SweepResult

from run_backtest import (
    align_bars,
    load_data_from_parquet,
    load_data_from_mt5,
    save_data_to_parquet,
    run_backtest,
    print_metrics,
)
from strategy import DarkConsensusStrategy

DATA_DIR = Path(__file__).resolve().parent / "data"


def sweep_objective(params: Dict[str, Any], symbol: str) -> float:
    _ = symbol
    pairs = ["EURJPY", "EURUSD", "GBPJPY"]

    strat_params = {
        "pairs": pairs,
        "mag_threshold": params.get("mag_threshold", 0.00018741),
        "hold_bars": int(params.get("hold_bars", 3)),
        "session_start": int(params.get("session_start", 7)),
        "session_end": int(params.get("session_end", 21)),
    }

    strategy = DarkConsensusStrategy(parameters=strat_params)

    data = load_data_from_parquet(pairs)
    has_data = any(d is not None and not d.empty for d in data.values())
    if not has_data:
        print("No cached data. Attempting MT5 download for sweep...")
        from_date = datetime(2025, 1, 1)
        to_date = datetime(2026, 7, 28)
        data = load_data_from_mt5(pairs, from_date, to_date)
        if any(d is not None and not d.empty for d in data.values()):
            save_data_to_parquet(data)
        else:
            return -999999.0

    timestamps, closes, volumes = align_bars(data, seed_bars=60)
    if not timestamps:
        return -999999.0

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

    if result["n_trades"] == 0:
        return -999999.0

    metrics = result["metrics"]
    sharpe = metrics.get("sharpe", 0.0)
    pf = metrics.get("profit_factor", 0.0)
    wr = metrics.get("win_rate", 0.0)
    n_trades = metrics.get("n_trades", 0)

    if n_trades < 10:
        return -999999.0

    if pf == float("inf"):
        pf = 10.0

    composite = sharpe * 0.4 + pf * 0.3 + wr * 0.2 + np.log1p(n_trades) * 0.1
    return composite


def main():
    print("Dark Consensus — Parameter Sweep\n")

    param_space = {
        "mag_threshold": (0.00005, 0.0005),
        "hold_bars": [1, 2, 3, 4, 5],
        "session_start": [0, 3, 5, 7, 9],
        "session_end": [15, 17, 19, 21, 23],
    }

    sweeper = ParameterSweep(
        param_space=param_space,
        metric_fn=sweep_objective,
        n_trials=200,
        method="optuna",
        seed=42,
    )

    t0 = time.perf_counter()
    results = sweeper.run(symbols=["DC"])
    elapsed = time.perf_counter() - t0

    primary = results[0] if results else None
    if primary:
        print(f"\nBest params:  {primary.best_params}")
        print(f"Best metric:  {primary.best_metric:.4f}")
        print(f"Trials:       {primary.n_trials}")
    else:
        print("No results returned.")

    print(f"\nElapsed: {elapsed:.2f}s")
    print(f"Method:  {sweeper.method}")

    report_path = Path(__file__).resolve().parent / "reports"
    report_path.mkdir(exist_ok=True)
    summary = sweeper.summarize(results) if results else {}

    import json
    with open(report_path / "sweep_result.json", "w") as f:
        json.dump({
            "best_params": primary.best_params if primary else {},
            "best_metric": primary.best_metric if primary else 0.0,
            "n_trials": primary.n_trials if primary else 0,
            "elapsed_seconds": elapsed,
            "method": sweeper.method,
            "summary": summary,
        }, f, indent=2, default=str)
    print(f"Results saved to {report_path / 'sweep_result.json'}")


if __name__ == "__main__":
    main()
