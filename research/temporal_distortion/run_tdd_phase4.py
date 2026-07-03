"""TDD Phase 4: Cross-Asset Audit — tick-based and bar-based TDD on all symbols."""
import sys, json
from pathlib import Path
import numpy as np
import polars as pl
import duckdb

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

from research.temporal_distortion.tdd_core import TDDCore, compute_sync_metrics
from research.temporal_distortion.tdd_counterfactual import compute_from_timestamps

TICK_SYMBOLS = ["EURJPY", "USDJPY", "GBPUSD", "EURUSD"]
BAR_SYMBOLS = ["eurjpy", "usdjpy", "gbpusd", "eurusd"]
BAR_FILES = {s: f"data/bars/{s}_bars_100t.parquet" for s in BAR_SYMBOLS}


def run_tick_tdd(symbol):
    core = TDDCore(symbol)
    n = core.load_ticks()
    core.detect_events()
    core.compute_event_rate(60)
    core.compute_acceleration(5)
    core.compute_distortion()
    core.build_bar_grid(300)
    core.compute_future_returns([5, 20, 50])

    fut = core.future_returns[50]
    valid = ~np.isnan(fut)
    uncond_pup = float(np.mean(fut[valid] > 0)) if np.sum(valid) > 0 else np.nan

    sync = compute_sync_metrics(core.bar_alpha, core.bar_delta, fut, f"{symbol}_H50")
    sync_up = sync.get("sync_up_accel_high_delta", {})
    sync_up_pup = sync_up.get("p_up", np.nan)
    n_sync = sync_up.get("n", 0)

    return {
        "symbol": symbol,
        "type": "tick",
        "sync_up_Pup_H50": round(float(sync_up_pup), 4) if sync_up_pup is not None and not np.isnan(sync_up_pup) else None,
        "n": int(n_sync) if n_sync else 0,
        "uncond_Pup": round(float(uncond_pup), 4) if uncond_pup is not None and not np.isnan(uncond_pup) else None,
    }


def run_bar_tdd(bar_symbol):
    root = Path(__file__).resolve().parent.parent.parent.parent
    bar_path = root / BAR_FILES[bar_symbol]
    bars = pl.read_parquet(str(bar_path)).sort("timestamp")

    timestamps = bars["timestamp"].to_numpy()
    close_prices = bars["close"].to_numpy()

    core = TDDCore(bar_symbol.upper())
    core.timestamps = timestamps

    compute_from_timestamps(core, window_seconds=3600, smooth=5)

    core.build_bar_grid(bar_seconds=300)

    ts_sec = timestamps.astype(np.float64) / 1_000_000
    n_bars = len(core.bar_times)
    fut_h50 = np.full(n_bars, np.nan)
    h_seconds = 50 * 300

    for i in range(n_bars):
        if np.isnan(core.bar_times[i]):
            continue
        target_time = core.bar_times[i] + h_seconds
        cur_idx = np.searchsorted(ts_sec, core.bar_times[i])
        fut_idx = np.searchsorted(ts_sec, target_time)
        if cur_idx < len(close_prices) and fut_idx < len(close_prices):
            fut_h50[i] = (close_prices[fut_idx] - close_prices[cur_idx]) / close_prices[cur_idx]

    core.future_returns = {50: fut_h50}

    valid = ~np.isnan(fut_h50)
    uncond_pup = float(np.mean(fut_h50[valid] > 0)) if np.sum(valid) > 0 else np.nan

    sync = compute_sync_metrics(core.bar_alpha, core.bar_delta, fut_h50, f"{bar_symbol.upper()}_H50")
    sync_up = sync.get("sync_up_accel_high_delta", {})
    sync_up_pup = sync_up.get("p_up", np.nan)
    n_sync = sync_up.get("n", 0)

    return {
        "symbol": bar_symbol.upper(),
        "type": "bar",
        "sync_up_Pup_H50": round(float(sync_up_pup), 4) if sync_up_pup is not None and not np.isnan(sync_up_pup) else None,
        "n": int(n_sync) if n_sync else 0,
        "uncond_Pup": round(float(uncond_pup), 4) if uncond_pup is not None and not np.isnan(uncond_pup) else None,
    }


if __name__ == "__main__":
    results = []
    print("=" * 80)
    print("TDD Phase 4 — Cross-Asset Audit")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("PART A: Tick-based TDD")
    print("=" * 80)
    for sym in TICK_SYMBOLS:
        try:
            r = run_tick_tdd(sym)
            results.append(r)
            print(f"  {r['symbol']:<8} (tick)  sync_up_H50={r['sync_up_Pup_H50']}  n={r['n']}  uncond_Pup={r['uncond_Pup']}")
        except Exception as e:
            print(f"  {sym:<8} (tick)  ERROR -- {e}")

    print("\n" + "=" * 80)
    print("PART B: Bar-based TDD")
    print("=" * 80)
    for sym in BAR_SYMBOLS:
        try:
            r = run_bar_tdd(sym)
            results.append(r)
            print(f"  {r['symbol']:<8} (bar)   sync_up_H50={r['sync_up_Pup_H50']}  n={r['n']}  uncond_Pup={r['uncond_Pup']}")
        except Exception as e:
            print(f"  {sym:<8} (bar)   ERROR -- {e}")

    print("\n" + "=" * 80)
    print("CROSS-ASSET TABLE")
    print("=" * 80)
    header = f"{'Symbol':<10} {'Type':<6} {'Sync_up_Pup_H50':<18} {'n':<8} {'Uncond_Pup':<12}"
    print(header)
    print("-" * 54)
    for r in results:
        sup = f"{r['sync_up_Pup_H50']:.4f}" if r['sync_up_Pup_H50'] is not None else "N/A"
        up = f"{r['uncond_Pup']:.4f}" if r['uncond_Pup'] is not None else "N/A"
        print(f"{r['symbol']:<10} {r['type']:<6} {sup:<18} {r['n']:<8} {up:<12}")

    print("\n" + "=" * 80)
    print("GENERALIZATION ASSESSMENT")
    print("=" * 80)

    tick_vals = [r['sync_up_Pup_H50'] for r in results if r['type'] == 'tick' and r['sync_up_Pup_H50'] is not None]
    bar_vals = [r['sync_up_Pup_H50'] for r in results if r['type'] == 'bar' and r['sync_up_Pup_H50'] is not None]

    tick_pass = sum(1 for v in tick_vals if v > 0.52)
    bar_pass = sum(1 for v in bar_vals if v > 0.52)

    print(f"  Tick-based: {tick_pass}/{len(TICK_SYMBOLS)} symbols with sync_up P(up) > 0.52")
    print(f"  Bar-based:  {bar_pass}/{len(BAR_SYMBOLS)} symbols with sync_up P(up) > 0.52")
    if tick_vals:
        print(f"  Tick sync_up range:  [{min(tick_vals):.4f}, {max(tick_vals):.4f}]")
    if bar_vals:
        print(f"  Bar sync_up range:   [{min(bar_vals):.4f}, {max(bar_vals):.4f}]")

    total_pass = sum(1 for r in results if r['sync_up_Pup_H50'] is not None and r['sync_up_Pup_H50'] > 0.52)
    total = len(results)
    ratio = total_pass / total

    if ratio >= 0.75:
        gen = "GENERALIZES -- TDD directional edge confirmed across assets and data types"
    elif ratio >= 0.5:
        gen = "PARTIALLY GENERALIZES -- TDD edge in majority but not universal"
    elif total_pass >= 1:
        gen = f"LIMITED GENERALIZATION -- TDD edge on {total_pass}/{total} combinations"
    else:
        gen = "DOES NOT GENERALIZE -- TDD edge not confirmed cross-asset"

    print(f"\n  VERDICT: {gen}")

    out_path = Path(__file__).resolve().parent / "reports"
    out_path.mkdir(exist_ok=True)
    report = {"phase": "TDD Phase 4 -- Cross-Asset Audit", "results": results, "verdict": gen}
    with open(out_path / "TDD_PHASE4_REPORT.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {out_path / 'TDD_PHASE4_REPORT.json'}")
