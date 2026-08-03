#!/usr/bin/env python3
"""Sunday H22 — Full sweep + 5-broker validation."""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import pandas as pd
import numpy as np
from proxima_honest_backtest.strategies.sunday_h22.strategy import SundayH22Strategy, ALL_PAIRS
from proxima_honest_backtest.strategies.multi_pair_engine import MultiPairBacktestEngine
from proxima_honest_backtest.execution.execution_simulator import ExecutionSimulator
from data.providers.mt5_provider import MT5Provider

BROKERS = ["exness","ftmo","fundednext","fusionmarkets","dukascopy"]
MONTHS = [(y,m) for y,m in [(2026,1),(2026,2),(2026,3),(2026,4),(2026,5),(2026,6),(2026,7)]]

def load_and_align():
    provider = MT5Provider()
    raw = {}
    for p in ALL_PAIRS:
        frames = [f for f in [provider.load_rates(p, y, m, "m5") for y,m in MONTHS] if not f.empty]
        if frames:
            d = pd.concat(frames, ignore_index=True)
            d.sort_values("time", inplace=True); d.reset_index(drop=True, inplace=True)
            raw[p] = d
    pieces = []
    for pair, df in raw.items():
        sub = df.set_index("time")[["close","open","high","low","tick_volume","spread"]]
        sub.columns = [pair, f"{pair}_open", f"{pair}_high", f"{pair}_low", f"{pair}_volume", f"{pair}_spread"]
        pieces.append(sub)
    aligned = pd.concat(pieces, axis=1, sort=True)
    aligned.sort_index(inplace=True)
    aligned.ffill(inplace=True); aligned.bfill(inplace=True)
    aligned.reset_index(inplace=True); aligned.rename(columns={"index": "time"}, inplace=True)
    return raw, aligned.to_dict("records")

def run_cfg(data, pre_align, top_n, min_gap, max_hold, broker):
    s = SundayH22Strategy({"top_n": top_n, "min_gap_pips": min_gap, "max_hold_bars": max_hold})
    e = MultiPairBacktestEngine(s, ExecutionSimulator(broker))
    return e.run(data, pre_aligned=pre_align)

def main():
    t0 = time.time()
    print("Loading & aligning M5 data...")
    raw, pre_align = load_and_align()
    print(f"  {len(raw)} pairs, {len(pre_align):,} bars ({time.time()-t0:.1f}s)")

    # Sweep: top_n=[2,3,5], min_gap=[10,15,20], max_hold=[12,18,24,36]
    TOP_N = [2, 3, 5]
    MIN_GAP = [10, 15, 20]
    MAX_HOLD = [12, 18, 24, 36]

    all_results = []
    print("\n--- Sweep on Exness ---")
    for top_n in TOP_N:
        for min_gap in MIN_GAP:
            for max_hold in MAX_HOLD:
                t1 = time.time()
                r = run_cfg(raw, pre_align, top_n, min_gap, max_hold, "exness")
                elapsed = time.time() - t1
                all_results.append({"top_n": top_n, "min_gap": min_gap, "max_hold": max_hold,
                                     "broker": "exness", "trades": r.n_trades,
                                     "net_pnl": r.net_pnl, "wr": r.win_rate,
                                     "pf": r.profit_factor, "sharpe": r.sharpe,
                                     "dd": r.max_drawdown_pct})
                print(f"  n={top_n} gap={min_gap} hold={max_hold} | "
                      f"T={r.n_trades:>3d} Net=${r.net_pnl:>+8.2f} "
                      f"WR={r.win_rate*100:>5.1f}% PF={r.profit_factor:>6.2f} ({elapsed:.1f}s)")

    print(f"\nSweep done in {time.time()-t0:.0f}s")
    print(f"\n{'='*70}")
    print("EXNESS TOP 5 (by Net PnL)")
    print('=' * 70)
    exness_only = [r for r in all_results if r["broker"] == "exness"]
    exness_only.sort(key=lambda x: x["net_pnl"], reverse=True)
    for i, r in enumerate(exness_only[:5]):
        print(f"  #{i+1} n={r['top_n']} gap≥{r['min_gap']} hold={r['max_hold']} | "
              f"T={r['trades']:>3d} Net=${r['net_pnl']:>+8.2f} "
              f"WR={r['wr']*100:>5.1f}% PF={r['pf']:>5.2f} Sh={r['sharpe']:>7.2f} DD={r['dd']:>5.2f}%")

    # Phase 2: Top 2 configs on all 5 brokers
    best_configs = exness_only[:2]
    print(f"\nPhase 2: All-broker validation...")
    for cfg in best_configs:
        for broker in BROKERS:
            if broker == "exness":
                continue
            t1 = time.time()
            r = run_cfg(raw, pre_align, cfg["top_n"], cfg["min_gap"], cfg["max_hold"], broker)
            elapsed = time.time() - t1
            entry = {**cfg, "broker": broker, "trades": r.n_trades,
                     "net_pnl": r.net_pnl, "wr": r.win_rate, "pf": r.profit_factor,
                     "sharpe": r.sharpe, "dd": r.max_drawdown_pct}
            all_results.append(entry)
            print(f"  n={cfg['top_n']} gap≥{cfg['min_gap']} hold={cfg['max_hold']} "
                  f"{broker:14s} | T={r.n_trades:>3d} Net=${r.net_pnl:>+8.2f} "
                  f"WR={r.win_rate*100:>5.1f}% PF={r.profit_factor:>5.2f} ({elapsed:.1f}s)")

    # Report
    print(f"\n{'='*70}")
    print(f"SUNDAY H22 SWEEP — COMPLETE ({time.time()-t0:.0f}s)")
    print(f"{'='*70}")
    best = exness_only[0]
    print(f"\nBEST CONFIG: n={best['top_n']} gap≥{best['min_gap']} hold={best['max_hold']}")
    print(f"\nAll broker scores:")
    print(f"{'Config':<35s} {'Broker':14s} {'Trades':>6s} {'Net PnL':>9s} {'WR':>6s} {'PF':>5s} {'DD':>6s}")
    print("-"*80)
    all_results.sort(key=lambda x: (x.get("top_n",0), x.get("broker","")))
    for r in all_results:
        config = f"n={r.get('top_n',0)} gap≥{r.get('min_gap',0)} hold={r.get('max_hold',0)}"
        print(f"{config:<35s} {r['broker']:14s} {r['trades']:>6d} ${r['net_pnl']:>+7.2f} "
              f"{r['wr']*100:>5.1f}% {r['pf']:>5.2f} {r['dd']:>5.2f}%")

    survivors = [r for r in all_results if r["net_pnl"] > 0 and r["pf"] > 1.0]
    print(f"\nSURVIVORS (PnL>0 & PF>1): {len(survivors)}/{len(all_results)}")

    out = Path(__file__).parent / "sweep_results.json"
    with open(out, "w") as f:
        json.dump({"results": all_results, "total_sec": round(time.time()-t0, 1)}, f, indent=2)
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    main()
