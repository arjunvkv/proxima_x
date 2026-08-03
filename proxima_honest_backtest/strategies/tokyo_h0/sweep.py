#!/usr/bin/env python3
"""Tokyo H0 — 27 config sweep on Exness, then validate top configs on all 5 brokers.

Optimized: pre-align bars once, reuse across runs (~4s per config).
"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import pandas as pd
from proxima_honest_backtest.strategies.tokyo_h0.strategy import TokyoH0Strategy
from proxima_honest_backtest.strategies.multi_pair_engine import MultiPairBacktestEngine
from proxima_honest_backtest.execution.execution_simulator import ExecutionSimulator
from data.providers.mt5_provider import MT5Provider

ALL_PAIRS = [
    "EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY",
    "GBPJPY","EURAUD","EURNZD","GBPAUD","GBPNZD",
    "GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP",
    "EURCHF","USDCHF","AUDJPY",
]
BROKERS = ["exness","ftmo","fundednext","fusionmarkets","dukascopy"]
LOOKBACK = [6, 12, 24]
HOLD = [3, 6, 12]
TOP_N = [1, 3, 5]

def load_and_align(tf="m5"):
    provider = MT5Provider()
    raw = {}
    for p in ALL_PAIRS:
        frames = [provider.load_rates(p, y, m, tf)
                  for y, m in [(2026,1),(2026,2),(2026,3),(2026,4),(2026,5),(2026,6),(2026,7)]]
        frames = [f for f in frames if not f.empty]
        if frames:
            d = pd.concat(frames, ignore_index=True)
            d.sort_values("time", inplace=True)
            d.reset_index(drop=True, inplace=True)
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

def run_cfg(data, pre_align, lb, hold, top_n, broker):
    s = TokyoH0Strategy({"lookback_bars": lb, "hold_bars": hold, "top_n": top_n})
    e = MultiPairBacktestEngine(s, ExecutionSimulator(broker))
    return e.run(data, pre_aligned=pre_align)

def main():
    t0 = time.time()
    print("Loading & aligning M5 data...")
    raw, pre_align = load_and_align()
    print(f"  {len(raw)} pairs, {len(pre_align):,} aligned bars ({time.time()-t0:.1f}s)")

    # Phase 1: Sweep all 27 configs on Exness ($0)
    print(f"\nPhase 1: Sweep 27 configs on Exness...")
    phase1 = []
    for lb in LOOKBACK:
        for hold in HOLD:
            for n in TOP_N:
                t1 = time.time()
                r = run_cfg(raw, pre_align, lb, hold, n, "exness")
                elapsed = time.time() - t1
                phase1.append({"lb": lb, "hold": hold, "n": n,
                               "trades": r.n_trades, "net_pnl": r.net_pnl,
                               "wr": r.win_rate, "pf": r.profit_factor,
                               "sharpe": r.sharpe, "dd": r.max_drawdown_pct})
                print(f"  lb={lb:>2d} hold={hold:>2d} n={n:>2d} | "
                      f"T={r.n_trades:>3d} Net=${r.net_pnl:>+8.2f} "
                      f"WR={r.win_rate*100:>5.1f}% PF={r.profit_factor:>5.2f} "
                      f"({elapsed:.1f}s)")

    print(f"\nPhase 1 done in {time.time()-t0:.0f}s")
    print(f"\n{'='*70}")
    print("EXNESS TOP 10 CONFIGS (by Net PnL)")
    print(f"{'='*70}")
    phase1.sort(key=lambda x: x["net_pnl"], reverse=True)
    for i, r in enumerate(phase1[:10]):
        print(f"  #{i+1} lb={r['lb']:>2d} hold={r['hold']:>2d} n={r['n']:>2d} | "
              f"T={r['trades']:>3d} Net=${r['net_pnl']:>+8.2f} "
              f"WR={r['wr']*100:>5.1f}% PF={r['pf']:>5.2f} Sh={r['sharpe']:>6.2f} DD={r['dd']:>5.2f}%")

    # Phase 2: Validate top 3 configs on all brokers
    top3 = phase1[:3]
    print(f"\nPhase 2: Validate top 3 configs on {len(BROKERS)} brokers...")
    phase2 = []
    for cfg in top3:
        for broker in BROKERS:
            if broker == "exness":
                continue
            t1 = time.time()
            r = run_cfg(raw, pre_align, cfg["lb"], cfg["hold"], cfg["n"], broker)
            elapsed = time.time() - t1
            entry = {**cfg, "broker": broker,
                     "trades": r.n_trades, "net_pnl": r.net_pnl,
                     "wr": r.win_rate, "pf": r.profit_factor,
                     "sharpe": r.sharpe, "dd": r.max_drawdown_pct}
            phase2.append(entry)
            print(f"  lb={cfg['lb']:>2d} hold={cfg['hold']:>2d} n={cfg['n']:>2d} "
                  f"{broker:14s} | T={r.n_trades:>3d} Net=${r.net_pnl:>+8.2f} "
                  f"WR={r.win_rate*100:>5.1f}% PF={r.profit_factor:>5.2f} ({elapsed:.1f}s)")

    total = time.time() - t0

    # Report
    print(f"\n{'='*70}")
    print(f"TOKYO H0 SWEEP — COMPLETE ({total:.0f}s)")
    print(f"{'='*70}")
    print(f"\nBEST CONFIG: lb={top3[0]['lb']} hold={top3[0]['hold']} n={top3[0]['n']}")
    print(f"\nAll 5-broker scores:")
    print(f"{'Config':<20s} {'Broker':14s} {'Trades':>6s} {'Net PnL':>9s} {'WR':>6s} {'PF':>5s} {'Sharpe':>7s} {'DD':>6s}")
    print("-"*75)
    for r in phase1 + [x for x in phase2 if x["broker"] != "exness"]:
        config = f"lb={r['lb']} hold={r['hold']} n={r['n']}"
        broker = r.get("broker", "exness")
        print(f"{config:<20s} {broker:14s} {r['trades']:>6d} ${r['net_pnl']:>+7.2f} "
              f"{r['wr']*100:>5.1f}% {r['pf']:>5.2f} {r['sharpe']:>7.2f} {r['dd']:>5.2f}%")

    survivors = [r for r in phase1 + phase2 if r["net_pnl"] > 0 and r["pf"] > 1.0]
    print(f"\nSURVIVORS (PnL>0 & PF>1): {len(survivors)}/{len(phase1) + len(phase2)}")

    out = Path(__file__).parent / "sweep_results.json"
    with open(out, "w") as f:
        json.dump({"phase1": phase1, "phase2": phase2, "total_sec": round(total, 1)}, f, indent=2)
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    main()
