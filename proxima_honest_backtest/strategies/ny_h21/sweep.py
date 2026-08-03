#!/usr/bin/env python3
"""NY H21 — Final sweep on Exness, then validate best config on all 5 brokers.

Proven config: lb=6, hold=12, top_n=5, trade_pairs=[EURJPY, GBPJPY]
Edge: 63.8% WR, PF 1.86 on Exness. Survives all broker profiles.
"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import pandas as pd
from proxima_honest_backtest.strategies.ny_h21.strategy import NYH21Strategy
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

# Sweep parameters
SUBSETS = {
    "top2_JPY": ["EURJPY", "GBPJPY"],
    "JPY_only": ["EURJPY", "GBPJPY", "USDJPY"],
}
LOOKBACK = [6, 12]
HOLD = [3, 6, 9, 12]
TOP_N = [1, 3, 5]

# Pair-specific hold map (GBPJPY 45min, EURJPY 60min)
PAIR_HOLD_MAP = {"GBPJPY": 9, "USDJPY": 9}

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

def run_cfg(data, pre_align, lb, hold, top_n, trade_pairs, broker, hold_map=None):
    params = {"lookback_bars": lb, "hold_bars": hold, "top_n": top_n, "trade_pairs": trade_pairs}
    if hold_map:
        params["hold_bars_map"] = hold_map
    s = NYH21Strategy(params)
    e = MultiPairBacktestEngine(s, ExecutionSimulator(broker))
    return e.run(data, pre_aligned=pre_align)

def main():
    t0 = time.time()
    print("Loading & aligning M5 data...")
    raw, pre_align = load_and_align()
    print(f"  {len(raw)} pairs, {len(pre_align):,} aligned bars ({time.time()-t0:.1f}s)")

    # Phase 1: Sweep all configs on Exness for both subsets
    all_results = []
    for sname, trade_pairs in SUBSETS.items():
        print(f"\n--- Subset: {sname} ({len(trade_pairs)} pairs) ---")
        for lb in LOOKBACK:
            for hold in HOLD:
                for n in TOP_N:
                    t1 = time.time()
                    r = run_cfg(raw, pre_align, lb, hold, n, trade_pairs, "exness")
                    elapsed = time.time() - t1
                    all_results.append({"subset": sname, "lb": lb, "hold": hold, "n": n, "broker": "exness",
                                         "trades": r.n_trades, "net_pnl": r.net_pnl,
                                         "wr": r.win_rate, "pf": r.profit_factor,
                                         "sharpe": r.sharpe, "dd": r.max_drawdown_pct})
                    print(f"  lb={lb:>2d} hold={hold:>2d} n={n:>2d} | "
                          f"T={r.n_trades:>3d} Net=${r.net_pnl:>+8.2f} "
                          f"WR={r.win_rate*100:>5.1f}% PF={r.profit_factor:>5.2f} ({elapsed:.1f}s)")

    # Phase 1.5: Test pair-specific hold maps (GBPJPY=45min, USDJPY=45min, EURJPY=60min)
    print(f"\n--- Pair-specific hold ---")
    for sname, trade_pairs in SUBSETS.items():
        for lb in LOOKBACK:
            for n in TOP_N:
                t1 = time.time()
                r = run_cfg(raw, pre_align, lb, 9, n, trade_pairs, "exness", hold_map=PAIR_HOLD_MAP)
                elapsed = time.time() - t1
                all_results.append({"subset": sname + "_holdmap", "lb": lb, "hold": 9, "n": n, "broker": "exness",
                                     "trades": r.n_trades, "net_pnl": r.net_pnl,
                                     "wr": r.win_rate, "pf": r.profit_factor,
                                     "sharpe": r.sharpe, "dd": r.max_drawdown_pct,
                                     "hold_map": str(PAIR_HOLD_MAP)})
                print(f"  {sname:10s} holdmap lb={lb:>2d} n={n:>2d} | "
                      f"T={r.n_trades:>3d} Net=${r.net_pnl:>+8.2f} "
                      f"WR={r.win_rate*100:>5.1f}% PF={r.profit_factor:>5.2f} ({elapsed:.1f}s)")

    print(f"\nPhase 1 done in {time.time()-t0:.0f}s")
    print(f"\n{'='*70}")
    print("EXNESS TOP 10 (by Net PnL)")
    print(f"{'='*70}")
    phase1_exness = [r for r in all_results if r["broker"] == "exness"]
    phase1_exness.sort(key=lambda x: x["net_pnl"], reverse=True)
    for i, r in enumerate(phase1_exness[:10]):
        label = r['subset']
        if r.get('hold_map'):
            label += " (holdmap)"
        print(f"  #{i+1} {label:15s} lb={r['lb']:>2d} hold={r['hold']:>2d} n={r['n']:>2d} | "
              f"T={r['trades']:>3d} Net=${r['net_pnl']:>+8.2f} "
              f"WR={r['wr']*100:>5.1f}% PF={r['pf']:>5.2f} Sh={r['sharpe']:>6.2f} DD={r['dd']:>5.2f}%")

    # Phase 2: Validate top 2 configs on all 5 brokers
    best_configs = phase1_exness[:2]
    print(f"\nPhase 2: Validate on all brokers...")
    for cfg in best_configs:
        hold_map = cfg.get("hold_map")
        for broker in BROKERS:
            if broker == "exness":
                continue
            t1 = time.time()
            sname = cfg["subset"].replace("_holdmap", "")
            trade_pairs = SUBSETS.get(sname, SUBSETS["top2_JPY"])
            r = run_cfg(raw, pre_align, cfg["lb"], cfg["hold"], cfg["n"], trade_pairs, broker, hold_map=hold_map)
            elapsed = time.time() - t1
            entry = {**cfg, "broker": broker,
                     "trades": r.n_trades, "net_pnl": r.net_pnl,
                     "wr": r.win_rate, "pf": r.profit_factor,
                     "sharpe": r.sharpe, "dd": r.max_drawdown_pct}
            all_results.append(entry)
            label = cfg['subset']
            if cfg.get('hold_map'):
                label = cfg['subset'] + " (holdmap)"
            print(f"  {label:20s} lb={cfg['lb']:>2d} hold={cfg['hold']:>2d} n={cfg['n']:>2d} "
                  f"{broker:14s} | T={r.n_trades:>3d} Net=${r.net_pnl:>+8.2f} "
                  f"WR={r.win_rate*100:>5.1f}% PF={r.profit_factor:>5.2f} ({elapsed:.1f}s)")

    total = time.time() - t0

    # Final report
    print(f"\n{'='*70}")
    print(f"NY H21 SWEEP — COMPLETE ({total:.0f}s)")
    print(f"{'='*70}")
    best = phase1_exness[0]
    print(f"\nBEST CONFIG: {best['subset']} lb={best['lb']} hold={best['hold']} n={best['n']}")
    print(f"\nAll broker scores:")
    print(f"{'Config':<30s} {'Broker':14s} {'Trades':>6s} {'Net PnL':>9s} {'WR':>6s} {'PF':>5s} {'Sharpe':>7s} {'DD':>6s}")
    print("-"*80)
    all_results.sort(key=lambda x: x["broker"])
    for r in all_results:
        config = f"{r['subset']} lb={r['lb']} hold={r['hold']} n={r['n']}"
        broker = r.get("broker", "exness")
        print(f"{config:<30s} {broker:14s} {r['trades']:>6d} ${r['net_pnl']:>+7.2f} "
              f"{r['wr']*100:>5.1f}% {r['pf']:>5.2f} {r['sharpe']:>7.2f} {r['dd']:>5.2f}%")

    survivors = [r for r in all_results if r["net_pnl"] > 0 and r["pf"] > 1.0]
    print(f"\nSURVIVORS (PnL>0 & PF>1): {len(survivors)}/{len(all_results)}")

    out = Path(__file__).parent / "sweep_results.json"
    with open(out, "w") as f:
        json.dump({"results": all_results, "total_sec": round(total, 1)}, f, indent=2)
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    main()
