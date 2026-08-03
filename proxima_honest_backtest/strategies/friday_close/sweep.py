#!/usr/bin/env python3
"""#3 Friday Close Window — LONG most declined Fri 17:00, hold to 21:00."""
import sys, time, json, math
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import pandas as pd
import numpy as np
from proxima_honest_backtest.strategies.inter_session.strategy import InterSessionStrategy, ALL_PAIRS
from proxima_honest_backtest.strategies.multi_pair_engine import MultiPairBacktestEngine
from proxima_honest_backtest.execution.execution_simulator import ExecutionSimulator
from data.providers.mt5_provider import MT5Provider

BROKERS = ["exness","ftmo","fundednext","fusionmarkets","dukascopy"]
MONTHS = [(2026, m) for m in range(1, 8)]
LOOKBACKS = [6, 12, 24]
HOLDS = [24, 48]  # 2hr or 4hr hold
TOP_N = [3, 5]

def load_and_align(tf="m5"):
    provider = MT5Provider()
    raw = {}
    for p in ALL_PAIRS:
        frames = [provider.load_rates(p, y, m, tf) for y, m in MONTHS]
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
    params = {"entry_hours": [17], "entry_weekdays": [4],
              "lookback_bars": lb, "hold_bars": hold, "top_n": top_n}
    s = InterSessionStrategy(params)
    e = MultiPairBacktestEngine(s, ExecutionSimulator(broker))
    return e.run(data, pre_aligned=pre_align)

def main():
    t0 = time.time()
    raw, pre_align = load_and_align()
    print(f"Aligned: {len(pre_align):,} bars ({time.time()-t0:.1f}s)")

    print(f"\n{'='*70}")
    print("PHASE 1: Friday Close sweep on Exness")
    print(f"{'='*70}")
    phase1 = []
    t1 = time.time()
    cfg_idx = 0
    total = len(LOOKBACKS) * len(HOLDS) * len(TOP_N)
    for lb in LOOKBACKS:
        for hold in HOLDS:
            for n in TOP_N:
                cfg_idx += 1
                r = run_cfg(raw, pre_align, lb, hold, n, "exness")
                et = time.time() - t1; t1 = time.time()
                e = {"lb": lb, "hold": hold, "n": n,
                     "trades": r.n_trades, "net_pnl": r.net_pnl,
                     "wr": r.win_rate, "pf": r.profit_factor}
                phase1.append(e)
                print(f"  [{cfg_idx:>2d}/{total}] lb={lb:>2d} h={hold:>2d} n={n:>2d} | "
                      f"T={r.n_trades:>3d} Net=${r.net_pnl:>+8.2f} WR={r.win_rate*100:>5.1f}% "
                      f"PF={r.profit_factor:>5.2f} ({et:.1f}s)")

    phase1.sort(key=lambda x: x["net_pnl"], reverse=True)
    print(f"\nExness results:")
    for i, r in enumerate(phase1):
        print(f"  #{i+1} lb={r['lb']:>2d} hold={r['hold']:>2d} n={r['n']:>2d} | "
              f"T={r['trades']:>3d} Net=${r['net_pnl']:>+8.2f} WR={r['wr']*100:>5.1f}% PF={r['pf']:>5.2f}")

    best = phase1[0]
    print(f"\n{'='*70}")
    print("PHASE 2: Multi-broker validation")
    print(f"{'='*70}")
    phase2 = []
    for broker in BROKERS:
        r = run_cfg(raw, pre_align, best["lb"], best["hold"], best["n"], broker)
        e = {**{k: best[k] for k in ["lb","hold","n"]},
             "broker": broker, "trades": r.n_trades, "net_pnl": r.net_pnl,
             "wr": r.win_rate, "pf": r.profit_factor, "dd": r.max_drawdown_pct,
             "avg_win": r.avg_win, "avg_loss": r.avg_loss, "recon": str(r.reconciliation_pass)}
        phase2.append(e)
        print(f"  {broker:>15s} | T={r.n_trades:>3d} Net=${r.net_pnl:>+8.2f} "
              f"WR={r.win_rate*100:>5.1f}% PF={r.profit_factor:>5.2f} DD={r.max_drawdown_pct:>5.2f}%")

    print(f"\n{'='*70}")
    print("PHASE 3: Sign-permutation + Walk-forward")
    print(f"{'='*70}")
    base = run_cfg(raw, pre_align, best["lb"], best["hold"], best["n"], "exness")
    exit_trades = [t for t in base.trades if t.pnl != 0]
    pnl = np.array([t.pnl for t in exit_trades])
    bs = base.sharpe
    print(f"  Base: {len(exit_trades)} trades, Sharpe={bs:.4f}")
    N = 10000; cnt = 0
    for _ in range(N):
        sgn = np.random.choice([1, -1], size=len(pnl))
        s = float(np.mean(pnl*sgn) / (np.std(pnl*sgn)+1e-12)) * math.sqrt(252*288/len(pnl))
        if s >= bs: cnt += 1
    pv = (cnt+1)/(N+1)
    print(f"  Perm: p={pv:.4f} ({cnt}/{N})")
    n = len(exit_trades); ws = max(n//5,3); wf = []
    for w in range(5):
        sx = w*ws; mx = sx+int(ws*0.7); ex = min(sx+ws, n)
        if mx >= ex or sx >= n: break
        train = pnl[sx:mx]; test = pnl[mx:ex]
        if len(train)<3 or len(test)<3: break
        trs = float(np.mean(train)/(np.std(train)+1e-12))*math.sqrt(252*288/len(train))
        tes = float(np.mean(test)/(np.std(test)+1e-12))*math.sqrt(252*288/len(test))
        wf.append({"w":w+1,"tr_n":len(train),"te_n":len(test),"tr_s":round(trs,2),"te_s":round(tes,2)})
        print(f"  W{w+1}: train={len(train)} (Sh={trs:.2f}) -> test={len(test)} (Sh={tes:.2f})")

    survivors = [r for r in phase2 if r["net_pnl"] > 0]
    print(f"  Broker survival: {len(survivors)}/{len(BROKERS)}")

    out = Path(__file__).parent / "sweep_results.json"
    with open(out, "w") as f:
        json.dump({"phase1": phase1, "phase2": phase2,
                    "validation": {"sharpe": round(bs,4),"perm_p": round(pv,4),
                                   "walkforward": wf, "broker_survivors": len(survivors)},
                    "total_sec": round(time.time()-t0,1)}, f, indent=2, default=str)
    print(f"Saved. Total: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
