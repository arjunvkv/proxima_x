#!/usr/bin/env python3
"""#2 Session Momentum Relay — LONG best performers at session transitions."""
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
HOURS_CFG = [[8], [16], [8, 16]]
LOOKBACKS = [6, 12, 24]
HOLDS = [6, 12, 24]
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

def run_cfg(data, pre_align, hours, lb, hold, top_n, broker):
    # sort_descending=True → LONG best performers (momentum)
    params = {"entry_hours": hours, "lookback_bars": lb, "hold_bars": hold,
              "top_n": top_n, "sort_descending": True}
    s = InterSessionStrategy(params)
    e = MultiPairBacktestEngine(s, ExecutionSimulator(broker))
    return e.run(data, pre_aligned=pre_align)

def run_p0(pre_align):
    print("=" * 70)
    print("PHASE 0: Session Momentum Relay — LONG best performers")
    print("=" * 70)
    pair_data = defaultdict(lambda: {"n":0,"w":0,"r":0.0})
    for i, r in enumerate(pre_align):
        h = r["time"].hour if hasattr(r["time"],"hour") else 0
        if h not in (8, 16): continue
        returns = []
        for pair in ALL_PAIRS:
            close = r.get(pair)
            if close is None or np.isnan(close): continue
            pi = i - 12
            if pi < 0: continue
            pv = pre_align[pi].get(pair)
            if pv is None or np.isnan(pv) or pv <= 0: continue
            returns.append((pair, (close - pv) / pv))
        if len(returns) < 8: continue
        returns.sort(key=lambda x: x[1], reverse=True)
        for pair, ret in returns[:5]:
            if ret <= 0: continue
            pair_data[pair]["n"] += 1
            fi = i + 12
            if fi >= len(pre_align): continue
            eo = r.get(f"{pair}_open", r.get(pair))
            fc = pre_align[fi].get(pair)
            if any(v is None or np.isnan(v) for v in (eo, fc)): continue
            fr = (fc - eo) / eo
            pair_data[pair]["r"] += fr
            if fr > 0: pair_data[pair]["w"] += 1
    print(f"{'Pair':>6s}  {'Events':>6s}  {'WR':>5s}  {'AvgRet%':>8s}")
    for pair in sorted(ALL_PAIRS, key=lambda p: pair_data[p]["w"]/max(pair_data[p]["n"],1), reverse=True):
        d = pair_data[pair]
        if d["n"] < 10: continue
        print(f"{pair:>6s}  {d['n']:>6d}  {d['w']/d['n']:>4.1%}  {d['r']/d['n']*100:>+8.3f}%")
    return pair_data

def main():
    t0 = time.time()
    raw, pre_align = load_and_align()
    pair_data = run_p0(pre_align)
    print(f"Total bars: {len(pre_align):,} ({time.time()-t0:.1f}s)")

    print(f"\n{'='*70}")
    print("PHASE 1: Parameter sweep on Exness")
    print(f"{'='*70}")
    phase1 = []
    t1 = time.time()
    cfg_idx = 0
    total = len(HOURS_CFG) * len(LOOKBACKS) * len(HOLDS) * len(TOP_N)
    for hours in HOURS_CFG:
        hname = "_".join(str(h) for h in hours)
        for lb in LOOKBACKS:
            for hold in HOLDS:
                for n in TOP_N:
                    cfg_idx += 1
                    r = run_cfg(raw, pre_align, hours, lb, hold, n, "exness")
                    et = time.time() - t1; t1 = time.time()
                    e = {"hours": hname, "lb": lb, "hold": hold, "n": n,
                         "trades": r.n_trades, "net_pnl": r.net_pnl,
                         "wr": r.win_rate, "pf": r.profit_factor}
                    phase1.append(e)
                    print(f"  [{cfg_idx:>2d}/{total}] {hname:>7s} lb={lb:>2d} h={hold:>2d} n={n:>2d} | "
                          f"T={r.n_trades:>4d} Net=${r.net_pnl:>+8.2f} WR={r.win_rate*100:>5.1f}% "
                          f"PF={r.profit_factor:>5.2f} ({et:.1f}s)")

    phase1.sort(key=lambda x: x["net_pnl"], reverse=True)
    print(f"\nExness Top 10:")
    for i, r in enumerate(phase1[:10]):
        print(f"  #{i+1} {r['hours']:>7s} lb={r['lb']:>2d} hold={r['hold']:>2d} n={r['n']:>2d} | "
              f"T={r['trades']:>4d} Net=${r['net_pnl']:>+8.2f} WR={r['wr']*100:>5.1f}% PF={r['pf']:>5.2f}")

    best = phase1[0]
    # Map hour name back to list
    best_hours = [int(h) for h in best["hours"].split("_")]

    print(f"\n{'='*70}")
    print(f"PHASE 2: Multi-broker validation")
    print(f"{'='*70}")
    phase2 = []
    for broker in BROKERS:
        r = run_cfg(raw, pre_align, best_hours, best["lb"], best["hold"], best["n"], broker)
        e = {**{k: best[k] for k in ["hours","lb","hold","n"]},
             "broker": broker, "trades": r.n_trades, "net_pnl": r.net_pnl,
             "wr": r.win_rate, "pf": r.profit_factor, "dd": r.max_drawdown_pct,
             "avg_win": r.avg_win, "avg_loss": r.avg_loss, "recon": str(r.reconciliation_pass)}
        phase2.append(e)
        print(f"  {broker:>15s} | T={r.n_trades:>4d} Net=${r.net_pnl:>+8.2f} "
              f"WR={r.win_rate*100:>5.1f}% PF={r.profit_factor:>5.2f} DD={r.max_drawdown_pct:>5.2f}%")

    # Phase 3: Perm + WF
    print(f"\n{'='*70}")
    print("PHASE 3: Sign-permutation (10,000) + Walk-forward")
    print(f"{'='*70}")
    base = run_cfg(raw, pre_align, best_hours, best["lb"], best["hold"], best["n"], "exness")
    exit_trades = [t for t in base.trades if t.pnl != 0]
    pnl = np.array([t.pnl for t in exit_trades])
    base_sharpe = base.sharpe
    print(f"  Base: {len(exit_trades)} exit trades, Sharpe={base_sharpe:.4f}")

    N = 10000
    cnt = 0
    for _ in range(N):
        sgn = np.random.choice([1, -1], size=len(pnl))
        s = float(np.mean(pnl * sgn) / (np.std(pnl * sgn) + 1e-12)) * math.sqrt(252 * 288 / len(pnl))
        if s >= base_sharpe: cnt += 1
    p_val = (cnt + 1) / (N + 1)
    print(f"  Permutation: p={p_val:.4f} ({cnt}/{N} exceed)")

    # Walk-forward
    n = len(exit_trades)
    ws = max(n // 5, 3)
    wf = []
    for w in range(5):
        s = w * ws
        m = s + int(ws * 0.7)
        e = min(s + ws, n)
        if m >= e or s >= n: break
        train = pnl[s:m]
        test = pnl[m:e]
        if len(train) < 3 or len(test) < 3: break
        tr_s = float(np.mean(train) / (np.std(train) + 1e-12)) * math.sqrt(252 * 288 / len(train))
        te_s = float(np.mean(test) / (np.std(test) + 1e-12)) * math.sqrt(252 * 288 / len(test))
        wf.append({"w": w+1, "tr_n": len(train), "te_n": len(test), "tr_s": round(tr_s,2), "te_s": round(te_s,2)})
        print(f"  W{w+1}: train={len(train)} (Sh={tr_s:.2f}) → test={len(test)} (Sh={te_s:.2f})")

    survivors = [r for r in phase2 if r["net_pnl"] > 0]
    print(f"\n  Broker survival: {len(survivors)}/{len(BROKERS)}")
    perm_pass = "PASS" if p_val < 0.05 else "FAIL"
    wf_pass = "PASS" if sum(1 for w in wf if w["te_s"] > 0) >= len(wf) * 0.6 else "FAIL"
    print(f"  Permutation: {perm_pass}  Walk-forward: {wf_pass}")

    out = Path(__file__).parent / "sweep_results.json"
    with open(out, "w") as f:
        json.dump({"p0": {pair: dict(d) for pair, d in pair_data.items()},
                    "phase1": phase1, "phase2": phase2,
                    "validation": {"sharpe": round(base_sharpe,4), "perm_p": round(p_val,4),
                                   "perm_exceed": cnt, "perm_N": N, "walkforward": wf,
                                   "broker_survivors": len(survivors)},
                    "total_sec": round(time.time()-t0,1)}, f, indent=2, default=str)
    print(f"\nSaved. Total: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
