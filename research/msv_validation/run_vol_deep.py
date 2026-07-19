"""Deep-dive on Vol Expansion contrarian — fast version."""

import sys, os, numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import deque

project_root = str(Path(__file__).resolve().parents[2])
os.chdir(project_root)
cd_root = os.path.join(project_root, "currency_decomposition")
sys.path.insert(0, cd_root)

import MetaTrader5 as mt5
if not mt5.initialize():
    raise RuntimeError("MT5 init failed")

from config.settings import BASE_CURRENCY_MAP
ALL_PAIRS = list(BASE_CURRENCY_MAP.keys())[:15]

def load_data():
    end = datetime.now()
    start = end - timedelta(days=120)
    all_data = {}
    for pair in ALL_PAIRS:
        rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M5, start, end)
        if rates is not None and len(rates) > 0:
            all_data[pair] = rates
    return all_data, [p for p in ALL_PAIRS if p in all_data]

def stats(trades):
    if not trades or len(trades) < 5: return None
    pnls = [t["pnl"] for t in trades]
    mu, s = float(np.mean(pnls)), float(np.std(pnls))
    wr = sum(1 for t in trades if t["won"]) / len(trades) * 100
    t_stat = mu / (s / np.sqrt(len(trades))) if s > 0 else 0
    return {"n":len(trades),"wr":wr,"mean_bp":mu,"mean_usd":mu*10,"t_stat":t_stat}

def compute_avg_atr(all_data, avail, idx, period=12):
    """Average pair ATR over period bars ending at idx."""
    total = 0.0
    for p in avail:
        tr_sum = 0.0
        for j in range(idx - period + 1, idx + 1):
            hi = float(all_data[p][j]["high"])
            lo = float(all_data[p][j]["low"])
            pc = float(all_data[p][j - 1]["close"])
            tr = max(hi - lo, abs(hi - pc), abs(lo - pc))
            tr_sum += tr / float(all_data[p][j]["close"])
        total += tr_sum / period
    return (total / len(avail)) * 10000 if avail else 0

def strat_vol(all_data, avail, start, end, hold=3, top_n=2, direction="contrarian",
              compression=48, thresh=1.5, costs_bp=0.3):
    N = min(len(v) for v in all_data.values() if v is not None)
    trades = []
    short_atr = deque(maxlen=12)
    long_atr = deque(maxlen=compression)
    for idx in range(max(compression + 12 + 5, start), min(N - hold, end - hold)):
        atr = compute_avg_atr(all_data, avail, idx, period=1)
        short_atr.append(atr)
        long_atr.append(atr)
        if len(long_atr) < compression or len(short_atr) < 12: continue
        ratio = float(np.mean(short_atr)) / float(np.mean(long_atr))
        if ratio < thresh: continue
        pair_moves = []
        for p in avail:
            cur = float(all_data[p][idx]["close"])
            bf = float(all_data[p][idx - 3]["close"])
            ret = (cur / bf - 1) if bf > 0 else 0
            pair_moves.append((p, abs(ret), ret))
        pair_moves.sort(key=lambda x: x[1], reverse=True)
        for p, mag, ret in pair_moves[:top_n]:
            if idx + 1 + hold >= N: continue
            entry = float(all_data[p][idx + 1]["open"])
            exit_ = float(all_data[p][idx + hold]["close"])
            dir_signal = 1 if (direction == "momentum" and ret > 0) or (direction == "contrarian" and ret < 0) else -1
            gross = dir_signal * (exit_ / entry - 1) if entry > 0 else 0
            net = gross - costs_bp / 10000
            trades.append({"pnl": net * 10000, "won": net > 0, "idx": idx})
    return trades

def main():
    all_data, avail = load_data()
    N = min(len(v) for v in all_data.values())
    split = int(N * 0.7)
    print(f"Data: {len(avail)} pairs, {N} bars, train={split}, test={N-split}")

    # ── FOCUSED PLATEAU ──
    print(f"\n{'='*60}")
    print("VOL EXPANSION CONTRARIAN — PLATEAU")
    print("="*60)
    print(f"  {'Comp':>5s}  {'Thresh':>7s}  {'H':>3s}  {'T':>3s}  {'n':>5s}  {'WR':>6s}  {'Mean':>8s}  {'t':>6s}")
    print(f"  {'-'*50}")

    results = []
    for cp in [24, 48, 96]:
        for th in [1.3, 1.5, 2.0, 3.0]:
            for h in [1, 2, 3]:
                for t in [1, 2]:
                    tr = strat_vol(all_data, avail, 100, split, hold=h, top_n=t,
                                    direction="contrarian", compression=cp, thresh=th)
                    s = stats(tr)
                    if s and s["n"] >= 10:
                        results.append((s["wr"], s["mean_bp"], s["n"], s["t_stat"], cp, th, h, t, tr))
                        print(f"  {cp:>4d}  {th:>5.1f}   H{h:>2d}  T{t:>2d}:  {s['n']:4d}  {s['wr']:5.1f}%  {s['mean_bp']:>+7.2f}bp  {s['t_stat']:>+5.2f}")

    if not results:
        print("  No valid configs found.")
        mt5.shutdown()
        return

    results.sort(key=lambda x: -x[0])
    strong = [r for r in results if r[0] >= 58 and r[2] >= 10]
    best = results[0]
    print(f"\n  Configs with WR >= 58%: {len(strong)}/{len(results)} ({(len(strong)/len(results)*100):.0f}%)")
    print(f"  Best: comp={best[4]} thresh={best[5]} H={best[6]} T={best[7]} → {best[0]:.1f}% WR")

    best_p = {"hold": best[6], "top_n": best[7], "direction": "contrarian",
              "compression": best[4], "thresh": best[5]}

    # Monthly breakdown (run on full data)
    tr_all = strat_vol(all_data, avail, 100, N, **best_p)
    by_month = {}
    for t in tr_all:
        dt = datetime.fromtimestamp(float(all_data[avail[0]][t["idx"]]["time"]), tz=timezone.utc)
        m = dt.strftime("%Y-%m")
        if m not in by_month: by_month[m] = []
        by_month[m].append(t["pnl"])
    print(f"\n  Monthly:")
    monthly_wrs = []
    for m in sorted(by_month.keys()):
        v = by_month[m]
        wr = sum(1 for x in v if x > 0) / len(v) * 100
        monthly_wrs.append(wr)
        print(f"    {m}:  n={len(v):4d}  wr={wr:5.1f}%  mean={float(np.mean(v)):>+.2f}bp")
    print(f"  All months >50%: {all(w > 50 for w in monthly_wrs)}")

    # OOS
    oos = strat_vol(all_data, avail, split, N, **best_p)
    s_oos = stats(oos)
    if s_oos:
        print(f"\n  OOS: n={s_oos['n']:4d}  wr={s_oos['wr']:5.1f}%  mean={s_oos['mean_bp']:>+7.2f}bp  t={s_oos['t_stat']:+.2f}")

    # Cost sweep
    print(f"\n  Cost sweep:")
    for c in [0, 0.3, 0.5, 0.8]:
        tc = strat_vol(all_data, avail, 100, split, costs_bp=c, **best_p)
        sc = stats(tc)
        if sc: print(f"    {c:>4.1f}bp:  n={sc['n']:4d}  wr={sc['wr']:5.1f}%  mean={sc['mean_bp']:>+.2f}bp")

    # Hour filter
    print(f"\n  Session filter test (all data):")
    for label, hrs in [("All", None), ("Asia 0-6", [0,1,2,3,4,5,6]),
                        ("London 7-15", [7,8,9,10,11,12,13,14,15]),
                        ("NY 16-23", [16,17,18,19,20,21,22,23])]:
        # Simulate hour filter by filtering trades
        filtered = []
        for t in tr_all:
            dt = datetime.fromtimestamp(float(all_data[avail[0]][t["idx"]]["time"]), tz=timezone.utc)
            if hrs is None or dt.hour in hrs:
                filtered.append(t)
        s_f = stats(filtered)
        if s_f:
            print(f"    {label:>15s}:  n={s_f['n']:4d}  wr={s_f['wr']:5.1f}%  mean={s_f['mean_bp']:>+.2f}bp")

    mt5.shutdown()

if __name__ == "__main__":
    main()
