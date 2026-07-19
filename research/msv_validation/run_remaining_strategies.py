"""Remaining FX strategies — fast version (no correlation)."""

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
    if not trades or len(trades) < 10: return None
    pnls = [t["pnl"] for t in trades]
    mu, s = float(np.mean(pnls)), float(np.std(pnls))
    wr = sum(1 for t in trades if t["won"]) / len(trades) * 100
    t_stat = mu / (s / np.sqrt(len(trades))) if s > 0 else 0
    return {"n":len(trades),"wr":wr,"mean_bp":mu,"mean_usd":mu*10,"t_stat":t_stat}

def run_strat(name, strat_fn, all_data, avail, split, configs):
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    N = min(len(v) for v in all_data.values())
    results = []
    for params in configs:
        trades = strat_fn(all_data, avail, 100, split, **params)
        s = stats(trades)
        if s: results.append((s["wr"], s["mean_bp"], s["n"], s["t_stat"], params, trades))
    if not results: print("  No trades."); return
    results.sort(key=lambda x: -x[0])
    best_p = results[0][4]

    print(f"  Top 6:")
    for wr, mb, n, t, p, _ in results[:6]:
        c = " ".join(f"{k}={v}" for k,v in p.items())
        print(f"    {c:>35s}:  n={n:4d}  wr={wr:5.1f}%  {mb:>+7.2f}bp  t={t:>+5.2f}")

    wr_best = results[0][0]
    oos = strat_fn(all_data, avail, split, N, **best_p)
    s_oos = stats(oos)
    tc = strat_fn(all_data, avail, 100, split, costs_bp=0.5, **best_p)
    sc = stats(tc)
    cost_ok = sc and sc["mean_bp"] > 0 if sc else False
    oos_ok = s_oos and s_oos["mean_bp"] > 0 if s_oos else False

    # Count configs within 5pp of best, n>=15
    plateau = len([r for r in results if r[0] >= wr_best - 5 and r[2] >= 15])

    print(f"    {'─'*50}")
    print(f"    Cost 0.5bp: {'OK' if cost_ok else 'FAIL'}  OOS: {'OK' if oos_ok else 'FAIL'}  "
          f"Plat: {plateau}/{len(results)}  Best: {wr_best:.1f}%")

    return {"name":name,"wr":wr_best,"cost_ok":cost_ok,"oos_ok":oos_ok,"plateau":plateau}

# ── 1: CARRY ROLLOVER (22:00) ──
def strat1(all_data, avail, start, end, hold=3, top_n=2, direction="contrarian", costs_bp=0.3):
    N = min(len(v) for v in all_data.values() if v is not None)
    trades = []
    for idx in range(max(3, start), min(N - hold, end - hold)):
        dt = datetime.fromtimestamp(float(all_data[avail[0]][idx]["time"]), tz=timezone.utc)
        if dt.hour != 22: continue
        pm = []
        for p in avail:
            cur = float(all_data[p][idx]["close"])
            bf = float(all_data[p][idx - 3]["close"])
            ret = (cur / bf - 1) if bf > 0 else 0
            pm.append((p, abs(ret), ret))
        pm.sort(key=lambda x: x[1], reverse=True)
        for p, mag, ret in pm[:top_n]:
            if idx + 1 + hold >= N: continue
            entry = float(all_data[p][idx + 1]["open"])
            exit_ = float(all_data[p][idx + hold]["close"])
            ds = 1 if (direction=="momentum" and ret>0) or (direction=="contrarian" and ret<0) else -1
            gross = ds * (exit_/entry - 1) if entry>0 else 0
            trades.append({"pnl": (gross - costs_bp/10000)*10000, "won": gross > costs_bp/10000, "idx": idx})
    return trades

# ── 2: SESSION CLOSE ──
def strat2(all_data, avail, start, end, hold=3, lookback=3, top_n=2, session="ny", direction="contrarian", costs_bp=0.3):
    hrs = {"tokyo":[5,6],"london":[15,16],"ny":[21,22]}.get(session,[21,22])
    N = min(len(v) for v in all_data.values() if v is not None)
    trades = []
    for idx in range(max(lookback, start), min(N - hold, end - hold)):
        dt = datetime.fromtimestamp(float(all_data[avail[0]][idx]["time"]), tz=timezone.utc)
        if dt.hour not in hrs: continue
        pm = []
        for p in avail:
            cur = float(all_data[p][idx]["close"])
            bf = float(all_data[p][idx - lookback]["close"])
            ret = (cur / bf - 1) if bf > 0 else 0
            pm.append((p, abs(ret), ret))
        pm.sort(key=lambda x: x[1], reverse=True)
        for p, mag, ret in pm[:top_n]:
            if idx + 1 + hold >= N: continue
            entry = float(all_data[p][idx + 1]["open"])
            exit_ = float(all_data[p][idx + hold]["close"])
            ds = 1 if (direction=="momentum" and ret>0) or (direction=="contrarian" and ret<0) else -1
            gross = ds * (exit_/entry - 1) if entry>0 else 0
            trades.append({"pnl": (gross - costs_bp/10000)*10000, "won": gross > costs_bp/10000, "idx": idx})
    return trades

# ── 3: WMR FIXING (16:00) ──
def strat3(all_data, avail, start, end, hold=3, top_n=2, direction="contrarian", costs_bp=0.3):
    N = min(len(v) for v in all_data.values() if v is not None)
    trades = []
    for idx in range(max(3, start), min(N - hold, end - hold)):
        dt = datetime.fromtimestamp(float(all_data[avail[0]][idx]["time"]), tz=timezone.utc)
        if not (dt.hour==16 and dt.minute<5): continue
        pm = []
        for p in avail:
            cur = float(all_data[p][idx]["close"])
            bf = float(all_data[p][idx - 3]["close"])
            ret = (cur / bf - 1) if bf > 0 else 0
            pm.append((p, abs(ret), ret))
        pm.sort(key=lambda x: x[1], reverse=True)
        for p, mag, ret in pm[:top_n]:
            if idx + 1 + hold >= N: continue
            entry = float(all_data[p][idx + 1]["open"])
            exit_ = float(all_data[p][idx + hold]["close"])
            ds = 1 if (direction=="momentum" and ret>0) or (direction=="contrarian" and ret<0) else -1
            gross = ds * (exit_/entry - 1) if entry>0 else 0
            trades.append({"pnl": (gross - costs_bp/10000)*10000, "won": gross > costs_bp/10000, "idx": idx})
    return trades

# ── 4: LONDON-NY OVERLAP ──
def strat4(all_data, avail, start, end, hold=3, lookback=3, top_n=2, direction="momentum", costs_bp=0.3):
    N = min(len(v) for v in all_data.values() if v is not None)
    trades = []
    for idx in range(max(lookback, start), min(N - hold, end - hold)):
        dt = datetime.fromtimestamp(float(all_data[avail[0]][idx]["time"]), tz=timezone.utc)
        if not (12 <= dt.hour < 16): continue
        pm = []
        for p in avail:
            cur = float(all_data[p][idx]["close"])
            bf = float(all_data[p][idx - lookback]["close"])
            ret = (cur / bf - 1) if bf > 0 else 0
            pm.append((p, abs(ret), ret))
        pm.sort(key=lambda x: x[1], reverse=True)
        for p, mag, ret in pm[:top_n]:
            if idx + 1 + hold >= N: continue
            entry = float(all_data[p][idx + 1]["open"])
            exit_ = float(all_data[p][idx + hold]["close"])
            ds = 1 if (direction=="momentum" and ret>0) or (direction=="contrarian" and ret<0) else -1
            gross = ds * (exit_/entry - 1) if entry>0 else 0
            trades.append({"pnl": (gross - costs_bp/10000)*10000, "won": gross > costs_bp/10000, "idx": idx})
    return trades

def main():
    all_data, avail = load_data()
    N = min(len(v) for v in all_data.values())
    split = int(N * 0.7)
    print(f"Data: {len(avail)} pairs, {N} bars, train={split}, test={N-split}")

    all_r = []

    # 1: Rollover
    cfg1 = [{"hold":h,"top_n":t,"direction":d}
            for h in [2,3,5] for t in [1,2] for d in ["contrarian","momentum"]]
    all_r.append(run_strat("Carry Rollover (22:00)", strat1, all_data, avail, split, cfg1))

    # 2: Session close
    cfg2 = [{"hold":h,"lookback":lb,"top_n":t,"session":s,"direction":d}
            for h in [2,3,5] for lb in [3,5] for t in [1,2]
            for s in ["tokyo","london","ny"] for d in ["contrarian","momentum"]]
    all_r.append(run_strat("Session Close Pressure", strat2, all_data, avail, split, cfg2))

    # 3: Fixing
    cfg3 = [{"hold":h,"top_n":t,"direction":d}
            for h in [2,3,5] for t in [1,2] for d in ["contrarian","momentum"]]
    all_r.append(run_strat("WMR Fixing (16:00)", strat3, all_data, avail, split, cfg3))

    # 4: Overlap
    cfg4 = [{"hold":h,"lookback":lb,"top_n":t,"direction":d}
            for h in [2,3,5] for lb in [3,6] for t in [1,2] for d in ["momentum","contrarian"]]
    all_r.append(run_strat("London-NY Overlap (12-16)", strat4, all_data, avail, split, cfg4))

    # FINAL
    print(f"\n{'='*55}")
    print("  FINAL COMPARISON")
    print(f"{'='*55}")
    print(f"  {'Strategy':>30s}  {'WR':>6s}  {'Cost':>6s}  {'OOS':>6s}  {'Plat':>5s}")
    print(f"  {'─'*55}")
    print(f"  {'Tokyo Hour 0 (bench)':>30s}:  80.2%  YES    YES     91")

    for r in sorted(all_r, key=lambda x: -x["wr"]) if all_r else []:
        print(f"  {r['name']:>30s}:  {r['wr']:5.1f}%  "
              f"{'YES' if r['cost_ok'] else 'NO':>6s}  {'YES' if r['oos_ok'] else 'NO':>6s}  {r['plateau']:>3d}")

    good = [r for r in all_r if r["cost_ok"] and r["oos_ok"] and r["plateau"] >= 3]
    print(f"\n  Non-overfitting strategies: {len(good)}/5 (excluding Tokyo benchmark)")
    print(f"  Verdict: {'YES' if good else 'NO'} second tradeable edge found")

    mt5.shutdown()

if __name__ == "__main__":
    main()
