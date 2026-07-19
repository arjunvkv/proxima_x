"""Alternative structurally-grounded FX strategies (fast version).

Tests 4 strategy families:
  1. Asian Range Breakout (London open 07:00)
  2. London Open Momentum (07:00-08:00)
  3. Vol Expansion (after compression)
  4. NY Open Continuation (14:00)
"""

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

# ── STRATEGY 1: ASIAN RANGE BREAKOUT ──
def strat_asian_breakout(all_data, avail, start_idx, end_idx,
                          hold=6, top_n=2, breakout_dir="both", costs_bp=0.3):
    N = min(len(v) for v in all_data.values() if v is not None)
    trades = []
    for idx in range(max(300, start_idx), min(N - hold, end_idx - hold)):
        dt = datetime.fromtimestamp(float(all_data[avail[0]][idx]["time"]), tz=timezone.utc)
        hour = dt.hour
        if hour != 7: continue
        today_start = dt.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        asia_start = None
        for j in range(idx - 288, idx):
            if float(all_data[avail[0]][j]["time"]) >= today_start:
                asia_start = j
                break
        if asia_start is None or asia_start < 0: continue

        breakouts = []
        for p in avail:
            asia_h = max(float(all_data[p][j]["high"]) for j in range(asia_start, idx))
            asia_l = min(float(all_data[p][j]["low"]) for j in range(asia_start, idx))
            cur = float(all_data[p][idx]["close"])
            if breakout_dir in ("both", "long") and cur > asia_h:
                ret = (cur - asia_h) / asia_h * 10000 if asia_h > 0 else 0
                breakouts.append((p, "long", ret))
            elif breakout_dir in ("both", "short") and cur < asia_l:
                ret = (cur - asia_l) / asia_l * 10000 if asia_l > 0 else 0
                breakouts.append((p, "short", ret))
        breakouts.sort(key=lambda x: abs(x[2]), reverse=True)

        for p, direction, strength in breakouts[:top_n]:
            if idx + 1 + hold >= N: continue
            entry = float(all_data[p][idx + 1]["open"])
            exit_ = float(all_data[p][idx + hold]["close"])
            dir_signal = 1 if direction == "long" else -1
            gross = dir_signal * (exit_ / entry - 1) if entry > 0 else 0
            net = gross - costs_bp / 10000
            trades.append({"pnl": net * 10000, "won": net > 0, "idx": idx})
    return trades

# ── STRATEGY 2: LONDON OPEN MOMENTUM ──
def strat_london_momentum(all_data, avail, start_idx, end_idx,
                           hold=3, lookback=3, top_n=2, direction="momentum",
                           costs_bp=0.3):
    N = min(len(v) for v in all_data.values() if v is not None)
    trades = []
    for idx in range(max(lookback, start_idx), min(N - hold, end_idx - hold)):
        dt = datetime.fromtimestamp(float(all_data[avail[0]][idx]["time"]), tz=timezone.utc)
        hour = dt.hour
        if hour != 7: continue
        pair_moves = []
        for p in avail:
            cur = float(all_data[p][idx]["close"])
            bf = float(all_data[p][idx - lookback]["close"])
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

# ── STRATEGY 3: VOL EXPANSION ──
def strat_vol_expansion(all_data, avail, start_idx, end_idx,
                         hold=3, top_n=2, direction="momentum",
                         compression=48, thresh=1.5, costs_bp=0.3):
    N = min(len(v) for v in all_data.values() if v is not None)
    trades = []
    atr_short = deque(maxlen=12)
    atr_long = deque(maxlen=compression)
    for idx in range(max(100, start_idx), min(N - hold, end_idx - hold)):
        trs = []
        for p in avail:
            hi = float(all_data[p][idx]["high"])
            lo = float(all_data[p][idx]["low"])
            pc = float(all_data[p][idx - 1]["close"])
            tr = max(hi - lo, abs(hi - pc), abs(lo - pc))
            trs.append(tr / float(all_data[p][idx]["close"]))
        atr_now = float(np.mean(trs)) * 10000
        atr_short.append(atr_now)
        atr_long.append(atr_now)
        if len(atr_long) < compression or len(atr_short) < 12: continue
        ratio = float(np.mean(atr_short)) / float(np.mean(atr_long))
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

# ── STRATEGY 4: NY OPEN CONTINUATION ──
def strat_ny_continuation(all_data, avail, start_idx, end_idx,
                           hold=3, lookback=6, top_n=2, costs_bp=0.3):
    N = min(len(v) for v in all_data.values() if v is not None)
    trades = []
    for idx in range(max(lookback, start_idx), min(N - hold, end_idx - hold)):
        dt = datetime.fromtimestamp(float(all_data[avail[0]][idx]["time"]), tz=timezone.utc)
        hour = dt.hour
        if hour != 14: continue
        pair_moves = []
        for p in avail:
            cur = float(all_data[p][idx]["close"])
            bf = float(all_data[p][idx - lookback]["close"])
            ret = (cur / bf - 1) if bf > 0 else 0
            pair_moves.append((p, abs(ret), ret))
        pair_moves.sort(key=lambda x: x[1], reverse=True)
        for p, mag, ret in pair_moves[:top_n]:
            if idx + 1 + hold >= N: continue
            entry = float(all_data[p][idx + 1]["open"])
            exit_ = float(all_data[p][idx + hold]["close"])
            dir_signal = 1 if ret > 0 else -1
            gross = dir_signal * (exit_ / entry - 1) if entry > 0 else 0
            net = gross - costs_bp / 10000
            trades.append({"pnl": net * 10000, "won": net > 0, "idx": idx})
    return trades

def run(name, strat_fn, all_data, avail, split, configs):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    results = []
    for params in configs:
        trades = strat_fn(all_data, avail, 100, split, **params, costs_bp=0.3)
        s = stats(trades)
        if s: results.append((s["wr"], s["mean_bp"], s["n"], s["t_stat"], params, trades))
    if not results: print("  No trades."); return None
    results.sort(key=lambda x: -x[0])
    best_p = results[0][4]

    print(f"  Top 8 (train):")
    for wr, mb, n, t, p, _ in results[:8]:
        c = " ".join(f"{k}={v}" for k,v in p.items())
        print(f"    {c:>40s}:  n={n:4d}  wr={wr:5.1f}%  {mb:>+7.2f}bp  t={t:>+5.2f}")

    oos_trades = strat_fn(all_data, avail, split + 1, split + 5000, **best_p)
    s_oos = stats(oos_trades)
    wr_best = results[0][0]
    plateau = len([r for r in results if r[0] >= wr_best - 5 and r[2] >= 15])

    # Cost check at 0.5bp
    t_cost = strat_fn(all_data, avail, 100, split, costs_bp=0.5, **best_p)
    s_cost = stats(t_cost)
    cost_ok = s_cost and s_cost["mean_bp"] > 0 if s_cost else False

    print(f"  OOS:   n={s_oos['n'] if s_oos else 0:4d}  wr={s_oos['wr'] if s_oos else 0:5.1f}%  mean={s_oos['mean_bp'] if s_oos else 0:>+7.2f}bp")
    print(f"  Cost 0.5bp: {'OK' if cost_ok else 'FAIL'}  |  Plateau: {plateau}/{len(results)}  |  Best WR: {wr_best:.1f}%")

    return {"name": name, "wr": wr_best, "mean_bp": results[0][1], "cost_ok": cost_ok,
            "oos_wr": s_oos["wr"] if s_oos else 0, "oos_mean": s_oos["mean_bp"] if s_oos else 0,
            "plateau": plateau, "total_configs": len(results)}

def main():
    all_data, avail = load_data()
    N = min(len(v) for v in all_data.values())
    split = int(N * 0.7)
    print(f"Data: {len(avail)} pairs, {N} bars, train={split}, test={N-split}")

    all_r = []

    # 1: Asian range breakout
    cfg1 = [{"hold": h, "top_n": t, "breakout_dir": d}
            for h in [3, 6, 12] for t in [1, 2, 3] for d in ["both", "long", "short"]]
    r1 = run("Asian Range Breakout (07:00)", strat_asian_breakout, all_data, avail, split, cfg1)
    if r1: all_r.append(r1)

    # 2: London momentum
    cfg2 = [{"hold": h, "lookback": lb, "top_n": t, "direction": d}
            for h in [3, 6, 12] for lb in [3, 6, 12] for t in [1, 2] for d in ["momentum", "contrarian"]]
    r2 = run("London Open Momentum (07:00)", strat_london_momentum, all_data, avail, split, cfg2)
    if r2: all_r.append(r2)

    # 3: Vol expansion
    cfg3 = [{"hold": h, "top_n": t, "direction": d, "compression": cp, "thresh": et}
            for h in [3, 6] for t in [1, 2] for d in ["momentum", "contrarian"]
            for cp in [48, 96] for et in [1.3, 1.5, 2.0]]
    r3 = run("Vol Expansion (after compression)", strat_vol_expansion, all_data, avail, split, cfg3)
    if r3: all_r.append(r3)

    # 4: NY continuation
    cfg4 = [{"hold": h, "lookback": lb, "top_n": t}
            for h in [3, 6, 12] for lb in [3, 6, 12] for t in [1, 2, 3]]
    r4 = run("NY Open Continuation (14:00)", strat_ny_continuation, all_data, avail, split, cfg4)
    if r4: all_r.append(r4)

    # ── COMPARISON ──
    print(f"\n{'='*60}")
    print("  COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Strategy':>35s}  {'WR':>6s}  {'Mean':>8s}  {'Cost':>6s}  {'OOS_WR':>7s}  {'Plat':>5s}")
    print(f"  {'-'*70}")
    print(f"  {'Tokyo Hour 0 (bench)':>35s}:  80.2%  +4.58bp  YES    78.9%    91")

    for r in sorted(all_r, key=lambda x: -x["wr"]):
        print(f"  {r['name']:>35s}:  {r['wr']:5.1f}%  {r['mean_bp']:>+7.2f}bp  "
              f"{'YES' if r['cost_ok'] else 'NO':>6s}  {r['oos_wr']:>5.1f}%  {r['plateau']:>3d}")

    mt5.shutdown()

if __name__ == "__main__":
    main()
