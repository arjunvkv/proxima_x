"""Focused search for non-Tokyo mean reversion edges. Faster version."""

import sys, os, numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import deque
from itertools import product

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
    available = [p for p in ALL_PAIRS if p in all_data]
    return all_data, available

def stats(trades):
    if not trades or len(trades) < 10:
        return None
    pnls = [t["pnl"] for t in trades]
    mu, s = float(np.mean(pnls)), float(np.std(pnls))
    wr = sum(1 for t in trades if t["won"]) / len(trades) * 100
    t = mu / (s / np.sqrt(len(trades))) if s > 0 else 0
    return {"n":len(trades),"wr":wr,"mean_bp":mu,"mean_usd":mu*10,"t_stat":t}

def bt(all_data, avail, session, hold=3, lookback=3, top_n=3,
       direction="both", max_pos=3, vol_filter=False, min_move_bp=0,
       costs_bp=0, exclude_usd=False):
    N = min(len(v) for v in all_data.values() if v is not None)
    atr_window = deque(maxlen=288)
    positions = {}
    trades = []

    for idx in range(lookback, N - hold):
        dt = datetime.fromtimestamp(float(all_data[avail[0]][idx]["time"]), tz=timezone.utc)
        hour = dt.hour

        if session == "ny" and not (16 <= hour < 24):
            continue
        if session == "london" and not (7 <= hour < 16):
            continue
        if session == "asia_0":
            if hour != 0:
                continue

        atr = 0.0
        for p in avail:
            hi = float(all_data[p][idx]["high"])
            lo = float(all_data[p][idx]["low"])
            pc = float(all_data[p][idx - 1]["close"])
            tr = max(hi - lo, abs(hi - pc), abs(lo - pc))
            atr += tr / float(all_data[p][idx]["close"])
        atr /= len(avail)
        atr_window.append(atr)

        if vol_filter and len(atr_window) >= 30:
            if atr <= sorted(atr_window)[2 * len(atr_window) // 3]:
                continue

        for p in list(positions.keys()):
            if idx >= positions[p]:
                del positions[p]
        if len(positions) >= max_pos:
            continue

        pair_moves = []
        for p in avail:
            if p in positions:
                continue
            if exclude_usd and "USD" in p:
                continue
            cur = float(all_data[p][idx]["close"])
            bf = float(all_data[p][idx - lookback]["close"])
            ret = (cur / bf - 1) if bf > 0 else 0
            if min_move_bp > 0 and abs(ret * 10000) < min_move_bp:
                continue
            pair_moves.append((p, abs(ret), ret))

        pair_moves.sort(key=lambda x: x[1], reverse=True)

        for p, mag, ret in pair_moves[:top_n]:
            if p in positions:
                continue
            if len(positions) >= max_pos:
                break
            if direction == "long" and ret > 0:
                continue
            if direction == "short" and ret < 0:
                continue
            dir_signal = 1 if ret < 0 else -1

            if idx + 1 + hold >= N:
                continue
            spread_cost = costs_bp / 10000
            entry = float(all_data[p][idx + 1]["open"])
            exit_ = float(all_data[p][idx + hold]["close"])
            gross = dir_signal * (exit_ / entry - 1) if entry > 0 else 0
            net_pnl = gross - spread_cost
            won = net_pnl > 0
            trades.append({"pnl": net_pnl * 10000, "won": won, "hour": hour,
                           "pair": p, "dt": dt})
            positions[p] = idx + hold
    return trades

def sweep_one(all_data, avail, session, label):
    """Single focused sweep."""
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")

    configs = []

    # Core: vary hold + lookback, both directions
    for hold in [1, 2, 3, 5]:
        for lookback in [1, 2, 3, 5]:
            for top_n in [1, 2, 3]:
                for direction in ["both", "short", "long"]:
                    configs.append((hold, lookback, top_n, direction, False, 0, False))

    # Vol filter variants
    for hold in [2, 3]:
        for lookback in [1, 2]:
            for top_n in [1, 2]:
                configs.append((hold, lookback, top_n, "both", True, 0, False))

    # Min move filter
    for mm in [0.3, 0.5]:
        for hold in [2, 3]:
            configs.append((hold, 2, 2, "both", False, mm, False))

    # Exclude USD (NY)
    if "ny" in session:
        configs.append((2, 2, 2, "both", False, 0, True))
        configs.append((3, 2, 2, "both", False, 0, True))
        configs.append((2, 1, 2, "both", False, 0, True))

    results = []
    total = len(configs)
    for i, (hold, lookback, top_n, direction, volF, mm, excl_usd) in enumerate(configs):
        trades = bt(all_data, avail, session=session, hold=hold, lookback=lookback,
                    top_n=top_n, direction=direction, vol_filter=volF,
                    min_move_bp=mm, exclude_usd=excl_usd)
        s = stats(trades)
        if s:
            config_str = f"H{hold} L{lookback} T{top_n} {direction[:4]}"
            if volF: config_str += " VF"
            if mm: config_str += f" MM{mm}"
            if excl_usd: config_str += " noUSD"
            results.append((s["wr"], s["mean_bp"], s["n"], s["t_stat"], config_str,
                           {"hold":hold,"lookback":lookback,"top_n":top_n,"direction":direction,
                            "vol_filter":volF,"min_move_bp":mm,"exclude_usd":excl_usd}))

    results.sort(key=lambda x: -x[0])

    print(f"  {'Config':>35s}  {'n':>5s}  {'WR':>6s}  {'Mean(bp)':>10s}  {'t':>7s}")
    print(f"  {'-'*67}")
    for wr, mean_bp, n, t, cfg, _ in results[:20]:
        print(f"  {cfg:>35s}:  {n:4d}  {wr:5.1f}%  {mean_bp:>+9.2f}bp  {t:>+6.2f}")

    # Top by t-stat (n>50)
    sig = [r for r in results if r[2] >= 50]
    if sig:
        sig.sort(key=lambda x: -abs(x[3]))
        print(f"\n  Top 10 by |t-stat| (n>=50):")
        print(f"  {'Config':>35s}  {'n':>5s}  {'WR':>6s}  {'Mean(bp)':>10s}  {'t':>7s}")
        print(f"  {'-'*67}")
        for wr, mean_bp, n, t, cfg, _ in sig[:10]:
            print(f"  {cfg:>35s}:  {n:4d}  {wr:5.1f}%  {mean_bp:>+9.2f}bp  {t:>+6.2f}")

    # Cost sweep on #1
    if results:
        best_params = results[0][5]
        print(f"\n  Cost sweep on best ({results[0][4]}):")
        print(f"  {'Cost':>6s}  {'n':>5s}  {'WR':>6s}  {'Mean(bp)':>10s}  {'$/trade':>8s}")
        print(f"  {'-'*39}")
        for cost in [0, 0.3, 0.5, 0.8, 1.0]:
            t2 = bt(all_data, avail, session=session, costs_bp=cost, **best_params)
            s2 = stats(t2)
            if s2:
                print(f"  {cost:>4.1f}bp  {s2['n']:4d}  {s2['wr']:5.1f}%  {s2['mean_bp']:>+9.2f}bp  ${s2['mean_usd']:>+6.1f}")

    # Hourly breakdown (NY)
    if "ny" in session and results:
        best_p = results[0][5]
        t_hourly = bt(all_data, avail, session=session, costs_bp=0.3, **best_p)
        by_hour = {}
        for t in t_hourly:
            h = t["hour"]
            if h not in by_hour: by_hour[h] = []
            by_hour[h].append(t["pnl"])
        print(f"\n  Hourly:")
        print(f"  {'Hour':>6s}  {'n':>5s}  {'WR':>6s}  {'Mean(bp)':>10s}")
        for h in sorted(by_hour.keys()):
            v = by_hour[h]
            if len(v) >= 10:
                mu = float(np.mean(v))
                wr = sum(1 for x in v if x > 0) / len(v) * 100
                print(f"  {h:>2d}:00   {len(v):4d}  {wr:5.1f}%  {mu:>+8.2f}bp")

    return results

def main():
    all_data, avail = load_data()
    N = min(len(v) for v in all_data.values())
    print(f"Data: {len(avail)} pairs, {N} bars, {N/288:.0f} days")
    print(f"Pairs: {avail}")

    # Baseline
    print(f"\n{'='*65}")
    print(f"  BASELINE: Tokyo Hour 0 (ref)")
    print(f"{'='*65}")
    t0 = bt(all_data, avail, "asia_0", 3, 3, 3, "long", vol_filter=True, costs_bp=0.3)
    s0 = stats(t0)
    if s0:
        print(f"  H3 L3 T3 long VF:  n={s0['n']:4d}  wr={s0['wr']:5.1f}%  mean={s0['mean_bp']:+.2f}bp  t={s0['t_stat']:+.2f}")
        # Hourly breakdown
        for h in range(7):
            subset = [t for t in t0 if t["hour"] == h]
            if len(subset) >= 10:
                pnls = [t["pnl"] for t in subset]
                mu = float(np.mean(pnls))
                wr = sum(1 for t in subset if t["won"]) / len(subset) * 100
                print(f"    Hour {h}:  n={len(subset):4d}  wr={wr:5.1f}%  mean={mu:+.2f}bp")

    # NY
    ny = sweep_one(all_data, avail, "ny", "NEW YORK (16-23 UTC)")

    # London
    ld = sweep_one(all_data, avail, "london", "LONDON (7-15 UTC)")

    # ── FINAL COMPARISON ──
    print(f"\n{'='*65}")
    print(f"  CROSS-SESSION COMPARISON")
    print(f"{'='*65}")
    print(f"  {'Session':>18s}  {'Config':>25s}  {'n':>5s}  {'WR':>6s}  {'Mean(bp)':>10s}  {'t':>7s}")
    print(f"  {'-'*75}")

    # Best for each session (WR, n>=50)
    for sname, skey, fixed_params in [
        ("Tokyo 0", "asia_0", {"hold":3,"lookback":3,"top_n":3,"direction":"long","vol_filter":True}),
    ]:
        t2 = bt(all_data, avail, skey, costs_bp=0.3, **fixed_params)
        s2 = stats(t2)
        if s2:
            print(f"  {sname:>18s}: {fixed_params_str(fixed_params):>25s}  {s2['n']:4d}  {s2['wr']:5.1f}%  {s2['mean_bp']:>+9.2f}bp  {s2['t_stat']:>+6.2f}")

    for sname, skey, all_results in [("New York", "ny", ny), ("London", "london", ld)]:
        if all_results:
            # best by WR, n>=50
            wr_best = max([r for r in all_results if r[2] >= 50], key=lambda x: x[0], default=None)
            t_best = max([r for r in all_results if r[2] >= 100], key=lambda x: abs(x[3]), default=None)
            if wr_best:
                print(f"  {sname:>18s}: {wr_best[4]:>25s}  {wr_best[2]:4d}  {wr_best[0]:5.1f}%  {wr_best[1]:>+9.2f}bp  {wr_best[3]:>+6.2f}")
            if t_best and t_best != wr_best:
                print(f"  {'':>18s}  {t_best[4]:>25s}  {t_best[2]:4d}  {t_best[0]:5.1f}%  {t_best[1]:>+9.2f}bp  {t_best[3]:>+6.2f}")

    mt5.shutdown()

def fixed_params_str(p):
    parts = []
    if "hold" in p: parts.append(f"H{p['hold']}")
    if "lookback" in p: parts.append(f"L{p['lookback']}")
    if "top_n" in p: parts.append(f"T{p['top_n']}")
    if "direction" in p: parts.append(p["direction"][:4])
    if p.get("vol_filter"): parts.append("VF")
    return " ".join(parts)

if __name__ == "__main__":
    main()
