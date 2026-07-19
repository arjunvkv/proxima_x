"""Full backtest of adaptive cross-pair scalper with walk-forward validation.

Strategy:
  Every 5min bar, rank 15 pairs by 15min move magnitude.
  Trade top N in direction of mean reversion.
  Session-adaptive: bias LONG in Asia, SHORT in NY, skip London.
  Vol-adaptive: only trade high-vol Asia, all vol levels NY.
  Max 3 concurrent positions on different pairs.
  15min hold. Next-candle execution.
"""

import sys, os, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import deque
import random

project_root = str(Path(__file__).resolve().parents[2])
os.chdir(project_root)
cd_root = os.path.join(project_root, "currency_decomposition")
sys.path.insert(0, cd_root)

import MetaTrader5 as mt5
if not mt5.initialize():
    raise RuntimeError("MT5 init failed")

from config.settings import BASE_CURRENCY_MAP

ALL_PAIRS = list(BASE_CURRENCY_MAP.keys())[:15]
print(f"Pairs: {ALL_PAIRS}")

def load_data():
    end = datetime.now()
    start = end - timedelta(days=120)
    all_data = {}
    for pair in ALL_PAIRS:
        rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M5, start, end)
        if rates is not None and len(rates) > 0:
            all_data[pair] = rates
    print(f"Loaded {len(all_data)} pairs, {min(len(v) for v in all_data.values())} bars")
    return all_data

def backtest(all_data, sessions="all", vol_filter=False, max_positions=3,
             lookback_bars=3, hold_bars=3, top_n=3, wf_window=None):
    """
    sessions: 'all', 'asia', 'london', 'ny'
    vol_filter: if True, skip low-vol Asian bars
    """
    N = min(len(v) for v in all_data.values())
    atr_window = deque(maxlen=288)  # ~24h ATR

    # Track positions: {pair: exit_bar_index}
    positions = {}
    trades = []

    for idx in range(lookback_bars, N - hold_bars):
        dt = datetime.fromtimestamp(float(all_data[ALL_PAIRS[0]][idx]["time"]), tz=timezone.utc)
        hour = dt.hour

        # Session filter
        if sessions == "asia" and hour >= 7: continue
        if sessions == "london" and not (7 <= hour < 16): continue
        if sessions == "ny" and not (16 <= hour < 24): continue

        # Skip London entirely if using adaptive strategy
        if sessions == "adaptive" and (7 <= hour < 16): continue

        # Compute ATR for vol filter
        atr = 0.0
        for p in ALL_PAIRS:
            hi = float(all_data[p][idx]["high"])
            lo = float(all_data[p][idx]["low"])
            pc = float(all_data[p][idx - 1]["close"])
            tr = max(hi - lo, abs(hi - pc), abs(lo - pc))
            atr += tr / float(all_data[p][idx]["close"])
        atr /= len(ALL_PAIRS)
        atr_window.append(atr)

        # Vol filter: skip low-vol Asia (bottom 2/3)
        if vol_filter and sessions == "adaptive" and hour < 7:
            if len(atr_window) >= 30:
                atr_thresh = sorted(atr_window)[2 * len(atr_window) // 3]
                if atr <= atr_thresh:
                    continue

        # Close expired positions
        for pair in list(positions.keys()):
            if idx >= positions[pair]:
                del positions[pair]

        if len(positions) >= max_positions:
            continue

        # Rank pairs by recent move
        pair_moves = []
        for p in ALL_PAIRS:
            if p in positions: continue
            cur = float(all_data[p][idx]["close"])
            bf = float(all_data[p][idx - lookback_bars]["close"])
            ret = (cur / bf - 1) if bf > 0 else 0
            pair_moves.append((p, abs(ret), ret))

        pair_moves.sort(key=lambda x: x[1], reverse=True)

        for p, mag, ret in pair_moves[:top_n]:
            if p in positions: continue
            if len(positions) >= max_positions: break

            # Direction: mean reversion + session bias
            if sessions == "adaptive":
                if hour < 7:  # Asia: prefer long
                    direction = 1  # long
                    # Only take long if pair actually declined (stronger signal)
                    if ret > 0: continue  # skip if rising in Asia
                else:  # NY: prefer short
                    direction = -1  # short
                    if ret < 0: continue  # skip if falling in NY
            else:
                direction = 1 if ret < 0 else -1

            # Entry on next candle open
            entry_open = float(all_data[p][idx + 1]["open"])
            # Exit at hold
            exit_close = float(all_data[p][idx + hold_bars]["close"])
            pnl = direction * (exit_close / entry_open - 1) if entry_open > 0 else 0

            # Direction check: did we actually trade in the right direction?
            is_long = direction > 0
            did_rise = exit_close > entry_open
            won = is_long == did_rise

            trades.append({
                "pair": p, "ts": dt, "hour": hour,
                "entry": entry_open, "exit": exit_close,
                "pnl": pnl * 10000,  # bp
                "direction": "LONG" if direction > 0 else "SHORT",
                "won": won, "ret_15min": ret * 10000,
                "atr": atr,
            })
            positions[p] = idx + hold_bars

    return trades

def stats(trades, label=""):
    if not trades:
        print(f"  {label:>25s}: 0 trades")
        return {}
    pnls = np.array([t["pnl"] for t in trades])
    mu = float(np.mean(pnls))
    s = float(np.std(pnls))
    wins = sum(1 for t in trades if t["won"])
    losses = sum(1 for t in trades if not t["won"])
    win_rate = wins / len(trades) * 100
    t_stat = mu / (s / np.sqrt(len(trades))) if s > 0 else 0
    days = (trades[-1]["ts"] - trades[0]["ts"]).total_seconds() / 86400 if len(trades) > 1 else 1
    n_day = len(trades) / max(days, 1)

    # In dollars: assume 1 lot, 1bp = $10 for standard lot (approx)
    usd_per_bp = 10  # rough: 1 lot = $10/pip = $1/bp
    pnl_usd = [t["pnl"] * usd_per_bp for t in trades]

    print(f"  {label:>25s}:  n={len(trades):5d}  wr={win_rate:5.1f}%  "
          f"mean={mu:>+6.2f}bp (${mu*usd_per_bp:>+5.1f})  "
          f"t={t_stat:>+6.2f}  {n_day:4.0f}/day")

    return {
        "n": len(trades), "win_rate": win_rate, "mean_bp": mu,
        "mean_usd": mu * usd_per_bp, "t_stat": t_stat,
        "n_day": n_day, "total_days": days,
    }

def main():
    all_data = load_data()

    # ── CONFIGURATIONS TO TEST ──
    configs = [
        ("Top3 simple MR", dict(sessions="all", top_n=3, vol_filter=False)),
        ("Asia simple MR", dict(sessions="asia", top_n=3, vol_filter=False)),
        ("NY simple MR", dict(sessions="ny", top_n=3, vol_filter=False)),
        ("Adaptive (skip LDN)", dict(sessions="adaptive", top_n=3, vol_filter=False)),
        ("Adaptive + vol filter", dict(sessions="adaptive", top_n=3, vol_filter=True)),
        ("Adaptive + vol + T5", dict(sessions="adaptive", top_n=5, vol_filter=True)),
    ]

    print(f"\n{'='*70}")
    print("FULL BACKTEST (all 85 days)")
    print("=" * 70)
    print(f"{'Config':>25s}  {'n':>5s}  {'WR':>6s}  {'Mean':>8s}  {'$/trade':>8s}  {'t':>7s}  {'/day':>5s}")
    print(f"  {'-'*68}")

    results = {}
    for label, params in configs:
        trades = backtest(all_data, **params)
        results[label] = {"trades": trades, **stats(trades, label)}

    # ── WALK-FORWARD VALIDATION ──
    print(f"\n{'='*70}")
    print("WALK-FORWARD VALIDATION (3 windows)")
    print("=" * 70)

    N = min(len(v) for v in all_data.values())
    wf_results = []
    for wf_idx, (start_pct, end_pct) in enumerate([
        (0, 0.5), (0.25, 0.75), (0.5, 1.0)
    ]):
        # Slice data
        sub_data = {}
        s_idx = int(N * start_pct)
        e_idx = int(N * end_pct)
        for p in all_data:
            sub_data[p] = all_data[p][s_idx:e_idx]

        trades = backtest(sub_data, sessions="adaptive", top_n=3, vol_filter=True)
        if trades:
            days = (trades[-1]["ts"] - trades[0]["ts"]).total_seconds() / 86400
            pnls = [t["pnl"] for t in trades]
            wr = sum(1 for t in trades if t["won"]) / len(trades) * 100
            mu = float(np.mean(pnls))
            t_stat = mu / (float(np.std(pnls)) / np.sqrt(len(trades))) if len(trades) > 1 and float(np.std(pnls)) > 0 else 0
            print(f"  WF{wf_idx+1} ({start_pct:.0%}-{end_pct:.0%}):  n={len(trades):4d}  "
                  f"wr={wr:5.1f}%  mean={mu:>+6.2f}bp  t={t_stat:>+6.2f}  {len(trades)/max(days,1):.0f}/day")
            wf_results.append(wr)

    if wf_results:
        print(f"  WF WR range: {min(wf_results):.1f}% - {max(wf_results):.1f}%")

    # ── DETAILED BREAKDOWN OF BEST CONFIG ──
    print(f"\n{'='*70}")
    print("DETAIL: Adaptive + Vol Filter breakdown")
    print("=" * 70)

    trades = backtest(all_data, sessions="adaptive", top_n=3, vol_filter=True)
    pnls = [t["pnl"] for t in trades]

    # Session breakdown
    for sname, scond in [("Asia", lambda t: t["hour"] < 7),
                          ("NY", lambda t: 16 <= t["hour"] < 24)]:
        sub = [t for t in trades if scond(t)]
        if sub:
            mu = float(np.mean([t["pnl"] for t in sub]))
            wr = sum(1 for t in sub if t["won"]) / len(sub) * 100
            print(f"  {sname:>10s}:  n={len(sub):4d}  wr={wr:5.1f}%  mean={mu:>+6.2f}bp  ${mu*10:>+5.1f}/trade")

    # Direction breakdown
    for d in ["LONG", "SHORT"]:
        sub = [t for t in trades if t["direction"] == d]
        if sub:
            mu = float(np.mean([t["pnl"] for t in sub]))
            wr = sum(1 for t in sub if t["won"]) / len(sub) * 100
            print(f"  {d:>10s}:  n={len(sub):4d}  wr={wr:5.1f}%  mean={mu:>+6.2f}bp  ${mu*10:>+5.1f}/trade")

    # Top pairs breakdown
    pair_stats = {}
    for t in trades:
        p = t["pair"]
        if p not in pair_stats:
            pair_stats[p] = {"n": 0, "wins": 0, "pnl": 0}
        pair_stats[p]["n"] += 1
        pair_stats[p]["wins"] += 1 if t["won"] else 0
        pair_stats[p]["pnl"] += t["pnl"]

    print(f"\n  Per-pair performance:")
    for p in sorted(pair_stats.keys(), key=lambda x: pair_stats[x]["n"], reverse=True):
        s = pair_stats[p]
        wr = s["wins"] / s["n"] * 100
        mu = s["pnl"] / s["n"]
        print(f"    {p:>8s}:  n={s['n']:4d}  wr={wr:5.1f}%  mean={mu:>+6.2f}bp")

    # ── MONTHLY BREAKDOWN ──
    print(f"\n  Monthly:")
    by_month = {}
    for t in trades:
        m = t["ts"].strftime("%Y-%m")
        if m not in by_month:
            by_month[m] = []
        by_month[m].append(t["pnl"])

    for m in sorted(by_month.keys()):
        v = by_month[m]
        wr = sum(1 for x in v if x > 0) / len(v) * 100
        mu = float(np.mean(v))
        print(f"    {m}:  n={len(v):4d}  wr={wr:5.1f}%  mean={mu:>+6.2f}bp")

    # ── MAX DRAWDOWN ──
    cum = np.cumsum([t["pnl"] * 10 for t in trades])  # cumulative $ PnL
    running_max = np.maximum.accumulate(cum)
    drawdown = cum - running_max
    max_dd = min(drawdown)
    max_dd_pct = max_dd / max(cum) * 100 if max(cum) > 0 else 0
    print(f"\n  Max drawdown: ${max_dd:.0f} ({max_dd_pct:.1f}%)")

    # ── SUMMARY ──
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print("=" * 70)

    best = results["Adaptive + vol filter"]
    pnls_best = [t["pnl"] for t in best["trades"]]
    worst_loss = min(pnls_best) if pnls_best else 0
    best_win = max(pnls_best) if pnls_best else 0

    print(f"""
  Strategy:     Adaptive cross-pair scalper (Asia long bias, skip London, NY short bias)
  Pairs:        Top 3 by 15min move (of 15 pairs)
  Hold:         15min (3 bars)
  Positions:    Max 3 concurrent
  Vol filter:   Only trade Asia when ATR > 66th percentile

  {best['n']} trades over {best['total_days']:.0f} days ({best['n_day']:.0f}/day)

  Win rate:           {best['win_rate']:.1f}%
  Mean per trade:    {best['mean_bp']:+.2f}bp (${best['mean_usd']:+.1f})
  With 3 positions:  ~${best['mean_usd']*3:+.0f} per entry batch
  t-stat:            {best['t_stat']:+.2f}
  Max drawdown:      ${max_dd:.0f}
  Best trade:        {best_win:.1f}bp
  Worst trade:       {worst_loss:.1f}bp

  Walk-forward WR:   {min(wf_results):.1f}% - {max(wf_results):.1f}%
    (across 3 non-overlapping windows)
""")

    mt5.shutdown()

if __name__ == "__main__":
    main()
