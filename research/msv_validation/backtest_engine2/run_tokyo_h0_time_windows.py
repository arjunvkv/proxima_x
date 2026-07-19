"""
Tokyo H0 Consistency Across Multiple Non-overlapping Time Windows.
Tests no-vol-filter strategy on 5 separate windows of ~30 trading days each.
"""
import sys, os, time, numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import deque

HOLD = 3
LOOKBACK = 3
TOP_N = 3
MAX_POS = 3
COSTS_BP = 0.3

np.random.seed(42)


def load_window(end_date, days=40):
    """Load MT5 data for a specific window ending on end_date."""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            return None, []
    except:
        return None, []

    project_root = str(Path(__file__).resolve().parents[3])
    cd_root = os.path.join(project_root, "currency_decomposition")
    sys.path.insert(0, cd_root)
    from config.settings import BASE_CURRENCY_MAP
    all_pairs = list(BASE_CURRENCY_MAP.keys())[:15]

    start = end_date - timedelta(days=days)
    all_data = {}
    for pair in all_pairs:
        rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M5, start, end_date)
        if rates is not None and len(rates) > 0:
            all_data[pair] = rates
    available = [p for p in all_pairs if p in all_data]
    mt5.shutdown()
    return all_data, available


def bt_hour0(all_data, avail_pairs, costs_bp=COSTS_BP, hold=HOLD,
             lookback=LOOKBACK, top_n=TOP_N, max_pos=MAX_POS):
    N = min(len(v) for v in all_data.values() if v is not None)
    positions = {}
    trades = []

    for idx in range(max(lookback, 0), min(N - hold, len(next(iter(all_data.values()))))):
        dt = datetime.fromtimestamp(float(all_data[avail_pairs[0]][idx]["time"]), tz=timezone.utc)
        if dt.hour != 0 or dt.minute != 0:
            continue

        for p in list(positions.keys()):
            if idx >= positions[p]:
                del positions[p]
        if len(positions) >= max_pos:
            continue

        pair_moves = []
        for p in avail_pairs:
            if p in positions:
                continue
            cur = float(all_data[p][idx]["close"])
            bf = float(all_data[p][idx - lookback]["close"])
            ret = (cur / bf - 1) if bf > 0 else 0
            pair_moves.append((p, abs(ret), ret))
        pair_moves.sort(key=lambda x: x[1], reverse=True)

        for p, mag, ret in pair_moves[:top_n]:
            if p in positions or len(positions) >= max_pos:
                break
            if ret > 0:
                continue
            if idx + 1 + hold >= N:
                continue
            spread_cost = costs_bp / 10000
            entry = float(all_data[p][idx + 1]["open"])
            exit_ = float(all_data[p][idx + hold]["close"])
            gross = (exit_ / entry - 1) if entry > 0 else 0
            net_pnl = gross - spread_cost
            trades.append({
                "pnl": net_pnl * 10000, "won": net_pnl > 0,
                "idx": idx, "pair": p, "dt": dt,
                "gross_bp": gross * 10000,
            })
            positions[p] = idx + hold
    return trades


def stats(trades):
    if not trades:
        return {"n": 0, "wr": 0, "mean_bp": 0, "total_bp": 0, "sharpe": 0, "tpd": 0,
                "max_dd": 0, "max_cl": 0, "max_cw": 0}
    pnls = [t["pnl"] for t in trades]
    n = len(pnls)
    wr = sum(1 for x in pnls if x > 0) / n * 100
    mu = float(np.mean(pnls))
    total = float(np.sum(pnls))
    std = float(np.std(pnls))
    sharpe = mu / std if std > 0 else 0

    days = set()
    for t in trades:
        days.add(t["dt"].date())
    tpd = n / max(len(days), 1)

    eq = np.cumsum(pnls)
    running_max = np.maximum.accumulate(eq)
    dd = eq - running_max
    max_dd = float(np.min(dd))

    max_cl = 0
    cur = 0
    for p in pnls:
        if p <= 0:
            cur += 1
            max_cl = max(max_cl, cur)
        else:
            cur = 0

    max_cw = 0
    cur = 0
    for p in pnls:
        if p > 0:
            cur += 1
            max_cw = max(max_cw, cur)
        else:
            cur = 0

    return {"n": n, "wr": wr, "mean_bp": mu, "total_bp": total, "sharpe": sharpe,
            "tpd": tpd, "max_dd": max_dd, "max_cl": max_cl, "max_cw": max_cw}


def print_result(r, label):
    print(f"  {label:<30s} n={r['n']:>4d}  WR={r['wr']:>5.1f}%  Mean={r['mean_bp']:>+7.2f}bp  "
          f"Total={r['total_bp']:>+7.0f}bp  t/d={r['tpd']:.1f}  Sharpe={r['sharpe']:.2f}  "
          f"DD={r['max_dd']:.1f}bp  CL={r['max_cl']}  CW={r['max_cw']}")


def main():
    t0 = time.time()

    # Define 5 non-overlapping ~30-trading-day windows
    # Working backwards from today
    now = datetime.now()
    windows = [
        ("Window 5 (latest)", now - timedelta(days=45), now),
        ("Window 4", now - timedelta(days=95), now - timedelta(days=45)),
        ("Window 3", now - timedelta(days=145), now - timedelta(days=95)),
        ("Window 2", now - timedelta(days=195), now - timedelta(days=145)),
        ("Window 1 (oldest)", now - timedelta(days=245), now - timedelta(days=195)),
    ]

    print("=" * 90)
    print("TOKYO H0 — NO VOL FILTER — MULTI-WINDOW CONSISTENCY TEST")
    print("=" * 90)

    all_results = []

    for label, w_end, _ in windows:
        print(f"\n{'─'*90}")
        print(f"  Loading: {label} (ending ~{w_end.date()})")
        print(f"{'─'*90}")

        all_data, avail_pairs = load_window(w_end, days=45)

        if not all_data or len(avail_pairs) < 5:
            print(f"  SKIP — insufficient data ({len(avail_pairs) if all_data else 0} pairs)")
            continue

        N = min(len(v) for v in all_data.values() if v is not None)
        days_data = N / 288
        print(f"  Data: {len(avail_pairs)} pairs, {N:,d} bars ({days_data:.0f} trading days)")

        trades = bt_hour0(all_data, avail_pairs)
        r = stats(trades)
        all_results.append((label, r, trades))
        print_result(r, label)

        if trades:
            date_range = f"{trades[0]['dt'].date()} to {trades[-1]['dt'].date()}"
            print(f"  Date range: {date_range}")
            by_pair = {}
            for t in trades:
                by_pair.setdefault(t["pair"], []).append(t)
            print(f"  Pairs hit: {len(by_pair)}")
            for pair in sorted(by_pair.keys()):
                v = by_pair[pair]
                pnls = [t["pnl"] for t in v]
                p_wr = sum(1 for x in pnls if x > 0) / len(pnls) * 100
                p_mu = float(np.mean(pnls))
                print(f"    {pair:<8s}: n={len(v):>3d}, WR={p_wr:>5.1f}%, Mean={p_mu:>+7.2f}bp")

    # ── Summary ──
    print(f"\n{'='*90}")
    print("SUMMARY ACROSS ALL TIME WINDOWS")
    print('=' * 90)
    print(f"  {'Window':<30s} {'n':>4s} {'WR':>6s} {'Mean(bp)':>10s} {'Total(bp)':>10s} {'t/d':>5s} {'Sharpe':>7s} {'MaxDD':>7s}")
    print(f"  {'-'*84}")

    wrs = []
    totals = []
    means = []
    for label, r, trades in all_results:
        wrs.append(r["wr"])
        totals.append(r["total_bp"])
        means.append(r["mean_bp"])
        print(f"  {label:<30s} {r['n']:>4d} {r['wr']:>5.1f}% {r['mean_bp']:>+9.2f} {r['total_bp']:>+9.0f} {r['tpd']:>4.1f} {r['sharpe']:>6.2f} {r['max_dd']:>6.1f}")

    if all_results:
        print(f"\n  {'─'*84}")
        print(f"  {'Mean':<30s} {np.mean([r['n'] for _,r,_ in all_results]):>4.0f} {np.mean(wrs):>5.1f}% {np.mean(means):>+9.2f} {np.mean(totals):>+9.0f}")
        print(f"  {'Range WR':<30s} {min(wrs):>5.1f}% – {max(wrs):>5.1f}%")
        print(f"  {'All WR > 60%':<30s} {all(w > 60 for w in wrs)}")
        print(f"  {'All WR > 70%':<30s} {all(w > 70 for w in wrs)}")
        print(f"  {'All Sharpe > 0':<30s} {all(r['sharpe'] > 0 for _,r,_ in all_results)}")
        print(f"  {'All total PnL > 0':<30s} {all(t > 0 for t in totals)}")
        print(f"  {'Min WR':<30s} {min(wrs):.1f}%")
        print(f"  {'Max WR':<30s} {max(wrs):.1f}%")
        print(f"  {'WR spread (max-min)':<30s} {max(wrs)-min(wrs):.1f}%")

        print(f"\n  {'Window':<30s} {'t/d':>5s} {'ConsecLoss':>10s} {'ConsecWin':>10s}")
        print(f"  {'-'*59}")
        for label, r, _ in all_results:
            print(f"  {label:<30s} {r['tpd']:>4.1f} {r['max_cl']:>10d} {r['max_cw']:>10d}")

    print(f"\n  Total elapsed: {time.time() - t0:.1f}s")
    print(f"\n  >>> VERDICT: The no-vol-filter strategy is {'CONSISTENT' if all(w > 60 for w in wrs) else 'MOSTLY CONSISTENT' if all(w > 50 for w in wrs) else 'INCONSISTENT'} across {len(all_results)} time windows")


if __name__ == '__main__':
    main()
