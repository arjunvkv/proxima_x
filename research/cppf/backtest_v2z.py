"""V2 signal (fade every M1 bar z-score) with proper OHLC-based trailing stop.

Uses full OHLC bar data for correct trailing stop simulation:
  - Entry at bar close (or next bar open)
  - Trailing stop checks bar HIGH/LOW for stop hits
  - Uses correct ATR formula: mean of bar ranges over window

Usage:
    python research/cppf/backtest_v2z.py                  # default run
    python research/cppf/backtest_v2z.py --scan            # sweep z thresholds
    python research/cppf/backtest_v2z.py --full --z 2.0    # full data
"""
import argparse, time
import numpy as np
import pandas as pd

DATA_DIR = "research/phase_dislocation/dukascopy_data"
PAIRS = ["eurusd", "eurjpy", "gbpjpy"]

SPREAD = {"eurusd": 0.000015, "eurjpy": 0.0050, "gbpjpy": 0.0060}

# V2 config
STOP_A = 0.15
TRIG_A = 0.20
GAP_A = 0.10
MAX_HOLD = 54
Z_WINDOW = 50
ATR_WINDOW = 20


def load_one(pair):
    df = pd.read_parquet(f"{DATA_DIR}/{pair}.parquet").set_index("timestamp")
    return df, df.index.tolist()


def backtest(pair, z_thresh=0.0, days=None):
    """V2 backtest with proper OHLC trailing stop."""
    df, times = load_one(pair)
    if days:
        cutoff = times[0] + pd.Timedelta(days=days)
        df = df[df.index < cutoff]

    n = len(df)
    if n < Z_WINDOW + ATR_WINDOW + 10:
        return []

    o = df["open"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    c = df["close"].values.astype(np.float64)

    # Returns and z-scores
    r = np.diff(np.log(np.maximum(c, 1e-12)))
    r = np.insert(r, 0, 0.0)
    z = np.full(n, np.nan)
    for i in range(Z_WINDOW, n):
        chunk = r[i - Z_WINDOW + 1 : i + 1]
        m, s = np.mean(chunk), np.std(chunk, ddof=1)
        z[i] = (r[i] - m) / s if s > 1e-12 else 0.0

    # ATR: mean of bar ranges over window
    atr = np.full(n, np.nan)
    bar_range = h - l
    for i in range(ATR_WINDOW, n):
        atr[i] = np.mean(bar_range[i - ATR_WINDOW + 1 : i + 1])

    trades = []
    for i in range(max(Z_WINDOW, ATR_WINDOW), n - 1):
        if not np.isfinite(z[i]) or not np.isfinite(atr[i]):
            continue
        if z_thresh > 0 and abs(z[i]) < z_thresh:
            continue

        direction = -1.0 if z[i] > 0 else 1.0
        entry = c[i]  # entry at bar close
        stop_dist = STOP_A * atr[i]
        trail_trig = TRIG_A * atr[i]
        trail_gap = GAP_A * atr[i]

        best = entry
        stop = entry - stop_dist if direction > 0 else entry + stop_dist
        exit_price = entry
        exit_reason = ""
        exited = False

        exit_k = min(n, i + MAX_HOLD + 1)
        for k in range(i + 1, exit_k):
            bar_high = h[k]
            bar_low = l[k]
            bar_close = c[k]

            if direction > 0:  # long
                # Check if stop was hit
                if bar_low <= stop:
                    exit_price = stop
                    exit_reason = "stop"
                    exited = True
                    break
                # Update best
                if bar_high > best:
                    best = bar_high
                    if best - entry > trail_trig:
                        stop = best - trail_gap
            else:  # short
                if bar_high >= stop:
                    exit_price = stop
                    exit_reason = "stop"
                    exited = True
                    break
                if bar_low < best:
                    best = bar_low
                    if entry - best > trail_trig:
                        stop = best + trail_gap

            # Also check close-based stop for expiry
            if not exited and k == exit_k - 1:
                exit_price = bar_close
                exit_reason = "expiry"

        if not exited:
            exit_price = c[min(n - 1, i + MAX_HOLD)]
            exit_reason = "expiry"

        raw = direction * (exit_price - entry)
        net = raw - SPREAD.get(pair, 0)

        trades.append({
            "time": df.index[i], "pair": pair, "dir": int(direction),
            "z": z[i], "atr": atr[i],
            "entry": entry, "exit": exit_price,
            "raw_pnl": raw, "net_pnl": net, "exit_reason": exit_reason,
        })

    return trades


def report(trades, elapsed=None):
    if not trades:
        print("  No trades.")
        return
    df = pd.DataFrame(trades)
    n_days = (df["time"].max() - df["time"].min()).total_seconds() / 86400 + 1
    tpd = len(df) / n_days

    print(f"\n{'='*60}")
    print("V2+z WITH PROPER OHLC TRAILING STOP")
    print(f"{'='*60}")
    print(f"  Total trades: {len(df):,}")
    print(f"  Trades/day:   {tpd:.0f}")
    print(f"  Data window:  {df['time'].min()}  ->  {df['time'].max()} ({n_days:.0f} days)")

    total_w = total_p = 0
    for p in sorted(df["pair"].unique()):
        sub = df[df["pair"] == p]
        n = len(sub)
        w = int((sub["net_pnl"] > 0).sum())
        tp = sub["net_pnl"].sum()
        total_w += w
        total_p += tp

        win_sub = sub[sub["net_pnl"] > 0]
        lose_sub = sub[sub["net_pnl"] <= 0]
        avg_w = win_sub["net_pnl"].mean() if len(win_sub) else 0
        avg_l = lose_sub["net_pnl"].mean() if len(lose_sub) else 0
        payoff = abs(avg_w / avg_l) if avg_l != 0 else float("inf")

        pip = 10000 if p == "eurusd" else 100
        print(f"\n  {p.upper():7s}: n={n:>6d} ({n/n_days:.0f}/d)  "
              f"WR={w:>4d}/{n:<4d} = {w/n:>5.1%}  "
              f"PnL={tp:>+11.6f} ({tp*pip:>+8.1f}p)  "
              f"avg={sub['net_pnl'].mean():>+9.6f}")
        print(f"         avg_win={avg_w:>+9.6f}  avg_loss={avg_l:>+9.6f}  payoff={payoff:.2f}")
        print(f"         stops={int((sub['exit_reason']=='stop').sum()):>5d}  "
              f"expiry={int((sub['exit_reason']=='expiry').sum()):>5d}")

        # By z bucket
        print(f"         By |z|:")
        sub2 = sub.copy()
        sub2["zb"] = pd.cut(sub2["z"].abs(), bins=[0, 0.5, 1, 1.5, 2, 3, 10])
        for b, g in sub2.groupby("zb", observed=True):
            gw = (g["net_pnl"] > 0).sum()
            print(f"           {str(b):>12s}: n={len(g):>4d}  WR={gw/len(g):.1%}  "
                  f"avg={g['net_pnl'].mean():>+9.6f}")

    wr_all = total_w / len(df)
    print(f"\n  {'─'*50}")
    print(f"  OVERALL: {total_w}/{len(df)} = {wr_all:.1%}  PnL={total_p:.6f}")
    if elapsed:
        print(f"  Runtime: {elapsed:.2f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", type=float, default=0.0)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--scan", action="store_true")
    args = ap.parse_args()

    t0 = time.time()

    if args.scan:
        days = args.days if not args.full else None
        print(f"V2+z SCAN (OHLC trailing) on {PAIRS}")
        best = {"pnl": -999, "cfg": None, "wr": 0, "n": 0}
        for z in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            all_trades = []
            for p in PAIRS:
                try:
                    tr = backtest(p, z, days=days)
                    all_trades.extend(tr)
                except Exception as e:
                    pass
            if not all_trades:
                continue
            df = pd.DataFrame(all_trades)
            n_days = (df["time"].max() - df["time"].min()).total_seconds() / 86400 + 1
            wr = (df["net_pnl"] > 0).mean()
            tp = df["net_pnl"].sum()
            n = len(df)
            print(f"  z>={z:.1f}: n={n:>6d}  {n/n_days:.0f}/d  WR={wr:>5.1%}  PnL={tp:>+9.6f}")
            if tp > best["pnl"]:
                best = {"pnl": tp, "cfg": z, "wr": wr, "n": n}
        print(f"\n  BEST: z>={best['cfg']}  n={best['n']}  WR={best['wr']:.1%}  PnL={best['pnl']:.6f}")
        print(f"  Runtime: {time.time()-t0:.2f}s")
    else:
        all_trades = []
        for p in PAIRS:
            tr = backtest(p, args.z, days=args.days if not args.full else None)
            all_trades.extend(tr)
        report(all_trades, time.time() - t0)
