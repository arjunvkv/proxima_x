"""V2+z on M1 bar data (Dukascopy) with correct OHLC trailing stop.

Matches the original V2 research methodology:
  1. Resample to M1 OHLC bars
  2. Compute z-score (50-bar rolling) + ATR (20-bar rolling)
  3. Entry: every bar close, direction = -sign(z_score)
  4. Exit: trailing stop checked on subsequent bar OHLC

Usage:
    python research/cppf/backtest_m1.py EURUSD
    python research/cppf/backtest_m1.py EURUSD --z 2.0
    python research/cppf/backtest_m1.py EURUSD --scan
"""
import argparse, time
import numpy as np
import pandas as pd

DATA_DIR = "research/phase_dislocation/dukascopy_data"

# V2 config
STOP_A = 0.15       # initial stop = STOP_A * ATR
TRIG_A = 0.20       # trail trigger after TRIG_A * ATR profit
GAP_A = 0.10        # trail gap = GAP_A * ATR behind best
MAX_HOLD = 54       # max bars to hold
Z_WINDOW = 50
ATR_WINDOW = 20

# Spread cost per pair (in raw price)
SPREAD_COST_RAW = {
    "eurusd": 0.000015,  # 0.15 pips
    "eurjpy": 0.0050,    # 50 MP
    "gbpjpy": 0.0060,    # 60 MP
}


def get_pip_mult(pair):
    return 10000 if pair == "eurusd" else 100


def load_bars(pair, days=None):
    """Load M1 OHLC data for a pair."""
    df = pd.read_parquet(f"{DATA_DIR}/{pair}.parquet").set_index("timestamp")
    if days:
        cutoff = df.index[0] + pd.Timedelta(days=days)
        df = df[df.index < cutoff]
    return df


def compute_signal(df, pair):
    """Compute z-scores and ATR from M1 bars. Returns modified df."""
    df = df.copy()
    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    n = len(df)

    # z-score (50-bar rolling of returns)
    r = np.diff(np.log(np.maximum(c, 1e-12)))
    r = np.insert(r, 0, 0.0)
    z = np.full(n, np.nan)
    for i in range(Z_WINDOW, n):
        chunk = r[i - Z_WINDOW + 1 : i + 1]
        m, s = np.mean(chunk), np.std(chunk, ddof=1)
        z[i] = (r[i] - m) / s if s > 1e-12 else 0.0

    # ATR (20-bar mean of bar ranges)
    atr = np.full(n, np.nan)
    br = h - l
    for i in range(ATR_WINDOW, n):
        atr[i] = np.mean(br[i - ATR_WINDOW + 1 : i + 1])

    df["z"] = z
    df["atr"] = atr
    return df


def run_backtest(df, pair, z_thresh=0.0, cost_raw=None):
    """Run V2 backtest with z-threshold filter, returns list of trade dicts."""
    if cost_raw is None:
        cost_raw = SPREAD_COST_RAW.get(pair, 0)

    n = len(df)
    trades = []

    for i in range(max(Z_WINDOW, ATR_WINDOW), n - 1):
        row = df.iloc[i]
        z_val = row["z"]
        atr_val = row["atr"]

        if not np.isfinite(z_val) or not np.isfinite(atr_val):
            continue
        if z_thresh > 0 and abs(z_val) < z_thresh:
            continue

        direction = -1.0 if z_val > 0 else 1.0
        entry = row["close"]
        stop_dist = STOP_A * atr_val
        trail_trig = TRIG_A * atr_val
        trail_gap = GAP_A * atr_val

        best = entry
        sl = entry - stop_dist if direction > 0 else entry + stop_dist
        exit_price = entry
        exit_reason = ""
        exited = False

        max_k = min(n, i + MAX_HOLD + 1)
        for k in range(i + 1, max_k):
            bar = df.iloc[k]
            bh = bar["high"]
            bl = bar["low"]
            bc = bar["close"]

            if direction > 0:  # long
                if bl <= sl:
                    exit_price = sl
                    exit_reason = "stop"
                    exited = True
                    break
                if bh > best:
                    best = bh
                    if best - entry > trail_trig:
                        sl = best - trail_gap
            else:  # short
                if bh >= sl:
                    exit_price = sl
                    exit_reason = "stop"
                    exited = True
                    break
                if bl < best:
                    best = bl
                    if entry - best > trail_trig:
                        sl = best + trail_gap

        if not exited:
            exit_price = df.iloc[min(n - 1, i + MAX_HOLD)]["close"]
            exit_reason = "expiry"

        raw_pnl = direction * (exit_price - entry)
        net_pnl = raw_pnl - cost_raw

        trades.append({
            "time": df.index[i],
            "pair": pair,
            "dir": int(direction),
            "z": z_val,
            "atr": atr_val,
            "entry": entry,
            "exit": exit_price,
            "raw_pnl": raw_pnl,
            "net_pnl": net_pnl,
            "exit_reason": exit_reason,
        })

    return trades


def report(trades):
    if not trades:
        print("  No trades.")
        return

    df = pd.DataFrame(trades)
    n_days = (df["time"].max() - df["time"].min()).total_seconds() / 86400 + 1
    tpd = len(df) / n_days

    pip_mult = get_pip_mult(trades[0]["pair"])
    pair = trades[0]["pair"]

    pnls = df["net_pnl"].values
    wins = (pnls > 0).sum()
    total = len(pnls)
    wr = wins / total
    gross = pnls.sum()

    win_p = pnls[pnls > 0]
    lose_p = pnls[pnls <= 0]
    avg_w = win_p.mean() if len(win_p) else 0
    avg_l = lose_p.mean() if len(lose_p) else 0
    pay = abs(avg_w / avg_l) if avg_l != 0 else float("inf")

    raw = df["raw_pnl"].values
    raw_gross = raw.sum()
    raw_wins = (raw > 0).sum()

    stops = (df["exit_reason"] == "stop").sum()
    expiries = (df["exit_reason"] == "expiry").sum()

    print(f"\n  {pair.upper():7s}: n={total:>5d}  {tpd:.0f}/d  "
          f"WR={wins:>4d}/{total:<4d} = {wr:>5.1%}")
    print(f"         Net PnL: {gross:>+10.6f} ({gross*pip_mult:>+8.1f}p)  "
          f"Per trade: {gross/total:>+9.6f}")
    print(f"         Raw PnL: {raw_gross:>+10.6f} ({raw_gross*pip_mult:>+8.1f}p)  "
          f"Raw WR: {raw_wins/total:.1%}")
    print(f"         Avg win: {avg_w:>+9.6f}  Avg loss: {avg_l:>+9.6f}  Payoff: {pay:.2f}")
    print(f"         Stops: {stops}  Expiry: {expiries}")

    print(f"         |z| buckets:")
    zb = pd.cut(df["z"].abs(), bins=[0, 0.5, 1, 1.5, 2, 3, 10])
    for b, g in df.groupby(zb, observed=True):
        gw = (g["net_pnl"] > 0).sum()
        print(f"           {str(b):>12s}: n={len(g):>4d}  WR={gw/len(g):.1%}  "
              f"avg={g['net_pnl'].mean():>+9.6f}")

    return wr, gross


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pair", nargs="?", default="EURUSD")
    ap.add_argument("--z", type=float, default=0.0)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--scan", action="store_true")
    args = ap.parse_args()

    pair = args.pair.lower()
    pkey = pair  # "eurusd"
    cost = SPREAD_COST_RAW.get(pkey, 0)

    t0 = time.time()

    if args.scan:
        days = args.days if not args.full else None
        name = pair.upper()
        print(f"Scanning {name} (cost={cost})...")
        df = load_bars(pkey, days=days)
        df = compute_signal(df, pkey)
        n_days = len(df) / 1440
        print(f"Bars: {len(df)}  Days: {n_days:.0f}")

        best = {"pnl": -999, "z": 0, "wr": 0, "n": 0}
        for z in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            trades = run_backtest(df, pkey, z_thresh=z, cost_raw=cost)
            if not trades:
                continue
            df2 = pd.DataFrame(trades)
            wr = (df2["net_pnl"] > 0).mean()
            tp = df2["net_pnl"].sum()
            n = len(df2)
            print(f"  z>={z:.1f}: n={n:>5d} ({n/n_days:.0f}/d)  "
                  f"WR={wr:>5.1%}  PnL={tp:>+10.6f}")
            if tp > best["pnl"]:
                best = {"pnl": tp, "z": z, "wr": wr, "n": n}
        print(f"\n  BEST: z>={best['z']}  n={best['n']}  WR={best['wr']:.1%}  PnL={best['pnl']:.6f}")
        print(f"  Runtime: {time.time()-t0:.2f}s")
    else:
        days = args.days if not args.full else None
        df = load_bars(pkey, days=days)
        print(f"Loaded {len(df)} bars for {pair.upper()}")
        df = compute_signal(df, pkey)
        trades = run_backtest(df, pkey, z_thresh=args.z, cost_raw=cost)
        report(trades)
        print(f"  Runtime: {time.time()-t0:.2f}s")
