"""CPPF (Cross-Pair Polarity Fade) — ultra-fast M1 bar backtest.

Usage:
    python research/cppf/backtest_cppf.py                    # 1 week sample
    python research/cppf/backtest_cppf.py --days 3           # 3 days
    python research/cppf/backtest_cppf.py --days 30          # 1 month
    python research/cppf/backtest_cppf.py --full             # full 73 days
    python research/cppf/backtest_cppf.py --z 1.0 --hold 1   # param sweep
"""
import argparse, time, sys
import numpy as np
import pandas as pd

DATA_DIR = "research/phase_dislocation/dukascopy_data"

# 6 USD-major pairs for net polarity
USD_PAIRS = ["eurusd", "gbpusd", "audusd", "nzdusd", "usdcad", "usdchf"]

# USD is QUOTE for these pairs (price moves OPPOSITE to USD strength)
USD_QUOTE_PAIRS = {"eurusd", "gbpusd", "audusd", "nzdusd"}
# USD is BASE for these pairs (price moves SAME as USD strength)
USD_BASE_PAIRS = {"usdcad", "usdchf"}

# Spread cost in raw price per pair (conservative estimates)
SPREAD_COST = {
    "eurusd": 0.000015,   # 0.15 pips
    "gbpusd": 0.000018,   # 0.18 pips
    "audusd": 0.000015,   # 0.15 pips
    "nzdusd": 0.000020,   # 0.20 pips
    "usdcad": 0.000020,   # 0.20 pips (×10000 = 2.0 MP)
    "usdchf": 0.000018,   # 0.18 pips
}


def load_pairs(pairs, days=None):
    """Load M1 bar data for multiple pairs. Returns dict of {pair: DataFrame}."""
    data = {}
    for p in pairs:
        df = pd.read_parquet(f"{DATA_DIR}/{p}.parquet")
        df = df.set_index("timestamp")
        data[p] = df
    return data


def compute_z_scores(df, window=50):
    """Compute rolling z-scores of log returns for a single pair DataFrame."""
    close = df["close"].values.astype(np.float64)
    ret = np.diff(np.log(np.maximum(close, 1e-12)))
    ret = np.insert(ret, 0, 0.0)  # first bar has no prior

    z = np.full(len(ret), np.nan)
    for i in range(window, len(ret)):
        chunk = ret[i - window + 1 : i + 1]
        m = np.mean(chunk)
        s = np.std(chunk, ddof=1)
        z[i] = (ret[i] - m) / s if s > 1e-12 else 0.0
    return ret, z


def backtest_cppf(
    data,
    z_thresh=1.0,
    hold_bars=1,
    window=50,
    days=None,
    use_spread=True,
):
    """Run CPPF (Cross-Pair Polarity Fade) backtest.

    Args:
        data: dict of {pair: DataFrame} with index=timestamp, columns=[open,high,low,close]
        z_thresh: minimum |z| to enter
        hold_bars: bars to hold position (1 = exit next bar close)
        window: rolling window for z-score
        days: if set, only use first N days of data
        use_spread: deduct spread cost from each trade

    Returns:
        list of trade dicts
    """
    pairs = list(data.keys())

    # Align all pairs by index
    common_idx = None
    for p in pairs:
        if common_idx is None:
            common_idx = set(data[p].index)
        else:
            common_idx = common_idx.intersection(set(data[p].index))
    common_idx = sorted(common_idx)

    if days:
        cutoff = common_idx[0] + pd.Timedelta(days=days)
        common_idx = [t for t in common_idx if t < cutoff]

    if len(common_idx) < window + 10:
        print(f"  Not enough bars: {len(common_idx)} (need {window + 10})")
        return []

    # Build aligned close price matrix: shape (N_bars, N_pairs)
    n_bars = len(common_idx)
    n_pairs = len(pairs)
    close_mat = np.zeros((n_bars, n_pairs), dtype=np.float64)

    for j, p in enumerate(pairs):
        for i, t in enumerate(common_idx):
            close_mat[i, j] = data[p].loc[t, "close"]

    # Compute log returns and z-scores per pair
    ret_mat = np.zeros((n_bars, n_pairs), dtype=np.float64)
    z_mat = np.full((n_bars, n_pairs), np.nan)

    for j in range(n_pairs):
        close_vec = close_mat[:, j]
        r = np.diff(np.log(np.maximum(close_vec, 1e-12)))
        r = np.insert(r, 0, 0.0)
        ret_mat[:, j] = r

        for i in range(window, n_bars):
            chunk = r[i - window + 1 : i + 1]
            m = np.mean(chunk)
            s = np.std(chunk, ddof=1)
            z_mat[i, j] = (r[i] - m) / s if s > 1e-12 else 0.0

    # Run strategy
    trades = []

    for i in range(window, n_bars):
        t = common_idx[i]
        if i + hold_bars >= n_bars:
            break

        z_row = z_mat[i, :]
        if np.any(np.isnan(z_row)):
            continue

        # Convert each pair's z to USD polarity vote
        usd_votes = np.zeros(n_pairs)
        for j, p in enumerate(pairs):
            if np.isnan(z_row[j]):
                usd_votes[j] = 0
            elif p in USD_QUOTE_PAIRS:
                # quote pair: z>0 means EUR up / USD down → vote = -1 (USD weak)
                usd_votes[j] = -1.0 if z_row[j] > 0 else (1.0 if z_row[j] < 0 else 0.0)
            else:
                # base pair: z>0 means USD up → vote = +1 (USD strong)
                usd_votes[j] = 1.0 if z_row[j] > 0 else (-1.0 if z_row[j] < 0 else 0.0)

        # Net consensus: majority vote
        mean_vote = np.mean(usd_votes)
        net_vote = 1 if mean_vote > 0 else (-1 if mean_vote < 0 else 0)
        if net_vote == 0:
            continue

        for j, p in enumerate(pairs):
            z_val = z_row[j]
            if np.isnan(z_val) or abs(z_val) < z_thresh:
                continue

            # This pair's vote disagrees with consensus → orphaned
            if usd_votes[j] != net_vote:
                entry_close = close_mat[i, j]
                exit_close = close_mat[i + hold_bars, j]

                direction = -1.0 if z_val > 0 else 1.0  # fade the z-score
                raw_pnl = direction * (exit_close - entry_close)
                cost = SPREAD_COST.get(p, 0) if use_spread else 0
                net_pnl = raw_pnl - cost

                trades.append({
                    "time": t,
                    "pair": p,
                    "direction": direction,
                    "entry": entry_close,
                    "exit": exit_close,
                    "hold_bars": hold_bars,
                    "z_score": z_val,
                    "raw_pnl": raw_pnl,
                    "cost": cost,
                    "net_pnl": net_pnl,
                    "usd_vote": int(usd_votes[j]),
                    "net_vote": net_vote,
                    "mean_vote": mean_vote,
                })

    return trades


def report(trades, elapsed=None):
    """Print summary report."""
    if not trades:
        print("  No trades generated.")
        return

    df = pd.DataFrame(trades)
    n_total = len(df)

    print(f"\n{'='*65}")
    print(f"CPPF RESULTS  ({' | '.join(USD_PAIRS)})")
    print(f"{'='*65}")
    print(f"  Total trades: {n_total}")
    print(f"  Trades/day:   {n_total / 73:.1f}")  # full range is ~73 days

    per_pair = df.groupby("pair")
    grand_wins = 0
    grand_pnl = 0.0
    for p in USD_PAIRS:
        if p not in per_pair.groups:
            continue
        sub = per_pair.get_group(p)
        n = len(sub)
        wins = int((sub["net_pnl"] > 0).sum())
        wr = wins / n
        total_pnl = sub["net_pnl"].sum()
        avg_pnl = sub["net_pnl"].mean()
        grand_wins += wins
        grand_pnl += total_pnl

        # Convert to pips
        if p in ("usdcad",):
            pip_mult = 10000  # 1 pip = 0.0001
        elif p in ("usdchf",):
            pip_mult = 10000
        else:
            pip_mult = 10000

        print(f"\n  {p.upper()}:")
        print(f"    Trades: {n:>5d}  ({n / 73:.1f}/day)")
        print(f"    WR:     {wins:>4d}/{n:<4d} = {wr:>5.1%}")
        print(f"    Net PnL: {total_pnl:>+9.6f}  ({total_pnl*pip_mult:>+8.2f} pips)")
        print(f"    Avg/PnL: {avg_pnl:>+9.6f}  ({avg_pnl*pip_mult:>+8.2f} pips)")
        print(f"    Avg |z|: {sub['z_score'].abs().mean():.2f}")

    overall_wr = grand_wins / n_total
    print(f"\n  {'─'*50}")
    print(f"  OVERALL:")
    print(f"    Win rate:  {grand_wins}/{n_total} = {overall_wr:.1%}")
    print(f"    Net PnL:   {grand_pnl:.6f} raw")
    print(f"    Per trade: {grand_pnl/n_total:.6f} raw")
    if elapsed:
        print(f"    Runtime:   {elapsed:.2f}s")

    # By z-score bucket
    df["z_bucket"] = pd.cut(df["z_score"].abs(), bins=[0, 0.5, 1.0, 1.5, 2.0, 3.0, 10.0])
    print(f"\n  {'─'*50}")
    print(f"  BY |z| BUCKET:")
    for bucket, sub in df.groupby("z_bucket", observed=True):
        n = len(sub)
        wr = (sub["net_pnl"] > 0).mean()
        avg = sub["net_pnl"].mean()
        print(f"    |z| in {bucket}: n={n:>4d}  WR={wr:>5.1%}  avg={avg:>+9.6f}")

    return df


def main():
    parser = argparse.ArgumentParser(description="CPPF Backtest")
    parser.add_argument("--z", type=float, default=1.0, help="z-score threshold")
    parser.add_argument("--hold", type=int, default=1, help="hold bars")
    parser.add_argument("--days", type=int, default=None, help="days of data to use")
    parser.add_argument("--full", action="store_true", help="use full 73 days")
    parser.add_argument("--no-spread", action="store_true", help="ignore spread cost")
    parser.add_argument("--scan", action="store_true", help="run parameter scan")
    args = parser.parse_args()

    t0 = time.time()

    print(f"Loading {len(USD_PAIRS)} pairs from Dukascopy...")
    data = load_pairs(USD_PAIRS)

    if args.scan:
        print(f"\n{'='*65}")
        print("PARAMETER SCAN")
        print(f"{'='*65}")
        best = {"pnl": -999, "cfg": None, "wr": 0, "n": 0}
        for z in [0.5, 1.0, 1.5, 2.0]:
            for h in [1, 2, 3, 5]:
                trades = backtest_cppf(
                    data, z_thresh=z, hold_bars=h,
                    days=args.days if not args.full else None,
                    use_spread=not args.no_spread,
                )
                if not trades:
                    continue
                df = pd.DataFrame(trades)
                wr = (df["net_pnl"] > 0).mean()
                tp = df["net_pnl"].sum()
                n = len(df)
                print(f"  z={z:.1f}  hold={h:>2d}  ->  n={n:>5d}  WR={wr:>5.1%}  PnL={tp:>+9.6f}")
                if tp > best["pnl"]:
                    best = {"pnl": tp, "cfg": (z, h), "wr": wr, "n": n}
        print(f"\n  BEST: z={best['cfg'][0]} hold={best['cfg'][1]}  n={best['n']}  WR={best['wr']:.1%}  PnL={best['pnl']:.6f}")
        t1 = time.time()
        print(f"  Runtime: {t1-t0:.2f}s")
        return

    days = 7 if args.days is None else args.days
    if args.full:
        days = None

    trades = backtest_cppf(
        data, z_thresh=args.z, hold_bars=args.hold,
        days=days, use_spread=not args.no_spread,
    )
    elapsed = time.time() - t0
    report(trades, elapsed)


if __name__ == "__main__":
    main()
