"""CPPF v2: Consensus Z Deviation — trade pairs whose z-score deviates
significantly from the cross-pair median z-score.

Thesis: When one pair moves far more (in z-score terms) than the peer group,
that excess is structural noise that reverts. The consensus z represents the
"true" market move; any pair deviating from it has local/retail flow.

Usage:
    python research/cppf/backtest_cppf_v2.py
    python research/cppf/backtest_cppf_v2.py --scan
    python research/cppf/backtest_cppf_v2.py --days 30 --z 1.5 --hold 5
"""
import argparse, time
import numpy as np
import pandas as pd

DATA_DIR = "research/phase_dislocation/dukascopy_data"

# All 8 pairs available
ALL_PAIRS = ["eurusd","gbpusd","audusd","nzdusd","usdcad","usdchf","eurjpy","gbpjpy"]

# Spread cost per pair (raw price)
SPREAD = {
    "eurusd": 0.000015, "gbpusd": 0.000018, "audusd": 0.000015, "nzdusd": 0.000020,
    "usdcad": 0.000020, "usdchf": 0.000018, "eurjpy": 0.0050, "gbpjpy": 0.0060,
}


def load_data(pairs):
    """Load Dukascopy data for all pairs, return aligned close matrix + timestamps."""
    data = {}
    for p in pairs:
        df = pd.read_parquet(f"{DATA_DIR}/{p}.parquet").set_index("timestamp")
        data[p] = df
    common = sorted(set.intersection(*[set(data[p].index) for p in pairs]))
    n = len(common)
    n_pairs = len(pairs)
    close = np.zeros((n, n_pairs))
    for j, p in enumerate(pairs):
        for i, t in enumerate(common):
            close[i, j] = data[p].loc[t, "close"]
    return close, common, pairs


def compute_z(close, window=50):
    """Precompute z-scores for all pairs. Returns (n_bars, n_pairs) matrix."""
    n, n_pairs = close.shape
    z_mat = np.full((n, n_pairs), np.nan)
    for j in range(n_pairs):
        r = np.diff(np.log(np.maximum(close[:, j], 1e-12)))
        r = np.insert(r, 0, 0.0)
        for i in range(window, n):
            c = r[i - window + 1 : i + 1]
            m, s = np.mean(c), np.std(c, ddof=1)
            z_mat[i, j] = (r[i] - m) / s if s > 1e-12 else 0.0
    return z_mat, r


def backtest(close, z_mat, common, pairs,
             z_thresh=1.5, hold_bars=5, window=50, days=None, use_spread=True):
    """Run consensus deviation backtest on pre-computed data."""
    n = len(common)
    trades = []
    for i in range(window, n):
        if i + hold_bars >= n:
            break
        z_row = z_mat[i, :]
        if np.any(np.isnan(z_row)):
            continue

        consensus_z = np.median(z_row)
        mad = np.median(np.abs(z_row - consensus_z)) + 1e-12
        z_dev = (z_row - consensus_z) / mad

        for j, p in enumerate(pairs):
            if np.isnan(z_row[j]) or abs(z_dev[j]) < z_thresh:
                continue
            entry = close[i, j]
            exit_ = close[i + hold_bars, j]
            direction = -1.0 if z_dev[j] > 0 else 1.0
            raw = direction * (exit_ - entry)
            cost = SPREAD.get(p, 0) if use_spread else 0
            trades.append({
                "time": common[i], "pair": p,
                "dir": direction, "entry": entry, "exit": exit_,
                "z": z_row[j], "consensus_z": consensus_z,
                "z_dev": z_dev[j], "raw_pnl": raw,
                "net_pnl": raw - cost,
            })
    return trades


def show(trades, elapsed=None):
    if not trades:
        print("  No trades.")
        return
    df = pd.DataFrame(trades)
    print(f"\n{'='*65}")
    print(f"CONSENSUS Z DEVIATION  ({', '.join(set(df['pair']))})")
    print(f"{'='*65}")
    print(f"  Total trades: {len(df)}")
    print(f"  Trades/day:   {len(df)/73:.1f}")

    grand_w = grand_p = 0
    for p in sorted(df["pair"].unique()):
        sub = df[df["pair"] == p]
        n = len(sub)
        w = int((sub["net_pnl"] > 0).sum())
        gw = sub["net_pnl"].sum()
        grand_w += w
        grand_p += gw
        pip = 10000 if p in ("eurusd","gbpusd","audusd","nzdusd","usdcad","usdchf") else 100
        print(f"\n  {p.upper():7s}: n={n:>4d} ({n/73:.1f}/d)  WR={w/n:>5.1%}  "
              f"PnL={gw:>+10.6f} ({gw*pip:>+8.2f}p)  avg={sub['net_pnl'].mean():>+9.6f}")

    print(f"\n  {'─'*50}")
    print(f"  OVERALL: WR={grand_w/len(df):.1%}  PnL={grand_p:.6f}")
    print(f"  Avg trade: {grand_p/len(df):.6f}")
    if elapsed:
        print(f"  Runtime: {elapsed:.2f}s")

    print(f"\n  BY |z_dev| BUCKET:")
    df["zb"] = pd.cut(df["z_dev"].abs(), bins=[0,1,2,3,5,10,50])
    for b, sub in df.groupby("zb", observed=True):
        print(f"    |zd| in {b}: n={len(sub):>4d}  WR={(sub['net_pnl']>0).mean():.1%}  "
              f"avg={sub['net_pnl'].mean():>+9.6f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", type=float, default=1.5)
    ap.add_argument("--hold", type=int, default=5)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--no-spread", action="store_true")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--pairs", nargs="+", default=None)
    args = ap.parse_args()

    t0 = time.time()
    pairs = args.pairs or ALL_PAIRS[:6]

    print(f"Loading {len(pairs)} pairs...")
    close, common, pair_list = load_data(pairs)
    if args.days and not args.full:
        cutoff = common[0] + pd.Timedelta(days=args.days)
        keep = [i for i, t in enumerate(common) if t < cutoff]
        close = close[keep, :]
        common = [common[i] for i in keep]
    print(f"  {len(common)} bars loaded.")

    print(f"Computing z-scores (window=50)...")
    z_mat, _ = compute_z(close)
    print(f"  z-scores computed.")

    n_bars = len(common)

    if args.scan:
        print(f"\nScanning parameter grid...")
        print(f"{'='*65}")
        best = {"pnl": -999, "cfg": None, "wr": 0, "n": 0}
        for z in [0.5, 1.0, 1.5, 2.0, 3.0]:
            for h in [1, 2, 3, 5, 10, 20]:
                tr = backtest(close, z_mat, common, pair_list, z, h, use_spread=not args.no_spread)
                if not tr:
                    continue
                d = pd.DataFrame(tr)
                wr = (d["net_pnl"] > 0).mean()
                tp = d["net_pnl"].sum()
                wct = (d["net_pnl"] > 0).sum()
                print(f"  z={z:.1f}  hold={h:>2d}  -> n={len(tr):>5d}  WR={wr:>5.1%}  "
                      f"Wins={wct}  PnL={tp:>+9.6f}")
                if tp > best["pnl"]:
                    best = {"pnl": tp, "cfg": (z, h), "wr": wr, "n": len(tr)}
        print(f"\n  BEST: z={best['cfg'][0]} hold={best['cfg'][1]}  n={best['n']}  "
              f"WR={best['wr']:.1%}  PnL={best['pnl']:.6f}")
        print(f"  Runtime: {time.time()-t0:.2f}s")

    else:
        tr = backtest(close, z_mat, common, pair_list, args.z, args.hold,
                      use_spread=not args.no_spread)
        show(tr, time.time() - t0)
