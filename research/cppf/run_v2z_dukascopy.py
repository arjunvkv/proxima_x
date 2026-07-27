"""V2+z on Dukascopy M1 data — test z-threshold filtering for trade reduction.

Uses exact V2 logic from run_hfdf_dukascopy.py + optional z-threshold filter.
Data: CSV files from dark_research/dukascopy_data/

Usage:
    python research/cppf/run_v2z_dukascopy.py EURUSD
    python research/cppf/run_v2z_dukascopy.py EURUSD --z 2.0
    python research/cppf/run_v2z_dukascopy.py EURUSD --scan
    python research/cppf/run_v2z_dukascopy.py EURUSD --full
"""
import argparse, time
import numpy as np
import pandas as pd
from pathlib import Path

DUKA_DIR = Path("research/dark_research/dukascopy_data")
COST = {"EURUSD": 0.15, "EURJPY": 50, "GBPJPY": 60}


def load_dukascopy(pair):
    files = sorted(DUKA_DIR.glob(f"{pair.lower()}-m1-bid-*.csv"))
    dfs = []
    for f in files:
        df = pd.read_csv(f, usecols=["timestamp","open","high","low","close"])
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp")
    df = df.set_index("timestamp").astype(float)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, unit="ms")
    df = df * 10000
    return df[["open","high","low","close"]]


def hfdf_m1(b, cost, z_thresh=0.0):
    """V2 backtest with optional z-threshold filter."""
    ret = b["close"].diff()
    z = (ret - ret.shift(1).rolling(50).mean()) / ret.shift(1).rolling(50).std().clip(1e-8)
    atr = (b["high"] - b["low"]).shift(1).rolling(20).mean().clip(1e-8)

    valid = z.notna() & atr.notna()
    if z_thresh > 0:
        valid &= z.abs() >= z_thresh
    idxs = np.where(valid)[0]
    if len(idxs) < 5:
        return np.array([])

    stop_a, trig_a, gap_a = 0.15, 0.20, 0.10
    max_bars = 54
    pnls = []
    closes = b["close"].values
    highs = b["high"].values
    lows = b["low"].values

    for pos in idxs:
        if pos + 2 >= len(b):
            continue
        direction = -1 if z.iloc[pos] > 0 else 1
        entry = closes[pos]
        atr_v = atr.iloc[pos]
        s = stop_a * atr_v
        tg = trig_a * atr_v
        gp = gap_a * atr_v
        best = entry
        exited = False
        for j in range(1, max_bars + 1):
            bar_pos = pos + j
            if bar_pos >= len(b):
                break
            if direction == 1:
                if highs[bar_pos] > best:
                    best = highs[bar_pos]
                sl = entry - s
                if best - entry > tg:
                    sl = best - gp
                if lows[bar_pos] <= sl:
                    pnls.append((sl - entry))
                    exited = True
                    break
            else:
                if lows[bar_pos] < best:
                    best = lows[bar_pos]
                sl = entry + s
                if entry - best > tg:
                    sl = best + gp
                if highs[bar_pos] >= sl:
                    pnls.append((sl - entry) * direction)
                    exited = True
                    break
        if not exited:
            exit_bar = min(pos + max_bars, len(b) - 1)
            pnls.append((closes[exit_bar] - entry) * direction)
    return np.array(pnls)


def run_test(pair, z_thresh=0.0, days=None):
    """Run V2+z on all available data for a pair."""
    b = load_dukascopy(pair)
    if days:
        cutoff = b.index[0] + pd.Timedelta(days=days)
        b = b[b.index < cutoff]

    splits = [
        ("2024-08-01","2025-01-01","Q4 2024"),
        ("2026-01-01","2026-05-01","Q1 2026"),
        ("2026-05-01","2026-07-01","Q2 2026"),
    ]
    all_p = []
    for sd, ed, label in splits:
        mask = (b.index >= sd) & (b.index < ed)
        if mask.sum() < 100:
            continue
        pnls = hfdf_m1(b[mask].copy(), COST[pair], z_thresh=z_thresh)
        all_p.append(pnls)
    return np.concatenate(all_p) if all_p else np.array([])


def summary(pnls, pair, z_thresh, label=""):
    if len(pnls) < 5:
        print(f"  No trades for {pair}")
        return 0, 0

    wr = np.mean(pnls > 0)
    avg = np.mean(pnls)
    net = avg - COST[pair]
    tpd = len(pnls) / 462  # ~462 trading days in dataset

    print(f"  {label} z>={z_thresh:.1f}: n={len(pnls):>6d}  {tpd:.0f}/d  "
          f"WR={wr:>5.1%}  avg={avg:>+7.2f}  net={net:>+7.2f}")
    return wr, net


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pair", nargs="?", default="EURUSD")
    ap.add_argument("--z", type=float, default=0.0)
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--per-split", action="store_true")
    args = ap.parse_args()

    pair = args.pair.upper()
    t0 = time.time()

    if args.scan:
        print(f"\nV2+z SCAN — {pair}")
        print(f"{'='*55}")
        for z in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            pnls = run_test(pair, z_thresh=z, days=args.days if not args.full else None)
            summary(pnls, pair, z, label="")
        print(f"\nRuntime: {time.time()-t0:.1f}s")
    elif args.per_split:
        print(f"\nV2+z PER-SPLIT — {pair}  (z>={args.z:.1f})")
        print(f"{'='*55}")
        b = load_dukascopy(pair)
        splits = [
            ("2024-08-01","2025-01-01","Q4 2024"),
            ("2026-01-01","2026-05-01","Q1 2026"),
            ("2026-05-01","2026-07-01","Q2 2026"),
        ]
        for sd, ed, label in splits:
            mask = (b.index >= sd) & (b.index < ed)
            if mask.sum() < 100:
                print(f"  {label}: insufficient data ({mask.sum()} rows)")
                continue
            pnls = hfdf_m1(b[mask].copy(), COST[pair], z_thresh=args.z)
            summary(pnls, pair, args.z, label=label)
        print(f"\nRuntime: {time.time()-t0:.1f}s")
    else:
        pnls = run_test(pair, z_thresh=args.z, days=args.days if not args.full else None)
        summary(pnls, pair, args.z, label="TOTAL")
        print(f"Runtime: {time.time()-t0:.1f}s")
