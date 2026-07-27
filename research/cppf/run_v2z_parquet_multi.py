"""V2+z on all Dukascopy parquet pairs. Uses exact CSV hfdf_m1 logic.
Data: 27 pairs from research/phase_dislocation/dukascopy_data/
"""
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

PARQUET_DIR = Path("research/phase_dislocation/dukascopy_data")

# Spread cost per pair in MP (1 MP = 0.0001 price change)
MP_COST = {
    "eurusd": 0.15, "eurjpy": 50, "gbpjpy": 60,
    "audusd": 0.15, "nzdusd": 0.18, "usdcad": 0.20, "usdchf": 0.18,
    "audjpy": 50, "nzdjpy": 60, "cadjpy": 50, "chfjpy": 50,
    "euraud": 0.25, "eurgbp": 0.20, "eurcad": 0.25, "eurchf": 0.25,
    "gbpaud": 0.30, "gbpcad": 0.30, "gbpchf": 0.30, "gbpnzd": 0.35,
    "audcad": 0.20, "audchf": 0.20, "audnzd": 0.20,
    "nzdcad": 0.25, "nzdchf": 0.25,
    "gbpusd": 0.18,
}

def get_cost(pair):
    if pair in MP_COST:
        return MP_COST[pair]
    if "jpy" in pair:
        return 50
    return 0.20

def pip_mult(pair):
    return 10000  # consistent MP units (1 MP = 0.0001 price change for all pairs)

def load_parquet(pair):
    df = pd.read_parquet(PARQUET_DIR / f"{pair}.parquet").set_index("timestamp")
    # Multiply by pip_mult to convert to MP-like units
    pm = pip_mult(pair)
    df = df * pm
    return df.astype(float)

def hfdf_m1(b, cost, z_thresh=0.0):
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

def scan_pair(pair):
    try:
        b = load_parquet(pair)
    except Exception as e:
        return None, str(e)
    cost = get_cost(pair)
    pm = pip_mult(pair)
    n_days = (b.index[-1] - b.index[0]).total_seconds() / 86400 + 1
    results = []
    for z in [0.0, 0.5, 1.0, 1.5, 2.0]:
        pnls = hfdf_m1(b, cost, z_thresh=z)
        if len(pnls) < 5:
            results.append((z, 0, 0, 0, 0, 0))
            continue
        wr = np.mean(pnls > 0)
        avg_mp = np.mean(pnls)
        net_mp = avg_mp - cost  # cost already in MP
        tpd = len(pnls) / n_days
        gross_mp = np.sum(pnls)
        results.append((z, len(pnls), tpd, wr, net_mp, gross_mp))
    return results, n_days

if __name__ == "__main__":
    all_pairs = sorted([f.stem for f in PARQUET_DIR.glob("*.parquet")])
    print(f"V2+z Multi-Pair Scan — {len(all_pairs)} pairs")
    print(f"{'Pair':>8s}  {'Days':>4s}  {'z>=0.0 WR':>9s}  {'t/d':>5s}"
          f"  {'z>=0.5 WR':>9s}  {'t/d':>5s}"
          f"  {'z>=1.0 WR':>9s}  {'t/d':>5s}"
          f"  {'z>=2.0 WR':>9s}  {'t/d':>5s}")
    print("=" * 80)

    good_pairs = []
    for pair in all_pairs:
        res, n_days = scan_pair(pair)
        if res is None:
            print(f"  {pair:>8s}  ERROR: {res[1][:40]}")
            continue
        parts = [f"{pair:>8s}", f"{n_days:>4.0f}"]
        for z, n, tpd, wr, net, gross in res:
            if n < 5:
                parts += [f"   N/A   ", "  N/A"]
            else:
                parts += [f"{wr:>7.1%}", f"{tpd:>4.0f}"]
        print("  ".join(parts))
        
        # Check if z>=0.5 gives WR > 60% and at least 30 trades/day
        for z, n, tpd, wr, net, gross in res:
            if abs(z - 0.5) < 0.01 and n >= 5 and wr >= 0.60 and tpd >= 30:
                good_pairs.append(pair)
                break

    print(f"\nGood pairs (z>=0.5 WR>60% + 30t/d): {len(good_pairs)}")
    for p in good_pairs:
        res, n_days = scan_pair(p)
        for rz, rn, rtpd, rwr, rnet, rgross in res:
            if abs(rz - 0.5) < 0.01:
                print(f"  {p}: WR={rwr:.1%}  {rtpd:.0f}/d  net={rnet:+.2f}MP")
                break
