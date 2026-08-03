"""V2+z on Dukascopy M1 bid — 6 cross pairs, fully numba, no look-ahead, with spread cost."""
import time
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit

DUKA_DIR = Path("research/dark_research/dukascopy_data")
PAIRS = ["EURAUD", "AUDNZD", "EURNZD", "GBPAUD", "GBPCAD", "GBPNZD"]
CONTRACT = 100000
LOT = 0.75
COMM = 3.0
PIP = 0.0001

@njit
def compute_z_numba(c, window):
    """z[i] = z-score of return c[i]-c[i-1] vs prior `window` returns."""
    n = len(c)
    z = np.full(n, np.nan)
    if n < window + 2:
        return z
    # ret[i] = c[i] - c[i-1]  (matching original vectorized with prepend)
    ret = np.empty(n)
    ret[0] = 0.0
    for i in range(1, n):
        ret[i] = c[i] - c[i-1]
    for i in range(window, n):
        avg = 0.0
        for k in range(i - window, i):
            avg += ret[k]
        avg /= window
        var = 0.0
        for k in range(i - window, i):
            var += (ret[k] - avg) ** 2
        var /= (window - 1)
        cur = ret[i]
        if var < 1e-14:
            z[i] = 0.0
        else:
            z[i] = (cur - avg) / np.sqrt(var)
    return z

@njit
def compute_atr_numba(h, l, period):
    n = len(h)
    atr = np.full(n, np.nan)
    if n < period:
        return atr
    for i in range(period - 1, n):
        s = 0.0
        for k in range(i - period + 1, i + 1):
            s += h[k] - l[k]
        atr[i] = s / period
    return atr

@njit
def run_backtest_kernel(o, h, l, c, z, atr, in_hours, z_thresh, max_hold,
                         stop_a, trig_a, gap_a, lot, contract, comm, sprd_price):
    n = len(o)
    cap = 100000
    entry_a = np.zeros(cap, dtype=np.float64)
    exit_a  = np.zeros(cap, dtype=np.float64)
    pnl_a   = np.zeros(cap, dtype=np.float64)
    dir_a   = np.zeros(cap, dtype=np.int32)
    bars_a  = np.zeros(cap, dtype=np.int32)
    z_a     = np.zeros(cap, dtype=np.float64)
    nt = 0
    i = 1

    while i < n and nt < cap:
        if not in_hours[i]:
            i += 1
            continue
        zi = z[i-1]
        if np.isnan(zi) or np.isnan(atr[i-1]) or atr[i-1] <= 0:
            i += 1
            continue
        if abs(zi) < z_thresh:
            i += 1
            continue

        direction = 1 if zi < 0 else -1
        entry = o[i]
        atr_v = atr[i-1]
        sl = entry - stop_a * atr_v if direction > 0 else entry + stop_a * atr_v
        best = entry
        exited = False
        exit_px_val = 0.0

        max_j = max_hold + 1
        if max_j > n - i:
            max_j = n - i

        for j in range(1, max_j):
            idx = i + j
            if direction > 0:
                if h[idx] > best:
                    best = h[idx]
                if best - entry > trig_a * atr_v:
                    ns = best - gap_a * atr_v
                    if ns > sl:
                        sl = ns
                if l[idx] <= sl:
                    exit_px_val = sl
                    exited = True
                    break
            else:
                if l[idx] < best:
                    best = l[idx]
                if entry - best > trig_a * atr_v:
                    ns = best + gap_a * atr_v
                    if ns < sl:
                        sl = ns
                if h[idx] >= sl:
                    exit_px_val = sl
                    exited = True
                    break

        if not exited:
            exit_px_val = c[i + max_j - 1]

        if direction > 0:
            pnl = (exit_px_val - entry) * lot * contract
        else:
            pnl = (entry - exit_px_val) * lot * contract
        pnl -= lot * comm
        pnl -= sprd_price * lot * contract

        entry_a[nt] = entry
        exit_a[nt]  = exit_px_val
        pnl_a[nt]   = pnl
        dir_a[nt]   = direction
        bars_a[nt]  = max_hold if not exited else j
        z_a[nt]     = zi
        nt += 1

        i += max_hold if not exited else j

    return entry_a[:nt], exit_a[:nt], pnl_a[:nt], dir_a[:nt], bars_a[:nt], z_a[:nt]

def load_dukascopy(pair):
    files = sorted(DUKA_DIR.glob(f"{pair.lower()}-m1-bid-*.csv"))
    dfs = []
    for f in files:
        dfs.append(pd.read_csv(f, usecols=["timestamp", "open", "high", "low", "close"]))
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def run_backtest(df, z_thresh, start_hour=0, end_hour=7, spread_pips=2):
    o = df['open'].values.astype(np.float64)
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)
    c = df['close'].values.astype(np.float64)

    z = compute_z_numba(c, 50)
    atr = compute_atr_numba(h, l, 20)

    hours = np.array([t.hour for t in df['datetime']], dtype=np.int32)
    in_hours = (hours >= start_hour) & (hours < end_hour) if end_hour > start_hour else (hours >= start_hour) | (hours < end_hour)
    in_hours = in_hours.astype(np.int8)

    sprd_price = spread_pips * PIP

    t0 = time.time()
    entry, ex, pnl, d, bars, zs = run_backtest_kernel(
        o, h, l, c, z, atr, in_hours, z_thresh, 54, 3.0, 1.0, 0.05,
        LOT, CONTRACT, COMM, sprd_price
    )
    return entry, ex, pnl, d, bars, zs, time.time() - t0

def print_result(pair, label, pnl, sprd_pips):
    n = len(pnl)
    if n == 0:
        print(f"{pair:<8} {label:<14}  NO TRADES")
        return
    sprd_cost = sprd_pips * PIP * LOT * CONTRACT
    gross = pnl + LOT * COMM + sprd_cost
    net = pnl.sum()
    wins = pnl > 0
    n_w = wins.sum()
    n_l = (pnl < 0).sum()
    n_z = (np.abs(pnl) < 0.01).sum()
    den = n - n_z
    wr = n_w / den * 100 if den else 0
    aw = pnl[wins].mean() if n_w else 0
    al = pnl[~wins & (np.abs(pnl) >= 0.01)].mean() if n_l else 0
    survive = "SURVIVES" if net > 0 else "DIES"
    print(f"{pair:<8} {label:<14} sprd={sprd_pips}  {n:>4d} trades  {wr:>5.1f}%  "
          f"gross ${gross.sum():>+8.2f}  net ${net:>+8.2f}  "
          f"avgW ${aw:>+7.2f}  avgL ${al:>+7.2f}  ← {survive}")

for SPREAD in [1, 2, 3]:
    print("=" * 130)
    print(f"V2+z ON DUKASCOPY — FIXED: spread cost + expiry off-by-one corrected")
    print(f"Spread={SPREAD} pip(s), $3/round-turn, 0.75 lot, 0-7 UTC")
    print("=" * 130)

    all_r = []
    for pair in PAIRS:
        df = load_dukascopy(pair)
        if len(df) == 0:
            print(f"\n{pair}: NO DATA"); continue
        dmin = df['datetime'].min().strftime('%Y-%m-%d')
        dmax = df['datetime'].max().strftime('%Y-%m-%d')
        print(f"\n--- {pair}: {dmin} to {dmax} ({len(df)} bars) ---")

        for z in [2.0, 2.5, 3.0, 3.5, 4.0]:
            entry, ex, pnl, d, bars, zs, elapsed = run_backtest(df, z_thresh=z, spread_pips=SPREAD)
            print_result(pair, f"z>={z:.1f}", pnl, SPREAD)
            all_r.append((pair, z, len(pnl), pnl.sum(), elapsed))

        entry, ex, pnl, d, bars, zs, elapsed = run_backtest(df, z_thresh=3.5, start_hour=0, end_hour=24, spread_pips=SPREAD)
        print_result(pair, "0-24h z>=3.5", pnl, SPREAD)

        secs = sum(r[4] for r in all_r if r[0] == pair)
        print(f"  kernel: {secs:.2f}s")

    print(f"\n--- Spread={SPREAD} pip BEST PER PAIR ---")
    print(f"{'PAIR':<8} {'BEST_Z':>7} {'TRADES':>7} {'NET_PNL':>10}")
    for pair in PAIRS:
        best = None
        for r in all_r:
            if r[0] == pair and r[2] > 10 and (best is None or r[3] > best[3]):
                best = r
        if best:
            survive = "SURVIVES" if best[3] > 0 else "DIES"
            print(f"  {best[0]:<6} z>={best[1]:.1f}  {best[2]:>4d} trades  ${best[3]:>+8.2f}  {survive}")

    total_net = sum(r[3] for r in all_r if r[2] > 0)
    total_tr  = sum(r[2] for r in all_r if r[2] > 0)
    print(f"  PORTFOLIO total (all z): ${total_net:>+.2f}  ({total_tr} trades)")
