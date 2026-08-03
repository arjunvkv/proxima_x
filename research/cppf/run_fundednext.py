"""V2+z on FundedNext-Server 3 M1 data — real spreads, $3 commission, numba JIT."""
import time
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit

DATA_DIR = Path(r"C:\Trading\Agentic_Trading\proxima_x\research\fundednext_data")
PAIRS = ["EURAUD", "AUDNZD", "EURNZD", "GBPAUD", "GBPCAD", "GBPNZD"]
CONTRACT = 100000
LOT = 0.75
COMM = 3.0
POINT = 0.00001  # 5-digit broker

def load_fundednext(pair):
    rates = np.load(str(DATA_DIR / f"{pair}.npy"))
    df = pd.DataFrame(rates, columns=["time","open","high","low","close","tick_volume","spread","real_volume"])
    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    return df

@njit
def compute_z_numba(c, window):
    n = len(c)
    z = np.full(n, np.nan)
    if n < window + 2:
        return z
    for i in range(window, n):
        cur = c[i] - c[i-1]
        s = 0.0
        for k in range(i - window, i):
            s += c[k] - c[k-1]
        avg = s / window
        var = 0.0
        for k in range(i - window, i):
            d = (c[k] - c[k-1]) - avg
            var += d * d
        var /= (window - 1)
        z[i] = (cur - avg) / np.sqrt(var) if var >= 1e-14 else 0.0
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
def run_kernel(o, h, l, c, z, atr, sprd_pts, in_hours, z_thresh, max_hold,
               stop_a, trig_a, gap_a, lot, contract, comm):
    n = len(o)
    cap = 100000
    entry_a = np.zeros(cap, dtype=np.float64)
    exit_a  = np.zeros(cap, dtype=np.float64)
    pnl_a   = np.zeros(cap, dtype=np.float64)
    dir_a   = np.zeros(cap, dtype=np.int32)
    bars_a  = np.zeros(cap, dtype=np.int32)
    z_a     = np.zeros(cap, dtype=np.float64)
    sprd_a  = np.zeros(cap, dtype=np.float64)
    nt = 0
    i = 1
    # spread cost in price = sprd_pts * POINT
    point = 0.00001

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
        # spread cost: entry at ask for long, exit at ask for short
        sprd_cost = sprd_pts[i] * point * lot * contract
        pnl -= sprd_cost

        entry_a[nt] = entry
        exit_a[nt]  = exit_px_val
        pnl_a[nt]   = pnl
        dir_a[nt]   = direction
        bars_a[nt]  = max_hold if not exited else j
        z_a[nt]     = zi
        sprd_a[nt]  = sprd_pts[i]
        nt += 1

        i += max_hold if not exited else j

    return entry_a[:nt], exit_a[:nt], pnl_a[:nt], dir_a[:nt], bars_a[:nt], z_a[:nt], sprd_a[:nt]

def run_backtest(df, z_thresh, start_hour=0, end_hour=7):
    o = df['open'].values.astype(np.float64)
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)
    c = df['close'].values.astype(np.float64)
    sp = df['spread'].values.astype(np.int32)

    z = compute_z_numba(c, 50)
    atr = compute_atr_numba(h, l, 20)

    hours = np.array([t.hour for t in df['datetime']], dtype=np.int32)
    in_hours = (hours >= start_hour) & (hours < end_hour) if end_hour > start_hour else (hours >= start_hour) | (hours < end_hour)
    in_hours = in_hours.astype(np.int8)

    t0 = time.time()
    entry, ex, pnl, d, bars, zs, sprd = run_kernel(
        o, h, l, c, z, atr, sp, in_hours, z_thresh, 54, 3.0, 1.0, 0.05,
        LOT, CONTRACT, COMM
    )
    return entry, ex, pnl, d, bars, zs, sprd, time.time() - t0

def print_result(pair, label, pnl, sprd_vals):
    n = len(pnl)
    if n == 0:
        print(f"{pair:<8} {label:<14}  NO TRADES")
        return
    net = pnl.sum()
    wins = pnl > 0
    n_w = wins.sum()
    n_l = (pnl < 0).sum()
    n_z = (np.abs(pnl) < 0.01).sum()
    den = n - n_z
    wr = n_w / den * 100 if den else 0
    aw = pnl[wins].mean() if n_w else 0
    al = pnl[~wins & (np.abs(pnl) >= 0.01)].mean() if n_l else 0
    med_sprd_pips = np.median(sprd_vals) / 10.0
    survive = "SURVIVES" if net > 0 else "DIES"
    print(f"{pair:<8} {label:<14}  {n:>4d} trades  {wr:>5.1f}%  "
          f"net ${net:>+8.2f}  avgW ${aw:>+7.2f}  avgL ${al:>+7.2f}  "
          f"med_sprd={med_sprd_pips:.1f}p  ← {survive}")

print("=" * 130)
print("V2+z ON FUNDEDNEXT-SERVER 3 M1 DATA — REAL SPREADS")
print(f"Apr 21 - Jul 1 2026, $3/round-turn, 0.75 lot, Asian session 0-7 UTC")
print("=" * 130)

all_r = []
for pair in PAIRS:
    df = load_fundednext(pair)
    dmin = df['datetime'].min().strftime('%m/%d')
    dmax = df['datetime'].max().strftime('%m/%d')
    print(f"\n--- {pair}: {dmin} to {dmax} ({len(df)} bars) ---")

    for z in [2.0, 2.5, 3.0, 3.5, 4.0]:
        entry, ex, pnl, d, bars, zs, sprd, elapsed = run_backtest(df, z_thresh=z)
        print_result(pair, f"z>={z:.1f}", pnl, sprd)
        all_r.append((pair, z, len(pnl), pnl.sum(), elapsed))

    entry, ex, pnl, d, bars, zs, sprd, elapsed = run_backtest(df, z_thresh=3.5, start_hour=0, end_hour=24)
    print_result(pair, "0-24h z>=3.5", pnl, sprd)
    secs = sum(r[4] for r in all_r if r[0] == pair)
    print(f"  kernel: {secs:.2f}s")

print(f"\n{'='*70}")
print("BEST NET PER PAIR (Asian session)")
print(f"{'PAIR':<8} {'BEST_Z':>7} {'TRADES':>6} {'NET_PNL':>10}")
print("-" * 35)
total_net = 0
for pair in PAIRS:
    best = None
    for r in all_r:
        if r[0] == pair and r[2] > 10 and (best is None or r[3] > best[3]):
            best = r
    if best:
        survive = "SURVIVES" if best[3] > 0 else "DIES"
        total_net += best[3]
        print(f"  {best[0]:<6} z>={best[1]:.1f}  {best[2]:>4d} trades  ${best[3]:>+8.2f}  {survive}")
print(f"  PORTFOLIO TOTAL: ${total_net:>+.2f}")
