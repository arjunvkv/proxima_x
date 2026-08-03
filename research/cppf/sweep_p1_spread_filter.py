"""P1: Spread Filter Sweep — add max_spread gate, sweep [5,7,9,10,12,15,20,50] at z=3.5"""
import time
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit

DATA_DIR = Path(r"C:\Trading\Agentic_Trading\proxima_x\research\fundednext_data")
PAIRS = ["EURAUD", "AUDNZD", "EURNZD", "GBPAUD", "GBPCAD", "GBPNZD"]
CONTRACT = 100000
COMM = 3.0
POINT = 0.00001

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
               stop_a, trig_a, gap_a, lot, contract, comm, max_sprd_pts):
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
    point = 0.00001

    while i < n and nt < cap:
        if not in_hours[i]:
            i += 1
            continue
        if sprd_pts[i] > max_sprd_pts:
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

def run_backtest(df, z_thresh, max_sprd_pts, lot=0.75, start_hour=0, end_hour=7):
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
        lot, CONTRACT, COMM, max_sprd_pts
    )
    return entry, ex, pnl, d, bars, zs, sprd, time.time() - t0

def print_result(pair, label, pnl, sprd_vals):
    n = len(pnl)
    if n == 0:
        return None
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
    return {'pair': pair, 'label': label, 'n': n, 'wr': wr, 'net': net,
            'avg_win': aw, 'avg_loss': al, 'med_sprd': med_sprd_pips}

SPRD_LIMITS = [5, 7, 9, 10, 12, 15, 20, 50]

print("=" * 140)
print("P1: SPREAD FILTER SWEEP — FundedNext Server 3")
print("z=3.5, stop=3.0/1.0/0.05, lot=0.75, 0-7 UTC")
print("=" * 140)

all_results = []
for pair in PAIRS:
    df = load_fundednext(pair)
    print(f"\n--- {pair}: {df['datetime'].min().strftime('%m/%d')} to {df['datetime'].max().strftime('%m/%d')} ({len(df)} bars) ---")

    for ms in SPRD_LIMITS:
        entry, ex, pnl, d, bars, zs, sprd, elapsed = run_backtest(df, z_thresh=3.5, max_sprd_pts=ms)
        r = print_result(pair, f"sprd<={ms:2d}", pnl, sprd)
        if r:
            survive = "SURVIVES" if r['net'] > 0 else "DIES"
            print(f"  sprd<={ms:2d}: {r['n']:>4d} trades  {r['wr']:>5.1f}%  "
                  f"net ${r['net']:>+8.2f}  avgW ${r['avg_win']:>+7.2f}  "
                  f"avgL ${r['avg_loss']:>+7.2f}  med_sprd={r['med_sprd']:.1f}p  "
                  f"{elapsed:.2f}s  {survive}")
            all_results.append(r)
        else:
            print(f"  sprd<={ms:2d}: NO TRADES")

print(f"\n{'='*140}")
print("BEST SPREAD THRESHOLD PER PAIR")
print(f"{'PAIR':<8} {'SPRD<=':>6} {'TRADES':>7} {'WR':>7} {'NET_PNL':>10} {'MED_SPRD':>9}")
print("-" * 45)
best_per_pair = {}
for pair in PAIRS:
    pr = [r for r in all_results if r['pair'] == pair and r['n'] > 0]
    if not pr:
        print(f"{pair:<8} NO SURVIVING CONFIG")
        continue
    # Best by net PnL
    best = max(pr, key=lambda r: r['net'])
    best_per_pair[pair] = best
    survive = "SURVIVES" if best['net'] > 0 else "DIES"
    print(f"{pair:<8} sprd<={best['label'][-2:]:>4s}  {best['n']:>5d}  "
          f"{best['wr']:>5.1f}%  ${best['net']:>+8.2f}  sprd={best['med_sprd']:.1f}p  {survive}")

total_net = sum(r['net'] for r in best_per_pair.values())
total_trades = sum(r['n'] for r in best_per_pair.values())
avg_wr = sum(r['wr'] for r in best_per_pair.values()) / len(best_per_pair) if best_per_pair else 0
print(f"\n  PORTFOLIO: {total_trades:>4d} trades  {avg_wr:.1f}% avg WR  ${total_net:>+8.2f} net")

# Also show which config had max net (including if negative, show least negative)
print(f"\n{'='*140}")
print("TOP 10 CONFIGS by PnL")
print(f"{'PAIR':<8} {'SPRD<=':>6} {'TRADES':>7} {'WR':>7} {'NET_PNL':>10} {'AVGW':>8} {'AVGL':>8}")
print("-" * 55)
top = sorted(all_results, key=lambda r: -r['net'])[:10]
for r in top:
    print(f"{r['pair']:<8} {r['label'][-2:]:>4s}  {r['n']:>5d}  {r['wr']:>5.1f}%  "
          f"${r['net']:>+8.2f}  ${r['avg_win']:>+6.2f}  ${r['avg_loss']:>+6.2f}")
