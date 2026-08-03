"""P3: Trailing Config Grid — sweep stop_a x trig_a x gap_a at best spread threshold per pair."""
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

# Best spread threshold per pair from P1 (tighter = higher quality)
BEST_SPRD = {"EURAUD": 12, "AUDNZD": 15, "EURNZD": 15, "GBPAUD": 12, "GBPCAD": 15, "GBPNZD": 15}
# Also try looser spread to see if trailing helps overcome wider costs
LOOSE_SPRD = {"EURAUD": 20, "AUDNZD": 20, "EURNZD": 20, "GBPAUD": 20, "GBPCAD": 20, "GBPNZD": 20}

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

def run_backtest(df, z_thresh, max_sprd_pts, stop_a, trig_a, gap_a, lot=0.75, start_hour=0, end_hour=7):
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
        o, h, l, c, z, atr, sp, in_hours, z_thresh, 54,
        stop_a, trig_a, gap_a, lot, CONTRACT, COMM, max_sprd_pts
    )
    return entry, ex, pnl, d, bars, zs, sprd, time.time() - t0

def analyze(pnl, sprd_vals):
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
    payoff = abs(aw/al) if al and al != 0 else 0
    return {'n': n, 'wr': wr, 'net': net, 'avg_win': aw, 'avg_loss': al, 'payoff': payoff}

# Trailing config sweep
STOPS = [2.0, 2.5, 3.0, 4.0, 5.0]
TRIGS = [0.5, 1.0, 2.0, 3.0]
GAPS  = [0.03, 0.05, 0.10]

print("=" * 140)
print("P3: TRAILING CONFIG GRID — FundedNext Server 3")
print("z=3.5, lot=0.75, 0-7 UTC")
print("=" * 140)

for sprd_mode, sprd_map in [("TIGHT(BEST)", BEST_SPRD), ("LOOSE", LOOSE_SPRD)]:
    print(f"\n{'='*140}")
    print(f"SPREAD MODE: {sprd_mode}")
    print("=" * 140)

    all_results = []
    for pair in PAIRS:
        max_sprd = sprd_map[pair]
        df = load_fundednext(pair)
        print(f"\n--- {pair}: sprd<={max_sprd} ---")

        for stop_a in STOPS:
            for trig_a in TRIGS:
                for gap_a in GAPS:
                    entry, ex, pnl, d, bars, zs, sprd, elapsed = run_backtest(
                        df, z_thresh=3.5, max_sprd_pts=max_sprd,
                        stop_a=stop_a, trig_a=trig_a, gap_a=gap_a
                    )
                    r = analyze(pnl, sprd)
                    if r and r['n'] >= 2:
                        survive = "SURVIVES" if r['net'] > 0 else "DIES"
                        label = f"s={stop_a:.0f}/t={trig_a:.0f}/g={gap_a:.2f}"
                        print(f"  {label:<17}  {r['n']:>3d}t  {r['wr']:>5.1f}%  "
                              f"net ${r['net']:>+8.2f}  W${r['avg_win']:>+6.2f}/L${r['avg_loss']:>+6.2f}  "
                              f"PF={r['payoff']:.2f}  {survive}")
                        all_results.append({**r, 'pair': pair, 'stop_a': stop_a, 'trig_a': trig_a,
                                            'gap_a': gap_a, 'label': label, 'sprd_mode': sprd_mode})

        # Best per pair
        pr = [r for r in all_results if r['pair'] == pair and r['n'] >= 3 and r['net'] > 0]
        survivors = [r for r in all_results if r['pair'] == pair and r['net'] > 0]
        if survivors:
            best = max(survivors, key=lambda r: r['net'] * r['n'] / (r['n'] + 5))  # weighted for confidence
            print(f"  >> BEST: s={best['stop_a']:.0f}/t={best['trig_a']:.0f}/g={best['gap_a']:.2f}  "
                  f"{best['n']}t  {best['wr']:.1f}%  ${best['net']:+.2f}")
        else:
            # Least bad
            bad = sorted([r for r in all_results if r['pair'] == pair], key=lambda r: -r['net'])
            if bad:
                b = bad[0]
                print(f"  >> LEAST BAD: s={b['stop_a']:.0f}/t={b['trig_a']:.0f}/g={b['gap_a']:.2f}  "
                      f"{b['n']}t  {b['wr']:.1f}%  ${b['net']:+.2f}")

    # Portfolio
    survivors = [r for r in all_results if r['net'] > 0]
    if survivors:
        print(f"\n{'='*140}")
        print(f"ALL SURVIVING CONFIGS ({sprd_mode})")
        print(f"{'PAIR':<8} {'CONFIG':<18} {'TRADES':>7} {'WR':>7} {'NET':>10} {'PAYOFF':>7}")
        print("-" * 60)
        survivors.sort(key=lambda r: -r['net'])
        for r in survivors:
            print(f"{r['pair']:<8} {r['label']:<18} {r['n']:>5d}  {r['wr']:>5.1f}%  "
                  f"${r['net']:>+8.2f}  {r['payoff']:.2f}")

print(f"\n{'='*140}")
print("BEST PER PAIR ACROSS ALL CONFIGS (tight spread, trades>=3)")
print(f"{'PAIR':<8} {'CONFIG':<18} {'TRADES':>7} {'WR':>7} {'NET':>10} {'PAYOFF':>7}")
print("-" * 60)
for pair in PAIRS:
    tight_r = [r for r in all_results if r['pair'] == pair and r['sprd_mode'] == 'TIGHT(BEST)' and r['n'] >= 3]
    if tight_r:
        best = max(tight_r, key=lambda r: r['net'])
        survive = "SURVIVES" if best['net'] > 0 else "DIES"
        print(f"{pair:<8} {best['label']:<18} {best['n']:>5d}  {best['wr']:>5.1f}%  "
              f"${best['net']:>+8.2f}  {best['payoff']:.2f}  {survive}")
    else:
        print(f"{pair:<8} {'NO CONFIG WITH >=3 TRADES':<18}")
