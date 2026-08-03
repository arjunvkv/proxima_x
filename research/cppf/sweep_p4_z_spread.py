"""P4: Z × Spread Combined Sweep — focus on surviving pairs, find best z at each spread level."""
import time
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit

DATA_DIR = Path(r"C:\Trading\Agentic_Trading\proxima_x\research\fundednext_data")
PAIRS = ["GBPAUD", "GBPCAD", "GBPNZD", "EURAUD", "AUDNZD", "EURNZD"]
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
    if n < window + 2: return z
    for i in range(window, n):
        cur = c[i] - c[i-1]
        s = 0.0
        for k in range(i - window, i): s += c[k] - c[k-1]
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
    if n < period: return atr
    for i in range(period - 1, n):
        s = 0.0
        for k in range(i - period + 1, i + 1): s += h[k] - l[k]
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
    nt = 0; i = 1; point = 0.00001
    while i < n and nt < cap:
        if not in_hours[i]: i += 1; continue
        if sprd_pts[i] > max_sprd_pts: i += 1; continue
        zi = z[i-1]
        if np.isnan(zi) or np.isnan(atr[i-1]) or atr[i-1] <= 0: i += 1; continue
        if abs(zi) < z_thresh: i += 1; continue
        direction = 1 if zi < 0 else -1
        entry = o[i]; atr_v = atr[i-1]
        sl = entry - stop_a * atr_v if direction > 0 else entry + stop_a * atr_v
        best = entry; exited = False; exit_px_val = 0.0
        max_j = min(max_hold + 1, n - i)
        for j in range(1, max_j):
            idx = i + j
            if direction > 0:
                if h[idx] > best: best = h[idx]
                if best - entry > trig_a * atr_v:
                    ns = best - gap_a * atr_v
                    if ns > sl: sl = ns
                if l[idx] <= sl: exit_px_val = sl; exited = True; break
            else:
                if l[idx] < best: best = l[idx]
                if entry - best > trig_a * atr_v:
                    ns = best + gap_a * atr_v
                    if ns < sl: sl = ns
                if h[idx] >= sl: exit_px_val = sl; exited = True; break
        if not exited: exit_px_val = c[i + max_j - 1]
        pnl = (exit_px_val - entry) * lot * contract if direction > 0 else (entry - exit_px_val) * lot * contract
        pnl -= lot * comm
        pnl -= sprd_pts[i] * point * lot * contract
        entry_a[nt]=entry; exit_a[nt]=exit_px_val; pnl_a[nt]=pnl; dir_a[nt]=direction
        bars_a[nt]=max_hold if not exited else j; z_a[nt]=zi; sprd_a[nt]=sprd_pts[i]
        nt += 1
        i += max_hold if not exited else j
    return entry_a[:nt], exit_a[:nt], pnl_a[:nt], dir_a[:nt], bars_a[:nt], z_a[:nt], sprd_a[:nt]

def run_backtest(df, z_thresh, max_sprd_pts, stop_a=2.0, trig_a=1.0, gap_a=0.03, lot=0.75, start_hour=0, end_hour=7):
    o=df['open'].values.astype(np.float64); h=df['high'].values.astype(np.float64)
    l=df['low'].values.astype(np.float64); c=df['close'].values.astype(np.float64)
    sp=df['spread'].values.astype(np.int32)
    z=compute_z_numba(c,50); atr=compute_atr_numba(h,l,20)
    hours=np.array([t.hour for t in df['datetime']], dtype=np.int32)
    in_hours = (hours>=start_hour)&(hours<end_hour) if end_hour>start_hour else (hours>=start_hour)|(hours<end_hour)
    in_hours=in_hours.astype(np.int8)
    t0=time.time()
    e,ex,pnl,d,b,zs,s=run_kernel(o,h,l,c,z,atr,sp,in_hours,z_thresh,54,stop_a,trig_a,gap_a,lot,CONTRACT,COMM,max_sprd_pts)
    return e,ex,pnl,d,b,zs,s,time.time()-t0

def analyze(pnl, sprd_vals):
    n=len(pnl)
    if n==0: return None
    net=pnl.sum(); wins=pnl>0; n_w=wins.sum(); n_l=(pnl<0).sum(); n_z=(np.abs(pnl)<0.01).sum()
    den=n-n_z; wr=n_w/den*100 if den else 0
    aw=pnl[wins].mean() if n_w else 0
    al=pnl[~wins&(np.abs(pnl)>=0.01)].mean() if n_l else 0
    payoff=abs(aw/al) if al and al!=0 else 0
    return {'n':n,'wr':wr,'net':net,'avg_win':aw,'avg_loss':al,'payoff':payoff}

# Best trailing from P3: s=2/t=1/g=0.03 for GBPAUD, s=2/t=1 for GBPCAD
# Sweep z x spread for each pair
Z_VALS = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
# Per-pair spread thresholds to sweep (tighter around where we saw survivors)
SPRD_SWEEP = {"GBPAUD": [12, 15, 20, 50], "GBPCAD": [12, 15, 20, 50],
              "GBPNZD": [12, 15, 20, 50], "EURAUD": [10, 12, 15, 20, 50],
              "AUDNZD": [12, 15, 20, 50], "EURNZD": [12, 15, 20, 50]}

print("=" * 130)
print("P4: Z × SPREAD COMBINED SWEEP — FundedNext Server 3")
print("Best trailing from P3: s=2.0/t=1.0/g=0.03, lot=0.75, 0-7 UTC")
print("=" * 130)

all_results = []
for pair in PAIRS:
    df = load_fundednext(pair)
    print(f"\n--- {pair}: ({len(df)} bars) ---")

    for z in Z_VALS:
        for ms in SPRD_SWEEP[pair]:
            entry, ex, pnl, d, bars, zs, sprd, elapsed = run_backtest(df, z_thresh=z, max_sprd_pts=ms)
            r = analyze(pnl, sprd)
            if r and r['n'] >= 2:
                survive = "SURVIVES" if r['net'] > 0 else "DIES"
                print(f"  z>={z:.1f} sprd<={ms:2d}: {r['n']:>3d}t {r['wr']:>5.1f}% "
                      f"net ${r['net']:>+8.2f} avgW${r['avg_win']:>+5.1f}/L${r['avg_loss']:>+5.1f} "
                      f"PF={r['payoff']:.2f}  {survive}")
                all_results.append({**r, 'pair':pair, 'z':z, 'max_sprd':ms})

# Best per pair
print(f"\n{'='*130}")
print("BEST CONFIG PER PAIR (trades>=5)")
print(f"{'PAIR':<8} {'z':<5} {'sprd<=':<6} {'TRADES':>7} {'WR':>7} {'NET':>10} {'PAYOFF':>7}")
print("-" * 55)
portfolio_total = 0
portfolio_trades = 0
for pair in PAIRS:
    pr = [r for r in all_results if r['pair']==pair and r['n']>=5 and r['net']>0]
    if not pr:
        pr = [r for r in all_results if r['pair']==pair and r['n']>=5]
    if pr:
        best = max(pr, key=lambda r: r['net'])
        survive = "SURVIVES" if best['net']>0 else "DIES"
        print(f"{pair:<8} z>={best['z']:.1f} sprd<={best['max_sprd']:>2d}  "
              f"{best['n']:>5d}  {best['wr']:>5.1f}%  ${best['net']:>+8.2f}  "
              f"{best['payoff']:.2f}  {survive}")
        if best['net'] > 0:
            portfolio_total += best['net']
            portfolio_trades += best['n']
    else:
        # Show least bad
        pr = [r for r in all_results if r['pair']==pair]
        if pr:
            best = max(pr, key=lambda r: r['net'])
            print(f"{pair:<8} z>={best['z']:.1f} sprd<={best['max_sprd']:>2d}  "
                  f"{best['n']:>5d}  {best['wr']:>5.1f}%  ${best['net']:>+8.2f}  "
                  f"{best['payoff']:.2f}  DIES")
        else:
            print(f"{pair:<8} NO CONFIG")

print(f"\n  SURVIVING PORTFOLIO: {portfolio_trades:>4d} trades  ${portfolio_total:>+8.2f}")

# Show top 10 configs overall
print(f"\n{'='*130}")
print("TOP 20 CONFIGS (trades>=10, by net PnL)")
print(f"{'PAIR':<8} {'z':<5} {'sprd<=':<6} {'TRADES':>7} {'WR':>7} {'NET':>10} {'PAYOFF':>7}")
print("-" * 55)
top = sorted([r for r in all_results if r['n']>=10], key=lambda r: -r['net'])[:20]
for r in top:
    survive = "SURVIVES" if r['net']>0 else "DIES"
    print(f"{r['pair']:<8} z>={r['z']:.1f} sprd<={r['max_sprd']:>2d}  "
          f"{r['n']:>5d}  {r['wr']:>5.1f}%  ${r['net']:>+8.2f}  "
          f"{r['payoff']:.2f}  {survive}")
