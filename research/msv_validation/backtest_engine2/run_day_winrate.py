"""Day-by-day consistency + WR improvement analysis for response deficit."""
import sys, os, numpy as np
from pathlib import Path
import importlib.util
from datetime import datetime

BACKTEST_DIR = str(Path(__file__).resolve().parent)
if BACKTEST_DIR in sys.path: sys.path.remove(BACKTEST_DIR)
sys.path.insert(0, BACKTEST_DIR)
spec = importlib.util.spec_from_file_location("data", os.path.join(BACKTEST_DIR, "data.py"))
data_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_mod)
TempCache = data_mod.TempCache
PAIRS = data_mod.PAIRS
pip = data_mod.pip

from numba import jit

@jit(nopython=True)
def rolling_beta(x, y, lb):
    n = len(x)
    beta = np.zeros(n)
    for i in range(lb, n):
        xw = x[i-lb:i]; yw = y[i-lb:i]
        xm = np.mean(xw); ym = np.mean(yw)
        num = np.sum((xw-xm)*(yw-ym))
        den = np.sum((xw-xm)**2)
        beta[i] = num/den if den != 0 else 0
    return beta

def test_pair(leader_name, follower_name, ohlc, times, lb=10, hold=10, zt=2.0):
    lc = ohlc[leader_name]["close"]
    fc = ohlc[follower_name]["close"]
    l_ret = np.diff(lc)
    f_ret = np.diff(fc)

    beta = rolling_beta(l_ret, f_ret, lb)
    ns = len(l_ret)

    deficits = np.zeros(ns)
    for i in range(lb, ns):
        deficits[i] = (beta[i] * l_ret[i-1] - f_ret[i-1]) * 100

    catchups = np.zeros(ns)
    for i in range(lb, ns - hold):
        catchups[i] = np.sum(f_ret[i:i+hold]) * 100

    std = np.std(deficits[lb:])
    if std == 0:
        return []
    sig = deficits > zt * std

    idx_sig = np.where(sig & (catchups != 0))[0]
    if len(idx_sig) < 5:
        return []

    # Day labels
    day_labels = []
    for idx in idx_sig:
        if idx < len(times_dt):
            t = times_dt[idx]
            day = t.strftime("%a")
            day_labels.append(day)
        else:
            day_labels.append("?")

    results = []
    for d in sorted(set(day_labels)):
        day_mask = np.array([dl == d for dl in day_labels])
        day_catchups = catchups[idx_sig][day_mask]
        wr = np.mean(day_catchups > 0)
        results.append({"day": d, "wr": wr, "trades": len(day_catchups), "avg": np.mean(day_catchups)})

    return results

def test_pair_multi_threshold(leader_name, follower_name, ohlc, times, lb=10, hold=10):
    lc = ohlc[leader_name]["close"]
    fc = ohlc[follower_name]["close"]
    l_ret = np.diff(lc)
    f_ret = np.diff(fc)
    beta = rolling_beta(l_ret, f_ret, lb)
    ns = len(l_ret)

    deficits = np.zeros(ns)
    for i in range(lb, ns):
        deficits[i] = (beta[i] * l_ret[i-1] - f_ret[i-1]) * 100

    catchups = np.zeros(ns)
    for i in range(lb, ns - hold):
        catchups[i] = np.sum(f_ret[i:i+hold]) * 100

    std = np.std(deficits[lb:])
    if std == 0:
        return []

    results = []
    thresholds = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    for zt in thresholds:
        sig = deficits > zt * std
        idx_sig = np.where(sig & (catchups != 0))[0]
        n = len(idx_sig)
        if n < 10:
            continue
        wr = np.mean(catchups[idx_sig] > 0)
        avg = np.mean(catchups[idx_sig])
        results.append({"thresh": zt, "wr": wr, "trades": n, "avg": avg})
    return results

# Load data
print("Loading data...")
cache = TempCache(7)
aligned, times, _ = cache.get()

ohlc = {}
for pi, pair in enumerate(PAIRS):
    ohlc[pair] = {"close": aligned[:, pi, 3]}

# Convert Unix timestamps to datetimes
times_dt = [datetime.fromtimestamp(t) for t in times]
print(f"Loaded {len(times_dt)} bars: {times_dt[0]} to {times_dt[-1]}")
day_labels = [t.strftime("%a") for t in times_dt]
unique_days = sorted(set(day_labels))
print(f"Days: {unique_days}")

# ============================================================
# QUESTION 1: Is divergence present EVERY day for each pair combo?
# ============================================================
print("\n" + "=" * 70)
print("Q1: DAY-BY-DAY DIVERGENCE CONSISTENCY")
print("Config: lb=10, hold=10, z>2.0 (strict, low trade count)")
print("=" * 70)

# Test key structural combos
combos_to_test = [
    ("EURJPY", "GBPJPY"),
    ("GBPJPY", "EURJPY"),
    ("EURUSD", "GBPUSD"),
    ("GBPUSD", "EURUSD"),
    ("GBPUSD", "GBPJPY"),
    ("EURUSD", "EURJPY"),
    ("EURJPY", "EURUSD"),
    ("USDJPY", "EURJPY"),
]

print(f"\n{'Pair':<24} | Day-by-day WR | Trades | Avg WR")
print("-" * 70)
for leader, follower in combos_to_test:
    day_results = test_pair(leader, follower, ohlc, times_dt, lb=10, hold=10, zt=2.0)
    if not day_results:
        day_results = test_pair(leader, follower, ohlc, times_dt, lb=10, hold=10, zt=1.5)

    info = {}
    for r in day_results:
        info[r["day"]] = (r["wr"], r["trades"])

    day_str = " | ".join([f"{d}: {info.get(d, ('-',0))[0]:.0%}({info.get(d, (0,0))[1]})" for d in unique_days if d in info])
    all_wrs = [info[d][0] for d in unique_days if d in info]
    all_trs = [info[d][1] for d in unique_days if d in info]
    avg_wr = np.mean(all_wrs) if all_wrs else 0
    total_tr = sum(all_trs) if all_trs else 0
    print(f"{leader}->{follower:<14} | {day_str} | {total_tr:4d} | {avg_wr:.1%}")

# ============================================================
# QUESTION 2: Can we increase winrate?
# ============================================================
print("\n" + "=" * 70)
print("Q2: WINRATE IMPROVEMENT ANALYSIS")
print("=" * 70)

print("\n--- A. Higher threshold (z-score) ---")
print(f"{'Pair':<22} {'lb':>3} {'hold':>4} | z=1.5 WR Tr | z=2.0 WR Tr | z=2.5 WR Tr | z=3.0 WR Tr | z=3.5 WR Tr")
print("-" * 90)
for leader, follower in combos_to_test:
    for lb in [10, 20]:
        for hold in [10, 20]:
            res = test_pair_multi_threshold(leader, follower, ohlc, times, lb=lb, hold=hold)
            if not res:
                continue
            parts = []
            for r in res:
                parts.append(f"{r['wr']:.0%} {r['trades']:3d}")
            if len(parts) >= 2:
                print(f"{leader}->{follower:<14} {lb:>3} {hold:>4} | " + " | ".join(parts[:5]))

print("\n--- B. Combined signals: BOTH EURUSD->EURJPY AND EURJPY->GBPJPY ---")
lc_eu = ohlc["EURUSD"]["close"]
lc_ej = ohlc["EURJPY"]["close"]
fc_gj = ohlc["GBPJPY"]["close"]

eu_ret = np.diff(lc_eu)
ej_ret = np.diff(lc_ej)
gj_ret = np.diff(fc_gj)

ns = len(eu_ret)
# EURUSD->EURJPY deficit
beta_eu_ej = rolling_beta(eu_ret, ej_ret, 10)
deficit_eu_ej = np.zeros(ns)
for i in range(10, ns):
    deficit_eu_ej[i] = (beta_eu_ej[i] * eu_ret[i-1] - ej_ret[i-1]) * 100

# EURJPY->GBPJPY deficit
beta_ej_gj = rolling_beta(ej_ret, gj_ret, 10)
deficit_ej_gj = np.zeros(ns)
for i in range(10, ns):
    deficit_ej_gj[i] = (beta_ej_gj[i] * ej_ret[i-1] - gj_ret[i-1]) * 100

catchouts = np.zeros(ns)
for i in range(10, ns - 10):
    catchouts[i] = np.sum(gj_ret[i:i+10]) * 100

std1 = np.std(deficit_eu_ej[10:])
std2 = np.std(deficit_ej_gj[10:])

for zt in [1.0, 1.5, 2.0]:
    sig1 = deficit_eu_ej > zt * std1
    sig2 = deficit_ej_gj > zt * std2
    sig_and = sig1 & sig2
    sig_or = sig1 | sig2

    n_and = np.sum(sig_and & (catchouts != 0))
    n_or = np.sum(sig_or & (catchouts != 0))
    wr_and = np.mean(catchouts[sig_and & (catchouts != 0)] > 0) if n_and >= 10 else 0
    wr_or = np.mean(catchouts[sig_or & (catchouts != 0)] > 0) if n_or >= 10 else 0
    avg_and = np.mean(catchouts[sig_and & (catchouts != 0)]) if n_and >= 10 else 0
    avg_or = np.mean(catchouts[sig_or & (catchouts != 0)]) if n_or >= 10 else 0

    print(f"  z>{zt:.1f}: AND={wr_and:.1%}({n_and}tr, {avg_and:.2f}p)  OR={wr_or:.1%}({n_or}tr, {avg_or:.2f}p)")

print("\n--- C. Session filtering (EURJPY->GBPJPY) ---")
hours = np.array([t.hour for t in times_dt])
deficits = deficit_ej_gj  # reuse
sig = deficits > 2.0 * std2

for session_name, hr_range in [("Tokyo", range(0, 8)), ("London", range(8, 17)), ("NY", range(13, 22)), ("Asia", range(0, 5))]:
    hr_mask = np.array([h in hr_range for h in hours[1:]])
    mask = sig & hr_mask & (catchouts != 0)
    n = np.sum(mask)
    if n >= 10:
        wr = np.mean(catchouts[mask] > 0)
        avg = np.mean(catchouts[mask])
        print(f"  {session_name:8s} ({hr_range[0]:2d}-{hr_range[-1]:2d}h): WR={wr:.1%} n={n:4d} avg={avg:.2f}p")

# D. Check deficit DIRECTION (negative deficits = follower overshot)
print("\n--- D. Negative deficits (follower OVERSHOT) ---")
sig_neg = deficits < -2.0 * std2
n_neg = np.sum(sig_neg & (catchouts != 0))
if n_neg >= 10:
    wr_neg = np.mean(catchouts[sig_neg & (catchouts != 0)] > 0)
    avg_neg = np.mean(catchouts[sig_neg & (catchouts != 0)])
    print(f"  Negative deficit (overshoot→revert down): WR={wr_neg:.1%} n={n_neg:4d} avg={avg_neg:.2f}p")
print(f"  Positive deficit (lag→catch up): WR={np.mean(catchouts[sig & (catchouts != 0)] > 0):.1%} n={np.sum(sig & (catchouts != 0)):4d}")

# E. Check if filtering by EURUSD->EURJPY magnitude increases WR for EURJPY->GBPJPY
print("\n--- E. EURUSD magnitude filter ---")
eu_mag = np.abs(np.diff(lc_eu)) * 10000  # EURUSD moves in pips
# Divide into quartiles
for q_name, (lo, hi) in [("Q1(weak)", (0, 25)), ("Q2(med)", (25, 50)), ("Q3(strong)", (50, 75)), ("Q4(max)", (75, 100))]:
    q_lo = np.percentile(eu_mag[10:], lo)
    q_hi = np.percentile(eu_mag[10:], hi)
    q_mask = (eu_mag >= q_lo) & (eu_mag < q_hi)
    mask = sig & q_mask & (catchouts != 0)
    n = np.sum(mask)
    if n >= 10:
        wr = np.mean(catchouts[mask] > 0)
        avg = np.mean(catchouts[mask])
        print(f"  {q_name:12s} [{q_lo:.1f}-{q_hi:.1f}p]: WR={wr:.1%} n={n:4d} avg={avg:.2f}p")
