"""Test ALL pair combos for response deficit + check tick cache."""
import sys, os, numpy as np
import pandas as pd
from pathlib import Path

# Set paths
BACKTEST_DIR = str(Path(__file__).resolve().parent)
os.chdir(BACKTEST_DIR)
if BACKTEST_DIR in sys.path:
    sys.path.remove(BACKTEST_DIR)
sys.path.insert(0, BACKTEST_DIR)

# Force-load the local data.py, not currency_decomposition/data/
import importlib.util
spec = importlib.util.spec_from_file_location("data", os.path.join(BACKTEST_DIR, "data.py"))

data_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_mod)
TempCache = data_mod.TempCache
PAIRS = data_mod.PAIRS
from numba import jit

PAIR_IDX = {p: i for i, p in enumerate(PAIRS)}

@jit(nopython=True)
def rolling_beta(x, y, lookback):
    n = len(x)
    beta = np.zeros(n)
    for i in range(lookback, n):
        xw = x[i-lookback:i]
        yw = y[i-lookback:i]
        xm = np.mean(xw)
        ym = np.mean(yw)
        num = np.sum((xw - xm) * (yw - ym))
        den = np.sum((xw - xm) ** 2)
        beta[i] = num / den if den != 0 else 0
    return beta

# Check tick cache files
PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
print("=" * 70)
print("TICK CACHE FILES")
print("=" * 70)
tick_dir = os.path.join(PROJECT_ROOT, "data", "cache")
for fname in sorted(os.listdir(tick_dir)):
    if not fname.endswith(".parquet"):
        continue
    fpath = os.path.join(tick_dir, fname)
    try:
        df = pd.read_parquet(fpath)
        print(f"  {fname}: shape={df.shape}, cols={list(df.columns)}")
    except:
        print(f"  {fname}: ERROR reading")

gbfiles = [f for f in os.listdir(tick_dir) if "GBPJPY" in f]
print(f"\nGBPJPY tick files: {gbfiles}")

# Check market parquets
print()
print("=" * 70)
print("MARKET PARQUETS (M1 bars, same as TempCache)")
print("=" * 70)
for pair in ["EURJPY", "GBPJPY", "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"]:
    fpath = os.path.join(PROJECT_ROOT, "data", "market", f"{pair}.parquet")
    if os.path.exists(fpath):
        df = pd.read_parquet(fpath)
        print(f"  {pair}: {df.shape}, {df.time.min()} to {df.time.max()}, cols={list(df.columns)}")

# Test ALL pair combos for response deficit
print()
print("=" * 70)
print("ALL PAIR COMBOS — Response Deficit Test")
print("=" * 70)

cache = TempCache(7)
aligned, times, _ = cache.get()
n = len(times)

ohlc = {}
for pi, pair in enumerate(PAIRS):
    ohlc[pair] = {"close": aligned[:, pi, 3]}

all_pairs_list = PAIRS
all_results = []

for leader_name in all_pairs_list:
    for follower_name in all_pairs_list:
        if leader_name == follower_name:
            continue
        
        lc = ohlc[leader_name]["close"]
        fc = ohlc[follower_name]["close"]
        l_ret = np.diff(lc)
        f_ret = np.diff(fc)
        
        scale = 10000 if leader_name in ("EURUSD", "GBPUSD", "AUDUSD", "NZDUSD") else 100
        
        for lb in [10, 20, 30]:
            beta = rolling_beta(l_ret, f_ret, lb)
            for hold in [5, 10, 20]:
                deficits, catchups = [], []
                for i in range(lb, len(l_ret) - hold):
                    exp_ret = beta[i] * l_ret[i-1]
                    act_ret = f_ret[i-1]
                    deficits.append((exp_ret - act_ret) * scale)
                    cum_ret = np.sum(f_ret[i:i+hold])
                    catchups.append(cum_ret * scale)
                
                deficits = np.array(deficits)
                catchups = np.array(catchups)
                if len(deficits) < 30 or np.std(deficits) == 0:
                    continue
                
                for zt in [1.5, 2.0]:
                    sig = deficits > zt * np.std(deficits)
                    ns = np.sum(sig)
                    if ns < 20:
                        continue
                    wr = np.sum(catchups[sig] > 0) / ns
                    avg_catchup = np.mean(catchups[sig])
                    all_results.append({
                        "pair": f"{leader_name}->{follower_name}",
                        "lb": lb, "hold": hold, "z": zt,
                        "wr": wr, "trades": ns, "avg": avg_catchup,
                    })

all_results.sort(key=lambda r: -r["wr"])
print(f"\nAll combos with WR >= 58% and >= 20 trades:")
print(f"{'Pair':<22} {'lb':>3} {'hold':>4} {'z':>3} {'WR':>6} {'Trades':>7} {'Avg(p)':>8}")
print("-" * 55)
for r in all_results:
    if r["wr"] < 0.58 or r["trades"] < 20:
        continue
    print(f"{r['pair']:<22} {r['lb']:>3} {r['hold']:>4} {r['z']:>3} "
          f"{r['wr']:>5.1%} {r['trades']:>7} {r['avg']:>7.2f}")

# Summary by leader-follower structural type
print()
print("=" * 70)
print("SUMMARY BY STRUCTURAL TYPE")
print("=" * 70)
print()
print("Cross-rate linked (EURJPY↔GBPJPY, EURUSD↔GBPUSD):")
for r in all_results:
    if r["wr"] < 0.55 or r["trades"] < 30:
        continue
    l, f = r["pair"].split("->")
    if (l == "EURJPY" and f == "GBPJPY") or (l == "GBPJPY" and f == "EURJPY"):
        print(f"  {r['pair']}: WR={r['wr']:.1%} hold={r['hold']} z>{r['z']} ({r['trades']}tr, {r['avg']:.2f}p)")
    if (l == "EURUSD" and f == "GBPUSD") or (l == "GBPUSD" and f == "EURUSD"):
        print(f"  {r['pair']}: WR={r['wr']:.1%} hold={r['hold']} z>{r['z']} ({r['trades']}tr, {r['avg']:.2f}p)")

print()
print("USD-correlated (AUDUSD↔NZDUSD):")
for r in all_results:
    if r["wr"] < 0.55 or r["trades"] < 30:
        continue
    l, f = r["pair"].split("->")
    if (l == "AUDUSD" and f == "NZDUSD") or (l == "NZDUSD" and f == "AUDUSD"):
        print(f"  {r['pair']}: WR={r['wr']:.1%} hold={r['hold']} z>{r['z']} ({r['trades']}tr, {r['avg']:.2f}p)")

print()
print("EURUSD->JPY crosses (leverage):")
for r in all_results:
    if r["wr"] < 0.55 or r["trades"] < 30:
        continue
    l, f = r["pair"].split("->")
    if l == "EURUSD" and f in ("EURJPY", "GBPJPY"):
        print(f"  {r['pair']}: WR={r['wr']:.1%} hold={r['hold']} z>{r['z']} ({r['trades']}tr, {r['avg']:.2f}p)")

print()
print("USDJPY->EURJPY (triangular via EURUSD):")
for r in all_results:
    if r["wr"] < 0.55 or r["trades"] < 30:
        continue
    l, f = r["pair"].split("->")
    if l == "USDJPY" and f == "EURJPY":
        print(f"  {r['pair']}: WR={r['wr']:.1%} hold={r['hold']} z>{r['z']} ({r['trades']}tr, {r['avg']:.2f}p)")
