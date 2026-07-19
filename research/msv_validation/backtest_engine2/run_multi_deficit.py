"""
Multi-bar response deficit test.

Earlier: EURJPY→GBPJPY LAG_CATCHUP at 60.2% but only 0.86 pips avg catchup.
Question: does the catchup ACCUMULATE over 5-10 bars to reach 2-5 pips?
"""
import sys, os
import numpy as np
from numba import jit

sys.path.insert(0, os.path.dirname(__file__))
from data import TempCache, PAIRS

PAIR_IDX = {p: i for i, p in enumerate(PAIRS)}

PAIRS_TO_TEST = [
    ("EURUSD", "GBPUSD"),
    ("AUDUSD", "NZDUSD"),
    ("EURJPY", "GBPJPY"),
]


@jit(nopython=True)
def rolling_beta(x, y, lookback=20):
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


print("=" * 70)
print("MULTI-BAR RESPONSE DEFICIT — Extended Catchup Horizon")
print("=" * 70)

cache = TempCache(7)
aligned, times, _ = cache.get()
n = len(times)

ohlc = {}
for pi, pair in enumerate(PAIRS):
    ohlc[pair] = {"close": aligned[:, pi, 3]}

for leader_name, follower_name in PAIRS_TO_TEST:
    print(f"\n{'─'*60}")
    print(f"  {leader_name} → {follower_name}")
    print(f"{'─'*60}")

    lc = ohlc[leader_name]["close"]
    fc = ohlc[follower_name]["close"]
    leader_scale = 10000 if leader_name in ("EURUSD", "GBPUSD", "AUDUSD", "NZDUSD") else 100

    l_ret = np.diff(lc)
    f_ret = np.diff(fc)

    for lookback in [10, 20, 30]:
        beta = rolling_beta(l_ret, f_ret, lookback)
        
        for hold_bars in [2, 3, 5, 10, 20, 30]:
            max_hold = min(hold_bars, n - lookback - 2)
            
            # Compute deficits and cumulative catchups
            deficits = []
            cum_catchups = []
            
            for i in range(lookback, n - max_hold - 1):
                exp_ret = beta[i] * l_ret[i-1]
                act_ret = f_ret[i-1]
                deficit = exp_ret - act_ret
                
                deficits.append(deficit * leader_scale)
                
                # Cumulative follower return over hold period
                cum_ret = np.sum(f_ret[i:i+max_hold])
                cum_catchups.append(cum_ret * leader_scale)
            
            deficits = np.array(deficits)
            cum_catchups = np.array(cum_catchups)
            
            if len(deficits) == 0:
                continue
            
            deficit_std = np.std(deficits)
            
            for z_thresh in [1.0, 1.5, 2.0]:
                sig = deficits > z_thresh * deficit_std
                ns = np.sum(sig)
                if ns < 20:
                    continue
                
                avg_catchup = np.mean(cum_catchups[sig])
                correct = np.sum(cum_catchups[sig] > 0)
                wr = correct / ns
                
                print(f"  lb={lookback} hold={hold_bars} z>{z_thresh}: "
                      f"WR={wr:.1%}({ns}) avg_catchup={avg_catchup:.2f}pips")

print()
print("=" * 70)
print("KEY FINDING: Does catchup accumulate?")
print("=" * 70)
print()
print("If avg_catchup grows with hold_bars → deficit is structural")
print("  (follower slowly converges over multiple bars)")
print("If avg_catchup stays flat → deficit is mostly noise")
print("  (one bar of catchup, then random walk)")
print()
print("For $20-30 profit on 1 lot, need 2-5 pips net catchup")
print("  (after 0.3-1.0 pip spread cost)")
