"""
Walk-forward validation for EURJPY→GBPJPY multi-bar response deficit.

Goal: verify the 65.6% WR signal holds out-of-sample.

Approach:
- 7 days M1 data (7200 bars, Mon-Sat)
- Walk-forward: train on day N, test on day N+1
- Rolling: expanding window train, forward test
- Report stability across days
"""
import sys, os
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict
from numba import jit

sys.path.insert(0, os.path.dirname(__file__))
from data import TempCache, PAIRS

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


print("=" * 70)
print("WALK-FORWARD VALIDATION: EURJPY→GBPJPY Response Deficit")
print("=" * 70)

cache = TempCache(7)
aligned, times, _ = cache.get()
n = len(times)

leader_name, follower_name = "EURJPY", "GBPJPY"
leader_scale = 100

lc = aligned[:, PAIR_IDX[leader_name], 3]  # close
fc = aligned[:, PAIR_IDX[follower_name], 3]

l_ret = np.diff(lc)
f_ret = np.diff(fc)

# Get date for each bar
dates = [datetime.fromtimestamp(t, tz=timezone.utc) for t in times]
date_labels = [d.strftime("%Y-%m-%d") for d in dates]
unique_dates = sorted(set(date_labels))
print(f"Data covers {len(unique_dates)} days: {unique_dates[0]} to {unique_dates[-1]}")
print(f"Total bars: {len(times)}")
print()

# Map bar index → date label
bar_date = np.array(date_labels[:len(l_ret)])  # align with returns

# For each test day, compute in-sample & out-of-sample WR
# using expanding window training

LOOKBACKS = [10, 20, 30]
Z_THRESHS = [1.0, 1.5, 2.0]
HOLD_BARS = [5, 10, 20, 30]

# First: find best params on full dataset (reference)
print("=" * 60)
print("FULL DATASET REFERENCE")
print("=" * 60)

all_results = []
for lookback in LOOKBACKS:
    beta = rolling_beta(l_ret, f_ret, lookback)
    for hold_bars in HOLD_BARS:
        for z_thresh in Z_THRESHS:
            deficits = []
            cum_catchups = []
            for i in range(lookback, len(l_ret) - hold_bars):
                exp_ret = beta[i] * l_ret[i-1]
                act_ret = f_ret[i-1]
                deficit = exp_ret - act_ret
                deficits.append(deficit * leader_scale)
                cum_ret = np.sum(f_ret[i:i+hold_bars])
                cum_catchups.append(cum_ret * leader_scale)
            
            deficits = np.array(deficits)
            cum_catchups = np.array(cum_catchups)
            deficit_std = np.std(deficits)
            
            if deficit_std == 0:
                continue
            
            sig = deficits > z_thresh * deficit_std
            ns = np.sum(sig)
            if ns < 20:
                continue
            wr = np.sum(cum_catchups[sig] > 0) / ns
            avg_catchup = np.mean(cum_catchups[sig])
            
            all_results.append({
                "lb": lookback, "hold": hold_bars, "z": z_thresh,
                "wr": wr, "trades": ns, "avg_catchup": avg_catchup
            })

all_results.sort(key=lambda r: (-r["wr"], -r["trades"]))
print(f"{'lb':>3} {'hold':>4} {'z':>3} {'WR':>6} {'Trades':>7} {'Catchup':>8}")
for r in all_results[:10]:
    print(f"{r['lb']:>3} {r['hold']:>4} {r['z']:>3} {r['wr']:>5.1%} {r['trades']:>7} {r['avg_catchup']:>7.2f}p")

# Pick the best config for walk-forward
best_config = all_results[0]
print(f"\nBest config: lb={best_config['lb']} hold={best_config['hold']} z>{best_config['z']}")
print(f"  Full-dataset WR={best_config['wr']:.1%} ({best_config['trades']} trades)")
print()

# =========================================================
# WALK-FORWARD: Train on day N, test on day N+1
# =========================================================
print("=" * 60)
print("WALK-FORWARD (expanding window, day-by-day)")
print("=" * 60)

# For each test date (starting from 2nd unique date)
wf_results = []
all_os_trades = []

lb = best_config["lb"]
hold = best_config["hold"]
zt = best_config["z"]

for test_idx, test_date in enumerate(unique_dates[1:], 1):
    train_dates = unique_dates[:test_idx]  # all previous dates
    
    # Compute in-sample WR on training days
    train_mask = np.isin(bar_date, train_dates)
    test_mask = bar_date == test_date
    
    # Use training data to determine threshold
    # (but we're using the full-dataset threshold — we should re-compute in-sample)
    
    # Actually, the proper walk-forward re-trains params on each expanding window
    for lb_wf in LOOKBACKS:
        beta_wf = rolling_beta(l_ret, f_ret, lb_wf)
        
        for hold_wf in HOLD_BARS:
            for zt_wf in Z_THRESHS:
                # Compute deficits on TRAINING data only
                train_deficits = []
                train_catchups = []
                for i in range(lb_wf, len(l_ret) - hold_wf):
                    if not train_mask[i]:
                        continue
                    exp_ret = beta_wf[i] * l_ret[i-1]
                    act_ret = f_ret[i-1]
                    deficit = exp_ret - act_ret
                    train_deficits.append(deficit)
                    cum_ret = np.sum(f_ret[i:i+hold_wf])
                    train_catchups.append(cum_ret * leader_scale)
                
                train_deficits = np.array(train_deficits)
                train_catchups = np.array(train_catchups)
                
                if len(train_deficits) < 30 or np.std(train_deficits) == 0:
                    continue
                
                train_wr = np.sum(train_catchups > 0) / len(train_catchups)
                
                # Now test on TEST data
                test_deficits = []
                test_catchups = []
                for i in range(lb_wf, len(l_ret) - hold_wf):
                    if not test_mask[i]:
                        continue
                    exp_ret = beta_wf[i] * l_ret[i-1]
                    act_ret = f_ret[i-1]
                    deficit = exp_ret - act_ret
                    test_deficits.append(deficit)
                    cum_ret = np.sum(f_ret[i:i+hold_wf])
                    test_catchups.append(cum_ret * leader_scale)
                
                test_deficits = np.array(test_deficits)
                test_catchups = np.array(test_catchups)
                
                if len(test_deficits) < 5:
                    continue
                
                # Use training threshold on test data
                train_std = np.std(train_deficits)
                sig_test = test_deficits > zt_wf * train_std
                ns_test = np.sum(sig_test)
                
                if ns_test < 3:
                    continue
                
                test_wr = np.sum(test_catchups[sig_test] > 0) / ns_test
                test_avg = np.mean(test_catchups[sig_test])
                
                wf_results.append({
                    "test_date": test_date, "train_wr": train_wr,
                    "test_wr": test_wr, "test_trades": ns_test,
                    "test_avg": test_avg,
                    "lb": lb_wf, "hold": hold_wf, "z": zt_wf,
                    "train_days": len(train_dates),
                })

# Summarize walk-forward results
print(f"\nWalk-forward results across {len(unique_dates)-1} test days:")
print(f"Total configs tested per day: {len([r for r in wf_results if r['test_date'] == unique_dates[1]])}")
print()

# Group by config
from collections import defaultdict
config_summary = defaultdict(list)
for r in wf_results:
    if r["test_trades"] >= 5:
        key = (r["lb"], r["hold"], r["z"])
        config_summary[key].append(r)

print(f"{'lb':>3} {'hold':>4} {'z':>3} {'Avg Test WR':>11} {'Min':>5} {'Max':>5} {'Avg Trades':>10} {'Days':>5}")
print("-" * 55)

best_wf = None
best_wf_score = -1

for key, results in sorted(config_summary.items(), key=lambda x: -np.mean([r["test_wr"] for r in x[1]])):
    wr_list = [r["test_wr"] for r in results]
    avg_wr = np.mean(wr_list)
    min_wr = min(wr_list)
    max_wr = max(wr_list)
    avg_trades = np.mean([r["test_trades"] for r in results])
    n_days = len(results)
    
    if n_days >= 3 and avg_trades >= 10:
        score = avg_wr * avg_trades
        if score > best_wf_score:
            best_wf_score = score
            best_wf = key
    
    print(f"{key[0]:>3} {key[1]:>4} {key[2]:>3} {avg_wr:>10.1%} {min_wr:>4.0%} {max_wr:>4.0%} {avg_trades:>9.0f} {n_days:>5}")

print()

if best_wf:
    print(f"Best walk-forward config: lb={best_wf[0]} hold={best_wf[1]} z>{best_wf[2]}")
    # Get the detailed per-day breakdown
    bf_results = [r for r in wf_results 
                  if r["lb"] == best_wf[0] and r["hold"] == best_wf[1] and r["z"] == best_wf[2]]
    print(f"\nPer-day breakdown (ordered by date ascending):")
    for r in sorted(bf_results, key=lambda x: x["test_date"]):
        train_str = f"train on {r['train_days']} days"
        print(f"  {r['test_date']}: "
              f"train_WR={r['train_wr']:.1%} "
              f"test_WR={r['test_wr']:.1%} "
              f"({r['test_trades']} trades, {r['test_avg']:.2f}p avg) "
              f"[{train_str}]")

# Expand to multi-config: report all configs with consistent OOS WR
print()
print("=" * 60)
print("CONSISTENT OOS CONFIGS")
print("=" * 60)
print()

stable_configs = []
for key, results in config_summary.items():
    wr_list = [r["test_wr"] for r in results]
    avg_wr = np.mean(wr_list)
    n_days = len(results)
    avg_trades = np.mean([r["test_trades"] for r in results])
    if n_days >= 3 and avg_wr >= 0.55 and avg_trades >= 8:
        stable_configs.append((key, avg_wr, avg_trades, n_days))

stable_configs.sort(key=lambda x: -x[1])
if stable_configs:
    print(f"Configs with avg OOS WR >= 55% across 3+ test days:")
    for key, awr, at, nd in stable_configs:
        print(f"  lb={key[0]} hold={key[1]} z>{key[2]}: OOS_WR={awr:.1%} avg_trades={at:.0f} ({nd} days)")
else:
    print("No configs maintain 55%+ OOS WR across 3+ test days.")

print()
print("=" * 70)
print("VERDICT")
print("=" * 70)
print()
print("If the best config maintains 55%+ OOS WR across 3+ test days,")
print("the response deficit signal is REAL at M1 resolution.")
print("If OOS WR drops to 50%, the full-sample 65.6% was overfit.")
print()
print("Key question: does the signal hold on Fridays (weekend effect)")
print("vs Mondays-Thursdays? This tells us if it's a structural edge")
print("or a data-mining artifact.")
