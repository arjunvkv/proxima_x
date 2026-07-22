"""Dealer Acceptance Transition — volatility regime analysis.
Edge may be about activity/volatility regime, not direction.
"""
import numpy as np
import pandas as pd

pair = 'EURJPY'
df = pd.read_parquet(f'data/market/{pair}.parquet').sort_values('timestamp')
n = len(df)

prices = df['close'].values
highs = df['high'].values
lows = df['low'].values
volumes = df['volume'].values

# Features
ranges = (highs - lows) / prices * 10000
norm_vol = volumes / np.mean(volumes)

window = 60
range_pct = np.full(n, np.nan)
vol_pct = np.full(n, np.nan)
for i in range(window, n):
    r_window = ranges[i-window:i]
    v_window = norm_vol[i-window:i]
    range_pct[i] = np.sum(ranges[i] >= r_window) / window
    vol_pct[i] = np.sum(norm_vol[i] >= v_window) / window

print(f"{pair}: {n} bars, mean range={np.mean(ranges):.2f} bps")

# ================================================================
# Idea 1: After stress, future volatility regime is predictable
# ================================================================
print("\n" + "=" * 60)
print("IDEA 1: Post-stress volatility prediction")
print("=" * 60)

for stress_rp in [0.90, 0.95]:
    for stress_vp in [0.80, 0.90]:
        events = []
        in_stress = False
        for i in range(window, n - 1):
            if range_pct[i] >= stress_rp and norm_vol[i] >= stress_vp and not in_stress:
                events.append(i)
                in_stress = True
            elif not (range_pct[i] >= stress_rp and norm_vol[i] >= stress_vp):
                in_stress = False

        if len(events) < 5:
            continue

        # For each stress event, check subsequent 60-min range percentile
        post_ranges = []
        for e in events[:50]:
            if e + 60 >= n:
                continue
            post_60_range = np.mean(ranges[e:e+60])
            baseline_range = np.mean(ranges[max(0,e-60):e])
            post_ranges.append(post_60_range / baseline_range if baseline_range > 0 else 1)

        avg_ratio = np.mean(post_ranges)
        print(f"  Stress>P{stress_rp*100:.0f} Vol>P{stress_vp*100:.0f}: {len(events)} events, post/baseline range ratio={avg_ratio:.2f}")

# ================================================================
# Idea 2: The "quiet before storm" — predict stress from extreme calm
# ================================================================
print("\n" + "=" * 60)
print("IDEA 2: Quiet-before-storm — does extreme calm predict stress?")
print("=" * 60)

for quiet_rp in [0.10, 0.15, 0.20]:
    for fwd_window in [30, 60, 120]:
        results = []
        for i in range(window, n - fwd_window):
            if range_pct[i] is not None and not np.isnan(range_pct[i]) and range_pct[i] <= quiet_rp:
                # Is there a stress event in the next fwd_window bars?
                fwd_ranges = ranges[i:i+fwd_window]
                fwd_max_range = np.max(fwd_ranges)
                stress_coming = np.any(range_pct[i:i+fwd_window] >= 0.90) if i+fwd_window < n else False
                results.append(stress_coming)

        if len(results) < 10:
            continue
        stress_rate = np.mean(results) * 100
        # Baseline: how often is the market stressed in any 60-bar window?
        baseline_stress_rate = np.mean([True if not np.isnan(range_pct[i]) and range_pct[i] >= 0.90 else False for i in range(window, n)]) * 100
        print(f"  Range<P{quiet_rp*100:.0f} → stress in {fwd_window}min: {stress_rate:.0f}% (baseline={baseline_stress_rate:.0f}%)")

# ================================================================
# Idea 3: After recovery, do successive bars have different directional efficiency?
# i.e., is the first bar after stress more directional than average?
# ================================================================
print("\n" + "=" * 60)
print("IDEA 3: Post-recovery directional efficiency")
print("=" * 60)

for stress_rp in [0.90, 0.95]:
    events = []
    in_stress = False
    for i in range(window, n - 1):
        if range_pct[i] >= stress_rp and norm_vol[i] >= 0.80 and not in_stress:
            events.append(i)
            in_stress = True
        elif not (range_pct[i] >= stress_rp and norm_vol[i] >= 0.80):
            in_stress = False

    if len(events) < 5:
        continue

    # Check: is the first 5-min return after stress larger than random 5-min returns?
    post_rets = []
    random_rets = []
    for e in events:
        if e + 5 >= n:
            continue
        post_rets.append(abs(prices[e+5] / prices[e] - 1))
        # Random 5-min return
        rand_i = np.random.randint(window, n - 5)
        random_rets.append(abs(prices[rand_i+5] / prices[rand_i] - 1))

    avg_post = np.mean(post_rets) * 10000
    avg_random = np.mean(random_rets) * 10000
    ratio = avg_post / avg_random if avg_random > 0 else 0
    print(f"  Stress>P{stress_rp*100:.0f}: post-stress 5m move={avg_post:.2f}bps vs random={avg_random:.2f}bps (ratio={ratio:.1f}x)")
