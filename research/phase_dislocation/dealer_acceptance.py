"""Dealer Acceptance Transition — liquidity regime shift edge.
When the market transitions from dealer protection mode (wide range, high volume)
back to normal (compressed range, normal volume), the first minutes after restoration
have exploitable properties.

Approximations using OHLC:
- Range = high - low (volatility/spread proxy)
- Volume = tick volume (activity proxy)
- Stress = range percentile > threshold AND volume percentile > threshold
- Recovery = range compresses AND volume normalizes
"""
import numpy as np
import pandas as pd
import os

# Load all pairs from 30-day market data
mdir = 'data/market'
files = [f for f in os.listdir(mdir) if f.endswith('.parquet') and f not in ('NAS100.parquet', 'XAUUSD.parquet')]
pairs_in_order = sorted([f.replace('.parquet', '') for f in files if f.replace('.parquet', '') not in ('NAS100', 'XAUUSD')])

print(f"Pairs: {len(pairs_in_order)}")

# For testing first: focus on EURJPY since we know its behavior best
test_pairs = ['EURJPY', 'EURUSD', 'USDJPY', 'GBPJPY', 'AUDUSD', 'GBPUSD', 'NZDUSD']

all_results = []

for pair in test_pairs:
    f = f'{mdir}/{pair}.parquet'
    if not os.path.exists(f):
        continue
    df = pd.read_parquet(f)
    time_col = 'timestamp' if 'timestamp' in df.columns else 'time'
    df = df.sort_values(time_col)
    n = len(df)
    if n < 100:
        continue

    prices = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    volumes = df['volume'].values
    # Normalize volume
    norm_vol = volumes / np.mean(volumes)

    # Compute 1-min returns and range
    rets = np.diff(np.log(prices))
    ranges = (highs - lows) / prices * 10000  # in bps
    range_pct = np.zeros(n)
    range_pct[:] = np.nan

    # Range percentile (rolling 60-min window)
    window = 60
    for i in range(window, n):
        r_window = ranges[i-window:i]
        range_pct[i] = np.sum(ranges[i] >= r_window) / window

    print(f"\n{pair}: {n} bars")

    # ================================================================
    # TEST 1: After stress event, first N minutes of recovery
    # ================================================================
    print(f"  STRESS → RECOVERY transition:")

    for stress_range_pct in [0.90, 0.95]:
        for stress_vol_pct in [0.80, 0.90]:
            for recovery_win in [10, 20, 30]:
                # Find stress events
                in_stress = False
                stress_end = 0
                stress_starts = []
                stress_ends = []

                for i in range(window, n - 1):
                    curr_stress = (
                        (range_pct[i] is not None and not np.isnan(range_pct[i]) and range_pct[i] >= stress_range_pct)
                        and (norm_vol[i] >= stress_vol_pct)
                    )

                    if curr_stress and not in_stress:
                        stress_starts.append(i)
                        in_stress = True
                    elif not curr_stress and in_stress:
                        stress_ends.append(i)
                        in_stress = False

                if in_stress:
                    stress_ends.append(n - 1)

                # For each stress event, check recovery
                for start, end in zip(stress_starts, stress_ends):
                    recovery_point = end
                    if recovery_point + recovery_win >= n:
                        continue

                    # Forward returns after recovery
                    fwd_rets = []
                    for fw in [5, 10, 15, 30]:
                        if recovery_point + fw >= n:
                            continue
                        fwd_ret = prices[recovery_point + fw] / prices[recovery_point] - 1
                        fwd_rets.append(fwd_ret)

                    # Range after recovery
                    post_range = np.mean(ranges[recovery_point:min(recovery_point + recovery_win, n)])

                    all_results.append({
                        'pair': pair,
                        'stress_dur': end - start,
                        'range_at_stress': ranges[end],
                        'post_range': post_range,
                        'range_shock_pct': range_pct[min(end, n-1)],
                        'rets_5m': fwd_rets[0] if len(fwd_rets) > 0 else 0,
                        'rets_10m': fwd_rets[1] if len(fwd_rets) > 1 else 0,
                        'rets_15m': fwd_rets[2] if len(fwd_rets) > 2 else 0,
                        'rets_30m': fwd_rets[3] if len(fwd_rets) > 3 else 0,
                        'stress_range_pct': stress_range_pct,
                        'stress_vol_pct': stress_vol_pct,
                        'rec_win': recovery_win,
                    })

rd = pd.DataFrame(all_results)
if len(rd) == 0:
    print("No stress events found")
else:
    print(f"\nTotal events: {len(rd)}")
    print(f"  Stress duration range: {rd['stress_dur'].min()}-{rd['stress_dur'].max()} min")
    print(f"  Mean: {rd['stress_dur'].mean():.0f} min")

    # Directional efficiency: are post-recovery moves more directional?
    print(f"\n  Post-recovery returns:")
    for fwd, col in [(5, 'rets_5m'), (10, 'rets_10m'), (15, 'rets_15m'), (30, 'rets_30m')]:
        vals = rd[col].values
        abs_mean = np.mean(np.abs(vals)) * 10000  # bps
        abs_median = np.median(np.abs(vals)) * 10000
        pos_pct = np.mean(vals > 0) * 100
        neg_pct = np.mean(vals < 0) * 100
        print(f"    {fwd:>2d}m: avg|move|={abs_mean:.1f}bps, median|move|={abs_median:.1f}bps, pos={pos_pct:.0f}% neg={neg_pct:.0f}%")

    # Test: does the first move after stress predict continuation?
    print(f"\n  First 5m direction predicts rest of recovery?")
    for threshold in [0.0003, 0.0005, 0.001]:
        sub = rd[abs(rd['rets_5m']) > threshold]
        if len(sub) < 5:
            continue
        # If first 5m is positive, does 5-30m continue positive?
        cont_pos = ((sub['rets_5m'] > 0) & (sub['rets_30m'] > sub['rets_5m'])).mean() * 100
        cont_neg = ((sub['rets_5m'] < 0) & (sub['rets_30m'] < sub['rets_5m'])).mean() * 100
        print(f"    |5m|>{threshold:.4f}: {len(sub)} events, cont_pos={cont_pos:.0f}% cont_neg={cont_neg:.0f}%")

    # Key test: after recovery, is the market more likely to continue the pre-stress direction or reverse?
    print(f"\n  Directional bias by pair:")
    for pair in rd['pair'].unique():
        sub = rd[rd['pair'] == pair]
        for fwd, col in [(5, 'rets_5m'), (15, 'rets_15m'), (30, 'rets_30m')]:
            vals = sub[col].values
            if len(vals) < 5:
                continue
            pos = np.mean(vals > 0) * 100
            abs_avg = np.mean(np.abs(vals)) * 10000
            print(f"    {pair:>7s} {fwd:>2d}m: pos={pos:.0f}% avg|move|={abs_avg:.1f}bps ({len(vals)} events)")
