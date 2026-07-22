"""Dealer Acceptance Transition — refined validation.
Focus on GBPJPY's post-stress positive bias.
"""
import numpy as np
import pandas as pd

pair = 'GBPJPY'
df = pd.read_parquet(f'data/market/{pair}.parquet').sort_values('timestamp')
n = len(df)

prices = df['close'].values
highs = df['high'].values
lows = df['low'].values
volumes = df['volume'].values

overall_ret_30m = np.mean(prices[30:] / prices[:-30] - 1) * 10000
print(f"{pair}: {n} bars over 30 days")
print(f"Overall 30m mean return: {overall_ret_30m:.1f} bps")
print(f"Overall pos/neg ratio: {np.mean(prices[30:] > prices[:-30])*100:.0f}% / {np.mean(prices[30:] < prices[:-30])*100:.0f}%")
print()

ranges = (highs - lows) / prices * 10000
norm_vol = volumes / np.mean(volumes)

window = 60
range_pct = np.full(n, np.nan)
for i in range(window, n):
    r_window = ranges[i-window:i]
    range_pct[i] = np.sum(ranges[i] >= r_window) / window

print(f"Mean 1-min range: {np.mean(ranges):.2f} bps")

all_data = []

for sr_pct in [0.85, 0.90, 0.95]:
    for sv_pct in [0.70, 0.80, 0.90]:
        events = []
        in_stress = False
        stress_start = 0

        for i in range(window, n - 1):
            is_stress = (range_pct[i] >= sr_pct) and (norm_vol[i] >= sv_pct)
            if is_stress and not in_stress:
                stress_start = i
                in_stress = True
            elif not is_stress and in_stress:
                recovery = i
                stress_dur = recovery - stress_start
                if stress_dur >= 1 and stress_dur <= 30:
                    events.append({
                        'stress_start': stress_start,
                        'stress_end': recovery - 1,
                        'stress_dur': stress_dur,
                        'recovery_bar': recovery,
                        'prices_at_recovery': prices[recovery],
                        'range_at_stress': ranges[stress_start:recovery].mean() if recovery > stress_start else ranges[stress_start],
                    })
                in_stress = False

        if not events:
            continue

        for ev in events:
            r = ev['recovery_bar']
            ev['price_recovery'] = prices[r]
            for fwd in [5, 10, 15, 20, 30, 60]:
                if r + fwd >= n:
                    continue
                ev[f'ret_{fwd}m'] = prices[r + fwd] / prices[r] - 1

        ev_df = pd.DataFrame(events)
        n_ev = len(ev_df)

        if n_ev < 5:
            continue

        print(f"  Stress>P{sr_pct*100:.0f} Vol>P{sv_pct*100:.0f}: {n_ev:>3d} events ", end="")
        for fwd in [5, 10, 15, 30]:
            col = f'ret_{fwd}m'
            valid = ev_df[col].dropna()
            if len(valid) < 5:
                continue
            wr = np.mean(valid > 0) * 100
            avg = np.mean(valid) * 10000
            print(f" {fwd:>2d}m: WR={wr:.0f}% avg={avg:+.1f}bps", end="")
        print()

        half = int(len(ev_df) / 2)
        if half > 10:
            first_wr = np.mean(ev_df['ret_15m'].iloc[:half].dropna() > 0) * 100
            second_wr = np.mean(ev_df['ret_15m'].iloc[half:half*2].dropna() > 0) * 100
            print(f"          15m WR: first half={first_wr:.0f}% second half={second_wr:.0f}%", end="")
            drift_15m = prices[15:] / prices[:-15] - 1
            drift_wr = np.mean(drift_15m > 0) * 100
            print(f" (drift WR={drift_wr:.0f}%)")

print()
print("Spread viability check:")
spread_bps = 4 * 0.01 / np.mean(prices) * 10000
print(f"  GBPJPY spread ~4 pips = {spread_bps:.1f} bps")
print(f"  Need avg move > {spread_bps:.1f} bps to be profitable")
