"""Calm-to-Storm breakout strategy.
After extreme calm (range<P10), the first breakout bar's direction has momentum.
"""
import numpy as np
import pandas as pd
import os

pairs = sorted([f.replace('.parquet', '') for f in os.listdir('data/market') 
                if f.endswith('.parquet') and f.replace('.parquet', '') not in ('NAS100', 'XAUUSD')])

window = 60
all_results = []

for pair in pairs:
    df = pd.read_parquet(f'data/market/{pair}.parquet').sort_values('timestamp')
    n = len(df)
    if n < window + 60:
        continue

    prices = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    volumes = df['volume'].values

    ranges = (highs - lows) / prices * 10000
    norm_vol = volumes / np.mean(volumes)

    # Rolling range percentile
    range_pct = np.full(n, np.nan)
    for i in range(window, n):
        rw = ranges[i-window:i]
        range_pct[i] = np.sum(ranges[i] >= rw) / window

    # Detect calm events
    for calm_thresh in [0.10, 0.15, 0.20]:
        calm_bars = []
        in_calm = False
        calm_start = 0
        calm_len = 0

        for i in range(window, n - 10):
            is_calm = (range_pct[i] <= calm_thresh)
            if is_calm and not in_calm:
                calm_start = i
                calm_len = 1
                in_calm = True
            elif is_calm and in_calm:
                calm_len += 1
            elif not is_calm and in_calm:
                if calm_len >= 3:  # Calm must last at least 3 bars
                    calm_bars.append({'start': calm_start, 'end': i, 'length': calm_len})
                in_calm = False

        for ev in calm_bars:
            breakout = ev['end']  # First non-calm bar after calm period
            if breakout + 30 >= n:
                continue

            # Breakout features
            breakout_range = ranges[breakout]
            breakout_dir = np.sign(prices[breakout] - prices[breakout - 1])
            breakout_vol = norm_vol[breakout]

            # Forward returns
            for fwd in [5, 10, 15, 30]:
                if breakout + fwd >= n:
                    continue
                fwd_ret = prices[breakout + fwd] / prices[breakout] - 1
                continued = (breakout_dir > 0 and fwd_ret > 0) or (breakout_dir < 0 and fwd_ret < 0)

                all_results.append({
                    'pair': pair,
                    'calm_thresh': calm_thresh,
                    'calm_len': calm_len,
                    'breakout_bar': breakout,
                    'breakout_dir': breakout_dir,
                    'breakout_range': breakout_range,
                    'breakout_vol': breakout_vol,
                    'fwd': fwd,
                    'fwd_ret': fwd_ret,
                    'continued': continued,
                })

rd = pd.DataFrame(all_results)
if len(rd) == 0:
    print("No events found")
else:
    print(f"Total calm-to-storm events: {len(rd)}")

    print("\n" + "=" * 60)
    print("CALM-TO-STORM BREAKOUT RESULTS")
    print("=" * 60)

    for calm_thresh in [0.10, 0.15, 0.20]:
        for fwd in [5, 10, 15, 30]:
            sub = rd[(rd['calm_thresh'] == calm_thresh) & (rd['fwd'] == fwd)]
            if len(sub) < 5:
                continue
            wr = np.mean(sub['continued']) * 100
            avg_ret = np.mean(sub['fwd_ret']) * 10000
            abs_avg = np.mean(np.abs(sub['fwd_ret'])) * 10000
            print(f"  Calm<P{calm_thresh*100:.0f} Fwd={fwd:>2d}m: n={len(sub):>4d} cont_WR={wr:.0f}% avg_ret={avg_ret:+.1f}bps avg|move|={abs_avg:.1f}bps")

    # Best per-pair breakdown
    print("\n" + "=" * 60)
    print("BEST PAIR BREAKDOWN")
    print("=" * 60)

    for pair in rd['pair'].unique():
        sub = rd[(rd['pair'] == pair) & (rd['calm_thresh'] == 0.10) & (rd['fwd'] == 15)]
        if len(sub) < 5:
            continue
        wr = np.mean(sub['continued']) * 100
        avg_ret = np.mean(sub['fwd_ret']) * 10000
        print(f"  {pair:>7s}: n={len(sub):>3d} cont_WR={wr:.0f}% avg={avg_ret:+.1f}bps")

    # Spread check per pair
    print("\n" + "=" * 60)
    print("SPREAD CHECK")
    print("=" * 60)
    spreads = {
        'AUDUSD': 1.5, 'EURJPY': 2.5, 'EURUSD': 1.5, 'GBPJPY': 4.0,
        'GBPUSD': 2.0, 'NZDUSD': 2.0, 'USDJPY': 1.8, 'AUDJPY': 2.5,
        'CHFJPY': 3.0, 'EURAUD': 2.5, 'EURCHF': 2.0, 'EURGBP': 2.0,
        'GBPCHF': 3.0, 'NZDCAD': 2.5, 'NZDJPY': 3.0, 'USDCAD': 2.0,
        'USDCHF': 2.0, 'AUDCHF': 2.5,
    }
    for pair in rd['pair'].unique():
        sub = rd[(rd['pair'] == pair) & (rd['calm_thresh'] == 0.10) & (rd['fwd'] == 15)]
        if len(sub) < 5:
            continue
        avg_ret_bps = np.mean(sub['fwd_ret']) * 10000
        sprd = spreads.get(pair, 2.0)
        sprd_bps = sprd * 0.0001 * 10000 if pair not in ('EURJPY', 'GBPJPY', 'USDJPY', 'AUDJPY', 'CHFJPY', 'CADJPY', 'NZDJPY') else sprd * 0.01 * 10000 / np.mean(prices)
        print(f"  {pair:>7s}: avg_15m={avg_ret_bps:+.1f}bps spread={sprd_bps:.1f}bps -> net={avg_ret_bps-sprd_bps:+.1f}bps")
