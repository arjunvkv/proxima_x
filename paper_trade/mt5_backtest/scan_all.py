"""Scan all bars for z-score extremes across entire dataset."""
import numpy as np
import pandas as pd

for pair in ['audusd', 'euraud', 'gbpaud']:
    data = np.load(f'paper_trade/mt5_backtest/fundednext_{pair}_m1.npy', allow_pickle=True)
    df = pd.DataFrame(data)
    closes = df['close'].values
    n = len(closes)
    
    z_window = 50
    close_buf = np.array([])
    z_abs_max = 0
    z_max_at = 0
    
    for i in range(n):
        close_buf = np.append(close_buf, closes[i])
        if len(close_buf) < z_window + 2:
            continue
        rets = np.diff(close_buf[-(z_window+2):])
        cur_ret = rets[-1]
        mean = np.mean(rets[:-1])
        var = np.var(rets[:-1], ddof=1)
        if var < 1e-14:
            continue
        z = (cur_ret - mean) / np.sqrt(var)
        if abs(z) > z_abs_max:
            z_abs_max = abs(z)
            z_max_at = i
    
    print(f"{pair.upper():>8}: max|z|={z_abs_max:.2f} at bar {z_max_at} ({z_max_at/n*100:.1f}% into data)")
    
    # Count z-score thresholds
    for thresh in [3.0, 3.5, 4.0, 5.0, 6.0]:
        count = 0
        close_buf = np.array([])
        for i in range(n):
            close_buf = np.append(close_buf, closes[i])
            if len(close_buf) < z_window + 2:
                continue
            rets = np.diff(close_buf[-(z_window+2):])
            cur_ret = rets[-1]
            mean = np.mean(rets[:-1])
            var = np.var(rets[:-1], ddof=1)
            if var < 1e-14:
                continue
            z = (cur_ret - mean) / np.sqrt(var)
            if abs(z) >= thresh:
                count += 1
        print(f"  |z|>={thresh:.1f}: {count} occurrences")
