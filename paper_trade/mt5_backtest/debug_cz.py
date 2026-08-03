"""Debug Challenge-Z on all 3 pairs."""
import numpy as np
import pandas as pd

z_window = 50
pairs = ['audusd', 'euraud', 'gbpaud']

for pair in pairs:
    data = np.load(f'paper_trade/mt5_backtest/fundednext_{pair}_m1.npy', allow_pickle=True)
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)

    n = len(df)
    days = (df['time'].max() - df['time'].min()).total_seconds() / 86400

    print(f"\n{'='*70}")
    print(f"{pair.upper()} ({n} bars, {days:.1f} days)")
    print(f"Spread: med={df['spread'].median()}, p90={df['spread'].quantile(0.90)}, p99={df['spread'].quantile(0.99)}")

    # Z-score counts
    print(f"\nZ-score trigger frequency:")
    for threshold in [4.0, 3.5, 3.0, 2.5]:
        count = 0
        close_buf = np.array([])
        for i in range(n):
            c = df['close'].iloc[i]
            close_buf = np.append(close_buf, c)
            if len(close_buf) < z_window + 2:
                continue
            rets = np.diff(close_buf[-(z_window+2):])
            cur_ret = rets[-1]
            mean = np.mean(rets[:-1])
            var = np.var(rets[:-1], ddof=1)
            if var < 1e-14:
                continue
            z = (cur_ret - mean) / np.sqrt(var)
            if abs(z) >= threshold:
                count += 1
        freq = count / days
        print(f"  |z| >= {threshold:.1f}: {count} ({freq:.1f}/day)")

    # Mean reversion test at z >= 3.5
    print(f"\nMean reversion: |z| >= 3.5, enter at open, exit at close+N")
    for hold in [5, 10, 15, 20]:
        long_pnl = 0.0
        short_pnl = 0.0
        long_trades = 0
        short_trades = 0
        close_buf = np.array([])
        for i in range(n):
            c = df['close'].iloc[i]
            close_buf = np.append(close_buf, c)
            if len(close_buf) < z_window + 2:
                continue
            rets = np.diff(close_buf[-(z_window+2):])
            cur_ret = rets[-1]
            mean = np.mean(rets[:-1])
            var = np.var(rets[:-1], ddof=1)
            if var < 1e-14:
                continue
            z = (cur_ret - mean) / np.sqrt(var)
            if z <= -3.5:
                exit_idx = min(i + hold, n - 1)
                entry = df['open'].iloc[i]
                exit_p = df['close'].iloc[exit_idx]
                long_pnl += (exit_p - entry) * 100000 * 0.5
                long_trades += 1
            elif z >= 3.5:
                exit_idx = min(i + hold, n - 1)
                entry = df['open'].iloc[i]
                exit_p = df['close'].iloc[exit_idx]
                short_pnl += (entry - exit_p) * 100000 * 0.5
                short_trades += 1
        total = long_pnl + short_pnl
        print(f"  Hold {hold:>2d}: LONG=${long_pnl:>+8.2f} ({long_trades}t)  "
              f"SHORT=${short_pnl:>+8.2f} ({short_trades}t)  "
              f"TOTAL=${total:>+8.2f}")

    # Try with spread limit filter at z >= 3.5
    print(f"\nWith spread filter (sprd <= 10): |z| >= 3.5")
    for hold in [5, 10, 15]:
        long_pnl = 0.0
        short_pnl = 0.0
        long_trades = 0
        short_trades = 0
        close_buf = np.array([])
        for i in range(n):
            c = df['close'].iloc[i]
            close_buf = np.append(close_buf, c)
            if len(close_buf) < z_window + 2:
                continue
            rets = np.diff(close_buf[-(z_window+2):])
            cur_ret = rets[-1]
            mean = np.mean(rets[:-1])
            var = np.var(rets[:-1], ddof=1)
            if var < 1e-14:
                continue
            z = (cur_ret - mean) / np.sqrt(var)
            if abs(z) >= 3.5 and df['spread'].iloc[i] <= 10:
                if z <= -3.5:
                    exit_idx = min(i + hold, n - 1)
                    entry = df['open'].iloc[i]
                    exit_p = df['close'].iloc[exit_idx]
                    long_pnl += (exit_p - entry) * 100000 * 0.5
                    long_trades += 1
                elif z >= 3.5:
                    exit_idx = min(i + hold, n - 1)
                    entry = df['open'].iloc[i]
                    exit_p = df['close'].iloc[exit_idx]
                    short_pnl += (entry - exit_p) * 100000 * 0.5
                    short_trades += 1
        total = long_pnl + short_pnl
        print(f"  Hold {hold:>2d}: LONG=${long_pnl:>+8.2f} ({long_trades}t)  "
              f"SHORT=${short_pnl:>+8.2f} ({short_trades}t)  "
              f"TOTAL=${total:>+8.2f}")

    # Try z >= 4.0
    print(f"\nz >= 4.0, sprd <= 10:")
    for hold in [5, 10, 15]:
        total_pnl = 0.0
        total_trades = 0
        close_buf = np.array([])
        for i in range(n):
            c = df['close'].iloc[i]
            close_buf = np.append(close_buf, c)
            if len(close_buf) < z_window + 2:
                continue
            rets = np.diff(close_buf[-(z_window+2):])
            cur_ret = rets[-1]
            mean = np.mean(rets[:-1])
            var = np.var(rets[:-1], ddof=1)
            if var < 1e-14:
                continue
            z = (cur_ret - mean) / np.sqrt(var)
            if abs(z) >= 4.0 and df['spread'].iloc[i] <= 10:
                exit_idx = min(i + hold, n - 1)
                entry = df['open'].iloc[i]
                exit_p = df['close'].iloc[exit_idx]
                if z <= -4.0:
                    total_pnl += (exit_p - entry) * 100000 * 0.5
                else:
                    total_pnl += (entry - exit_p) * 100000 * 0.5
                total_trades += 1
        print(f"  Hold {hold:>2d}: ${total_pnl:>+8.2f} ({total_trades}t)")

    # NOW: test with the ORIGINAL sim_recon logic to see if THAT works
    # (which uses trailing stop, not fixed hold)
    print(f"\nORIGINAL trailing-stop logic (z >= 3.5, sprd <= 10):")
