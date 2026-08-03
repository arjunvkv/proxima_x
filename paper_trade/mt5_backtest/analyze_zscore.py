"""Analyze M1 z-score distribution on FTMO data to find right threshold."""
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import datetime

pairs = ["AUDNZD", "EURAUD", "EURNZD", "GBPAUD"]

if not mt5.initialize():
    print("MT5 init failed:", mt5.last_error())
    exit(1)

print(f"Connected: {mt5.terminal_info().name}")

def compute_z_scores(closes, z_window=50):
    rets = np.diff(closes)
    scores = np.full(len(closes), np.nan)
    for i in range(z_window, len(rets)):
        cur = rets[i]
        past = rets[i - z_window:i]
        mu = np.mean(past)
        sigma = np.std(past, ddof=1)
        if sigma < 1e-14:
            scores[i + 1] = 0.0
        else:
            scores[i + 1] = (cur - mu) / sigma
    return scores

from_dt = datetime(2025, 6, 1)
to_dt = datetime(2026, 7, 26)

for pair in pairs:
    print(f"\n{'='*60}")
    print(f"  {pair}")
    print(f"{'='*60}")

    rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M1, from_dt, to_dt)
    if rates is None or len(rates) == 0:
        print(f"  No data")
        continue

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    closes = df['close'].values
    print(f"  Bars: {len(closes):,}  |  {df.time.iloc[0].date()} to {df.time.iloc[-1].date()}")

    zs = compute_z_scores(closes, z_window=50)
    df['z'] = zs
    valid = df.dropna(subset=['z'])

    # Asian hours (0-7 UTC)
    df['hour'] = df['time'].dt.hour
    asia = df[df['hour'].between(0, 6)]
    asia_valid = asia.dropna(subset=['z'])

    for label, subset in [("ALL HOURS", valid), ("ASIA 0-6 UTC", asia_valid)]:
        zvals = subset['z'].values
        n = len(zvals)
        print(f"\n  [{label}] n={n:,}")
        print(f"  {'Threshold':<12} {'Count':>8} {'Pct':>8} {'/day':>8}")
        print(f"  {'-'*40}")
        for thresh in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
            above = np.sum(np.abs(zvals) >= thresh)
            pct = 100.0 * above / n
            freq_day = pct * 1440 / 100
            print(f"  |z|>={thresh:<3.1f}   {above:>8,}  {pct:>7.2f}%  {freq_day:>7.1f}")

    # Direction split at various thresholds
    print(f"\n  Direction splits (ASIA):")
    for thresh in [2.0, 2.5, 3.0, 3.5]:
        extreme = np.abs(asia_valid['z'].values) >= thresh
        if np.sum(extreme) > 0:
            longs = np.sum(asia_valid['z'].values[extreme] < 0)
            shorts = np.sum(asia_valid['z'].values[extreme] > 0)
            print(f"    |z|>={thresh}: LONG={longs} SHORT={shorts}")

    # Basic stats
    print(f"\n  Stats (ASIA):")
    print(f"    Mean return: {asia_valid['z'].mean():.4f}")
    print(f"    Std:         {asia_valid['z'].std():.4f}")
    print(f"    Skew:        {asia_valid['z'].skew():.4f}")
    print(f"    Kurtosis:    {asia_valid['z'].kurtosis():.4f}")

mt5.shutdown()
print("\nDone")
