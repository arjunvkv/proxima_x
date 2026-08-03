"""Analyze M1 z-score on FTMO data using copy_rates_from (from a specific date)."""
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

for pair in pairs:
    print(f"\n{'='*60}")
    print(f"  {pair}")
    print(f"{'='*60}")

    mt5.symbol_select(pair, True)

    # Try getting data from start of Asian session on a known date
    start = datetime(2026, 6, 8)
    rates = mt5.copy_rates_from(pair, mt5.TIMEFRAME_M1, start, 50000)
    if rates is None or len(rates) < 1000:
        print(f"  copy_rates_from failed: {mt5.last_error()}")
        continue

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    closes = df['close'].values
    print(f"  Bars: {len(closes):,}  |  {df.time.iloc[0]} to {df.time.iloc[-1]}")

    zs = compute_z_scores(closes, z_window=50)
    df['z'] = zs
    df['hour'] = df['time'].dt.hour
    asia = df[df['hour'].between(0, 6)]
    asia_valid = asia.dropna(subset=['z'])
    all_valid = df.dropna(subset=['z'])

    for label, subset in [("ALL HOURS", all_valid), ("ASIA 0-6 UTC", asia_valid)]:
        zvals = subset['z'].values
        n = len(zvals)
        print(f"\n  [{label}] n={n:,}")
        print(f"  {'Threshold':<12} {'Count':>8} {'Pct':>8} {'/day':>8}")
        print(f"  {'-'*40}")
        for thresh in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
            above = np.sum(np.abs(zvals) >= thresh)
            pct = 100.0 * above / n
            freq_day = pct * 1440 / 100
            print(f"  |z|>={thresh:<3.1f}   {above:>8,}  {pct:>7.2f}%  {freq_day:>7.1f}")

    for thresh in [1.5, 2.0, 2.5, 3.0, 3.5]:
        extreme = np.abs(asia_valid['z'].values) >= thresh
        if np.sum(extreme) > 0:
            longs = np.sum(asia_valid['z'].values[extreme] < 0)
            shorts = np.sum(asia_valid['z'].values[extreme] > 0)
            print(f"\n  Direction |z|>={thresh} ASIA: LONG={longs} SHORT={shorts}")

    zv = asia_valid['z']
    print(f"\n  ASIA stats: mean={zv.mean():.4f} std={zv.std():.4f} "
          f"skew={zv.skew():.4f} kurt={zv.kurtosis():.4f}")

mt5.shutdown()
print("\nDone")
