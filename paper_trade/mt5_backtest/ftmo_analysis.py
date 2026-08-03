"""FTMO z-score entries surviving spread filter."""
import MetaTrader5 as mt5
from datetime import datetime
import pandas as pd
import numpy as np

mt5.initialize()
pairs = ['AUDNZD','EURAUD','EURNZD','GBPAUD']
fwd_from = datetime(2026,6,8)
fwd_to = datetime(2026,7,25)

for pair in pairs:
    mt5.symbol_select(pair, True)
    rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M1, fwd_from, fwd_to)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['hour'] = df['time'].dt.hour

    closes = df['close'].values
    rets = np.diff(closes)
    scores = np.full(len(closes), np.nan)
    for i in range(50, len(rets)):
        cur = rets[i]
        past = rets[i-50:i]
        mu = np.mean(past)
        sigma = np.std(past, ddof=1)
        if sigma >= 1e-14:
            scores[i+1] = (cur - mu) / sigma
    df['z'] = scores

    print(f"\n{'='*60}")
    print(f"  {pair}")
    print(f"{'='*60}")

    for spread_max in [5, 10, 15, 20, 25, 30]:
        # Asian hours 1-6 (skip hour 0, spread is 100+)
        asia = df[(df['hour'].between(1,6)) & (df['spread'] <= spread_max)]
        filtered = asia.dropna(subset=['z'])
        n_total = len(filtered)
        n_z35 = np.sum(np.abs(filtered['z']) >= 3.5)
        n_z30 = np.sum(np.abs(filtered['z']) >= 3.0)
        n_z25 = np.sum(np.abs(filtered['z']) >= 2.5)
        print(f"  spread<={spread_max:>2d}: bars={n_total:>6,} |z|>=2.5={n_z25:>5,} |z|>=3.0={n_z30:>5,} |z|>=3.5={n_z35:>5,}")

    # Also show total without spread filter for comparison
    asia_all = df[df['hour'].between(1,6)].dropna(subset=['z'])
    print(f"  No spread:    bars={len(asia_all):>6,} |z|>=3.5={np.sum(np.abs(asia_all['z'])>=3.5):>5,}")

mt5.shutdown()
