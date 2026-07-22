"""Verify actual spread costs from tick data."""
import numpy as np, pandas as pd, time
from pathlib import Path

TICK_DIR = Path(__file__).resolve().parents[3] / 'data' / 'exness_ticks'
t0 = time.time()

for pair in ['EURUSD', 'EURJPY', 'GBPJPY']:
    for y,m in [(2025,10),(2025,11),(2025,12)]:
        fn = TICK_DIR / f'{pair}_Raw_Spread_{y}_{m:02d}.zip'
        d = pd.read_csv(fn, compression='zip',
            names=['E','S','Ts','B','A'], skiprows=1, header=None,
            dtype={'Ts':str,'B':np.float64,'A':np.float64})
        spread = (d['A'] - d['B']).values
        spread_pips = spread * (10000 if pair == 'EURUSD' else 100)
        p50 = np.percentile(spread_pips, 50)
        p25 = np.percentile(spread_pips, 25)
        p75 = np.percentile(spread_pips, 75)
        mean = np.mean(spread_pips)
        print(f"  {pair} {y}-{m:02d}: mean={mean:.2f}p  median={p50:.2f}p  Q25={p25:.2f}p  Q75={p75:.2f}p "
              f"  ticks={len(d):,d}")
print(f"Time: {time.time()-t0:.1f}s")
