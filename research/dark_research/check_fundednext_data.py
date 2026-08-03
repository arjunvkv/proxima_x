"""Check FundedNext data quality for Dark Consensus pairs."""
import numpy as np
import pandas as pd
import os

ROOT = os.path.dirname(__file__)
for pair in ['eurjpy', 'eurusd', 'gbpjpy']:
    fpath = os.path.join(ROOT, f'fundednext_{pair}_m1.npy')
    d = np.load(fpath, allow_pickle=True)
    df = pd.DataFrame(d)
    s = df['spread']
    z = (s == 0).sum()
    print(f'{pair.upper()}: {len(df)} bars, zero_sprd={z} ({z/len(df)*100:.2f}%), med={s.median()}, p90={s.quantile(0.90)}')
    if z > 0:
        zrows = df[s == 0].head(3)
        for _, r in zrows.iterrows():
            print(f'  time={r["time"]}  close={r["close"]:.5f}  sprd={r["spread"]}')
