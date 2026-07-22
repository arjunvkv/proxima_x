"""Peek at raw tick data structure."""
import pandas as pd
from pathlib import Path

fn = Path(r'C:\Trading\Agentic_Trading\proxima_x\data\exness_ticks\EURUSD_Raw_Spread_2025_10.zip')
d = pd.read_csv(fn, names=['E','S','Ts','B','A'], skiprows=1, header=None, nrows=20,
    dtype={'Ts':str,'B':float,'A':float})
for i, r in d.iterrows():
    sprd_raw = r['A']-r['B']
    sprd_pips = sprd_raw * 10000
    mid = (r['B']+r['A'])/2
    print(f'{i:2d}  B={r["B"]:.6f}  A={r["A"]:.6f}  mid={mid:.5f}  sprd_raw={sprd_raw:.6f}  sprd_pips={sprd_pips:.4f}')

# Check count of zero-spread ticks
print()
for pair in ['EURUSD','EURJPY','GBPJPY']:
    fn2 = Path(rf'C:\Trading\Agentic_Trading\proxima_x\data\exness_ticks\{pair}_Raw_Spread_2025_10.zip')
    d2 = pd.read_csv(fn2, names=['E','S','Ts','B','A'], skiprows=1, header=None,
        dtype={'Ts':str,'B':float,'A':float})
    sprd = (d2['A']-d2['B']) * 10000
    zero_count = (sprd <= 0.001).sum()
    total = len(sprd)
    print(f'{pair}: zero-spread ticks = {zero_count}/{total} = {100*zero_count/total:.1f}%')
