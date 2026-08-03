import numpy as np, os, csv
from datetime import datetime

ROOT = r'C:\Trading\Agentic_Trading\proxima_x\research\dark_research'
OUT = r'C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\custom_data'
os.makedirs(OUT, exist_ok=True)

pairs = ['eurjpy', 'eurusd', 'gbpjpy']
pair_names = ['EURJPY', 'EURUSD', 'GBPJPY']

data = {}
for p in pairs:
    d = np.load(os.path.join(ROOT, f'fundednext_{p}_m1.npy'), allow_pickle=True)
    data[p] = {row['time']: row for row in d}

common = sorted(set(data['eurjpy'].keys()) & set(data['eurusd'].keys()) & set(data['gbpjpy'].keys()))
print(f'Common bars: {len(common)}')

for i, p in enumerate(pairs):
    fpath = os.path.join(OUT, f'FN_{pair_names[i]}.csv')
    with open(fpath, 'w', newline='') as f:
        w = csv.writer(f)
        for ts in common:
            row = data[p][ts]
            dt = datetime.utcfromtimestamp(row['time'])
            date_str = dt.strftime('%Y.%m.%d')
            time_str = dt.strftime('%H:%M')
            if p == 'eurusd':
                o = f'{row["open"]:.5f}'
                h = f'{row["high"]:.5f}'
                l = f'{row["low"]:.5f}'
                c = f'{row["close"]:.5f}'
            else:
                o = f'{row["open"]:.3f}'
                h = f'{row["high"]:.3f}'
                l = f'{row["low"]:.3f}'
                c = f'{row["close"]:.3f}'
            w.writerow([date_str, time_str, o, h, l, c, int(row['tick_volume'])])
    print(f'{pair_names[i]}: {len(common)} bars -> {fpath}')
    with open(fpath) as f:
        lines = f.readlines()
        print(f'  First: {lines[0].strip()}  Last: {lines[-1].strip()}')

print('Done!')
