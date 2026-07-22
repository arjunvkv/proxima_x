"""
Combined DC + 10sMR backtest — matches original research methodology.
DC: 1-min bar level (z > thr, sr > 2.0), enter at bar close, hold 10min.
MR: 10s bar level (z > 3.5), enter next bar, hold 18 bars (3min).
"""
import numpy as np, pandas as pd, time
from pathlib import Path

TICK_DIR = Path(__file__).resolve().parents[3] / 'data' / 'exness_ticks'
MONTHS = [(2025, 10), (2025, 11), (2025, 12)]
COST = {'EURUSD': 0.15, 'EURJPY': 50, 'GBPJPY': 60}
DC_HOLD_MIN = 10
MR_HOLD_BARS = 18

t0 = time.time()

def load_pair(pair):
    dfs = []
    for y, m in MONTHS:
        fn = TICK_DIR / f'{pair}_Raw_Spread_{y}_{m:02d}.zip'
        if not fn.exists(): continue
        d = pd.read_csv(fn, compression='zip',
            names=['E','S','Ts','B','A'], skiprows=1, header=None,
            dtype={'Ts':str,'B':np.float64,'A':np.float64})
        d['Ts'] = pd.to_datetime(d['Ts'].str.replace('Z','',regex=False),
            format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
        dfs.append(d.dropna(subset=['Ts']))
    df = pd.concat(dfs, ignore_index=True).sort_values('Ts').reset_index(drop=True)
    df['MP'] = ((df['B']+df['A'])/2) * 10000
    df['Sprd'] = (df['A']-df['B']) * 10000
    return df.set_index('Ts')

pairs_info = [('EURUSD', 1.5), ('EURJPY', 1.75), ('GBPJPY', 1.75)]
all_data = {}
for name, _ in pairs_info:
    all_data[name] = load_pair(name)
    print(f"  {name}: {len(all_data[name]):,d} ticks ({time.time()-t0:.1f}s)")

all_trades = []

for name, dc_z_thr in pairs_info:
    t = all_data[name]
    cost = COST[name]
    print(f"\n{'='*60}\n{name}\n{'='*60}")

    # --- MR: 10s bars (EURUSD only) ---
    if name == 'EURUSD':
        b10 = t['MP'].resample('10s').agg({'open':'first','close':'last'}).dropna()
        b10['ret'] = b10['close'] - b10['open']
        b10['z'] = (b10['ret'] - b10['ret'].rolling(50).mean()) / b10['ret'].rolling(50).std().clip(1e-8)

        for i in range(50, len(b10) - MR_HOLD_BARS):
            if abs(b10['z'].iloc[i]) > 3.5:
                fwd = b10['ret'].iloc[i+1:i+1+MR_HOLD_BARS].sum()
                dir_ = -1 if b10['z'].iloc[i] > 0 else 1
                all_trades.append({
                    'pair': name, 'type': 'MR', 'pnl': fwd * dir_,
                    'dt': b10.index[i], 'z': float(b10['z'].iloc[i]),
                })
        c = len([x for x in all_trades if x['pair']==name and x['type']=='MR'])
        print(f"  MR: {c} signals")

    # --- DC: 1m bars ---
    b1 = t['MP'].resample('1min').agg({'open':'first','close':'last'}).dropna()
    b1['ret'] = b1['close'] - b1['open']
    b1['v'] = b1['ret'].rolling(20).std()
    b1['z'] = b1['ret'] / b1['v'].clip(1e-8)  # DC original: ret/std, not (ret-mean)/std

    sp_max = t['Sprd'].resample('1min').max()
    sp_med_1m = t['Sprd'].resample('1min').median().fillna(method='ffill')
    b1['rm'] = sp_med_1m.rolling(20, min_periods=1).median().fillna(method='ffill')
    b1['sr'] = sp_max / b1['rm'].clip(1e-8)

    # Forward returns: as in original DC (shift(-N) for N-min forward)
    b1[f'fwd{DC_HOLD_MIN}'] = b1['ret'].rolling(DC_HOLD_MIN).sum().shift(-DC_HOLD_MIN)
    b1['af'] = -np.sign(b1['z'].fillna(0)) * b1[f'fwd{DC_HOLD_MIN}'].fillna(0)

    mask = (b1['z'].abs() > dc_z_thr) & (b1['sr'] > 2.0)
    events = b1[mask].dropna(subset=['af'])
    for dt, row in events.iterrows():
        all_trades.append({
            'pair': name, 'type': 'DC', 'pnl': row['af'],
            'dt': dt, 'z': float(row['z']),
        })
    print(f"  DC: {len(events)} signals")

# Report
print(f"\n{'='*60}\nRESULTS\n{'='*60}")
for name, _ in pairs_info:
    for st in ['MR', 'DC']:
        sub = [x for x in all_trades if x['pair']==name and x['type']==st]
        if not sub: continue
        n = len(sub); tpd = n / 66
        wr = sum(1 for x in sub if x['pnl'] > 0) / n
        avg = np.mean([x['pnl'] for x in sub])
        net = avg - COST[name]
        print(f"  {name} {st:<5s}: n={n:>4d} ({tpd:.1f}/d) WR={wr:.1%} avg={avg:+.2f} net={net:+.2f}")
        pre = [x for x in sub if x['dt'] < pd.Timestamp('2025-12-01')]
        post = [x for x in sub if x['dt'] >= pd.Timestamp('2025-12-01')]
        if pre:
            n1=len(pre); w1=sum(1 for x in pre if x['pnl']>0)/n1; a1=np.mean([x['pnl'] for x in pre])
            print(f"        IS  n={n1} WR={w1:.1%} avg={a1:+.2f}")
        if post:
            n2=len(post); w2=sum(1 for x in post if x['pnl']>0)/n2; a2=np.mean([x['pnl'] for x in post])
            print(f"        OOS n={n2} ({n2/22:.1f}/d) WR={w2:.1%} avg={a2:+.2f} net={a2-COST[name]:+.2f}")

print(f"\n  COMBINED")
for stype in ['MR', 'DC', 'ALL']:
    sub = all_trades if stype=='ALL' else [x for x in all_trades if x['type']==stype]
    if not sub: continue
    n = len(sub); tpd = n / 66
    wr = sum(1 for x in sub if x['pnl'] > 0) / n
    avg = np.mean([x['pnl'] for x in sub])
    avg_cost = np.mean([COST[x['pair']] for x in sub])
    print(f"  {stype:<8s}: n={n} ({tpd:.1f}/d) WR={wr:.1%} avg={avg:+.2f} "
          f"net={avg-avg_cost:+.2f}  avg_z={np.mean([x['z'] for x in sub]):.2f}")

print(f"\nTime: {time.time()-t0:.1f}s")
