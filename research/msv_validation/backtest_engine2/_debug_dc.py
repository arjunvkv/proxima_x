"""Debug DC detection for EURUSD."""
import pandas as pd, numpy as np
from pathlib import Path

TICK_DIR = Path(__file__).resolve().parents[3] / 'data' / 'exness_ticks'
dfs = []
for y,m in [(2025,10),(2025,11),(2025,12)]:
    fn = TICK_DIR / f'EURUSD_Raw_Spread_{y}_{m:02d}.zip'
    d = pd.read_csv(fn, compression='zip', names=['E','S','Ts','B','A'],
        skiprows=1, header=None, dtype={'Ts':str,'B':np.float64,'A':np.float64})
    d['Ts'] = pd.to_datetime(d['Ts'].str.replace('Z','',regex=False),
        format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    dfs.append(d.dropna(subset=['Ts']))
df = pd.concat(dfs, ignore_index=True).sort_values('Ts').reset_index(drop=True)
df['MP'] = ((df['B']+df['A'])/2) * 10000
df['Sprd'] = (df['A']-df['B']) * 10000
df = df.set_index('Ts')
print(f"EURUSD: {len(df):,d} ticks")

b1 = df['MP'].resample('1min').agg({'open':'first','close':'last'}).dropna()
b1['ret'] = b1['close'] - b1['open']
b1['z'] = (b1['ret'] - b1['ret'].rolling(20).mean()) / b1['ret'].rolling(20).std().clip(1e-8)
sp_max = df['Sprd'].resample('1min').max()
sp_med = df['Sprd'].resample('1min').median().fillna(method='ffill')
b1['sr'] = sp_max / sp_med.clip(1e-8)

events = b1[(b1['z'].abs()>1.5) & (b1['sr']>2.0)]
print(f"DC events: {len(events)}")

# Show z distribution of events
print(f"z range: {events['z'].min():.2f} to {events['z'].max():.2f}")
print(f"z mean: {events['z'].mean():.2f}")
print(f"sr range: {events['sr'].min():.2f} to {events['sr'].max():.2f}")

# Analyze first event
t = df
first = events.index[0]
print(f"\nFirst event: {first}")
print(f"  z={events.loc[first,'z']:.2f} ret={events.loc[first,'ret']:+.2f} sr={events.loc[first,'sr']:.2f}")
print(f"  sp_med={sp_med.loc[first]:.2f} sp_max={sp_max.loc[first]:.2f}")
print(f"  rec_thr={1.3*sp_med.loc[first]:.2f}")

# Check spreads around this event
window = t.loc[first - pd.Timedelta(minutes=2):first + pd.Timedelta(minutes=5)]
print(f"\n  Spreads around event:")
print(f"  {window['Sprd'].describe()}")
print(f"  Any spread < rec_thr? {(window['Sprd'] < 1.3*sp_med.loc[first]).any()}")

# Check recovery
future = t.loc[first:]
rec_thr = 1.3 * sp_med.loc[first]
rec = future[future['Sprd'] < rec_thr]
print(f"  Recovery ticks found: {len(rec)}")
if len(rec) >= 2:
    entry_time = rec.index[1]
    print(f"  Entry time: {entry_time}")
    print(f"  Entry spread: {rec['Sprd'].iloc[1]:.2f}")
    exit_time = entry_time + pd.Timedelta(seconds=600)
    sl = t.loc[entry_time:exit_time]
    print(f"  Exit window: {len(sl)} ticks over 10min")
    if len(sl) >= 2:
        ep = sl['MP'].iloc[0]
        xp = sl['MP'].iloc[-1]
        pnl = (xp - ep) * (-1 if events.loc[first,'z'] > 0 else 1)
        print(f"  entry_mp={ep:.1f} exit_mp={xp:.1f} pnl={pnl:+.1f}")

print(f"\nAll events summary:")
pnls = []
for dt in events.index[:50]:
    z = events.loc[dt,'z']
    med = sp_med.loc[dt]
    rec_thr = 1.3 * med
    future = t.loc[dt:]
    rec = future[future['Sprd'] < rec_thr]
    if len(rec) < 2: continue
    et = rec.index[1]
    xt = et + pd.Timedelta(seconds=600)
    sl = t.loc[et:xt]
    if len(sl) < 2: continue
    ep = sl['MP'].iloc[0]
    xp = sl['MP'].iloc[-1]
    pnl = (xp - ep) * (-1 if z > 0 else 1)
    pnls.append(pnl)
    if len(pnls) <= 10:
        print(f"  {dt}: z={z:.2f} pnl={pnl:+.1f}")

if pnls:
    print(f"\n  First 50 events: n={len(pnls)} WR={sum(1 for p in pnls if p>0)/len(pnls):.1%} avg={np.mean(pnls):+.1f}")
