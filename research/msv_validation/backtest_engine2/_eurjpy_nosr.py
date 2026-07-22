"""EURJPY no-sr on Exness 10s bars — fast vectorized."""
import numpy as np, pandas as pd, time
from pathlib import Path

TICK_DIR = Path(__file__).resolve().parents[3] / 'data' / 'exness_ticks'

def load(pair):
    dfs = []
    for y,m in [(2025,10),(2025,11),(2025,12)]:
        fn = TICK_DIR / f'{pair}_Raw_Spread_{y}_{m:02d}.zip'
        d = pd.read_csv(fn, compression='zip',
            names=['E','S','Ts','B','A'], skiprows=1, header=None,
            dtype={'Ts':str,'B':np.float64,'A':np.float64})
        d['Ts'] = pd.to_datetime(d['Ts'].str.replace('Z','',regex=False),
            format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
        dfs.append(d.dropna(subset=['Ts']))
    df = pd.concat(dfs, ignore_index=True).sort_values('Ts').reset_index(drop=True)
    df['MP'] = ((df['B']+df['A'])/2) * 10000
    return df.set_index('Ts')

t0 = time.time()
t = load('EURJPY')
print(f"Loaded: {time.time()-t0:.0f}s  ticks={len(t):,d}")

b = t['MP'].resample('10s').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
print(f"Bars: {len(b):,d}  10s bars")

ret = b['close'].diff()
z = (ret - ret.shift(1).rolling(50).mean()) / ret.shift(1).rolling(50).std().clip(1e-8)
atr = (b['high'] - b['low']).shift(1).rolling(20).mean().clip(1e-8)

valid = z.notna() & atr.notna()
idxs = np.where(valid)[0]
print(f"Valid entries: {len(idxs):,d}")

stop_a, trig_a, gap_a = 0.15, 0.20, 0.10
max_bars = 54
closes = b['close'].values
highs = b['high'].values
lows = b['low'].values
z_vals = z.values
atr_vals = atr.values
index_dates = b.index.date

pnls = []
nday_set = set()

for pos in idxs:
    if pos + 2 >= len(b): continue
    direction = -1 if z_vals[pos] > 0 else 1
    entry = closes[pos]
    atr_v = atr_vals[pos]
    s = stop_a * atr_v; tg = trig_a * atr_v; gp = gap_a * atr_v
    best = entry; exited = False
    for j in range(1, max_bars + 1):
        bp = pos + j
        if bp >= len(b): break
        if direction == 1:
            if highs[bp] > best: best = highs[bp]
            sl = entry - s
            if best - entry > tg: sl = best - gp
            if lows[bp] <= sl: pnls.append(sl - entry); exited = True; break
        else:
            if lows[bp] < best: best = lows[bp]
            sl = entry + s
            if entry - best > tg: sl = best + gp
            if highs[bp] >= sl: pnls.append((sl - entry) * direction); exited = True; break
    if not exited:
        pnls.append((closes[min(pos+max_bars, len(b)-1)] - entry) * direction)
    nday_set.add(index_dates[pos])

pnls = np.array(pnls)
if len(pnls) >= 5:
    wr = np.mean(pnls > 0)
    avg = np.mean(pnls)
    net = avg - 50  # EURJPY cost
    tpd = len(pnls) / max(len(nday_set), 1)
    print(f"\nEURJPY no-sr: n={len(pnls):,d}  {tpd:.1f}/d  WR={wr:.1%}  avg={avg:+.2f}  net={net:+.2f}")
    print(f"Days: {len(nday_set)}")
print(f"Total: {time.time()-t0:.1f}s")
