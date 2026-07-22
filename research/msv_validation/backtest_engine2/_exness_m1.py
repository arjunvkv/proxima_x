"""Exness tick data resampled to M1 instead of 10s — test EURJPY."""
import numpy as np, pandas as pd, time
from pathlib import Path

TICK_DIR = Path(__file__).resolve().parents[3] / 'data' / 'exness_ticks'
COST = {'EURUSD': 0.15, 'EURJPY': 50, 'GBPJPY': 60}

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

def run(pair, cost, bar_sec):
    t = load(pair)
    rule = f'{bar_sec}s'
    b = t['MP'].resample(rule).agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
    ret = b['close'].diff()
    z = (ret - ret.shift(1).rolling(50).mean()) / ret.shift(1).rolling(50).std().clip(1e-8)
    atr = (b['high'] - b['low']).shift(1).rolling(20).mean().clip(1e-8)
    valid = z.notna() & atr.notna()
    idxs = np.where(valid)[0]
    closes = b['close'].values; highs = b['high'].values; lows = b['low'].values
    z_vals = z.values; atr_vals = atr.values
    index_dates = b.index
    stop_a, trig_a, gap_a = 0.15, 0.20, 0.10
    pnls = []; ndays = set()
    for pos in idxs:
        if pos + 2 >= len(b): continue
        direction = -1 if z_vals[pos] > 0 else 1
        entry = closes[pos]; atr_v = atr_vals[pos]
        if np.isnan(atr_v) or atr_v < 1e-10: continue
        s = stop_a * atr_v; tg = trig_a * atr_v; gp = gap_a * atr_v
        best = entry; exited = False
        for j in range(1, 55):
            bp = pos + j
            if bp >= len(b): break
            if direction == 1:
                best = max(best, highs[bp])
                sl = entry - s
                if best - entry > tg: sl = best - gp
                if lows[bp] <= sl: pnls.append(sl - entry); exited = True; break
            else:
                best = min(best, lows[bp])
                sl = entry + s
                if entry - best > tg: sl = best + gp
                if highs[bp] >= sl: pnls.append((sl - entry) * direction); exited = True; break
        if not exited:
            pnls.append((closes[min(pos+54, len(b)-1)] - entry) * direction)
        ndays.add(index_dates[pos].date())
    pnls = np.array(pnls)
    if len(pnls) >= 5:
        wr = np.mean(pnls > 0); avg = np.mean(pnls); net = avg - cost
        tpd = len(pnls) / max(len(ndays), 1)
        print(f"  {pair:7s} {bar_sec:3d}s  n={len(pnls):>8,d}  {tpd:>5.1f}/d  WR={wr:.1%}  avg={avg:+.2f}  net={net:+.2f}  days={len(ndays)}")

t0 = time.time()
print(f"{'Pair':7s} {'Bar':>3s}  {'n':>8s}  {'tpd':>5s}  {'WR':>5s}  {'avg':>7s}  {'net':>7s}  days")
for pair in ['EURUSD','EURJPY','GBPJPY']:
    for sec in [10, 60]:
        run(pair, COST[pair], sec)
print(f"\nTotal: {time.time()-t0:.0f}s")
