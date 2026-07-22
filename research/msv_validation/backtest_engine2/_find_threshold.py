"""Find optimal z-score threshold: balance trades/day vs WR vs net profit."""
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

def run(pair, cost, z_thresh):
    t = load(pair)
    b = t['MP'].resample('1min').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
    ret = b['close'].diff()
    z = (ret - ret.shift(1).rolling(50).mean()) / ret.shift(1).rolling(50).std().clip(1e-8)
    atr = (b['high'] - b['low']).shift(1).rolling(20).mean().clip(1e-8)

    valid = z.notna() & atr.notna() & (z.abs() > z_thresh)
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
    if len(pnls) < 5: return None
    wr = np.mean(pnls > 0); avg = np.mean(pnls); net = avg - cost
    tpd = len(pnls) / max(len(ndays), 1)
    return {'n': len(pnls), 'tpd': tpd, 'wr': wr, 'avg': avg, 'net': net, 'days': len(ndays),
            'pct': len(pnls) / b[z.notna() & atr.notna()].shape[0]}

print(f"{'Pair':>7s}  {'z_thr':>5s}  {'tpd':>6s}  {'WR':>5s}  {'avg':>7s}  {'net':>7s}  {'%bars':>6s}")
print("-" * 55)
t0 = time.time()
for pair in ['EURUSD', 'EURJPY', 'GBPJPY']:
    for zt in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        r = run(pair, COST[pair], zt)
        if r:
            print(f"{pair:>7s}  {zt:>5.2f}  {r['tpd']:>6.0f}  {r['wr']:>5.1%}  {r['avg']:>+7.2f}  {r['net']:>+7.2f}  {r['pct']:>6.1%}")
print(f"\nTotal: {time.time()-t0:.0f}s")
