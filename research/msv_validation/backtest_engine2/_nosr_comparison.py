"""Quick no-sr comparison: EURUSD + GBPJPY on Exness 10s bars."""
import numpy as np, pandas as pd, time
from pathlib import Path

TICK_DIR = Path(__file__).resolve().parents[3] / 'data' / 'exness_ticks'
COST = {'EURUSD': 0.15, 'GBPJPY': 60}

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

def run_experiment(pair, cost, with_sr):
    t = load(pair)
    b = t['MP'].resample('10s').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
    ret = b['close'].diff()
    z = (ret - ret.shift(1).rolling(50).mean()) / ret.shift(1).rolling(50).std().clip(1e-8)
    atr = (b['high'] - b['low']).shift(1).rolling(20).mean().clip(1e-8)

    if with_sr:
        sp_max = t['Sprd'].resample('10s').max()
        sp_med = t['Sprd'].resample('10s').median().ffill()
        rm = sp_med.shift(1).rolling(30, min_periods=1).median().ffill()
        sr = sp_max / rm.clip(1e-8)
        valid = z.notna() & atr.notna() & (sr > 1.05)
    else:
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
            exit_bar = min(pos + 54, len(b) - 1)
            pnls.append((closes[exit_bar] - entry) * direction)
        ndays.add(index_dates[pos].date())

    pnls = np.array(pnls)
    if len(pnls) >= 5:
        wr = np.mean(pnls > 0); avg = np.mean(pnls); net = avg - cost
        tpd = len(pnls) / max(len(ndays), 1)
        label = "sr>1.05" if with_sr else "no-sr"
        print(f"  {pair} {label}: n={len(pnls):>7,d}  {tpd:>5.1f}/d  WR={wr:.1%}  avg={avg:+.2f}  net={net:+.2f}  days={len(ndays)}")

t0 = time.time()
for pair in ['EURUSD','GBPJPY']:
    run_experiment(pair, COST[pair], with_sr=False)
print(f"Total: {time.time()-t0:.0f}s")
