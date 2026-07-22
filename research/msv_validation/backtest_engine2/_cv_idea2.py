"""
Idea 2 cross-validation: z>2.0 + ATR>25% on ALL available data sources.
"""
import numpy as np, pandas as pd, time
from pathlib import Path

EXNESS_DIR = Path(__file__).resolve().parents[3] / 'data' / 'exness_ticks'
DUKA_CSV_DIR = Path(__file__).resolve().parents[3] / 'research' / 'dark_research' / 'dukascopy_data'
DUKA_PAR_DIR = Path(__file__).resolve().parents[3] / 'research' / 'phase_dislocation' / 'dukascopy_data'

COST = {'EURUSD': 0.15, 'EURJPY': 50, 'GBPJPY': 60}

def load_exness(pair):
    dfs = []
    for y,m in [(2025,10),(2025,11),(2025,12)]:
        fn = EXNESS_DIR / f'{pair}_Raw_Spread_{y}_{m:02d}.zip'
        d = pd.read_csv(fn, compression='zip',
            names=['E','S','Ts','B','A'], skiprows=1, header=None,
            dtype={'Ts':str,'B':np.float64,'A':np.float64})
        d['Ts'] = pd.to_datetime(d['Ts'].str.replace('Z','',regex=False),
            format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
        dfs.append(d.dropna(subset=['Ts']))
    df = pd.concat(dfs, ignore_index=True).sort_values('Ts').reset_index(drop=True)
    df['MP'] = ((df['B']+df['A'])/2) * 10000
    return df.set_index('Ts')

def load_duka_csv(pair):
    files = sorted(DUKA_CSV_DIR.glob(f'{pair.lower()}-m1-bid-*.csv'))
    dfs = []
    for f in files:
        d = pd.read_csv(f)
        d['timestamp'] = pd.to_datetime(d['timestamp'], unit='ms')
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True).sort_values('timestamp').drop_duplicates('timestamp')
    df = df.set_index('timestamp').astype(float) * 10000
    return df[['open','high','low','close']]

def load_duka_par(pair):
    df = pd.read_parquet(DUKA_PAR_DIR / f'{pair.lower()}.parquet')
    df = df.set_index('timestamp').astype(float) * 10000
    return df[['open','high','low','close']]

def run_on_bars(b, cost):
    ret = b['close'].diff()
    z = (ret - ret.shift(1).rolling(50).mean()) / ret.shift(1).rolling(50).std().clip(1e-8)
    atr = (b['high'] - b['low']).shift(1).rolling(20).mean().clip(1e-8)
    atr_gate = atr.rolling(100, min_periods=10).quantile(0.25).bfill()
    atr_pass = atr > atr_gate

    valid = z.notna() & atr.notna() & (z.abs() > 2.0) & atr_pass
    idxs = np.where(valid)[0]
    closes = b['close'].values; highs = b['high'].values; lows = b['low'].values
    z_vals = z.values; atr_vals = atr.values
    index_dates = b.index

    pnls = []; ndays = set()
    stop_a, trig_a, gap_a = 0.15, 0.20, 0.10
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
                best = max(best, highs[bp]); sl = entry - s
                if best - entry > tg: sl = best - gp
                if lows[bp] <= sl: pnls.append(sl - entry); exited = True; break
            else:
                best = min(best, lows[bp]); sl = entry + s
                if entry - best > tg: sl = best + gp
                if highs[bp] >= sl: pnls.append((sl - entry) * direction); exited = True; break
        if not exited:
            pnls.append((closes[min(pos+54, len(b)-1)] - entry) * direction)
        ndays.add(index_dates[pos].date())

    pnls = np.array(pnls)
    if len(pnls) < 10: return None
    wr = np.mean(pnls > 0); avg = np.mean(pnls); net = avg - cost
    tpd = len(pnls) / max(len(ndays), 1)
    win_avg = np.mean(pnls[pnls>0]) if np.any(pnls>0) else 0
    loss_avg = np.mean(pnls[pnls<=0]) if np.any(pnls<=0) else 0
    payoff = win_avg/abs(loss_avg) if loss_avg!=0 else 0
    return {'n':len(pnls),'tpd':tpd,'wr':wr,'avg':avg,'net':net,'payoff':payoff,'days':len(ndays),
            'pct': len(pnls)/len(b)*100}

print(f"{'Source':>12s}  {'Pair':>7s}  {'Period':>16s}  {'tpd':>6s}  {'WR':>5s}  {'net':>9s}  {'payoff':>6s}  {'n':>6s}  {'n/days':>5s}")
print("=" * 85)
t0 = time.time()

# 1. Exness tick data (Oct-Dec 2025)
for pair in ['EURUSD','EURJPY','GBPJPY']:
    t = load_exness(pair)
    b = t['MP'].resample('1min').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
    r = run_on_bars(b, COST[pair])
    if r:
        print(f"{'Exness tick':>12s}  {pair:>7s}  {'Oct-Dec 2025':>16s}  {r['tpd']:>6.0f}  {r['wr']:>5.1%}  {r['net']:>+9.2f}  {r['payoff']:>6.1f}  {r['n']:>6d}  {r['n']//78:>5d}")

# 2. Dukascopy CSV (Oct 2024 - Jun 2026)
for pair in ['EURUSD','EURJPY','GBPJPY']:
    b = load_duka_csv(pair)
    r = run_on_bars(b, COST[pair])
    if r:
        nd = len(set(b.index.date)) if len(b) > 0 else 1
        print(f"{'Dukascopy CSV':>12s}  {pair:>7s}  {'Oct24-Jun26':>16s}  {r['tpd']:>6.0f}  {r['wr']:>5.1%}  {r['net']:>+9.2f}  {r['payoff']:>6.1f}  {r['n']:>6d}  {r['n']//nd:>5d}")

# 3. Dukascopy Parquet (Apr-Jun 2026)
for pair in ['EURUSD','EURJPY','GBPJPY']:
    b = load_duka_par(pair)
    r = run_on_bars(b, COST[pair])
    if r:
        nd = len(set(b.index.date)) if len(b) > 0 else 1
        print(f"{'Duka Parquet':>12s}  {pair:>7s}  {'Apr-Jun 2026':>16s}  {r['tpd']:>6.0f}  {r['wr']:>5.1%}  {r['net']:>+9.2f}  {r['payoff']:>6.1f}  {r['n']:>6d}  {r['n']//nd:>5d}")

print(f"\nTotal: {time.time()-t0:.0f}s")
