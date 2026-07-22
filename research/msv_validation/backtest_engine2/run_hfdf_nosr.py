"""
HF-DF without sr filter on Exness tick 10s bars.
Tests whether the trailing stop signal works without spread-widening filter.
Also tests EURJPY to see if removing sr filter fixes the negative net issue.
"""
import numpy as np, pandas as pd, time
from pathlib import Path

TICK_DIR = Path(__file__).resolve().parents[3] / 'data' / 'exness_ticks'
COST = {'EURUSD': 0.15, 'EURJPY': 50, 'GBPJPY': 60}

def load_pair(pair):
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
    df['Sprd'] = (df['A']-df['B']) * 10000
    return df.set_index('Ts')

def run_experiment(pair, cost, use_sr, label):
    t = load_pair(pair)
    b = t['MP'].resample('10s').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
    b['ret'] = b['close'].diff()
    b['z'] = (b['ret'] - b['ret'].shift(1).rolling(50).mean()) / b['ret'].shift(1).rolling(50).std().clip(1e-8)
    b['atr_v'] = (b['high'] - b['low']).shift(1).rolling(20).mean().clip(1e-8)

    if use_sr:
        sp_max = t['Sprd'].resample('10s').max()
        sp_med = t['Sprd'].resample('10s').median().ffill()
        b['rm'] = sp_med.shift(1).rolling(30, min_periods=1).median().ffill()
        b['sr'] = sp_max / b['rm'].clip(1e-8)
        events = b[b['sr'] > 1.05].dropna(subset=['z','atr_v'])
    else:
        events = b.dropna(subset=['z','atr_v'])

    print(f"\n{'='*60}")
    print(f"{pair}  {label}  (sr_filter={use_sr})")
    print(f"{'='*60}")

    for y,m,mname in [(2025,10,'Oct'),(2025,11,'Nov'),(2025,12,'Dec')]:
        mask = (events.index.year == y) & (events.index.month == m)
        phase = events[mask]
        if len(phase) < 5: continue
        ndays = len(set(idx.date() for idx in phase.index))

        pnls = []
        for idx in phase.index:
            pos = b.index.get_loc(idx)
            if pos + 2 >= len(b): continue
            row = b.loc[idx]
            direction = -1 if row['z'] > 0 else 1
            entry = row['close']; atr = row['atr_v']
            s = 0.15*atr; tg=0.20*atr; gp=0.10*atr
            best = entry; exited = False
            for j in range(1, 55):
                if pos + j >= len(b): break
                bar = b.iloc[pos+j]
                if direction == 1:
                    best = max(best, bar['high'])
                    sl = entry - s
                    if best - entry > tg: sl = best - gp
                    if bar['low'] <= sl: pnls.append((sl-entry)); exited=True; break
                else:
                    best = min(best, bar['low'])
                    sl = entry + s
                    if entry - best > tg: sl = best + gp
                    if bar['high'] >= sl: pnls.append((sl-entry)*direction); exited=True; break
            if not exited:
                pnls.append((b['close'].iloc[min(pos+54,len(b)-1)]-entry)*direction)

        if len(pnls) >= 5:
            wr = np.mean(np.array(pnls)>0)
            avg = np.mean(pnls); net = avg - cost
            tpd = len(pnls)/max(ndays,1)
            print(f"  {mname:3s}: n={len(pnls):>6d}  {tpd:>5.1f}/d  WR={wr:.1%}  avg={avg:+.2f}  net={net:+.2f}")

    # All months combined
    b_clean = events
    if len(b_clean) >= 5:
        pnls = []
        for idx in b_clean.index:
            pos = b.index.get_loc(idx)
            if pos + 2 >= len(b): continue
            row = b.loc[idx]
            direction = -1 if row['z']>0 else 1; entry=row['close']; atr=row['atr_v']
            if np.isnan(atr) or atr<1e-10: continue
            s=0.15*atr; tg=0.20*atr; gp=0.10*atr; best=entry; exited=False
            for j in range(1,55):
                if pos+j>=len(b): break
                bar=b.iloc[pos+j]
                if direction==1:
                    best=max(best,bar['high']); sl=entry-s
                    if best-entry>tg: sl=best-gp
                    if bar['low']<=sl: pnls.append((sl-entry)); exited=True; break
                else:
                    best=min(best,bar['low']); sl=entry+s
                    if entry-best>tg: sl=best+gp
                    if bar['high']>=sl: pnls.append((sl-entry)*direction); exited=True; break
            if not exited:
                pnls.append((b['close'].iloc[min(pos+54,len(b)-1)]-entry)*direction)
        ndays = len(set(idx.date() for idx in b_clean.index))
        wr=np.mean(np.array(pnls)>0); avg=np.mean(pnls); net=avg-cost; tpd=len(pnls)/max(ndays,1)
        print(f"  ALL: n={len(pnls):>6d}  {tpd:>5.1f}/d  WR={wr:.1%}  avg={avg:+.2f}  net={net:+.2f}")

t0 = time.time()
for pair in ['EURUSD','EURJPY','GBPJPY']:
    run_experiment(pair, COST[pair], use_sr=True, label='with sr>1.05')
    run_experiment(pair, COST[pair], use_sr=False, label='no sr filter')
print(f"\nTotal: {time.time()-t0:.1f}s")
