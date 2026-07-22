"""
Cross-pair momentum spillover: EURUSD → EURJPY.
Structural rationale: EURUSD leads (most liquid), EURJPY lags as dealers hedge.
When EURUSD moves sharply, EURJPY catch-up is predictable.
"""
import numpy as np, pandas as pd, time, gc
from pathlib import Path

TICK_DIR = Path(__file__).resolve().parents[3] / 'data' / 'exness_ticks'
SC = {'EURJPY':100, 'EURUSD':10000}

def load(pair):
    s=[]
    for y,m in [(2025,10),(2025,11),(2025,12)]:
        fn=TICK_DIR/f'{pair}_Raw_Spread_{y}_{m:02d}.zip'
        if not fn.exists(): continue
        d=pd.read_csv(fn, compression='zip', names=['E','S','Ts','B','A'],
            skiprows=1, header=None, dtype={'Ts':str,'B':np.float64,'A':np.float64})
        d['Ts']=pd.to_datetime(d['Ts'].str.replace('Z','',regex=False),
            format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
        s.append(d.dropna(subset=['Ts']))
    df=pd.concat(s,ignore_index=True).sort_values('Ts').reset_index(drop=True)
    df['Mid']=(df['B']+df['A'])/2
    sc=SC[pair]
    df['MP']=df['Mid']*sc
    df['Sprd']=(df['A']-df['B'])*sc
    return df.set_index('Ts')

def make_bars(t, freq='1min'):
    mp = t['MP'].resample(freq).last().dropna()
    ret = mp.pct_change().fillna(0)
    sp = t['Sprd'].resample(freq).mean()
    bars = pd.DataFrame({'mp':mp, 'ret':ret, 'sp':sp})
    bars['z'] = (bars['ret'] - bars['ret'].rolling(50).mean()) / bars['ret'].rolling(50).std().clip(lower=1e-8)
    return bars.dropna(subset=['mp','ret'])

print("="*60)
print("CROSS-PAIR SPILLOVER: EURUSD → EURJPY")
print("="*60)

t_eur = load('EURUSD')
t_jpy = load('EURJPY')

for phase_name, emask, jmask in [
    ('IS', (t_eur.index.year==2025)&(t_eur.index.month.isin([10,11])),
            (t_jpy.index.year==2025)&(t_jpy.index.month.isin([10,11]))),
    ('OOS', (t_eur.index.year==2025)&(t_eur.index.month==12),
            (t_jpy.index.year==2025)&(t_jpy.index.month==12)),
]:
    print(f"\n--- {phase_name} ---")
    eur_bars = make_bars(t_eur.loc[emask])
    jpy_bars = make_bars(t_jpy.loc[jmask])
    
    # Align by time index
    aligned = pd.DataFrame({
        'eur_z': eur_bars['z'],
        'eur_mp': eur_bars['mp'],
        'jpy_mp': jpy_bars['mp'],
        'jpy_sp': jpy_bars['sp'],
    }).dropna()
    
    # Forward JPY returns
    aligned['jpy_ret5'] = aligned['jpy_mp'].pct_change(5).shift(-5).fillna(0)
    aligned['jpy_ret10'] = aligned['jpy_mp'].pct_change(10).shift(-10).fillna(0)
    aligned['jpy_ret20'] = aligned['jpy_mp'].pct_change(20).shift(-20).fillna(0)
    
    for hold, col in [(5,'jpy_ret5'),(10,'jpy_ret10'),(20,'jpy_ret20')]:
        aligned[f'af{hold}'] = np.sign(aligned['eur_z'].fillna(0).values) * aligned[col].values * SC['EURJPY']
    
    aligned = aligned.dropna(subset=['af5','af10','af20'])
    ndays = int(aligned.index.normalize().nunique() * 5 / 7)
    
    if phase_name == 'IS':
        print(f"  bars={len(aligned):,d} estimate {ndays}d")
    
    for z_thr in [1.0, 1.5, 2.0, 2.5, 3.0]:
        events = aligned['eur_z'].abs() > z_thr
        n = events.sum()
        if n < 5: continue
        for hold in [5, 10, 20]:
            col = f'af{hold}'
            sub = aligned.loc[events]
            wr = (sub[col] > 0).mean()
            avg = sub[col].mean()
            nd = n / ndays if ndays > 0 else 0
            if phase_name == 'OOS' or (phase_name == 'IS' and wr >= 0.55):
                print(f"  {phase_name} z>{z_thr:.1f} h={hold:<2}min n={n:>4d} {nd:>6.1f}/d "
                      f"WR={wr:.1%} avg={avg:+.3f}p")

print("\nDone")
