"""
Order Flow Imbalance signal on 1min bars.
Fast vectorized implementation. Tests multiple thresholds.
Walk-forward: IS=Oct+Nov, OOS=Dec.
"""
import numpy as np, pandas as pd, time, gc
from pathlib import Path

TICK_DIR = Path(__file__).resolve().parents[3] / 'data' / 'exness_ticks'
SCALE = {'EURJPY': 100, 'GBPJPY': 100, 'EURUSD': 10000}

def load(pair):
    s = []
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
    sc=SCALE[pair]
    df['MP']=df['Mid']*sc
    df['Sprd']=(df['A']-df['B'])*sc
    return df.set_index('Ts')

def compute_ofi(t, pair):
    t0=time.time()
    sc=SCALE[pair]
    mp=t['MP'].values
    up=(np.diff(mp, prepend=mp[0])>0).astype(float)
    dn=(np.diff(mp, prepend=mp[0])<0).astype(float)
    
    # Minute bars
    idx=t.index
    m_idx=idx.floor('1min')
    ofi=pd.DataFrame({'up':up,'dn':dn}, index=idx).groupby(m_idx)[['up','dn']].sum()
    ofi['fl']=ofi['up']+ofi['dn']
    ofi['ofi']=(ofi['up']-ofi['dn'])/ofi['fl'].clip(lower=1)
    
    # Price from last tick of each minute
    mp_s=pd.Series(mp, index=idx)
    ofi['mp']=mp_s.groupby(m_idx).last()
    ofi['ret1']=ofi['mp'].pct_change(1).fillna(0)
    ofi['ret5']=ofi['mp'].pct_change(5).shift(-5).fillna(0)
    ofi['ret10']=ofi['mp'].pct_change(10).shift(-10).fillna(0)
    
    ofi['ofi_z']=(ofi['ofi']-ofi['ofi'].rolling(50).mean())/ofi['ofi'].rolling(50).std().clip(lower=1e-8)
    for h,col in [(1,'ret1'),(5,'ret5'),(10,'ret10')]:
        ofi[f'af{h}']=np.sign(ofi['ofi_z'].fillna(0).values)*ofi[col].values*sc
    print(f"  {pair}: {len(ofi)} bars, {time.time()-t0:.1f}s")
    return ofi

def run():
    for pair in ['EURJPY','GBPJPY']:
        print(f"\n{'='*60}")
        print(f"{pair}")
        print(f"{'='*60}")
        t=load(pair)
        sc=SCALE[pair]
        cost = 0.6 if pair=='GBPJPY' else 0.5
        
        for phase_name, mask in [('IS',(t.index.year==2025)&(t.index.month.isin([10,11]))),
                                  ('OOS',(t.index.year==2025)&(t.index.month==12))]:
            sub_t=t.loc[mask]
            if len(sub_t)<1000: continue
            ndays=int(sub_t.index.normalize().nunique()*5/7)
            ofi=compute_ofi(sub_t, pair)
            if len(ofi)<100: continue
            
            for z_thr in [0.5,0.75,1.0,1.25,1.5,2.0,2.5,3.0]:
                events=ofi['ofi_z'].abs()>z_thr
                n=events.sum()
                if n<5: continue
                for hold in [1,5,10]:
                    col=f'af{hold}'
                    sub=ofi.loc[events]
                    wr=(sub[col]>0).mean()
                    avg=sub[col].mean()
                    net=avg-cost
                    nd=n/ndays
                    if phase_name=='OOS':
                        if wr>=0.60 or (hold>1 and net>0):
                            print(f"  OOS z>{z_thr:.2f} h={hold:<2}min n={n:>4d} {nd:>5.2f}/d "
                                  f"WR={wr:.1%} avg={avg:+.3f}p net={net:+.3f}p")
                    elif phase_name=='IS' and n>=30:
                        if wr>=0.55 or (hold>1 and net>0):
                            print(f"  IS  z>{z_thr:.2f} h={hold:<2}min n={n:>4d} {nd:>5.2f}/d "
                                  f"WR={wr:.1%} avg={avg:+.3f}p net={net:+.3f}p")
            gc.collect()

if __name__=='__main__':
    run()
