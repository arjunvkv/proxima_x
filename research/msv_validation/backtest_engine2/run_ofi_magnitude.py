"""
OFI with magnitude filter: trade only when OFI_z is extreme AND
the 1-min price move exceeds a threshold. This selects momentum
that's already established and likely to continue.
Walk-forward: IS=Oct+Nov, OOS=Dec.
"""
import numpy as np, pandas as pd, time, gc
from pathlib import Path

TICK_DIR = Path(__file__).resolve().parents[3] / 'data' / 'exness_ticks'
SC = {'EURJPY':100, 'GBPJPY':100}

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

def run():
    for pair in ['EURJPY','GBPJPY']:
        print(f"\n{'='*70}")
        print(f"{pair}")
        print(f"{'='*70}")
        t=load(pair)
        sc=SC[pair]
        cost = 0.6 if pair=='GBPJPY' else 0.5
        
        for phase_name, tmask in [
            ('IS',(t.index.year==2025)&(t.index.month.isin([10,11]))),
            ('OOS',(t.index.year==2025)&(t.index.month==12)),
        ]:
            sub_t=t.loc[tmask]
            if len(sub_t)<1000: continue
            ndays=int(sub_t.index.normalize().nunique()*5/7)
            
            # Minute bars
            mp_arr=sub_t['MP'].values
            idx=sub_t.index
            m_idx=idx.floor('1min')
            
            up=(np.diff(mp_arr, prepend=mp_arr[0])>0).astype(float)
            dn=(np.diff(mp_arr, prepend=mp_arr[0])<0).astype(float)
            
            of=pd.DataFrame({'up':up,'dn':dn,'mp':mp_arr}, index=idx)
            ob=of.groupby(m_idx).agg({'up':'sum','dn':'sum','mp':'last'})
            ob['fl']=ob['up']+ob['dn']
            ob['ofi']=(ob['up']-ob['dn'])/ob['fl'].clip(lower=1)
            ob['ofi_z']=(ob['ofi']-ob['ofi'].rolling(50).mean())/ob['ofi'].rolling(50).std().clip(lower=1e-8)
            
            # 1-min return magnitude
            ob['ret1']=ob['mp'].diff().fillna(0)
            ob['ret1_abs']=ob['ret1'].abs()
            
            # Forward returns
            for h in [3,5,10]:
                ob[f'fwd{h}']=ob['mp'].diff(h).shift(-h).fillna(0)
                ob[f'af{h}']=np.sign(ob['ofi_z'].fillna(0))*ob[f'fwd{h}']
            
            ob=ob.dropna(subset=['ofi_z','ret1'])
            if len(ob)<100: continue
            
            # Debug: print distributions for IS
            if phase_name=='IS':
                q99=np.quantile(ob['ret1_abs'].dropna().values, 0.99)
                q95=np.quantile(ob['ret1_abs'].dropna().values, 0.95)
                q90=np.quantile(ob['ret1_abs'].dropna().values, 0.90)
                q50=np.quantile(ob['ret1_abs'].dropna().values, 0.50)
                print(f"  IS ret1_abs: p50={q50:.3f}p p90={q90:.3f}p p95={q95:.3f}p p99={q99:.3f}p")
                nz=(ob['ofi_z'].abs()>2.0).sum()
                print(f"  IS ofi_z>2: n={nz} ({nz/ndays:.1f}/d)")
            
            # Test: OFI_z threshold AND 1-min return magnitude threshold
            for z_thr in [1.0,1.5,2.0,2.5]:
                for mag_thr in [0.05,0.1,0.15,0.25]:
                    events=(ob['ofi_z'].abs()>z_thr)&(ob['ret1_abs']>mag_thr)
                    n=events.sum()
                    if n<5: continue
                    
                    for hold in [3,5,10]:
                        col=f'af{hold}'
                        sub=ob.loc[events]
                        wr=(sub[col]>0).mean()
                        avg=sub[col].mean()
                        net=avg-cost
                        nd=n/ndays
                        
                        if phase_name=='OOS':
                            if wr>=0.60:
                                print(f"  OOS z>{z_thr:.1f} mag>{mag_thr:.2f}p h={hold:<2}min "
                                      f"n={n:>3d} {nd:>5.2f}/d WR={wr:.1%} avg={avg:+.3f}p net={net:+.3f}p")
                            elif net>0 and wr>=0.55 and hold>=5:
                                print(f"  OOS z>{z_thr:.1f} mag>{mag_thr:.2f}p h={hold:<2}min "
                                      f"n={n:>3d} {nd:>5.2f}/d WR={wr:.1%} avg={avg:+.3f}p net={net:+.3f}p")
                        elif phase_name=='IS' and n>=20:
                            if wr>=0.55:
                                print(f"  IS  z>{z_thr:.1f} mag>{mag_thr:.2f}p h={hold:<2}min "
                                      f"n={n:>3d} {nd:>5.2f}/d WR={wr:.1%} avg={avg:+.3f}p net={net:+.3f}p")
            gc.collect()

if __name__=='__main__':
    run()
