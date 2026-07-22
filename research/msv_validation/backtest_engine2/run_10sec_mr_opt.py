"""
Optimize EURUSD 10s mean reversion for max net + 30+/day.
"""
import numpy as np, pandas as pd, gc
from pathlib import Path

TICK_DIR = Path(__file__).resolve().parents[3] / 'data' / 'exness_ticks'

def load():
    s=[]
    for y,m in [(2025,10),(2025,11),(2025,12)]:
        fn=TICK_DIR/f'EURUSD_Raw_Spread_{y}_{m:02d}.zip'
        if not fn.exists(): continue
        d=pd.read_csv(fn, compression='zip', names=['E','S','Ts','B','A'],
            skiprows=1, header=None, dtype={'Ts':str,'B':np.float64,'A':np.float64})
        d['Ts']=pd.to_datetime(d['Ts'].str.replace('Z','',regex=False),
            format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
        s.append(d.dropna(subset=['Ts']))
    df=pd.concat(s,ignore_index=True).sort_values('Ts').reset_index(drop=True)
    df['MP']=((df['B']+df['A'])/2)*10000
    df['Sprd']=(df['A']-df['B'])*10000
    return df.set_index('Ts')

cost=0.15
print("EURUSD 10s MR Optimization")
t=load()

for phase_name, mask in [
    ('IS',(t.index.year==2025)&(t.index.month.isin([10,11]))),
    ('OOS',(t.index.year==2025)&(t.index.month==12)),
]:
    sub_t=t.loc[mask]
    if len(sub_t)<1000: continue
    ndays=int(sub_t.index.normalize().nunique()*5/7)
    
    mp=sub_t['MP'].values
    sp=sub_t['Sprd'].values
    idx=sub_t.index
    b_idx=idx.floor('10s')
    
    bars=pd.DataFrame({'mp':mp,'sp':sp}, index=idx)
    bars=bars.groupby(b_idx).agg({'mp':'last','sp':'mean'}).dropna(subset=['mp'])
    
    bars['ret']=bars['mp'].diff().fillna(0)
    bars['z']=(bars['ret']-bars['ret'].rolling(50).mean())/bars['ret'].rolling(50).std().clip(lower=1e-8)
    bars['sp_pct']=bars['sp'].rolling(200).rank(pct=True)
    
    for h in [6,12,18,24,36]:
        bars[f'fwd{h}']=bars['mp'].diff(h).shift(-h).fillna(0)
        bars[f'mr{h}']=-np.sign(bars['z'].fillna(0).values)*bars[f'fwd{h}'].values
    
    bars=bars.dropna(subset=['z','sp'])
    
    if phase_name=='IS':
        print(f"\n{phase_name}: {len(bars):,d} bars ({len(bars)/ndays:.0f}/d)")
    
    for z_thr in [3.0,3.25,3.5,3.75,4.0]:
        for sp_max in [1.0,0.75,0.50,0.25]:
            if sp_max<1.0:
                sp_ok=bars['sp_pct']<=sp_max
                events=(bars['z'].abs()>z_thr)&sp_ok
                sl=f'sp<p{int(sp_max*100)}'
            else:
                events=bars['z'].abs()>z_thr
                sl='all_sp'
            
            n=events.sum()
            if n<5: continue
            
            for hold in [6,12,18,24,36]:
                col=f'mr{hold}'
                sub=bars.loc[events]
                wr=(sub[col]>0).mean()
                avg=sub[col].mean()
                net=avg-cost
                nd=n/ndays
                
                if phase_name=='OOS' and nd>=20:
                    if net>0 or wr>=0.55:
                        print(f"  OOS z>{z_thr:.2f} {sl:<10s} h={hold:<2}bars n={n:>4d} {nd:>5.1f}/d "
                              f"WR={wr:.1%} avg={avg:+.4f}p net={net:+.4f}p")
                elif phase_name=='IS' and n>=30:
                    if net>0:
                        print(f"  IS  z>{z_thr:.2f} {sl:<10s} h={hold:<2}bars n={n:>4d} {nd:>5.1f}/d "
                              f"WR={wr:.1%} avg={avg:+.4f}p net={net:+.4f}p")

print("\nDone")
