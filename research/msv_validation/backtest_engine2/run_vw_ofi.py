"""
Volume-Weighted OFI: weight up/down ticks by spread deviation from median.
Ticks during wide spreads are more likely informed (dealer flow).
Ticks during tight spreads are more likely noise.
Weighted OFI should have better forward predictive power.
"""
import numpy as np, pandas as pd, gc
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

def compute_vw_ofi(t, pair):
    t0=pd.Timestamp.now()
    sc=SC[pair]
    mp=t['MP'].values
    sp=t['Sprd'].values
    
    # Tick direction
    diff=np.diff(mp, prepend=mp[0])
    up=(diff>0).astype(float)
    dn=(diff<0).astype(float)
    
    # Spread weight: tick spread - rolling median spread (minute-level)
    # Use a simple EMA of minute-median spread as the baseline
    sp_ma=pd.Series(sp).rolling(1000, min_periods=100).mean().values
    sp_dev=np.maximum(sp-sp_ma, 0)  # only positive deviation (wide spreads)
    
    idx=t.index
    m_idx=idx.floor('1min')
    
    of=pd.DataFrame({
        'up':up,'dn':dn,'mp':mp,
        'up_w':up*(1+sp_dev),  # weighted up-ticks
        'dn_w':dn*(1+sp_dev),  # weighted down-ticks
        'sp':sp,'sp_dev':sp_dev,
    }, index=idx)
    
    ob=of.groupby(m_idx).agg({'up':'sum','dn':'sum','up_w':'sum','dn_w':'sum','mp':'last','sp':'mean','sp_dev':'mean'})
    ob['fl']=ob['up']+ob['dn']
    ob['ofi']=(ob['up']-ob['dn'])/ob['fl'].clip(lower=1)
    ob['vw_ofi']=(ob['up_w']-ob['dn_w'])/(ob['up_w']+ob['dn_w']).clip(lower=1)
    
    ob['ofi_z']=(ob['ofi']-ob['ofi'].rolling(50).mean())/ob['ofi'].rolling(50).std().clip(lower=1e-8)
    ob['vw_z']=(ob['vw_ofi']-ob['vw_ofi'].rolling(50).mean())/ob['vw_ofi'].rolling(50).std().clip(lower=1e-8)
    
    # Forward returns
    for h in [3,5,10,15]:
        ob[f'fwd{h}']=ob['mp'].diff(h).shift(-h).fillna(0)
        ob[f'afi{h}']=np.sign(ob['ofi_z'])*ob[f'fwd{h}']
        ob[f'afv{h}']=np.sign(ob['vw_z'])*ob[f'fwd{h}']
    
    return ob.dropna(subset=['ofi_z','vw_z','fwd3'])

def run():
    for pair in ['EURJPY','GBPJPY']:
        print(f"\n{'='*70}")
        print(f"{pair}")
        print(f"{'='*70}")
        t=load(pair)
        cost=0.6 if pair=='GBPJPY' else 0.5
        
        for phase_name, tmask in [
            ('IS',(t.index.year==2025)&(t.index.month.isin([10,11]))),
            ('OOS',(t.index.year==2025)&(t.index.month==12)),
        ]:
            sub_t=t.loc[tmask]
            if len(sub_t)<1000: continue
            ndays=int(sub_t.index.normalize().nunique()*5/7)
            ob=compute_vw_ofi(sub_t, pair)
            if len(ob)<100: continue
            
            if phase_name=='IS':
                # Compare OFI vs VW-OFI distributions
                corr=np.corrcoef(ob['ofi_z'].fillna(0), ob['vw_z'].fillna(0))[0,1]
                print(f"  {phase_name}: {len(ob)} bars, OFI_z vs VW_z corr={corr:.3f}")
            
            for label, zcol, afcol in [('OFI','ofi_z','afi'), ('VW_OFI','vw_z','afv')]:
                for z_thr in [1.5,2.0,2.5,3.0]:
                    events=ob[zcol].abs()>z_thr
                    n=events.sum()
                    if n<5: continue
                    for hold in [3,5,10,15]:
                        col=f'{afcol}{hold}'
                        sub=ob.loc[events]
                        wr=(sub[col]>0).mean()
                        avg=sub[col].mean()
                        net=avg-cost
                        nd=n/ndays
                        if phase_name=='OOS' and wr>=0.57:
                            print(f"  OOS {label:<7s} z>{z_thr:.1f} h={hold:<2}min n={n:>4d} {nd:>5.2f}/d "
                                  f"WR={wr:.1%} avg={avg:+.3f}p net={net:+.3f}p")
                        elif phase_name=='IS' and n>=20 and wr>=0.55:
                            print(f"  IS  {label:<7s} z>{z_thr:.1f} h={hold:<2}min n={n:>4d} {nd:>5.2f}/d "
                                  f"WR={wr:.1%} avg={avg:+.3f}p net={net:+.3f}p")
            gc.collect()

if __name__=='__main__':
    run()
