"""
Spread Asymmetry: dealers quote tighter on the side they want to trade.
When bid is close to mid (tight) while ask is far (wide), dealers want to sell → short.
When ask is close to mid while bid is far, dealers want to buy → long.
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
    df['Sprd']=df['A']-df['B']
    df['BDev']=df['Mid']-df['B']  # bid deviation from mid
    df['ADev']=df['A']-df['Mid']  # ask deviation from mid
    # Asymmetry: >1 means bid is wider = dealers want to sell → short
    # <1 means ask is wider = dealers want to buy → long
    df['Asym']=df['BDev']/df['ADev'].clip(lower=1e-8)
    df['MP']=df['Mid']*SC[pair]
    return df.set_index('Ts')

def run():
    for pair in ['EURJPY', 'GBPJPY']:
        print(f"\n{'='*60}")
        print(f"{pair}")
        print(f"{'='*60}")
        t=load(pair)
        sc=SC[pair]
        cost = 0.6 if pair=='GBPJPY' else 0.5
        
        for phase_name, mask in [('IS',(t.index.year==2025)&(t.index.month.isin([10,11]))),
                                  ('OOS',(t.index.year==2025)&(t.index.month==12))]:
            sub_t=t.loc[mask]
            if len(sub_t)<1000: continue
            ndays=int(sub_t.index.normalize().nunique()*5/7)
            
            # Aggregate to 1min bars
            b=pd.DataFrame({
                'asym': np.log(sub_t['Asym'].resample('1min').mean().clip(lower=0.01, upper=100)),
                'mp': sub_t['MP'].resample('1min').last(),
            }).dropna()
            b['asym_z']=(b['asym']-b['asym'].rolling(50).mean())/b['asym'].rolling(50).std().clip(lower=1e-8)
            
            for hold in [1, 3, 5, 10]:
                col=f'ret{hold}'
                b[col]=b['mp'].pct_change(hold).shift(-hold).fillna(0)*sc
                b[f'af{hold}']=-np.sign(b['asym_z'].fillna(0).values)*b[col].values
            
            for z_thr in [1.0,1.5,2.0,2.5,3.0]:
                events=b['asym_z'].abs()>z_thr
                n=events.sum()
                if n<5: continue
                for hold in [1,3,5,10]:
                    col=f'af{hold}'
                    sub=b.loc[events]
                    wr=(sub[col]>0).mean()
                    avg=sub[col].mean()
                    net=avg-cost
                    nd=n/ndays
                    if phase_name=='OOS' and wr>=0.55:
                        print(f"  OOS z>{z_thr:.1f} h={hold:<2}min n={n:>4d} {nd:>6.1f}/d "
                              f"WR={wr:.1%} avg={avg:+.3f}p net={net:+.3f}p")
                    elif phase_name=='IS' and n>=30 and wr>=0.53:
                        print(f"  IS  z>{z_thr:.1f} h={hold:<2}min n={n:>4d} {nd:>6.1f}/d "
                              f"WR={wr:.1%} avg={avg:+.3f}p net={net:+.3f}p")
            gc.collect()

if __name__=='__main__':
    run()
