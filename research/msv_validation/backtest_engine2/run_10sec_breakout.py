"""
10-second bar breakout strategy.
For each 10s bar: z-score of price change over last 50 bars.
When |z| > threshold + spread is tight (normal liquidity), trade.
Hold for 30-120 seconds (3-12 bars forward).
Produces hundreds of signals/day.
"""
import numpy as np, pandas as pd, gc
from pathlib import Path

TICK_DIR = Path(__file__).resolve().parents[3] / 'data' / 'exness_ticks'
SC = {'EURJPY':100, 'GBPJPY':100, 'EURUSD':10000}
COST = {'EURJPY':0.5, 'GBPJPY':0.6, 'EURUSD':0.15}

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
    df['MP']=((df['B']+df['A'])/2)*SC[pair]
    df['Sprd']=(df['A']-df['B'])*SC[pair]
    return df.set_index('Ts')

def compute_10s_bars(t, pair):
    sc=SC[pair]
    mp=t['MP'].values
    sp=t['Sprd'].values
    idx=t.index
    b_idx=idx.floor('10s')
    
    b=pd.DataFrame({'mp':mp,'sp':sp}, index=idx)
    bars=b.groupby(b_idx).agg({'mp':'last','sp':'mean'}).dropna()
    bars['ret']=bars['mp'].diff().fillna(0)
    bars['ret_abs']=bars['ret'].abs()
    # Rolling z-score of returns
    bars['z']=(bars['ret']-bars['ret'].rolling(50).mean())/bars['ret'].rolling(50).std().clip(lower=1e-8)
    bars['sp_z']=(bars['sp']-bars['sp'].rolling(50).mean())/bars['sp'].rolling(50).std().clip(lower=1e-8)
    
    # Forward returns over 3,6,12 bars (30s, 60s, 120s)
    for h in [3,6,12]:
        bars[f'fwd{h}']=bars['mp'].diff(h).shift(-h).fillna(0)
        bars[f'af{h}']=np.sign(bars['z'].fillna(0).values)*bars[f'fwd{h}'].values
    
    return bars.dropna(subset=['z','sp_z'])

def run():
    for pair in ['EURJPY','GBPJPY','EURUSD']:
        print(f"\n{'='*60}")
        print(f"{pair}")
        print(f"{'='*60}")
        t=load(pair)
        cost=COST[pair]
        
        for phase_name, mask in [
            ('IS',(t.index.year==2025)&(t.index.month.isin([10,11]))),
            ('OOS',(t.index.year==2025)&(t.index.month==12)),
        ]:
            sub_t=t.loc[mask]
            if len(sub_t)<1000: continue
            ndays=int(sub_t.index.normalize().nunique()*5/7)
            bars=compute_10s_bars(sub_t, pair)
            if len(bars)<200: continue
            nbars=len(bars)
            nbars_day=nbars/ndays
            
            if phase_name=='IS':
                print(f"  IS: {nbars:,d} bars (~{nbars_day:.0f}/d)")
            
            for z_thr in [1.5,2.0,2.5,3.0]:
                # Also filter by spread: only trade when spread is NOT extreme (>3 z)
                spread_ok=bars['sp_z'].abs()<3
                events=(bars['z'].abs()>z_thr)&spread_ok
                n=events.sum()
                if n<5: continue
                
                for hold in [3,6,12]:
                    col=f'af{hold}'
                    sub=bars.loc[events]
                    wr=(sub[col]>0).mean()
                    avg=sub[col].mean()
                    net=avg-cost
                    nd=n/ndays
                    
                    info=f"  {phase_name} z>{z_thr:.1f} h={hold:>2}bars n={n:>5d} {nd:>7.1f}/d WR={wr:.1%} avg={avg:+.3f}p net={net:+.3f}p"
                    
                    if phase_name=='OOS' and net>0 and nd>=30:
                        print(info)
                    elif phase_name=='IS' and n>=50:
                        print(info)
            gc.collect()

if __name__=='__main__':
    run()
