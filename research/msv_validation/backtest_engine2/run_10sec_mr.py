"""
10-second bar MEAN REVERSION.
Trade AGAINST large 10s moves (z-score > threshold).
The empirical finding: 10s moves are mean-reverting (~55% WR).
Filter by tight spread to minimize cost.
Target: 30+ trades/day with positive net expectancy.
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

def run():
    for pair in ['EURJPY','GBPJPY','EURUSD']:
        print(f"\n{'='*70}")
        print(f"{pair}")
        print(f"{'='*70}")
        t=load(pair)
        cost=COST[pair]
        
        for phase_name, mask in [
            ('IS',(t.index.year==2025)&(t.index.month.isin([10,11]))),
            ('OOS',(t.index.year==2025)&(t.index.month==12)),
        ]:
            sub_t=t.loc[mask]
            if len(sub_t)<1000: continue
            ndays=int(sub_t.index.normalize().nunique()*5/7)
            
            # 10s bars
            mp=sub_t['MP'].values
            sp=sub_t['Sprd'].values
            idx=sub_t.index
            b_idx=idx.floor('10s')
            
            bars=pd.DataFrame({'mp':mp,'sp':sp}, index=idx)
            bars=bars.groupby(b_idx).agg({'mp':'last','sp':['mean','min','max']})
            bars.columns=['mp','sp_avg','sp_min','sp_max']
            bars=bars.dropna(subset=['mp'])
            bars['sp']=bars['sp_avg']
            
            bars['ret']=bars['mp'].diff().fillna(0)
            bars['z']=(bars['ret']-bars['ret'].rolling(50).mean())/bars['ret'].rolling(50).std().clip(lower=1e-8)
            
            # Spread percentile (compression filter)
            bars['sp_pct']=bars['sp'].rolling(200).rank(pct=True)
            
            # Forward returns: mean reversion = trade AGAINST 10s move
            for h in [3,6,12,24]:  # 30s, 60s, 120s, 240s
                bars[f'fwd{h}']=bars['mp'].diff(h).shift(-h).fillna(0)
                # Mean reversion: -sign(z) * fwd (trade against the move)
                bars[f'mr{h}']=-np.sign(bars['z'].fillna(0).values)*bars[f'fwd{h}'].values
            
            bars=bars.dropna(subset=['z','sp'])
            if len(bars)<200: continue
            
            if phase_name=='IS':
                print(f"  {phase_name}: {len(bars):,d} bars ({len(bars)/ndays:.0f}/d)")
            
            for z_thr in [2.0,2.5,3.0,3.5]:
                # Test multiple spread filters
                for sp_pct_thr in [0.0, 0.25, 0.50]:  # 0=no filter, <25th pct, <50th pct
                    if sp_pct_thr>0:
                        sp_ok=bars['sp_pct']<=sp_pct_thr
                        events=(bars['z'].abs()>z_thr)&sp_ok
                        sp_label=f'sp<p{int(sp_pct_thr*100)}'
                    else:
                        events=bars['z'].abs()>z_thr
                        sp_label='all_sp'
                    
                    n=events.sum()
                    if n<10: continue
                    
                    z_dir_label='mr'
                    for hold in [3,6,12,24]:
                        col=f'{z_dir_label}{hold}'
                        sub=bars.loc[events]
                        wr=(sub[col]>0).mean()
                        avg=sub[col].mean()
                        net=avg-cost
                        nd=n/ndays
                        
                        if phase_name=='OOS' and nd>=30:
                            if net>0 or wr>=0.55:
                                print(f"  OOS z>{z_thr:.1f} {sp_label:<10s} h={hold:<2}bars n={n:>5d} {nd:>7.0f}/d "
                                      f"WR={wr:.1%} avg={avg:+.3f}p net={net:+.3f}p")
                        elif phase_name=='IS' and n>=50:
                            if net>0 or wr>=0.55:
                                print(f"  IS  z>{z_thr:.1f} {sp_label:<10s} h={hold:<2}bars n={n:>5d} {nd:>7.0f}/d "
                                      f"WR={wr:.1%} avg={avg:+.3f}p net={net:+.3f}p")
            gc.collect()

if __name__=='__main__':
    run()
