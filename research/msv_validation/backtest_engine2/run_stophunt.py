"""
Stop Hunt detection: price spikes + spread widens + immediate reversal.
Trade direction: OPPOSITE the spike (reversal trade).
This is structurally different from DC (continuation).
"""
import numpy as np, pandas as pd, gc
from pathlib import Path
from datetime import timedelta

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
    df['Sprd']=(df['A']-df['B'])*SC[pair]
    df['MP']=df['Mid']*SC[pair]
    return df.set_index('Ts')

def run():
    for pair in ['EURJPY','GBPJPY']:
        print(f"\n{'='*60}")
        print(f"{pair} — Stop Hunt Detection")
        print(f"{'='*60}")
        t=load(pair)
        sc=SC[pair]
        cost = 0.6 if pair=='GBPJPY' else 0.5
        
        # Minute bars for z-score
        mb = pd.DataFrame({
            'mp': t['MP'].resample('1min').last(),
            'msp': t['Sprd'].resample('1min').max(),
            'tc': t['MP'].resample('1min').count(),
        }).dropna(subset=['mp'])
        mb['ret1']=mb['mp'].diff().fillna(0)
        mb['v5']=mb['ret1'].rolling(20).std()
        mb['z']=mb['ret1']/mb['v5'].clip(lower=1e-8)
        mb['rm']=mb['msp'].rolling(20).median()
        mb['sr']=mb['msp']/mb['rm'].clip(lower=1e-8)
        mb['sw']=mb['sr']>2.0
        
        # Forward returns
        for h in [5,10,15]:
            mb[f'fwd{h}']=mb['mp'].diff(h).shift(-h).fillna(0)
            mb[f'af{h}']=-np.sign(mb['z'].fillna(0))*mb[f'fwd{h}']
        
        for phase_name, tmask, bmask in [
            ('IS', (t.index.year==2025)&(t.index.month.isin([10,11])),
                   (mb.index.year==2025)&(mb.index.month.isin([10,11]))),
            ('OOS', (t.index.year==2025)&(t.index.month==12),
                    (mb.index.year==2025)&(mb.index.month==12)),
        ]:
            b=mb.loc[bmask].copy()
            if len(b)<100: continue
            sub_t=t.loc[tmask]
            ndays=int(b.index.normalize().nunique()*5/7)
            
            for z_thr in [1.5, 2.0, 2.5, 3.0]:
                # Events: big move + spread widen
                events=b['z'].abs()>z_thr
                if events.sum()<5: continue
                
                # Check each event for immediate reversal at tick level
                trades={h:[] for h in [5,10,15]}
                for idx in np.where(events.values)[0]:
                    dt=b.index[idx]
                    z_dir=np.sign(b['z'].iloc[idx])
                    
                    sl_arr=sub_t.loc[dt:dt+timedelta(seconds=10)]
                    if len(sl_arr)<5: continue
                    
                    sp_arr=sl_arr['Sprd'].values
                    mp_arr=sl_arr['MP'].values
                    pp_int=int(sp_arr.argmax())
                    af_len=len(sl_arr)-pp_int
                    if af_len<5: continue
                    
                    peak_mp=float(mp_arr[pp_int])
                    post_ticks=mp_arr[pp_int+1:pp_int+5]
                    if len(post_ticks)<3: continue
                    
                    # Immediate reversal: 2 of first 3 post-peak ticks go opposite to spike direction
                    if z_dir>0:
                        immediate_reversal = sum(peak_mp > t for t in post_ticks[:3]) >= 2
                    else:
                        immediate_reversal = sum(peak_mp < t for t in post_ticks[:3]) >= 2
                    
                    if immediate_reversal:
                        # Trade the REVERSAL: opposite of the spike direction
                        for h in [5,10,15]:
                            col=f'fwd{h}'
                            v=b[col].iloc[idx]
                            trades[h].append({
                                'dt':dt,'ret':v,
                                'win':-z_dir*v>0,  # Long if spike was down, short if spike was up
                                'af':-z_dir*v,
                            })
                
                for h in [5,10,15]:
                    tr=trades[h]
                    if len(tr)<5: continue
                    tdf=pd.DataFrame(tr)
                    wr=tdf['win'].mean()
                    avg=tdf['af'].mean()
                    net=avg-cost
                    nd=len(tr)/ndays
                    if phase_name=='OOS' and wr>=0.60:
                        print(f"  OOS z>{z_thr:.1f} h={h:<2}min n={len(tr):>3d} {nd:>5.2f}/d "
                              f"WR={wr:.1%} avg={avg:+.3f}p net={net:+.3f}p")
                    elif phase_name=='IS' and len(tr)>=10 and wr>=0.55:
                        print(f"  IS  z>{z_thr:.1f} h={h:<2}min n={len(tr):>3d} {nd:>5.2f}/d "
                              f"WR={wr:.1%} avg={avg:+.3f}p net={net:+.3f}p")
            gc.collect()

if __name__=='__main__':
    run()
