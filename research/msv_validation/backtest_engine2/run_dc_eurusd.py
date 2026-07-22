"""
Dealer Capitulation on EURUSD (tightest spreads, highest liquidity).
"""
import numpy as np, pandas as pd, gc
from pathlib import Path

TICK_DIR = Path(__file__).resolve().parents[3] / 'data' / 'exness_ticks'
SC=10000

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
    df['Mid']=(df['B']+df['A'])/2
    df['Sprd']=(df['A']-df['B'])*SC
    df['MP']=df['Mid']*SC
    return df.set_index('Ts')

cost=0.15
print("="*60)
print("EURUSD — DC")
print("="*60)
t=load()

for phase_name, mask in [
    ('IS',(t.index.year==2025)&(t.index.month.isin([10,11]))),
    ('OOS',(t.index.year==2025)&(t.index.month==12)),
]:
    sub_t=t.loc[mask]
    if len(sub_t)<1000: continue
    ndays=int(sub_t.index.normalize().nunique()*5/7)
    
    # Minute bars
    mb=pd.DataFrame({
        'mp':sub_t['MP'].resample('1min').last(),
        'ms':sub_t['Sprd'].resample('1min').median(),
        'xs':sub_t['Sprd'].resample('1min').max(),
    }).dropna(subset=['mp'])
    mb['ret']=mb['mp'].diff().fillna(0)
    mb['v5']=mb['ret'].rolling(20).std()
    mb['z']=mb['ret']/mb['v5'].clip(lower=1e-8)
    mb['rm']=mb['ms'].rolling(20).median()
    mb['sr']=mb['xs']/mb['rm'].clip(lower=1e-8)
    mb['sw']=mb['sr']>2.0
    for h in [5,10,15]:
        f=mb['mp'].diff(h).shift(-h).fillna(0)
        mb[f'af{h}']=-np.sign(mb['z'].fillna(0))*f
    
    if phase_name=='IS':
        print(f"  IS: {len(mb)} bars, {ndays}d")
    
    for z_thr in [2.0,1.75,1.5]:
        events=(mb['z'].abs()>z_thr)&mb['sw']
        eidx=np.where(events.values)[0]
        if len(eidx)<5: continue
        
        all_trades=[]
        all_rec=[]
        for idx in eidx:
            dt=mb.index[idx]
            rm=mb['rm'].iloc[idx]
            sl=sub_t.loc[dt:dt+pd.Timedelta(minutes=5)]
            if len(sl)<3:
                sl=sub_t.loc[dt:dt+pd.Timedelta(minutes=10)]
            if len(sl)<3: continue
            
            sp=sl['Sprd'].values
            pp=int(sp.argmax())
            af=sp[pp:]
            if len(af)<2: continue
            thr=max(1.3*rm, rm+0.03)
            rec=np.where(af<thr)[0]
            rt=rec[0]+1 if len(rec)>0 else 999
            if rt<999: all_rec.append(rt)
            
            for h in [5,10,15]:
                if idx+h>=len(mb): continue
                all_trades.append({'h':h,'rt':rt,'v':mb[f'af{h}'].iloc[idx]})
        
        if len(all_trades)<5: continue
        
        if phase_name=='IS':
            rec_arr=np.array(all_rec)
            if len(rec_arr)<5: continue
            med_r=np.median(rec_arr)
            q3_r=np.quantile(rec_arr,0.75)
            print(f"  IS z>{z_thr}: {len(eidx)} events, med={med_r:.0f} Q3={q3_r:.0f}")
        
        for label,cond,thr_str in [
            ('ALL', lambda r:r<999, 'rec<999'),
            ('SLOW', lambda r:r>med_r and r<999, f'rec>{med_r:.0f}'),
            ('Q4', lambda r:r>q3_r and r<999, f'rec>{q3_r:.0f}'),
        ]:
            if phase_name=='IS' and label!='ALL': continue  # Only print ALL for IS
            relevant=[x for x in all_trades if cond(x['rt'])]
            if len(relevant)<3: continue
            for h in [5,10,15]:
                sub=[x for x in relevant if x['h']==h]
                if len(sub)<3: continue
                v=np.array([x['v'] for x in sub])
                wr=(v>0).mean()
                avg=np.mean(v)
                net=avg-cost
                nd=len(sub)/ndays
                if phase_name=='OOS' and wr>=0.57:
                    print(f"  OOS z>{z_thr:.1f} {label:<5s} h={h:<2}min n={len(sub):>3d} {nd:>5.2f}/d "
                          f"WR={wr:.1%} avg={avg:+.3f}p net={net:+.3f}p {thr_str}")
                elif phase_name=='IS' and wr>=0.55:
                    print(f"  IS  z>{z_thr:.1f} {label:<5s} h={h:<2}min n={len(sub):>3d} {nd:>5.2f}/d "
                          f"WR={wr:.1%} avg={avg:+.3f}p net={net:+.3f}p")
        
        # OOS: never-recovered
        if phase_name=='OOS':
            never=[x for x in all_trades if x['rt']>=999]
            if len(never)>=5:
                for h in [5,10,15]:
                    sub=[x for x in never if x['h']==h]
                    if len(sub)<3: continue
                    v=np.array([x['v'] for x in sub])
                    wr=(v>0).mean()
                    avg=np.mean(v)
                    net=avg-cost
                    nd=len(sub)/ndays
                    if wr>=0.55:
                        print(f"  OOS z>{z_thr:.1f} NEVER  h={h:<2}min n={len(sub):>3d} {nd:>5.2f}/d "
                              f"WR={wr:.1%} avg={avg:+.3f}p net={net:+.3f}p")

print("\nDone")
