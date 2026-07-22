"""
Unified DC + Stop Hunt model:
Every z>threshold + spread widen event is classified by recovery speed.
- Fast recovery → reversal (stop hunt): trade OPPOSITE spike direction
- Slow recovery → continuation (DC): trade WITH spike direction
Combined: nearly every large spike is tradeable.
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
    df['Sprd']=(df['A']-df['B'])*SC[pair]
    df['MP']=df['Mid']*SC[pair]
    return df.set_index('Ts')

def run():
    for pair in ['EURJPY','GBPJPY']:
        print(f"\n{'='*70}")
        print(f"{pair} — UNIFIED: DC + Stop Hunt")
        print(f"{'='*70}")
        t=load(pair)
        sc=SC[pair]
        cost = 0.6 if pair=='GBPJPY' else 0.5
        
        # 1min bars
        mb=pd.DataFrame({
            'mp':t['MP'].resample('1min').last(),
            'ms':t['Sprd'].resample('1min').median(),
            'xs':t['Sprd'].resample('1min').max(),
            'tc':t['MP'].resample('1min').count(),
        }).dropna(subset=['mp'])
        mb['ret']=mb['mp'].diff().fillna(0)
        mb['v']=mb['ret'].rolling(20).std()
        mb['z']=mb['ret']/mb['v'].clip(lower=1e-8)
        mb['rm']=mb['ms'].rolling(20).median()
        mb['sr']=mb['xs']/mb['rm'].clip(lower=1e-8)
        mb['sw']=mb['sr']>2.0
        
        for h in [5,10,15]:
            mb[f'fwd{h}']=mb['mp'].diff(h).shift(-h).fillna(0)
        
        for phase_name, tmask, bmask in [
            ('IS',(t.index.year==2025)&(t.index.month.isin([10,11])),
                  (mb.index.year==2025)&(mb.index.month.isin([10,11]))),
            ('OOS',(t.index.year==2025)&(t.index.month==12),
                   (mb.index.year==2025)&(mb.index.month==12)),
        ]:
            b=mb.loc[bmask].copy()
            if len(b)<100: continue
            sub_t=t.loc[tmask]
            ndays=int(b.index.normalize().nunique()*5/7)
            
            for z_thr in [1.5,1.75,2.0,2.5]:
                events=b['z'].abs()>z_thr
                n=events.sum()
                if n<10: continue
                
                # Measure recovery for each event
                results={'dc':[],'sh':[],'all':[]}
                for idx in np.where(events.values)[0]:
                    dt=b.index[idx]
                    rm=b['rm'].iloc[idx]
                    z_dir=np.sign(b['z'].iloc[idx])
                    
                    # Measure spread recovery
                    sl=sub_t.loc[dt:dt+pd.Timedelta(minutes=5)]
                    if len(sl)<3:
                        sl=sub_t.loc[dt:dt+pd.Timedelta(minutes=10)]
                    if len(sl)<3: continue
                    
                    sp_arr=sl['Sprd'].values
                    mp_arr=sl['MP'].values
                    pp_int=int(sp_arr.argmax())
                    af=sp_arr[pp_int:]
                    if len(af)<3: continue
                    
                    thr=1.3*rm
                    recov=np.where(af<thr)[0]
                    rec_t=recov[0]+1 if len(recov)>0 else 999
                    
                    # Classify: fast recovery = stop hunt, slow = DC
                    # Stop hunt also requires immediate reversal check
                    if pp_int+3<len(mp_arr):
                        pk_mp=mp_arr[pp_int]
                        post=mp_arr[pp_int+1:pp_int+4]
                        if z_dir>0:
                            reversal_ticks=sum(pk_mp>t for t in post)
                        else:
                            reversal_ticks=sum(pk_mp<t for t in post)
                    else:
                        reversal_ticks=0
                    
                    # DC: signal_dir = z_dir (same direction), but filtered by slow recovery
                    # Stop hunt: signal_dir = -z_dir (opposite), filtered by fast recovery
                    for h in [5,10,15]:
                        col=f'fwd{h}'
                        if idx+h>=len(b): continue
                        v=b[col].iloc[idx]
                        dc_af=z_dir*v  # DC: trade with spike
                        sh_af=-z_dir*v  # SH: trade against spike
                        
                        results['all'].append({'dt':dt,'rec_t':rec_t,'rev':reversal_ticks>=2,
                                               'dir':z_dir,'h':h,'dc_af':dc_af,'sh_af':sh_af,
                                               'v':v, 'dc_win':dc_af>0, 'sh_win':sh_af>0})
                        
                        # DC: slow recovery (rec_t > threshold from IS)
                        # SH: fast recovery (rec_t <= threshold) AND reversal ticks
                
                if not results['all']: continue
                rdf=pd.DataFrame(results['all'])
                
                if phase_name=='IS':
                    # Compute IS thresholds
                    rec_vals=rdf.groupby('dt')['rec_t'].first().values
                    rec_clean=rec_vals[rec_vals<999]
                    if len(rec_clean)<5: continue
                    dc_thr=np.quantile(rec_clean,0.5)
                    print(f"  IS: {len(rdf['dt'].unique())} events, rec_med={np.median(rec_clean):.0f}")
                    
                    # Print IS results for DC (slow recovery) and SH (fast reversal)
                    for label, cond_dir, af_col, win_col, thr_cond in [
                        ('DC', lambda r:r['rec_t']>dc_thr and r['rec_t']<999, 'dc_af', 'dc_win', f'r>{dc_thr:.0f}'),
                        ('SH', lambda r:r['rec_t']<=dc_thr and r['rec_t']<999 and r['rev'], 'sh_af', 'sh_win', f'r<={dc_thr:.0f}+rev'),
                    ]:
                        for h in [5,10,15]:
                            sub=rdf[(rdf['h']==h) & rdf.apply(cond_dir,axis=1)]
                            if len(sub)<5: continue
                            wr=sub[win_col].mean()
                            avg=sub[af_col].mean()
                            net=avg-cost
                            nd=len(sub)/ndays
                            if wr>=0.60:
                                print(f"    IS {label:<3s} z>{z_thr:.1f} h={h:<2}min n={len(sub):>3d} {nd:>5.2f}/d "
                                      f"WR={wr:.1%} avg={avg:+.3f}p net={net:+.3f}p {thr_cond}")
                
                else:  # OOS
                    for label, cond_dir, af_col, win_col, thr_cond in [
                        ('DC', lambda r:r['rec_t']>dc_thr and r['rec_t']<999, 'dc_af', 'dc_win', f'r>{dc_thr:.0f}'),
                        ('SH', lambda r:r['rec_t']<=dc_thr and r['rec_t']<999 and r['rev'], 'sh_af', 'sh_win', f'r<={dc_thr:.0f}+rev'),
                    ]:
                        for h in [5,10,15]:
                            sub=rdf[(rdf['h']==h) & rdf.apply(cond_dir,axis=1)]
                            if len(sub)<5: continue
                            wr=sub[win_col].mean()
                            avg=sub[af_col].mean()
                            net=avg-cost
                            nd=len(sub)/ndays
                            if wr>=0.60:
                                print(f"    OOS {label:<3s} z>{z_thr:.1f} h={h:<2}min n={len(sub):>3d} {nd:>5.2f}/d "
                                      f"WR={wr:.1%} avg={avg:+.3f}p net={net:+.3f}p {thr_cond}")
            gc.collect()

if __name__=='__main__':
    run()
