"""P6: Direction Asymmetry — long vs short at best config per pair."""
import time, numpy as np, pandas as pd
from pathlib import Path
from numba import njit

DATA_DIR = Path(r"C:\Trading\Agentic_Trading\proxima_x\research\fundednext_data")
CONTRACT = 100000; COMM = 3.0; POINT = 0.00001

def load(pair):
    rates = np.load(str(DATA_DIR / f"{pair}.npy"))
    df = pd.DataFrame(rates, columns=["time","open","high","low","close","tick_volume","spread","real_volume"])
    df['datetime'] = pd.to_datetime(df['time'], unit='s'); return df

@njit
def compute_z_numba(c, window):
    n=len(c); z=np.full(n,np.nan)
    if n<window+2: return z
    for i in range(window,n):
        cur=c[i]-c[i-1]; s=0.0
        for k in range(i-window,i): s+=c[k]-c[k-1]
        avg=s/window; var=0.0
        for k in range(i-window,i):
            d=(c[k]-c[k-1])-avg; var+=d*d
        var/=(window-1)
        z[i]=(cur-avg)/np.sqrt(var) if var>=1e-14 else 0.0
    return z

@njit
def compute_atr_numba(h,l,period):
    n=len(h); atr=np.full(n,np.nan)
    if n<period: return atr
    for i in range(period-1,n):
        s=0.0
        for k in range(i-period+1,i+1): s+=h[k]-l[k]
        atr[i]=s/period
    return atr

@njit
def run_kernel(o,h,l,c,z,atr,sprd_pts,in_hours,z_thresh,max_hold,
               stop_a,trig_a,gap_a,lot,contract,comm,max_sprd_pts,dir_filter):
    n=len(o); cap=100000
    pnl_a=np.zeros(cap,dtype=np.float64); nt=0; point=0.00001
    while i<n and nt<cap:
        if not in_hours[i]: i+=1; continue
        if sprd_pts[i]>max_sprd_pts: i+=1; continue
        zi=z[i-1]
        if np.isnan(zi) or np.isnan(atr[i-1]) or atr[i-1]<=0: i+=1; continue
        if abs(zi)<z_thresh: i+=1; continue
        direction=1 if zi<0 else -1
        if dir_filter==1 and direction==-1: i+=1; continue  # long only
        if dir_filter==2 and direction==1: i+=1; continue   # short only
        entry=o[i]; atr_v=atr[i-1]
        sl=entry-stop_a*atr_v if direction>0 else entry+stop_a*atr_v
        best=entry; exited=False
        max_j=min(max_hold+1,n-i)
        for j in range(1,max_j):
            idx=i+j
            if direction>0:
                if h[idx]>best: best=h[idx]
                if best-entry>trig_a*atr_v:
                    ns=best-gap_a*atr_v
                    if ns>sl: sl=ns
                if l[idx]<=sl: exited=True; break
            else:
                if l[idx]<best: best=l[idx]
                if entry-best>trig_a*atr_v:
                    ns=best+gap_a*atr_v
                    if ns<sl: sl=ns
                if h[idx]>=sl: exited=True; break
        exit_px=c[i+max_j-1] if not exited else (sl if direction>0 else sl)
        if direction>0: pnl=(exit_px-entry)*lot*contract
        else: pnl=(entry-exit_px)*lot*contract
        pnl-=lot*comm; pnl-=sprd_pts[i]*point*lot*contract
        pnl_a[nt]=pnl; nt+=1
        i+=max_hold if not exited else j
    return pnl_a[:nt]

# Fix: need to declare i before the loop
@njit
def run_kernel_fixed(o,h,l,c,z,atr,sprd_pts,in_hours,z_thresh,max_hold,
               stop_a,trig_a,gap_a,lot,contract,comm,max_sprd_pts,dir_filter):
    n=len(o); cap=100000
    pnl_a=np.zeros(cap,dtype=np.float64); nt=0; i=1; point=0.00001
    while i<n and nt<cap:
        if not in_hours[i]: i+=1; continue
        if sprd_pts[i]>max_sprd_pts: i+=1; continue
        zi=z[i-1]
        if np.isnan(zi) or np.isnan(atr[i-1]) or atr[i-1]<=0: i+=1; continue
        if abs(zi)<z_thresh: i+=1; continue
        direction=1 if zi<0 else -1
        if dir_filter==1 and direction==-1: i+=1; continue
        if dir_filter==2 and direction==1: i+=1; continue
        entry=o[i]; atr_v=atr[i-1]
        sl=entry-stop_a*atr_v if direction>0 else entry+stop_a*atr_v
        best=entry; exited=False; exit_px=0.0
        max_j=min(max_hold+1,n-i)
        for j in range(1,max_j):
            idx=i+j
            if direction>0:
                if h[idx]>best: best=h[idx]
                if best-entry>trig_a*atr_v:
                    ns=best-gap_a*atr_v
                    if ns>sl: sl=ns
                if l[idx]<=sl: exit_px=sl; exited=True; break
            else:
                if l[idx]<best: best=l[idx]
                if entry-best>trig_a*atr_v:
                    ns=best+gap_a*atr_v
                    if ns<sl: sl=ns
                if h[idx]>=sl: exit_px=sl; exited=True; break
        if not exited: exit_px=c[i+max_j-1]
        if direction>0: pnl=(exit_px-entry)*lot*contract
        else: pnl=(entry-exit_px)*lot*contract
        pnl-=lot*comm; pnl-=sprd_pts[i]*point*lot*contract
        pnl_a[nt]=pnl; nt+=1
        i+=max_hold if not exited else j
    return pnl_a[:nt]

def run_bt(df, z_thresh, max_sprd_pts, start_hour, end_hour, dir_filter=0, stop_a=2.0, trig_a=1.0, gap_a=0.03, lot=0.75):
    o=df['open'].values.astype(np.float64); h=df['high'].values.astype(np.float64)
    l=df['low'].values.astype(np.float64); c=df['close'].values.astype(np.float64)
    sp=df['spread'].values.astype(np.int32)
    z=compute_z_numba(c,50); atr=compute_atr_numba(h,l,20)
    hrs=np.array([t.hour for t in df['datetime']], dtype=np.int32)
    if end_hour>start_hour: inh=(hrs>=start_hour)&(hrs<end_hour)
    else: inh=(hrs>=start_hour)|(hrs<end_hour)
    return run_kernel_fixed(o,h,l,c,z,atr,sp,inh.astype(np.int8),z_thresh,54,stop_a,trig_a,gap_a,lot,CONTRACT,COMM,max_sprd_pts,dir_filter)

def analyze(pnl):
    n=len(pnl)
    if n==0: return None
    net=pnl.sum(); wins=pnl>0; n_w=wins.sum(); n_l=(pnl<0).sum(); n_z=(np.abs(pnl)<0.01).sum()
    den=n-n_z; wr=n_w/den*100 if den else 0
    aw=pnl[wins].mean() if n_w else 0; al=pnl[~wins&(np.abs(pnl)>=0.01)].mean() if n_l else 0
    payoff=abs(aw/al) if al and al!=0 else 0
    return {'n':n,'wr':wr,'net':net,'avg_win':aw,'avg_loss':al,'payoff':payoff}

CONFIGS = [
    ("EURAUD", 6.0, 50, (0,24), "z>=6.0 full-day"),
    ("GBPAUD", 6.0, 50, (0,24), "z>=6.0 full-day"),
    ("GBPCAD", 3.5, 15, (12,16), "z>=3.5 sprd<=15 12-16UTC"),
    ("EURNZD", 3.0, 15, (16,20), "z>=3.0 sprd<=15 16-20UTC"),
]

print("=" * 130)
print("P6: DIRECTION ASYMMETRY")
print("=" * 130)

for pair, z_thresh, max_sprd, (sh, eh), label in CONFIGS:
    df = load(pair)
    print(f"\n--- {pair} ({label}) ---")

    for dir_name, dir_code in [("LONG+SHORT", 0), ("LONG ONLY", 1), ("SHORT ONLY", 2)]:
        pnl = run_bt(df, z_thresh, max_sprd, sh, eh, dir_filter=dir_code)
        r = analyze(pnl)
        if r:
            survive = "SURVIVES" if r['net']>0 else "DIES"
            print(f"  {dir_name:<15} {r['n']:>3d}t  {r['wr']:>5.1f}%  "
                  f"net ${r['net']:>+8.2f}  W${r['avg_win']:>+7.2f}/L${r['avg_loss']:>+7.2f}  {survive}")
