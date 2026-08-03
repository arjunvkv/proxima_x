"""Refined walk-forward: direction-constrained + high-z only."""
import time, numpy as np, pandas as pd
from pathlib import Path
from numba import njit

DATA_DIR = Path(r"C:\Trading\Agentic_Trading\proxima_x\research\fundednext_data")
CONTRACT = 100000; COMM = 3.0; POINT = 0.00001

# Direction bias from P6 (structural, not period-specific)
PAIRS = ["EURAUD", "GBPAUD", "GBPCAD", "EURNZD"]
DIR_BIAS = {"EURAUD": (1, "LONG"), "GBPAUD": (1, "LONG"),
            "GBPCAD": (1, "LONG"), "EURNZD": (2, "SHORT")}

def load(pair):
    rates = np.load(str(DATA_DIR / f"{pair}.npy"))
    df = pd.DataFrame(rates, columns=["time","open","high","low","close","tick_volume","spread","real_volume"])
    df['datetime'] = pd.to_datetime(df['time'], unit='s'); return df

@njit
def compute_z_numba(c, w):
    n=len(c); z=np.full(n,np.nan)
    if n<w+2: return z
    for i in range(w,n):
        cur=c[i]-c[i-1]; s=0.0
        for k in range(i-w,i): s+=c[k]-c[k-1]
        avg=s/w; var=0.0
        for k in range(i-w,i):
            d=(c[k]-c[k-1])-avg; var+=d*d
        var/=(w-1); z[i]=(cur-avg)/np.sqrt(var) if var>=1e-14 else 0.0
    return z

@njit
def compute_atr_numba(h,l,p):
    n=len(h); a=np.full(n,np.nan)
    if n<p: return a
    for i in range(p-1,n):
        s=0.0
        for k in range(i-p+1,i+1): s+=h[k]-l[k]
        a[i]=s/p; return a

@njit
def compute_atr_numba_fixed(h,l,p):
    n=len(h); a=np.full(n,np.nan)
    if n<p: return a
    for i in range(p-1,n):
        s=0.0
        for k in range(i-p+1,i+1): s+=h[k]-l[k]
        a[i]=s/p
    return a

@njit
def kernel(o,h,l,c,z,atr,sp,inh,zt,mh,sa,tg,ga,lot,co,cm,ms,dfi):
    n=len(o); cap=50000; pnl=np.zeros(cap,dtype=np.float64); nt=0; i=1
    while i<n and nt<cap:
        if not inh[i]: i+=1; continue
        if sp[i]>ms: i+=1; continue
        zi=z[i-1]
        if np.isnan(zi) or np.isnan(atr[i-1]) or atr[i-1]<=0: i+=1; continue
        if abs(zi)<zt: i+=1; continue
        d=1 if zi<0 else -1
        if dfi==1 and d==-1: i+=1; continue
        if dfi==2 and d==1: i+=1; continue
        e=o[i]; av=atr[i-1]; sl=e-sa*av if d>0 else e+sa*av; b=e; ex=False; xp=0.0
        mj=min(mh+1,n-i)
        for j in range(1,mj):
            idx=i+j
            if d>0:
                if h[idx]>b: b=h[idx]
                if b-e>tg*av:
                    ns=b-ga*av
                    if ns>sl: sl=ns
                if l[idx]<=sl: xp=sl; ex=True; break
            else:
                if l[idx]<b: b=l[idx]
                if e-b>tg*av:
                    ns=b+ga*av
                    if ns<sl: sl=ns
                if h[idx]>=sl: xp=sl; ex=True; break
        if not ex: xp=c[i+mj-1]
        p=(xp-e)*lot*co if d>0 else (e-xp)*lot*co
        p-=lot*cm; p-=sp[i]*0.00001*lot*co
        pnl[nt]=p; nt+=1; i+=mh if not ex else j
    return pnl[:nt]

def run_bt(df, zt, ms, dfi):
    o=df['open'].values.astype(np.float64); h=df['high'].values.astype(np.float64)
    l=df['low'].values.astype(np.float64); c=df['close'].values.astype(np.float64)
    sp=df['spread'].values.astype(np.int32)
    z=compute_z_numba(c,50); atr=compute_atr_numba_fixed(h,l,20)
    hrs=np.array([t.hour for t in df['datetime']], dtype=np.int32)
    return kernel(o,h,l,c,z,atr,sp,(hrs>=0).astype(np.int8),zt,54,2.0,1.0,0.03,0.75,CONTRACT,COMM,ms,dfi)

def analyze(pnl):
    n=len(pnl)
    if n==0: return None
    net=pnl.sum(); w=pnl>0; nw=w.sum(); nl=(pnl<0).sum(); nz=(np.abs(pnl)<0.01).sum()
    dn=n-nz; wr=nw/dn*100 if dn else 0
    aw=pnl[w].mean() if nw else 0; al=pnl[~w&(np.abs(pnl)>=0.01)].mean() if nl else 0
    return {'n':n,'wr':wr,'net':net,'aw':aw,'al':al,'pr':abs(aw/al) if al and al!=0 else 0}

Z_VALS = [4.0, 5.0, 6.0]
SPRDS = [12, 15, 20, 50]

print("=" * 130)
print("REFINED WALK-FORWARD (direction-constrained, z>=4)")
print("Train: Apr 21 - May 31 | Test: Jun 1 - Jul 1")
print("=" * 130)

for phase, split_df_fn, label in [("TRAIN", lambda df: df[df['datetime']<'2026-06-01'], "Train"), 
                                    ("TEST", lambda df: df[df['datetime']>='2026-06-01'], "Test")]:
    if phase == "TEST":
        print(f"\n{'='*130}")
        print("APPLYING BEST TRAIN CONFIGS TO TEST DATA")
        print("=" * 130)
    
    all_r = []
    for pair in PAIRS:
        df = load(pair)
        sd = split_df_fn(df)
        d_code, d_name = DIR_BIAS[pair]
        
        if phase == "TRAIN":
            # Search for best config on training data
            best = None
            for z in Z_VALS:
                for ms in SPRDS:
                    pnl = run_bt(sd, z, ms, d_code)
                    r = analyze(pnl)
                    if r and r['n'] >= 3:
                        if best is None or r['net'] > best['net']:
                            best = {**r, 'z':z, 'sprd':ms}
            
            if best:
                print(f"{pair:<8} {label:<5} BEST: z>={best['z']:.0f} sprd<={best['sprd']} {d_name:<6} "
                      f"{best['n']:>3d}t {best['wr']:>5.1f}% ${best['net']:>+8.2f}")
                # Store best config for test phase
                globals()[f'{pair}_CONFIG'] = (best['z'], best['sprd'], d_code)
            else:
                print(f"{pair:<8} {label:<5} NO CONFIG found")
                globals()[f'{pair}_CONFIG'] = None
        
        else:  # TEST
            cfg = globals().get(f'{pair}_CONFIG')
            if cfg is None:
                print(f"{pair:<8} {label:<5} SKIP (no train config)")
                continue
            z_b, ms_b, d_c = cfg
            pnl = run_bt(sd, z_b, ms_b, d_c)
            r = analyze(pnl)
            if r and r['n'] > 0:
                survive = "SURVIVES" if r['net']>0 else "DIES"
                all_r.append({**r, 'pair':pair, 'z':z_b, 'sprd':ms_b, 'dir':d_name})
                print(f"{pair:<8} {label:<5} z>={z_b:.0f} sprd<={ms_b} {d_name:<6} "
                      f"{r['n']:>3d}t {r['wr']:>5.1f}% ${r['net']:>+8.2f} pr={r['pr']:.2f}  {survive}")
                # Show first 5 trades
                for i in range(min(5, len(pnl))):
                    print(f"         t{i+1}: ${pnl[i]:+7.2f}")
            else:
                print(f"{pair:<8} {label:<5} z>={z_b:.0f} sprd<={ms_b} NO TRADES")

    if phase == "TEST":
        print(f"\n{'='*130}")
        print("PORTFOLIO WALK-FORWARD RESULT")
        print(f"{'PAIR':<8} {'CONFIG':<26} {'TRADES':>7} {'WR':>7} {'NET':>10}")
        print("-" * 60)
        total_n=0; total_net=0
        for r in sorted(all_r, key=lambda x: x['pair']):
            label=f"z>={r['z']:.0f} sprd<={r['sprd']} {r['dir']}"
            sv="SURVIVES" if r['net']>0 else "DIES"
            print(f"{r['pair']:<8} {label:<26} {r['n']:>5d} {r['wr']:>5.1f}% ${r['net']:>+8.2f}  {sv}")
            total_n+=r['n']; total_net+=r['net']
        print(f"{'TOTAL':<8} {'':26} {total_n:>5d} {'':5} ${total_net:>+8.2f}")
        print(f"\nVERDICT: ", end="")
        if total_net > 0 and total_n >= 10:
            print("PASS (positive PnL on unseen data, direction-constrained)")
        else:
            print("FAIL")
        print("=" * 130)
