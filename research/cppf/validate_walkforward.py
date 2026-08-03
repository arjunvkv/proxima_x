"""Walk-forward validation: train on Apr 21-May 31, test on Jun 1-Jul 1."""
import time, numpy as np, pandas as pd
from pathlib import Path
from numba import njit

DATA_DIR = Path(r"C:\Trading\Agentic_Trading\proxima_x\research\fundednext_data")
CONTRACT = 100000; COMM = 3.0; POINT = 0.00001
PAIRS = ["EURAUD", "GBPAUD", "GBPCAD", "EURNZD"]

def load(pair):
    rates = np.load(str(DATA_DIR / f"{pair}.npy"))
    df = pd.DataFrame(rates, columns=["time","open","high","low","close","tick_volume","spread","real_volume"])
    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    return df

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
def run_kernel(o,h,l,c,z,atr,sprd_pts,in_hours,z_thresh,max_hold,stop_a,trig_a,
               gap_a,lot,contract,comm,max_sprd_pts,dir_filter):
    n=len(o); cap=50000
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
        best_e=entry; exited=False; exit_px=0.0
        max_j=min(max_hold+1,n-i)
        for j in range(1,max_j):
            idx=i+j
            if direction>0:
                if h[idx]>best_e: best_e=h[idx]
                if best_e-entry>trig_a*atr_v:
                    ns=best_e-gap_a*atr_v
                    if ns>sl: sl=ns
                if l[idx]<=sl: exit_px=sl; exited=True; break
            else:
                if l[idx]<best_e: best_e=l[idx]
                if entry-best_e>trig_a*atr_v:
                    ns=best_e+gap_a*atr_v
                    if ns<sl: sl=ns
                if h[idx]>=sl: exit_px=sl; exited=True; break
        if not exited: exit_px=c[i+max_j-1]
        pnl=(exit_px-entry)*lot*contract if direction>0 else (entry-exit_px)*lot*contract
        pnl-=lot*comm; pnl-=sprd_pts[i]*point*lot*contract
        pnl_a[nt]=pnl; nt+=1
        i+=max_hold if not exited else j
    return pnl_a[:nt]

def run_bt(df, z_thresh, max_sprd, dir_f, stop_a=2.0, trig_a=1.0, gap_a=0.03, lot=0.75):
    o=df['open'].values.astype(np.float64); h=df['high'].values.astype(np.float64)
    l=df['low'].values.astype(np.float64); c=df['close'].values.astype(np.float64)
    sp=df['spread'].values.astype(np.int32)
    z=compute_z_numba(c,50); atr=compute_atr_numba(h,l,20)
    hrs=np.array([t.hour for t in df['datetime']], dtype=np.int32)
    inh=(hrs>=0)&(hrs<24)
    return run_kernel(o,h,l,c,z,atr,sp,inh.astype(np.int8),z_thresh,54,stop_a,trig_a,gap_a,lot,CONTRACT,COMM,max_sprd,dir_f)

def analyze(pnl):
    n=len(pnl)
    if n==0: return None
    net=pnl.sum(); wins=pnl>0; n_w=wins.sum(); n_l=(pnl<0).sum(); n_z=(np.abs(pnl)<0.01).sum()
    den=n-n_z; wr=n_w/den*100 if den else 0
    aw=pnl[wins].mean() if n_w else 0; al=pnl[~wins&(np.abs(pnl)>=0.01)].mean() if n_l else 0
    payoff=abs(aw/al) if al and al!=0 else 0
    return {'n':n,'wr':wr,'net':net,'avg_win':aw,'avg_loss':al,'payoff':payoff}

# Search space
Z_VALS = [3.0, 3.5, 4.0, 5.0, 6.0]
SPRDS = [12, 15, 20, 50]
DIRS = [(0,"BOTH"), (1,"LONG"), (2,"SHORT")]

print("=" * 130)
print("WALK-FORWARD VALIDATION")
print("Train: Apr 21 - May 31 | Test: Jun 1 - Jul 1")
print("=" * 130)

all_train_results = []
all_test_results = []

for pair in PAIRS:
    df = load(pair)
    train_df = df[df['datetime'] < '2026-06-01'].copy()
    test_df = df[df['datetime'] >= '2026-06-01'].copy()
    
    print(f"\n--- {pair} ---")
    print(f"  Train: {len(train_df)} bars | Test: {len(test_df)} bars")
    
    # === TRAINING: find best config ===
    best_train = None
    for z in Z_VALS:
        for ms in SPRDS:
            for d_code, d_name in DIRS:
                pnl = run_bt(train_df, z, ms, d_code)
                r = analyze(pnl)
                if r and r['n'] >= 3:
                    all_train_results.append({**r, 'pair':pair, 'z':z, 'sprd':ms, 'dir':d_name})
                    if best_train is None or r['net'] > best_train['net']:
                        best_train = {**r, 'z':z, 'sprd':ms, 'dir':d_name, 'dir_code':d_code}
    
    if best_train:
        z_b, ms_b, d_b, d_c = best_train['z'], best_train['sprd'], best_train['dir'], best_train['dir_code']
        print(f"  TRAIN BEST: z>={z_b:.0f} sprd<={ms_b} {d_b:<6} "
              f"{best_train['n']:>3d}t {best_train['wr']:>5.1f}% "
              f"${best_train['net']:>+8.2f} payoff={best_train['payoff']:.2f}")
        
        # === TESTING: apply best config ===
        test_pnl = run_bt(test_df, z_b, ms_b, d_c)
        r_test = analyze(test_pnl)
        if r_test:
            survive = "SURVIVES" if r_test['net']>0 else "DIES"
            all_test_results.append({**r_test, 'pair':pair, 'z':z_b, 'sprd':ms_b, 'dir':d_b})
            print(f"  TEST:      z>={z_b:.0f} sprd<={ms_b} {d_b:<6} "
                  f"{r_test['n']:>3d}t {r_test['wr']:>5.1f}% "
                  f"${r_test['net']:>+8.2f} payoff={r_test['payoff']:.2f}  {survive}")
            
            # Show individual trades
            for i in range(min(len(test_pnl), 15)):
                print(f"    t{i+1}: ${test_pnl[i]:+7.2f}")
        else:
            print(f"  TEST: NO TRADES  DIES")
    else:
        print(f"  TRAIN: NO CONFIG with >=3 trades found")

# Portfolio summary
print(f"\n{'='*130}")
print("PORTFOLIO WALK-FORWARD RESULTS")
print(f"{'PAIR':<8} {'CONFIG':<28} {'TRADES':>7} {'WR':>7} {'NET':>10} {'PAYOFF':>7}")
print("-" * 70)
total_net = 0; total_trades = 0
for r in sorted(all_test_results, key=lambda x: x['pair']):
    survive = "SURVIVES" if r['net']>0 else "DIES"
    label = f"z>={r['z']:.0f} sprd<={r['sprd']} {r['dir']}"
    print(f"{r['pair']:<8} {label:<28} {r['n']:>5d}  {r['wr']:>5.1f}%  "
          f"${r['net']:>+8.2f}  {r['payoff']:.2f}  {survive}")
    total_net += r['net']; total_trades += r['n']

print(f"{'TOTAL':<8} {'':28} {total_trades:>5d}  {'':5} ${total_net:>+8.2f}")
print(f"\nWALK-FORWARD VERDICT: ", end="")
if total_net > 0 and total_trades >= 10:
    print("PASS (positive PnL on unseen data)")
else:
    print("FAIL (negative PnL or too few trades on unseen data)")
print("=" * 130)
