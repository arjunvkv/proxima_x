"""Analyze whether we can pass FundedNext 5-day challenge ($2,000 on $25K)."""
import time, numpy as np, pandas as pd
from pathlib import Path
from numba import njit

DATA_DIR = Path(r"C:\Trading\Agentic_Trading\proxima_x\research\fundednext_data")
CONTRACT = 100000; COMM = 3.0; POINT = 0.00001

@njit
def compute_z(c, w):
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
def compute_atr(h,l,p):
    n=len(h); a=np.full(n,np.nan)
    if n<p: return a
    for i in range(p-1,n):
        s=0.0
        for k in range(i-p+1,i+1): s+=h[k]-l[k]
        a[i]=s/p
    return a

@njit
def run_kernel(o,h,l,c,z,atr,sp,inh,zt,mh,sa,tg,ga,lot,co,cm,ms,dfi):
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
        pnl[nt]=(xp-e)*lot*co if d>0 else (e-xp)*lot*co
        pnl[nt]-=lot*cm; pnl[nt]-=sp[i]*0.00001*lot*co; nt+=1
        i+=mh if not ex else j
    return pnl[:nt]

def bt(df, zt, ms, dfi=0):
    o=df['open'].values.astype(np.float64); h=df['high'].values.astype(np.float64)
    l=df['low'].values.astype(np.float64); c=df['close'].values.astype(np.float64)
    sp=df['spread'].values.astype(np.int32)
    z=compute_z(c,50); atr=compute_atr(h,l,20)
    hrs=np.array([t.hour for t in df['datetime']], dtype=np.int32)
    return run_kernel(o,h,l,c,z,atr,sp,(hrs>=0).astype(np.int8),zt,54,2.0,1.0,0.03,0.75,CONTRACT,COMM,ms,dfi)

# Load data
data = {}
for pair in ["EURAUD","GBPAUD","AUDNZD","EURNZD","GBPCAD","GBPNZD"]:
    d = np.load(str(DATA_DIR / f"{pair}.npy"))
    df = pd.DataFrame(d, columns=["time","open","high","low","close","tick_volume","spread","real_volume"])
    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    data[pair] = df
    print(f"{pair}: {df['datetime'].min().strftime('%m/%d')} to {df['datetime'].max().strftime('%m/%d')} ({len(df)} bars)")

# Get pnl for all pairs at z>=3.5 and z>=5, both directions, no spread filter
print("\n=== ALL 6 PAIRS — z>=3.5 and z>=5, FULL-DAY, BOTH DIRS ===")
for zt in [3.5, 5.0]:
    total_net, total_trades = 0, 0
    for pair in ["EURAUD","GBPAUD","AUDNZD","EURNZD","GBPCAD","GBPNZD"]:
        pnl = bt(data[pair], zt, 50, 0)
        n=len(pnl)
        if n==0: continue
        net=pnl.sum(); w=pnl>0; nw=w.sum(); nl=(pnl<0).sum()
        den=n-(np.abs(pnl)<0.01).sum()
        wr=nw/den*100 if den else 0
        aw=pnl[w].mean() if nw else 0; al=pnl[~w].mean() if nl else 0
        trades_per_day = n / 70
        print(f"  z>={zt} {pair:<8}: {n:>3d}t ({trades_per_day:.2f}/d) {wr:>5.1f}% ${net:>+8.2f} W${aw:>+6.2f} L${al:>+6.2f}")
        total_net+=net; total_trades+=n
    print(f"  z>={zt} PORTFOLIO: {total_trades}t in 70d = {total_trades/70:.2f}/d, net ${total_net:>.2f}")

# Now check what lot size needed for $2,000 in 5 days
print("\n\n=== WHAT LOT SIZE FOR $2K IN 5 DAYS? ===")
for zt in [3.5, 5.0, 6.0]:
    all_pnls = []
    for pair in ["EURAUD","GBPAUD","AUDNZD","EURNZD","GBPCAD","GBPNZD"]:
        pnl = bt(data[pair], zt, 50, 0)
        all_pnls.extend(pnl.tolist())
    all_pnls = np.array(all_pnls)
    n = len(all_pnls)
    if n == 0: continue
    net = all_pnls.sum()
    days = 70
    trades_per_day = n / days
    pnl_per_day = net / days
    pnl_per_trade = net / n
    
    # What lot multiplier gives $400/day?
    multiplier_needed = 400.0 / pnl_per_day if pnl_per_day > 0 else float('inf')
    
    # At that multiplier, what's the worst single-day loss look like?
    # Simulate: pick the worst 1, 2, 3 consecutive trades
    worst_3_sum = sum(sorted(all_pnls)[:3]) * multiplier_needed
    
    print(f"z>={zt}: {n}t in 70d ({trades_per_day:.2f}/d), ${pnl_per_day:.2f}/d at 0.75 lot")
    if multiplier_needed < 100:
        print(f"  Need {multiplier_needed:.1f}x lot = {0.75*multiplier_needed:.2f} lots to get $400/day")
        print(f"  At that size: 3 worst trades would lose ${worst_3_sum:.2f}")
        print(f"  Daily limit: $1,250 — {'SAFE' if abs(worst_3_sum) < 1250 else 'BLOWN'} ({abs(worst_3_sum)/1250*100:.0f}% of limit)")
    else:
        print(f"  Impossible — pnl_per_day is negative or negligible")

# Now check: what if we restrict to only the best pair-direction combos?
print("\n\n=== BEST DIRECTION-BIASED PAIRS ONLY ===")
dir_biases = {"EURAUD": 1, "GBPAUD": 1, "GBPCAD": 1, "EURNZD": 2}
for zt in [5.0, 6.0]:
    all_pnls = []
    for pair, dfi in dir_biases.items():
        pnl = bt(data[pair], zt, 50, dfi)
        all_pnls.extend(pnl.tolist())
    all_pnls = np.array(all_pnls)
    n = len(all_pnls)
    if n == 0: continue
    net = all_pnls.sum()
    tpd = n / 70
    ppd = net / 70
    
    print(f"z>={zt} bias-dir ({len(dir_biases)} pairs): {n}t ({tpd:.2f}/d), ${net:.2f} net, ${ppd:.2f}/d at 0.75lot")
    
    mult = 400.0 / ppd if ppd > 0 else float('inf')
    if mult < 100:
        lot_needed = 0.75 * mult
        worst_3 = sum(sorted(all_pnls)[:3]) * mult
        print(f"  Need {lot_needed:.1f} lots for $400/day")
        print(f"  Worst 3 trades at that size: ${worst_3:.2f} ({'SAFE' if abs(worst_3)<1250 else 'BLOWN'})")
    
    # Risk-optimized: what lot size keeps max daily loss under $1,250?
    # Find the worst single trade
    worst_trade = all_pnls.min()
    max_lot_2loss = (1250 / 2) / abs(worst_trade) * 0.75  # 2 worst consecutive losses = max daily limit
    max_lot_3loss = (1250 / 3) / abs(worst_trade) * 0.75  # 3 worst consecutive losses
    print(f"  Max lot for 2-loss daily safety: {max_lot_2loss:.1f}")
    print(f"  Max lot for 3-loss daily safety: {max_lot_3loss:.1f}")
    
    # What daily PnL at the safe lot size?
    safe_lot = max_lot_3loss
    safe_ppd = ppd * (safe_lot / 0.75)
    print(f"  At safe lot ({safe_lot:.1f}): ${safe_ppd:.2f}/day")
    print(f"  Days to reach $2,000: {2000/safe_ppd:.0f}")
