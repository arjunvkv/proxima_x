"""P8: Final combined portfolio — best config per pair on FundedNext data."""
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
def run_kernel(o,h,l,c,z,atr,sprd_pts,in_hours,z_thresh,max_hold,stop_a,trig_a,
               gap_a,lot,contract,comm,max_sprd_pts,dir_filter):
    n=len(o); cap=100000
    entry_a=np.zeros(cap,dtype=np.float64); exit_a=np.zeros(cap,dtype=np.float64)
    pnl_a=np.zeros(cap,dtype=np.float64); z_a=np.zeros(cap,dtype=np.float64)
    sprd_a=np.zeros(cap,dtype=np.float64); bars_a=np.zeros(cap,dtype=np.int32)
    dir_a=np.zeros(cap,dtype=np.int32); nt=0; i=1; point=0.00001
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
        entry_a[nt]=entry; exit_a[nt]=exit_px; pnl_a[nt]=pnl; dir_a[nt]=direction
        z_a[nt]=zi; sprd_a[nt]=sprd_pts[i]; bars_a[nt]=max_hold if not exited else j
        nt+=1
        i+=max_hold if not exited else j
    return entry_a[:nt],exit_a[:nt],pnl_a[:nt],dir_a[:nt],bars_a[:nt],z_a[:nt],sprd_a[:nt]

def run_bt(df, z_thresh, max_sprd, sh, eh, dir_f, stop_a=2.0, trig_a=1.0, gap_a=0.03, lot=0.75):
    o=df['open'].values.astype(np.float64); h=df['high'].values.astype(np.float64)
    l=df['low'].values.astype(np.float64); c=df['close'].values.astype(np.float64)
    sp=df['spread'].values.astype(np.int32)
    z=compute_z_numba(c,50); atr=compute_atr_numba(h,l,20)
    hrs=np.array([t.hour for t in df['datetime']], dtype=np.int32)
    if eh>sh: inh=(hrs>=sh)&(hrs<eh)
    else: inh=(hrs>=sh)|(hrs<eh)
    return run_kernel(o,h,l,c,z,atr,sp,inh.astype(np.int8),z_thresh,54,stop_a,trig_a,gap_a,lot,CONTRACT,COMM,max_sprd,dir_f)

# === FINAL PORTFOLIO CONFIGS ===
CONFIGS = [
    # (pair, z, max_sprd, sh, eh, dir_f, dir_name, label)
    ("EURAUD", 6.0, 50, 0, 24, 1, "LONG", "z>=6.0 full-day LONG"),
    ("GBPAUD", 6.0, 50, 0, 24, 1, "LONG", "z>=6.0 full-day LONG"),
    ("GBPCAD", 3.5, 15, 12, 16, 1, "LONG", "z>=3.5 sprd<=15 12-16UTC LONG"),
    ("EURNZD", 3.0, 15, 16, 20, 2, "SHORT","z>=3.0 sprd<=15 16-20UTC SHORT"),
]

print("=" * 140)
print("P8: FINAL PORTFOLIO — Best config per pair on FundedNext Server 3")
print("Apr 21 - Jul 1 2026, lot=0.75, $3/round-turn")
print("=" * 140)

all_pairs = []
grand_total = 0
grand_trades = 0

for pair, z, ms, sh, eh, d_f, d_name, label in CONFIGS:
    df = load(pair)
    entry, ex, pnl, d, bars, zs, sprd = run_bt(df, z, ms, sh, eh, d_f)
    n = len(pnl)
    net = pnl.sum()
    wins = pnl>0
    n_w = wins.sum()
    n_l = (pnl<0).sum()
    n_z = (np.abs(pnl)<0.01).sum()
    den = n - n_z
    wr = n_w/den*100 if den else 0
    aw = pnl[wins].mean() if n_w else 0
    al = pnl[~wins & (np.abs(pnl)>=0.01)].mean() if n_l else 0
    payoff = abs(aw/al) if al and al!=0 else 0
    pf = sum(pnl[wins])/abs(sum(pnl[~wins])) if sum(pnl[~wins])!=0 else float('inf')
    peak=0; dd=0; run=0
    for p in pnl: run+=p; peak=max(peak,run); dd=max(dd,peak-run)
    hrs_label = f"{sh:02d}-{eh:02d}UTC"
    survive = "SURVIVES" if net>0 else "DIES"
    
    print(f"\n--- {pair} ({label}) ---")
    print(f"  {n:>4d} trades  {d_name:<7} {hrs_label:<10} "
          f"{wr:>5.1f}% WR  net ${net:>+8.2f}  PF={pf:.2f}  "
          f"avgW ${aw:>+6.2f}  avgL ${al:>+6.2f}  DD ${dd:.2f}  {survive}")
    
    # Show recent trades
    if n > 0:
        df['datetime_str'] = df['datetime'].dt.strftime('%m/%d %H:%M')
        print(f"  Recent trades (last 10):")
        for idx in range(max(0,n-10), n):
            dt_str = "?"
            print(f"    {d_name:<6} z={zs[idx]:+5.2f} entry={entry[idx]:.5f} "
                  f"exit={ex[idx]:.5f} sprd={sprd[idx]:.0f} held={bars[idx]:2d} "
                  f"pnl=${pnl[idx]:+6.2f}")
    
    all_pairs.append({'pair':pair,'n':n,'net':net,'wr':wr,'payoff':payoff,'pf':pf,'dd':dd})
    grand_total += net
    grand_trades += n

print(f"\n{'='*140}")
print("PORTFOLIO SUMMARY")
print(f"{'PAIR':<8} {'TRADES':>7} {'WR':>7} {'NET':>10} {'PAYOFF':>7} {'PF':>7} {'MAXDD':>8}")
print("-" * 55)
for r in all_pairs:
    print(f"{r['pair']:<8} {r['n']:>5d}  {r['wr']:>5.1f}%  ${r['net']:>+8.2f}  "
          f"{r['payoff']:.2f}  {r['pf']:.2f}  ${r['dd']:>+7.2f}")
print(f"\n  PORTFOLIO TOTAL: {grand_trades:>3d} trades  ${grand_total:>+8.2f}")
print(f"  Period: Apr 21 - Jul 1 2026 (~2.3 months)")
print(f"  Est. monthly: ${grand_total/2.3:>+7.2f} on $25K = {grand_total/2.3/25000*100:.2f}%/mo")
print(f"  Est. annualized: {grand_total/2.3/25000*100*12:.1f}%")
print(f"\n{'='*140}")
print("WARNING: All configs were optimized on this exact dataset.")
print("Results WILL be lower out of sample. This is a best-case upper bound.")
print("=" * 140)
