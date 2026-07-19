#!/usr/bin/env python3
"""Calculate dollar value of best_pair+P95+H3 strategy at 1 lot."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

BASE = r"C:\Trading\Agentic_Trading\proxima_x\research\dark_research\tick_data"
raw = {}
for p in ["eurjpy","eurusd","gbpjpy"]:
    raw[p] = {k: np.load(f"{BASE}/{p}_m1_{k}.npy") for k in ["prices","times"]}
common = sorted(set(raw["eurjpy"]["times"])&set(raw["eurusd"]["times"])&set(raw["gbpjpy"]["times"]))
idx_map = {k:{t:i for i,t in enumerate(raw[k]["times"])} for k in raw}
close = np.column_stack([raw[k]["prices"][[idx_map[k][c] for c in common],3] for k in ["eurjpy","eurusd","gbpjpy"]])
times=np.array(common,dtype=np.int64); T=close.shape[0]
rets=np.diff(np.log(close),axis=0)
up=rets>0; consensus=up.all(axis=1)|(~up).all(axis=1); direction=np.where(up.all(axis=1),1.0,-1.0)
avg_mag=np.mean(np.abs(rets),axis=1); pair_mags=np.abs(rets)

MIN_IDX=1440; costs_a=np.array([0.00008*2, 0.00005*2, 0.00010*2])
tr_idx=np.where((np.arange(len(avg_mag))>=MIN_IDX)&consensus)[0]
mag95=np.percentile(avg_mag[tr_idx],95)
te_idx=np.where(consensus&(avg_mag>mag95))[0]
te_idx=te_idx[te_idx+3<T-1]
bi=np.argmax(pair_mags[te_idx],axis=1)
usdjpy_proxy = close[:,0] / close[:,1]
LOT=100000
dollars=[]
for j,i in enumerate(te_idx):
    p=bi[j]
    net_ret = np.log(close[i+3,p]/close[i,p])*direction[te_idx[j]] - costs_a[p]
    if p==1:  # EURUSD - quote is USD
        usd_val = LOT * net_ret
    else:  # EURJPY or GBPJPY - quote is JPY, convert at USDJPY
        # P&L in JPY = LOT * ret * entry_price (notional × log return × pair price)
        jpy_pnl = LOT * net_ret * close[i, p]
        usd_val = jpy_pnl / usdjpy_proxy[i]
    dollars.append(usd_val)
dollars=np.array(dollars)

print("="*70)
print("DOLLAR VALUE: 1 lot per trade, P95+best_pair+H3")
print("="*70)
print(f"  Trades:                {len(dollars)}")
print(f"  Avg per trade:         ${np.mean(dollars):.2f}")
print(f"  Total over 3 months:   ${np.sum(dollars):.0f}")
print(f"  Win rate:              {np.mean(dollars>0)*100:.1f}%")
print(f"  Trades/day:            {len(dollars)/(T/1440):.1f}")
print(f"  Est daily:             ${np.mean(dollars)*len(dollars)/(T/1440):.0f}")
print(f"  Est monthly (21d):     ${np.mean(dollars)*len(dollars)/(T/1440)*21:.0f}")
print(f"  Est yearly (252d):     ${np.mean(dollars)*len(dollars)/(T/1440)*252:.0f}")
print(f"  Max win:               ${np.max(dollars):.0f}")
print(f"  Max loss:              ${np.min(dollars):.0f}")
print(f"  Avg win:               ${np.mean(dollars[dollars>0]):.0f}")
print(f"  Avg loss:              ${np.mean(dollars[dollars<0]):.0f}")
sh=np.mean(dollars)/np.std(dollars)*np.sqrt(1440/3)
print(f"  Sharpe (dollar):        {sh:.2f}")
pair_pct=np.bincount(bi,minlength=3)/len(bi)*100
print(f"  EURUSD: {pair_pct[1]:.0f}%  EURJPY: {pair_pct[0]:.0f}%  GBPJPY: {pair_pct[2]:.0f}%")
print(f"  Avg USDJPY proxy:      {np.mean(usdjpy_proxy):.2f}")

# OOS validation
print("\n" + "="*70)
print("OOS (Jun-Jul 2026 MT5 data)")
print("="*70)
df=pd.read_parquet(r"C:\Trading\Agentic_Trading\proxima_x\data\temp\mt5_m1_9day.parquet")
piv=df.pivot_table(index="time",columns="pair",values=["open","high","low","close"])
piv.columns=[f"{c[1]}_{c[0]}" for c in piv.columns]; piv=piv.sort_index()
coos=np.column_stack([piv[f"{pn}_close"].values.astype(np.float64) for pn in ["EURJPY","EURUSD","GBPJPY"]])
usdjpy_oos=piv["USDJPY_close"].values.astype(np.float64)
roos=np.diff(np.log(coos),axis=0)
up_oos=roos>0; cons_oos=up_oos.all(axis=1)|(~up_oos).all(axis=1); dir_oos=np.where(up_oos.all(axis=1),1.0,-1.0)
am_oos=np.mean(np.abs(roos),axis=1); pm_oos=np.abs(roos)
Toos=coos.shape[0]; dollars_oos=[]
for t in range(1440, Toos-1-3):
    if not cons_oos[t]: continue
    if am_oos[t] <= mag95: continue
    p=np.argmax(pm_oos[t])
    nr=np.log(coos[t+3,p]/coos[t,p])*dir_oos[t]-costs_a[p]
    if p==1: dv=LOT*nr
    else: dv=LOT*nr*coos[t,p]/float(usdjpy_oos[min(t,len(usdjpy_oos)-1)])
    dollars_oos.append(dv)
do=np.array(dollars_oos)
print(f"  Trades:                {len(do)}")
print(f"  Avg per trade:         ${np.mean(do):.2f}")
print(f"  Total over 18 days:    ${np.sum(do):.0f}")
print(f"  Win rate:              {np.mean(do>0)*100:.1f}%")
sh_oos=np.mean(do)/np.std(do)*np.sqrt(1440/3)
print(f"  Sharpe (dollar):        {sh_oos:.2f}")
