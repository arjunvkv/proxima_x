#!/usr/bin/env python3
"""
EXHAUSTIVE CROSS-VALIDATION: 6 two-week blocks, expanding windows, reverse.
Config: P95 mag + best_pair + H3 (no ES filter).
"""
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

BASE = r"C:\Trading\Agentic_Trading\proxima_x\research\dark_research\tick_data"
raw = {}
for p in ["eurjpy","eurusd","gbpjpy"]:
    raw[p] = {k: np.load(os.path.join(BASE,f"{p}_m1_{k}.npy")) for k in ["prices","times"]}
common = sorted(set(raw["eurjpy"]["times"])&set(raw["eurusd"]["times"])&set(raw["gbpjpy"]["times"]))
idx_map = {k:{t:i for i,t in enumerate(raw[k]["times"])} for k in raw}
close = np.column_stack([raw[k]["prices"][[idx_map[k][c] for c in common],3] for k in ["eurjpy","eurusd","gbpjpy"]])
times = np.array(common,dtype=np.int64); T=close.shape[0]
rets=np.diff(np.log(close),axis=0)
up=rets>0; consensus=up.all(axis=1)|(~up).all(axis=1); direction=np.where(up.all(axis=1),1.0,-1.0)
avg_mag=np.mean(np.abs(rets),axis=1); pair_mags=np.abs(rets)
dt_all=pd.to_datetime(times,unit="s"); hour_arr=dt_all.hour.values[1:]
MIN_IDX = 1440
costs_a = np.array([0.00008*2, 0.00005*2, 0.00010*2])

day_of_data = dt_all[1:].date
unique_days = sorted(set(day_of_data))
n_days = len(unique_days)
print(f"Unique trading days: {n_days} ({unique_days[0]} -- {unique_days[-1]})")

block_size = max(1, n_days // 6)
blocks = [unique_days[i:i+block_size] for i in range(0, n_days, block_size)][:6]

def run_config(days_train, days_test):
    tr_mask = np.isin(day_of_data, days_train)
    te_mask = np.isin(day_of_data, days_test)
    tr_mask[:MIN_IDX] = False
    tr_idx = np.where(tr_mask & consensus)[0]
    if len(tr_idx) < 50:
        return None
    mag95 = np.percentile(avg_mag[tr_idx], 95)
    te_idx = np.where(te_mask & consensus & (hour_arr >= 7) & (hour_arr <= 21) & (avg_mag > mag95))[0]
    te_idx = te_idx[te_idx + 3 < T - 1]
    if len(te_idx) < 5:
        return None
    bi = np.argmax(pair_mags[te_idx], axis=1)
    rf = np.array([np.log(close[i+3, bi[j]] / close[i, bi[j]]) for j,i in enumerate(te_idx)])
    pnl = rf * direction[te_idx] - costs_a[bi]
    n = len(pnl)
    wr = np.mean(pnl > 0) * 100
    sh = np.mean(pnl)/(np.std(pnl)+1e-10)*np.sqrt(1440/3)
    return {"n": n, "wr": wr, "sharpe": sh, "total_pips": np.sum(pnl)*10000, "mean_pips": np.mean(pnl)*10000}

print("="*80)
print("METHOD 1: Leave-one-block-out CV (train on 5 blocks, test on 1)")
print("P95 mag thresholds trained on each 5-block pool, tested on held-out block")
print("="*80)
print("Test Block          n    WR%  Sharpe  avg(p)  tot(p)  TrainM95")
print("-"*80)
xval_results = []
for i_test, test_block in enumerate(blocks):
    test_days = set(test_block)
    train_days = [d for j, b in enumerate(blocks) if j != i_test for d in b]
    res = run_config(train_days, list(test_days))
    if res:
        print(f"Block {i_test+1:<15d} {res['n']:5d} {res['wr']:5.1f} {res['sharpe']:7.3f} {res['mean_pips']:7.2f} {res['total_pips']:7.0f} {np.percentile(avg_mag[np.isin(day_of_data, train_days)&consensus&(np.arange(len(avg_mag))>=MIN_IDX)],95):.6f}")
        xval_results.append(res)

if xval_results:
    ws = [r['sharpe'] for r in xval_results]
    print(f"\n  CV Summary: min={min(ws):.3f}  max={max(ws):.3f}  mean={np.mean(ws):.3f}  all_pos={all(s>0 for s in ws)}")

print("\n"+"="*80)
print("METHOD 2: Expanding window (train on first N blocks, test on N+1)")
print("="*80)
print("Train Blocks        Test Block         n    WR%  Sharpe  avg(p)  tot(p)")
print("-"*80)
for i in range(1, len(blocks)):
    train_days = [d for j in range(i) for d in blocks[j]]
    test_days = list(blocks[i])
    res = run_config(train_days, test_days)
    if res:
        print(f"Blocks 1-{i:<7d}  Block {i+1:<10d} {res['n']:5d} {res['wr']:5.1f} {res['sharpe']:7.3f} {res['mean_pips']:7.2f} {res['total_pips']:7.0f}")

print("\n"+"="*80)
print("METHOD 3: Forward chaining (train on 7 days, test on next 7 days)")
print("="*80)
print("Train Days           Test Days            n    WR%  Sharpe  avg(p)  tot(p)")
print("-"*80)
chain_results = []
for start in range(0, n_days - 14, 7):
    mid = min(start + 7, n_days)
    end = min(mid + 7, n_days)
    tr = unique_days[start:mid]
    te = unique_days[mid:end]
    if len(te) < 3: continue
    res = run_config(tr, te)
    if res:
        chain_results.append(res)
        print(f"{str(tr[0]):10s}..{str(tr[-1]):10s} {str(te[0]):10s}..{str(te[-1]):10s} {res['n']:5d} {res['wr']:5.1f} {res['sharpe']:7.3f} {res['mean_pips']:7.2f} {res['total_pips']:7.0f}")

if chain_results:
    ws = [r['sharpe'] for r in chain_results]
    print(f"\n  Forward chaining: min={min(ws):.3f}  max={max(ws):.3f}  mean={np.mean(ws):.3f}  pos={sum(1 for w in ws if w>0)}/{len(ws)}")

print("\n"+"="*80)
print("METHOD 4: Reverse chaining (train on later 7d, test on earlier 7d)")
print("="*80)
print("Train Days           Test Days            n    WR%  Sharpe  avg(p)  tot(p)")
print("-"*80)
rev_results = []
for start in range(n_days - 14, 0, -7):
    mid = min(start + 7, n_days)
    end = min(mid + 7, n_days)
    tr = unique_days[mid:end]
    te = unique_days[start:mid]
    if len(te) < 3 or len(tr) < 3: continue
    res = run_config(tr, te)
    if res:
        rev_results.append(res)
        print(f"{str(tr[0]):10s}..{str(tr[-1]):10s} {str(te[0]):10s}..{str(te[-1]):10s} {res['n']:5d} {res['wr']:5.1f} {res['sharpe']:7.3f} {res['mean_pips']:7.2f} {res['total_pips']:7.0f}")

if rev_results:
    ws = [r['sharpe'] for r in rev_results]
    print(f"\n  Reverse chaining: min={min(ws):.3f}  max={max(ws):.3f}  mean={np.mean(ws):.3f}  pos={sum(1 for w in ws if w>0)}/{len(ws)}")

print("\n"+"="*80)
print("METHOD 5: Random 50/50 split (100 trials)")
print("="*80)
np.random.seed(42)
sub_sharpes, sub_wrs = [], []
for _ in range(100):
    tr_days = sorted(np.random.choice(unique_days, n_days//2, replace=False))
    te_days = sorted(set(unique_days) - set(tr_days))
    res = run_config(tr_days, te_days)
    if res:
        sub_sharpes.append(res['sharpe'])
        sub_wrs.append(res['wr'])
if sub_sharpes:
    print(f"  Sharpe: mean={np.mean(sub_sharpes):.3f}  min={np.min(sub_sharpes):.3f}  max={np.max(sub_sharpes):.3f}")
    print(f"  WR: mean={np.mean(sub_wrs):.1f}%  min={np.min(sub_wrs):.1f}%  max={np.max(sub_wrs):.1f}%")
    print(f"  p(neg Sharpe)={np.mean(np.array(sub_sharpes)<0)*100:.1f}%")

print("\n"+"="*80)
print("METHOD 6: All-data reference (full-sample backtest)")
print("="*80)
res = run_config(unique_days, unique_days)
if res:
    print(f"  n={res['n']:5d}  WR={res['wr']:5.1f}%  Sharpe={res['sharpe']:.3f}  avg={res['mean_pips']:.2f}p  tot={res['total_pips']:.0f}p")
