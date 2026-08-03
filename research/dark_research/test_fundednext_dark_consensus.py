#!/usr/bin/env python3
"""Dark Consensus on FundedNext data — correct pip values and cost model."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, os

ROOT = os.path.dirname(__file__)
pairs = ["eurjpy", "eurusd", "gbpjpy"]
pair_names = ["EURJPY", "EURUSD", "GBPJPY"]
data = {}
for p in pairs:
    f = os.path.join(ROOT, f"fundednext_{p}_m1.npy")
    d = np.load(f, allow_pickle=True)
    df = pd.DataFrame(d)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    data[p] = df.set_index("time")

common = sorted(set(data["eurjpy"].index) & set(data["eurusd"].index) & set(data["gbpjpy"].index))
print(f"Common bars: {len(common)}")

close = np.column_stack([data[p].loc[common, "close"].values for p in pairs])
opens = np.column_stack([data[p].loc[common, "open"].values for p in pairs])
high = np.column_stack([data[p].loc[common, "high"].values for p in pairs])
low = np.column_stack([data[p].loc[common, "low"].values for p in pairs])
spreads = np.column_stack([data[p].loc[common, "spread"].values for p in pairs])
times_arr = np.array([int(t.timestamp()) for t in common], dtype=np.int64)

# Fix EURUSD zero spreads
for c in range(3):
    nnz = spreads[:, c][spreads[:, c] > 0]
    if len(nnz) > 0:
        print(f"  {pair_names[c]}: spread med={np.median(nnz):.1f} p90={np.percentile(nnz, 90):.1f} (non-zero)")

eurusd_nnz = spreads[:, 1][spreads[:, 1] > 0]
eurusd_med = np.median(eurusd_nnz) if len(eurusd_nnz) > 0 else 3.0
spreads[:, 1] = np.maximum(spreads[:, 1], eurusd_med)

T = close.shape[0]
rets = np.diff(np.log(close), axis=0)
up = rets > 0
consensus = up.all(axis=1) | (~up).all(axis=1)
direction = np.where(up.all(axis=1), 1.0, -1.0)
avg_mag = np.mean(np.abs(rets), axis=1)
pair_mags = np.abs(rets)
dt_all = pd.to_datetime(times_arr, unit="s")
hour_arr = dt_all.hour.values[1:]
usdjpy_proxy = close[:, 0] / close[:, 1]

MAG95 = 0.00018741
HOLD = 3
FN_COMM = 3.0
SLIP_PIPS = 0.5

def pv(pair_idx, usdjpy):
    """Pip value in USD for 1 lot."""
    if pair_idx == 1:
        return 10.0
    return 1000.0 / usdjpy

def sprd_cost(pair_idx, spread_pts, usdjpy):
    """Spread cost in USD."""
    pip_v = pv(pair_idx, usdjpy)
    return (spread_pts / 10.0) * pip_v

def slip_cost(pair_idx, usdjpy):
    """Slippage cost in USD."""
    return SLIP_PIPS * pv(pair_idx, usdjpy)

def total_cost(pair_idx, spread_pts, usdjpy):
    """Total cost per round-turn trade in USD."""
    return sprd_cost(pair_idx, spread_pts, usdjpy) + slip_cost(pair_idx, usdjpy) + FN_COMM

avg_uj = np.median(usdjpy_proxy)
print(f"\nCost model: $3 comm + {SLIP_PIPS}p slip + spread (USDJPY~{avg_uj:.1f})")
for c in range(3):
    ms = np.median(spreads[:, c])
    sp = ms / 10.0
    print(f"  {pair_names[c]}: med_sprd={ms:.0f}pt = {sp:.1f}pips -> cost=${total_cost(c, ms, avg_uj):.2f}/trade")

# Build trade list
gross_list = []
net_list = []
pair_idx_list = []
sprd_at_entry = []
entry_dts = []

for t in range(1440, T - HOLD - 1):
    if not consensus[t]: continue
    h = hour_arr[t]
    if h < 7 or h > 21: continue
    if avg_mag[t] <= MAG95: continue
    bi = int(np.argmax(pair_mags[t]))
    ep = close[t, bi]
    xp = close[t + HOLD, bi]
    raw_pnl = (xp - ep) * direction[t]  # direction = +1 LONG, -1 SHORT
    if bi == 1:
        gross = raw_pnl * 100000
    else:
        gross = raw_pnl * 100000 / usdjpy_proxy[t]
    s = spreads[t, bi]
    cost = total_cost(bi, s, usdjpy_proxy[t])
    gross_list.append(gross)
    net_list.append(gross - cost)
    pair_idx_list.append(bi)
    sprd_at_entry.append(s)
    entry_dts.append(dt_all[t])

gross_arr = np.array(gross_list)
net_arr = np.array(net_list)
n = len(net_arr)
dts_idx = pd.DatetimeIndex(entry_dts)

print(f"\n{'='*70}")
print(f"DARK CONSENSUS ON FUNDEDNEXT (Jun 8 - Jul 24, 2026)")
print(f"{'='*70}")
print(f"Bars per pair: {T}  |  Trades: {n}  |  Trades/day: {n/(T/1440):.1f}")
print(f"Consensus events: {np.sum(consensus)}")

if n < 10:
    print("Not enough trades.")
else:
    print(f"\n--- GROSS (before costs) ---")
    print(f"  Total: ${np.sum(gross_arr):,.2f}")
    print(f"  Avg/trade: ${np.mean(gross_arr):.2f}")
    print(f"  WR: {np.mean(gross_arr>0)*100:.1f}%")
    sh_g = np.mean(gross_arr) / (np.std(gross_arr)+1e-10) * np.sqrt(1440/HOLD)
    print(f"  Sharpe: {sh_g:.2f}")

    print(f"\n--- NET (FundedNext costs) ---")
    print(f"  Total: ${np.sum(net_arr):,.2f}")
    print(f"  Avg/trade: ${np.mean(net_arr):.2f}")
    print(f"  WR: {np.mean(net_arr>0)*100:.1f}%")
    sh_n = np.mean(net_arr) / (np.std(net_arr)+1e-10) * np.sqrt(1440/HOLD)
    print(f"  Sharpe: {sh_n:.2f}")
    cum = np.cumsum(net_arr)
    rm = np.maximum.accumulate(cum)
    print(f"  Max DD: ${np.min(cum - rm):.2f}")

    # Daily
    dfd = pd.DataFrame({"date": dts_idx.date, "pnl": net_arr, "pair": [pair_names[p] for p in pair_idx_list]})
    daily = dfd.groupby("date").agg(n=("pnl","count"), pnl=("pnl","sum")).reset_index()
    print(f"\n--- DAILY ---")
    print(f"  Trading days: {len(daily)}")
    print(f"  Avg trades/day: {daily['n'].mean():.1f}")
    print(f"  Avg daily PnL: ${daily['pnl'].mean():.2f}")
    print(f"  Daily WR: {np.mean(daily['pnl']>0)*100:.1f}%")
    print(f"  Best: ${daily['pnl'].max():,.2f}  Worst: ${daily['pnl'].min():,.2f}")

    print(f"\n--- PAIR DISTRIBUTION ---")
    for pn in pair_names:
        sub = dfd[dfd["pair"] == pn]
        if len(sub) == 0: continue
        p = sub["pnl"].values
        print(f"  {pn}: n={len(p):4d} ({len(p)/n*100:.0f}%)  WR={np.mean(p>0)*100:.1f}%  Avg=${np.mean(p):.2f}  Tot=${np.sum(p):,.0f}")

    print(f"\n--- COST BREAKDOWN ---")
    avg_sprd = np.mean([sprd_cost(pair_idx_list[i], sprd_at_entry[i], usdjpy_proxy[i+1440]) for i in range(n)])
    avg_slip = np.mean([slip_cost(pair_idx_list[i], usdjpy_proxy[i+1440]) for i in range(n)])
    print(f"  Commission: ${FN_COMM:.2f}/trade = ${n*FN_COMM:,.0f} total")
    print(f"  Avg spread: ${avg_sprd:.2f}/trade")
    print(f"  Avg slippage: ${avg_slip:.2f}/trade")
    print(f"  Avg total cost: ${FN_COMM+avg_sprd+avg_slip:.2f}/trade")

    print(f"\n--- GROSS vs NET ---")
    print(f"  Gross: ${np.sum(gross_arr):,.2f}")
    print(f"  Net:   ${np.sum(net_arr):,.2f}")
    drain = np.sum(gross_arr) - np.sum(net_arr)
    if np.sum(gross_arr) != 0:
        print(f"  Cost drain: ${drain:,.2f} ({drain/np.sum(gross_arr)*100:.0f}% of gross)")
