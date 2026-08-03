"""Check worst-case metrics for FundedNext Dark Consensus."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, os, itertools

ROOT = os.path.dirname(__file__)
pairs = ["eurjpy", "eurusd", "gbpjpy"]
data = {}
for p in pairs:
    d = np.load(os.path.join(ROOT, f"fundednext_{p}_m1.npy"), allow_pickle=True)
    df = pd.DataFrame(d)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    data[p] = df.set_index("time")

common = sorted(set(data["eurjpy"].index) & set(data["eurusd"].index) & set(data["gbpjpy"].index))
close = np.column_stack([data[p].loc[common, "close"].values for p in pairs])
spreads = np.column_stack([data[p].loc[common, "spread"].values for p in pairs])
eurusd_nnz = spreads[:, 1][spreads[:, 1] > 0]
spreads[:, 1] = np.maximum(spreads[:, 1], np.median(eurusd_nnz) if len(eurusd_nnz) > 0 else 3.0)

rets = np.diff(np.log(close), axis=0)
up = rets > 0
consensus = up.all(axis=1) | (~up).all(axis=1)
direction = np.where(up.all(axis=1), 1.0, -1.0)
avg_mag = np.mean(np.abs(rets), axis=1)
pair_mags = np.abs(rets)
hour_arr = pd.DatetimeIndex(common).hour.values[1:]
usdjpy = close[:, 0] / close[:, 1]
MAG95 = 0.00018741

def pv(pi, uj):
    return 10.0 if pi == 1 else 1000.0 / uj

n_list = []
g_list = []
t_dts = []
for t in range(1440, len(close) - 4):
    if not consensus[t]: continue
    if hour_arr[t] < 7 or hour_arr[t] > 21: continue
    if avg_mag[t] <= MAG95: continue
    bi = int(np.argmax(pair_mags[t]))
    ep = close[t, bi]
    xp = close[t + 3, bi]
    raw = (xp - ep) * direction[t]
    gross = raw * 100000 if bi == 1 else raw * 100000 / usdjpy[t]
    sp = spreads[t, bi] / 10.0
    c = sp * pv(bi, usdjpy[t]) + 0.5 * pv(bi, usdjpy[t]) + 3.0
    n_list.append(gross - c)
    g_list.append(gross)
    t_dts.append(common[t])

n = np.array(n_list)
g = np.array(g_list)

print(f"Total trades: {len(n)}")
print(f"Worst single trade: ${np.min(n):.2f}")
print(f"Avg trade: ${np.mean(n):.2f}")
print(f"Median trade: ${np.median(n):.2f}")
print(f"Trade WR: {np.mean(n > 0) * 100:.1f}%")
print(f"Max consecutive wins: {max(len(list(g)) for k,g in itertools.groupby(n>0) if k)}")
print(f"Max consecutive losses: {max(len(list(g)) for k,g in itertools.groupby(n>0) if not k)}")

# Worst 5-trade sliding window
if len(n) >= 5:
    w5 = np.convolve(n, np.ones(5), "valid")
    w_idx = np.argmin(w5)
    print(f"Worst 5-trade period: ${w5[w_idx]:.2f}")
    print(f"  Trades {w_idx}-{w_idx+4}: {n[w_idx:w_idx+5]}")

# Daily loss risk
dfd = pd.DataFrame({"date": [d.date() for d in t_dts], "pnl": n})
daily = dfd.groupby("date")["pnl"].sum()
print(f"\nDaily stats (34 days):")
print(f"  Worst day: ${daily.min():.2f}")
print(f"  All days positive? {all(daily > 0)}")
print(f"  Daily std: ${daily.std():.2f}")

# Value at Risk
print(f"\nRisk at different lot sizes:")
for lot in [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]:
    adj = n * lot
    worst_trade = np.min(adj)
    worst_day = (daily * lot).min()
    avg_day = (daily * lot).mean()
    five_day = (daily * lot).rolling(5).sum().min()
    print(f"  {lot:.1f} lot: avg_day=${avg_day:.0f}  worst_trade=${worst_trade:.0f}  worst_day=${worst_day:.0f}  worst_5d=${five_day if not np.isnan(five_day) else 0:.0f}")
