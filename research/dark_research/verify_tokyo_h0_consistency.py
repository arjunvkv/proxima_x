#!/usr/bin/env python3
"""Tokyo H0 cross-sample backtest — optimized version."""

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

COSTS_BP = 0.3
HOLD = 3
LOOKBACK = 3
TOP_N = 3
MAX_POS = 3

df = pd.read_parquet(r"C:\Trading\Agentic_Trading\proxima_x\data\temp\mt5_m1_9day.parquet")
pairs_all = sorted(df["pair"].unique())

piv = df.pivot_table(index="time", columns="pair", values="close")
times = piv.index.values
T = len(piv)
hour = times.astype('datetime64[m]').astype(int) // 60 % 24
minute = times.astype('datetime64[m]').astype(int) % 60

# Pre-compute log returns as numpy arrays
close = piv.values.astype(np.float64)
rets = np.diff(np.log(close), axis=0)

# Only process 00:00 UTC bars
h0_mask = (hour[1:] == 0) & (minute[1:] == 0)
h0_idx = np.where(h0_mask)[0] + 1  # align with close indices

print(f"Pairs: {len(pairs_all)} ({', '.join(pairs_all)})")
print(f"Bars: {T:,}")
print(f"00:00 UTC bars: {len(h0_idx)}")
print()

def bt_h0(pair_list, costs_bp=COSTS_BP, hold=HOLD, lookback=LOOKBACK, top_n=TOP_N, max_pos=MAX_POS):
    pair_idx = [pairs_all.index(p) for p in pair_list]
    pair_names = [pairs_all[i] for i in pair_idx]
    trade_data = []
    
    for ii, i in enumerate(h0_idx):
        if i + hold >= T:
            continue

        # Remove expired positions
        trade_data = [t for t in trade_data if t.get("close_at", 0) > i]
        open_pairs = set(t["pair"] for t in trade_data)
        if len(open_pairs) >= max_pos:
            continue

        pair_moves = []
        for pi in pair_idx:
            pname = pairs_all[pi]
            if pname in open_pairs:
                continue
            if i - lookback < 0:
                continue
            p0 = close[i - lookback, pi]
            p1 = close[i, pi]
            if p0 <= 0:
                continue
            ret = (p1 / p0 - 1)
            pair_moves.append((pname, abs(ret), ret))

        pair_moves.sort(key=lambda x: x[1], reverse=True)

        for pname, mag, ret in pair_moves[:top_n]:
            if pname in open_pairs or len(open_pairs) >= max_pos:
                break
            if ret > 0:
                continue
            if i + 1 + hold >= T:
                continue

            pi = pairs_all.index(pname)
            entry = close[i + 1, pi]
            exit_ = close[i + hold, pi]
            if entry <= 0:
                continue
            gross = (exit_ / entry - 1)
            net = gross - costs_bp / 10000
            trade_data.append({
                "pair": pname, "pnl": net * 10000, "won": net > 0,
                "gross_bp": gross * 10000, "dt": piv.index[i],
                "close_at": i + hold,
            })
            open_pairs.add(pname)

    return trade_data

def stats(trades, label):
    if not trades:
        print(f"  {label}: 0 trades")
        return
    pnls = np.array([t["pnl"] for t in trades])
    n = len(pnls)
    wr = np.mean(pnls > 0) * 100
    mu = np.mean(pnls)
    std = np.std(pnls)
    sh = mu / std if std > 0 else 0
    t_stat = mu / (std / np.sqrt(n)) if std > 0 else 0
    gross = np.mean([t["gross_bp"] for t in trades])
    print(f"  {label}: n={n:3d}, WR={wr:5.1f}%, Mean={mu:+.2f}bp, "
          f"Sharpe={sh:.2f}, t={t_stat:.2f}, Gross={gross:+.2f}bp")

# ══════════════════════════════════════════════════════════════
# 1. BASELINE — All 7 pairs
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("1. BASELINE (7 pairs)")
print("=" * 70)
trades = bt_h0(pairs_all)
stats(trades, "Tokyo H0")

# ══════════════════════════════════════════════════════════════
# 2. PAIR UNIVERSE SIZE — 5/4/3 pairs
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. PAIR UNIVERSE SIZE")
print("=" * 70)
for n in [5, 4, 3]:
    pt = pairs_all[:n]
    stats(bt_h0(pt), "%d pairs" % n)

# ══════════════════════════════════════════════════════════════
# 3. LEAVE-ONE-PAIR-OUT
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. LEAVE-ONE-PAIR-OUT")
print("=" * 70)
for p in pairs_all:
    rp = [x for x in pairs_all if x != p]
    stats(bt_h0(rp), "Remove %s" % p)

# ══════════════════════════════════════════════════════════════
# 4. DAY-OF-WEEK
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. DAY-OF-WEEK")
print("=" * 70)
dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
by_dow = {}
for t in trades:
    d = t["dt"].dayofweek
    by_dow.setdefault(d, []).append(t["pnl"])
for d in range(5):
    pnls = np.array(by_dow.get(d, []))
    if len(pnls) < 2: continue
    print(f"  {dow_names[d]}: n={len(pnls):3d}, WR={np.mean(pnls>0)*100:.1f}%, "
          f"Mean={np.mean(pnls):+.2f}bp")

# ══════════════════════════════════════════════════════════════
# 5. WEEKLY WALK-FORWARD
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. WEEKLY CHUNKS")
print("=" * 70)
by_week = {}
for t in sorted(trades, key=lambda x: x["dt"]):
    w = t["dt"].isocalendar()[1]
    by_week.setdefault(w, []).append(t["pnl"])
for w in sorted(by_week):
    pnls = np.array(by_week[w])
    if len(pnls) < 2: continue
    print(f"  Week {w}: n={len(pnls):3d}, WR={np.mean(pnls>0)*100:.1f}%, "
          f"Mean={np.mean(pnls):+.2f}bp")

# ══════════════════════════════════════════════════════════════
# 6. COST SENSITIVITY
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("6. COST SENSITIVITY")
print("=" * 70)
for cost in [0, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0]:
    stats(bt_h0(pairs_all, costs_bp=cost), "Cost %.1fbp" % cost)

# ══════════════════════════════════════════════════════════════
# 7. ROLLING WR
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("7. ROLLING 15-TRADE WR")
print("=" * 70)
pnls_arr = np.array([t["pnl"] for t in sorted(trades, key=lambda x: x["dt"])])
window = 15
if len(pnls_arr) >= window + 3:
    wrs = [np.mean(pnls_arr[i:i+window] > 0) * 100 for i in range(len(pnls_arr) - window + 1)]
    print(f"  Min: {min(wrs):.1f}%, Max: {max(wrs):.1f}%, Mean: {np.mean(wrs):.1f}%")
    print(f"  % below 60% WR: {np.mean([w < 60 for w in wrs]) * 100:.1f}%")
else:
    print(f"  Too few trades ({len(pnls_arr)})")

# ══════════════════════════════════════════════════════════════
# 8. DRAWDOWN
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("8. MAX DRAWDOWN")
print("=" * 70)
pnls_sorted = np.array([t["pnl"] for t in sorted(trades, key=lambda x: x["dt"])])
eq = np.cumsum(pnls_sorted)
running_max = np.maximum.accumulate(eq)
dd = eq - running_max
max_dd = float(np.min(dd))

max_consec_loss = 0
cur = 0
for p in pnls_sorted:
    if p <= 0:
        cur += 1
        max_consec_loss = max(max_consec_loss, cur)
    else:
        cur = 0

print(f"  Max DD: {max_dd:.2f}bp")
print(f"  Max consec losses: {max_consec_loss}")
print(f"  Total PnL: {np.sum(pnls_sorted):.2f}bp")

# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
by_day = {}
for t in trades:
    d = t["dt"].date()
    by_day[d] = by_day.get(d, 0) + 1
print(f"  Period: {piv.index[0]} — {piv.index[-1]}")
print(f"  Trading days: {len(by_day)}")
print(f"  Trades: {len(trades)} ({len(trades)/max(len(by_day),1):.1f}/day)")
print(f"  Overall WR: {np.mean([t['won'] for t in trades])*100:.1f}%")
print(f"  Mean PnL: {np.mean([t['pnl'] for t in trades]):+.2f}bp")
print(f"  Total PnL: {np.sum(pnls_sorted):.2f}bp")
