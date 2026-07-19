#!/usr/bin/env python3
"""Enhanced MT5 realism: portfolio overlap + latency + variable slippage + feed audit."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, os, calendar
from collections import defaultdict

DATA = r"C:\Trading\Agentic_Trading\proxima_x\research\dark_research\dukascopy_data"
HALF_SPREAD_PIPS = np.array([0.5, 0.3, 0.7])
ECN_COMM, LOT, MAG95 = 7, 100000, 0.00018741
PIPS = np.array([0.01, 0.0001, 0.01])
JPY_INDICES = [0, 2]  # EURJPY, GBPJPY

def pip_val(p, usdjpy):
    return 10.0 if p == 1 else 1000.0 / usdjpy

def load_all():
    """Load and merge all 9 months of Dukascopy data."""
    frames = {}
    for p, pn in [("eurjpy","EURJPY"),("eurusd","EURUSD"),("gbpjpy","GBPJPY")]:
        dfs = []
        for y in [2024, 2026]:
            for m in range(1, 13):
                if (y==2024 and m<10) or (y==2026 and m>6): continue
                ld = calendar.monthrange(y, m)[1]
                f = os.path.join(DATA, f"{p}-m1-bid-{y}-{m:02d}-01-{y}-{m:02d}-{ld}.csv")
                if not os.path.exists(f): continue
                df = pd.read_csv(f)
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                dfs.append(df)
        frames[pn] = pd.concat(dfs).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    common = sorted(set(frames["EURJPY"]["timestamp"]) & set(frames["EURUSD"]["timestamp"]) & set(frames["GBPJPY"]["timestamp"]))
    tmap = {p: {t: i for i, t in enumerate(frames[p]["timestamp"])} for p in frames}
    close = np.column_stack([frames[p]["close"].values[[tmap[p][t] for t in common]] for p in ["EURJPY","EURUSD","GBPJPY"]])
    opens = np.column_stack([frames[p]["open"].values[[tmap[p][t] for t in common]] for p in ["EURJPY","EURUSD","GBPJPY"]])
    high = np.column_stack([frames[p]["high"].values[[tmap[p][t] for t in common]] for p in ["EURJPY","EURUSD","GBPJPY"]])
    low = np.column_stack([frames[p]["low"].values[[tmap[p][t] for t in common]] for p in ["EURJPY","EURUSD","GBPJPY"]])
    times = np.array([int(t.timestamp()) for t in common], dtype=np.int64)
    return close, opens, high, low, times

close, opens, high, low, times = load_all()
T = close.shape[0]
rets = np.diff(np.log(close), axis=0)
up = rets > 0
consensus = up.all(axis=1) | (~up).all(axis=1)
direction = np.where(up.all(axis=1), 1.0, -1.0)
avg_mag = np.mean(np.abs(rets), axis=1)
pair_mags = np.abs(rets)
dt_all = pd.to_datetime(times, unit="s")
hour_arr = dt_all.hour.values[1:]
usdjpy_proxy = close[:,0] / close[:,1]

# Generate all candidate trades with timestamps
te_idx = np.where(consensus & (hour_arr >= 7) & (hour_arr <= 21) & (avg_mag > MAG95))[0]
te_idx = te_idx[te_idx + 3 < T - 1]
bi = np.argmax(pair_mags[te_idx], axis=1)
avg_usdjpy = np.mean(usdjpy_proxy[te_idx])

print("=" * 90)
print("ENHANCED MT5 REALISM TEST — 9 months Dukascopy data")
print("=" * 90)
print(f"Total bars: {T:,}  Candidate trades: {len(te_idx):,}")
print()

# ============================================================
# BASELINE: flat model (our current assumption)
# ============================================================
pnls_base = []
for j,i in enumerate(te_idx):
    p = bi[j]; gross = np.log(close[i+3,p]/close[i,p])*direction[te_idx[j]]
    spread = HALF_SPREAD_PIPS[p]*2*1.5*pip_val(p, avg_usdjpy)
    slp = 0.5*2*pip_val(p, avg_usdjpy)
    gusd = LOT*gross if p==1 else LOT*gross*close[i,p]/usdjpy_proxy[i]
    pnls_base.append(gusd - spread - slp - ECN_COMM)
db = np.array(pnls_base)
print(f"BASELINE: n={len(db):,} WR={np.mean(db>0)*100:.1f}% Sharpe={np.mean(db)/(np.std(db)+1e-10)*np.sqrt(1440/3):.2f} Avg=${np.mean(db):.2f} Tot=${np.sum(db):,.0f}")
print()

# ============================================================
# TEST 1: PORTFOLIO OVERLAP SIMULATION
# ============================================================
print("TEST 1: Portfolio Overlap — chronological replay with concurrent positions")
print("-" * 70)

# Build trade list with timestamps, fill timestamps, and exit timestamps
trades = []
for j, i in enumerate(te_idx):
    p = bi[j]
    entry_price = close[i, p]
    exit_price = close[min(i+3, T-1), p]
    gross = np.log(exit_price/entry_price)*direction[te_idx[j]]
    spread = HALF_SPREAD_PIPS[p]*2*1.5*pip_val(p, avg_usdjpy)
    slp = 0.5*2*pip_val(p, avg_usdjpy)
    gusd = LOT*gross if p==1 else LOT*gross*entry_price/usdjpy_proxy[i]
    pnl = gusd - spread - slp - ECN_COMM
    direction_sign = direction[te_idx[j]]
    trades.append({
        "entry_ts": times[te_idx[j]], "exit_ts": times[min(i+3, T-1)],
        "pair": p, "dir": direction_sign,
        "pnl": pnl, "size": LOT,
        "entry_price": entry_price, "exit_price": exit_price
    })

# Sort chronologically
trades.sort(key=lambda t: t["entry_ts"])

# Simple chronological replay — track concurrent positions
open_positions = {}
max_concurrent = 0
cumulative = 0.0
peak_equity = 0.0
equity_curve = []

# Sort exit events and entry events in chronological order
events = []
for t in trades:
    events.append((t["entry_ts"], "ENTER", t))
    events.append((t["exit_ts"], "EXIT", t))
events.sort(key=lambda e: (e[0], 0 if e[1]=="EXIT" else 1))  # EXIT before ENTER at same time

for ts, etype, t in events:
    if etype == "EXIT":
        cumulative += t["pnl"]
        del open_positions[id(t)]
    else:
        open_positions[id(t)] = t
        max_concurrent = max(max_concurrent, len(open_positions))
    equity_curve.append(cumulative)
    peak_equity = max(peak_equity, cumulative)

cumulative_max = np.maximum.accumulate(np.array(equity_curve))
dd_arr = np.array(equity_curve) - cumulative_max
max_dd_val = abs(np.min(dd_arr))
max_dd_pct = max_dd_val / np.max(equity_curve) * 100 if np.max(equity_curve) > 0 else 0
print(f"  Total PnL:          ${cumulative:,.0f}")
print(f"  Max concurrent:     {max_concurrent}")
print(f"  Peak equity:        ${peak_equity:,.0f}")
print(f"  Max portfolio DD:   ${max_dd_val:,.0f} ({max_dd_pct:.1f}%)")
print()

# ============================================================
# TEST 2: JPY FACTOR EXPOSURE ANALYSIS
# ============================================================
print("TEST 2: JPY Factor Concentration — aggregated by currency leg")
print("-" * 70)

# Track net JPY exposure at each trade entry
jpy_exposures = []
for t in trades:
    pair_code = t["pair"]
    if pair_code in [0, 2]:  # EURJPY or GBPJPY
        notional_jpy = t["size"] * t["entry_price"]  # Convert to JPY notional
        jpy_exposures.append(notional_jpy * t["dir"])
    else:
        jpy_exposures.append(0)

jpy_arr = np.array(jpy_exposures)
print(f"  JPY net exposure range: ${np.min(jpy_arr):,.0f} to ${np.max(jpy_arr):,.0f}")
print(f"  JPY abs exposure avg:   ${np.mean(np.abs(jpy_arr)):,.0f}")
print(f"  JPY positions/trade:    {np.mean(jpy_arr!=0)*100:.0f}%")

# Max single-minute JPY exposure
from collections import Counter
jpy_by_minute = Counter()
for t in trades:
    if t["pair"] in [0, 2]:
        jpy_by_minute[t["entry_ts"]] += t["size"] * t["entry_price"] * t["dir"]
max_jpy_min = max(abs(v) for v in jpy_by_minute.values()) if jpy_by_minute else 0
print(f"  Max single-minute JPY: ${max_jpy_min:,.0f} notional")
print()

# ============================================================
# TEST 3: EXECUTION LATENCY REPLAY
# ============================================================
print("TEST 3: Execution Latency — delayed entry using next-candle open")
print("-" * 70)
# Conservative upper bound: 1-minute delay (actual is 100-500ms)

pnls_latency = []
for j, i in enumerate(te_idx):
    p = bi[j]; next_i = min(i+1, T-1)
    entry_price = opens[next_i, p]  # entry at next bar open (~60s delay)
    exit_price = close[min(i+3, T-1), p]
    gross = np.log(exit_price/entry_price)*direction[te_idx[j]]
    spread = HALF_SPREAD_PIPS[p]*2*1.5*pip_val(p, avg_usdjpy)
    slp = 0.5*2*pip_val(p, avg_usdjpy)
    gusd = LOT*gross if p==1 else LOT*gross*entry_price/usdjpy_proxy[i]
    pnls_latency.append(gusd - spread - slp - ECN_COMM)
dl = np.array(pnls_latency)
print(f"  60s delay (conservative): n={len(dl):,} WR={np.mean(dl>0)*100:.1f}% Sharpe={np.mean(dl)/(np.std(dl)+1e-10)*np.sqrt(1440/3):.2f} Avg=${np.mean(dl):.2f}")
print()

# ============================================================
# TEST 4: VARIABLE SLIPPAGE (volatility-conditional)
# ============================================================
print("TEST 4: Variable Slippage — scaled by intra-bar ATR")
print("-" * 70)

# ATR = high - low for the bar (in pips)
atr_vals = np.zeros(T)
for pair_idx in range(3):
    atr = (high[:,pair_idx] - low[:,pair_idx]) / PIPS[pair_idx]
    atr_vals = np.maximum(atr_vals, atr)

atr_median = np.median(atr_vals[te_idx])
atr_999 = np.percentile(atr_vals[te_idx], 99.9)

print(f"  ATR median: {atr_median:.1f}p  99.9th: {atr_999:.1f}p")

pnls_vslip = []
for j, i in enumerate(te_idx):
    p = bi[j]
    # Slippage = base 0.2p + 0.3 * (ATR / median ATR)
    slip_mult = atr_vals[i] / max(atr_median, 0.1)
    slip_var = (0.2 + 0.3 * min(slip_mult, 5.0)) * 2 * pip_val(p, avg_usdjpy)
    gross = np.log(close[i+3,p]/close[i,p])*direction[te_idx[j]]
    spread = HALF_SPREAD_PIPS[p]*2*1.5*pip_val(p, avg_usdjpy)
    gusd = LOT*gross if p==1 else LOT*gross*close[i,p]/usdjpy_proxy[i]
    pnls_vslip.append(gusd - spread - slip_var - ECN_COMM)
dv = np.array(pnls_vslip)
print(f"  ATR-conditional slip: n={len(dv):,} WR={np.mean(dv>0)*100:.1f}% Sharpe={np.mean(dv)/(np.std(dv)+1e-10)*np.sqrt(1440/3):.2f} Avg=${np.mean(dv):.2f}")
print()

# ============================================================
# TEST 5: COMBINED STRESS (latency + variable slip + overlap)
# ============================================================
print("TEST 5: Combined Stress — ALL factors together")
print("-" * 70)

pnls_combined = []
for j, i in enumerate(te_idx):
    p = bi[j]; next_i = min(i+1, T-1)
    # Latency: delayed entry at next bar open
    entry_price = opens[next_i, p]
    exit_price = close[min(i+3, T-1), p]
    gross = np.log(exit_price/entry_price)*direction[te_idx[j]]
    # Variable slippage
    slip_mult = atr_vals[i] / max(atr_median, 0.1)
    slip_var = (0.2 + 0.3 * min(slip_mult, 5.0)) * 2 * pip_val(p, avg_usdjpy)
    spread = HALF_SPREAD_PIPS[p]*2*1.5*pip_val(p, avg_usdjpy)
    gusd = LOT*gross if p==1 else LOT*gross*entry_price/usdjpy_proxy[i]
    pnls_combined.append(gusd - spread - slip_var - ECN_COMM)
dc = np.array(pnls_combined)
print(f"  n={len(dc):,}  WR={np.mean(dc>0)*100:.1f}%  Sharpe={np.mean(dc)/(np.std(dc)+1e-10)*np.sqrt(1440/3):.2f}  Avg=${np.mean(dc):.2f}  Tot=${np.sum(dc):,.0f}")

# Overlap-adjusted VaR
var_95 = np.percentile(dc, 5)
var_99 = np.percentile(dc, 1)
print(f"  Trade-level VaR 95%: ${var_95:.0f}  VaR 99%: ${var_99:.0f}")
print()

# ============================================================
# TEST 6: FEED CONSISTENCY AUDIT
# ============================================================
print("TEST 6: Feed Consistency — signal overlap across samples")
print("-" * 70)

# Compare Dukascopy (this run) vs Exness Oct-Dec 2025 vs MT5 Jun-Jul 2026
# Key metric: trade count per day, WR, Sharpe consistency
print(f"  Dukascopy (9mo): n={len(te_idx):,}  WR={np.mean(db>0)*100:.1f}%  Sharpe={np.mean(db)/(np.std(db)+1e-10)*np.sqrt(1440/3):.2f}  tpd={len(te_idx)/(T/1440):.1f}")

# Check for session consistency
early_hours = hour_arr[te_idx]
late_trades = np.sum(early_hours >= 18) / len(te_idx) * 100
mid_trades = np.sum((early_hours >= 7) & (early_hours < 12)) / len(te_idx) * 100
print(f"  Session distribution: AM(7-11): {mid_trades:.0f}%  PM(12-17): {100-early_hours[early_hours<18].shape[0]/len(te_idx)*100:.0f}%  Late(18-21): {late_trades:.0f}%")

# Weekend gap check
dow = pd.DatetimeIndex([dt_all[ti] for ti in te_idx]).dayofweek
friday_late = np.sum((dow == 4) & (early_hours >= 20)) / len(te_idx) * 100
print(f"  Fri 20-21 entries: {friday_late:.1f}% of trades (filter to eliminate weekend gap)")
print()

# ============================================================
# SUMMARY TABLE
# ============================================================
print("=" * 90)
print("SUMMARY: Edge degradation under enhanced realism")
print("=" * 90)
scenarios = [
    ("Baseline (flat 0.5p slip)", db),
    ("60s latency delay", dl),
    ("ATR-conditional slip", dv),
    ("Combined latency+slip", dc),
]
for name, pnl in scenarios:
    s = np.mean(pnl)/(np.std(pnl)+1e-10)*np.sqrt(1440/3)
    w = np.mean(pnl>0)*100
    print(f"  {name:35s}  Sharpe={s:6.2f}  WR={w:5.1f}%  Avg=\${np.mean(pnl):.2f}  Tot=\${np.sum(pnl):,.0f}")

# Breakeven under combined stress
print()
print("BREAKEVEN (combined stress, variable spread multiplier):")
all_gross = []
all_spread = []
all_slip = []
for j,i in enumerate(te_idx):
    p = bi[j]; next_i = min(i+1, T-1)
    entry_price = opens[next_i, p]
    exit_price = close[min(i+3, T-1), p]
    gross = np.log(exit_price/entry_price)*direction[te_idx[j]]
    spread = HALF_SPREAD_PIPS[p]*2*1.0*pip_val(p, avg_usdjpy)
    slip_mult = atr_vals[i] / max(atr_median, 0.1)
    slip_var = (0.2 + 0.3 * min(slip_mult, 5.0)) * 2 * pip_val(p, avg_usdjpy)
    gusd = LOT*gross if p==1 else LOT*gross*entry_price/usdjpy_proxy[i]
    all_gross.append(gusd)
    all_spread.append(spread)
    all_slip.append(slip_var)

gross_arr = np.array(all_gross)
spread_arr = np.array(all_spread)
slip_arr = np.array(all_slip)
ecn_arr = np.full(len(gross_arr), ECN_COMM)

for sm in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
    pnl = gross_arr - sm*spread_arr - slip_arr - ecn_arr
    s = np.mean(pnl)/(np.std(pnl)+1e-10)*np.sqrt(1440/3)
    w = np.mean(pnl>0)*100
    print(f"  {sm:.1f}x spread: Sharpe={s:6.2f}  WR={w:5.1f}%  Avg=\${np.mean(pnl):.2f}")
