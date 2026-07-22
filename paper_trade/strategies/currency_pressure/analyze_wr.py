"""DARK RESEARCH: 3 out-of-box approaches to exploit CP sub-friction edge.
Approach 1: Limit order entry (earn spread, don't pay it)
Approach 2: Dynamic trailing stop (let winners run, cut losers)
Approach 3: Z-based exit (exit when Z normalizes, not after fixed hold)
All preserve entry signals — only exit/execution changes."""
import MetaTrader5 as mt5
import numpy as np
from time import time

ALL_CURRENCIES = ['USD','EUR','JPY','GBP','AUD','NZD','CAD','CHF']
ALL_PAIRS = ["EURUSD","GBPUSD","USDJPY","AUDUSD","NZDUSD","USDCAD","USDCHF",
    "EURJPY","GBPJPY","EURGBP","EURAUD","EURCHF","EURCAD","EURNZD",
    "GBPAUD","GBPCAD","GBPCHF","GBPNZD","AUDJPY","AUDCAD","AUDCHF","AUDNZD",
    "NZDJPY","NZDCAD","NZDCHF","CADJPY","CADCHF","CHFJPY"]
BEST_PAIR = {"USD":"AUDUSD","EUR":"EURUSD","JPY":"NZDJPY","GBP":"GBPUSD",
    "AUD":"AUDUSD","NZD":"NZDUSD","CAD":"NZDCAD","CHF":"USDCHF"}
PAIR_SPREAD_PIPS = {"AUDUSD":1.5,"EURUSD":1.5,"GBPUSD":2.0,"NZDUSD":2.0,
    "USDCAD":2.0,"USDCHF":2.0,"USDJPY":1.8,"EURJPY":2.5,"GBPJPY":4.0,
    "EURGBP":2.0,"EURAUD":2.5,"EURCHF":2.0,"EURCAD":2.0,"EURNZD":2.5,
    "GBPAUD":2.5,"GBPCAD":2.5,"GBPCHF":3.0,"GBPNZD":3.0,"AUDJPY":2.5,
    "AUDCAD":2.5,"AUDCHF":2.5,"AUDNZD":2.5,"NZDJPY":3.0,"NZDCAD":2.5,
    "NZDCHF":2.5,"CADJPY":2.5,"CADCHF":2.5,"CHFJPY":3.0}
VOL_W = 200; NC = len(ALL_CURRENCIES); NP = len(ALL_PAIRS)
SPREAD_COST = np.array([PAIR_SPREAD_PIPS.get(p, 2.0) * 5.0 for p in ALL_PAIRS])
HALF_SPREAD = SPREAD_COST / 2

def base_quote(p):
    for c in ALL_CURRENCIES:
        if p.startswith(c): return c, p[len(c):]
    return None, None

CM = np.zeros((NC, NP), dtype=np.float64)
for ci, c in enumerate(ALL_CURRENCIES):
    for pj, p in enumerate(ALL_PAIRS):
        b, q = base_quote(p)
        if b == c: CM[ci, pj] = 1.0
        elif q == c: CM[ci, pj] = -1.0

t0 = time()
if not mt5.initialize(): exit(1)
for p in ALL_PAIRS: mt5.symbol_select(p, True)
N = 50000
PH = {}  # prices with high/low
for p in ALL_PAIRS:
    r = mt5.copy_rates_from_pos(p, mt5.TIMEFRAME_M1, 0, N)
    if r is not None: PH[p] = [(int(x[0]), float(x[2]), float(x[3]), float(x[4])) for x in r]
n = min(len(v) for v in PH.values())
# P array: rows=bar, cols={close, high, low} per pair
PC = np.zeros((n, NP)); PHIGH = np.zeros((n, NP)); PLOW = np.zeros((n, NP))
for pj, p in enumerate(ALL_PAIRS):
    for i in range(min(n, len(PH[p]))):
        PC[i, pj] = PH[p][i][3]
        PHIGH[i, pj] = PH[p][i][1]
        PLOW[i, pj] = PH[p][i][2]

lr = np.diff(np.log(PC + 1e-12), axis=0)
fv = np.std(lr[:VOL_W], axis=0) + 1e-10
W_raw = 1.0 / fv; W = np.zeros((NC, NP))
for ci in range(NC):
    mask = np.abs(CM[ci]) > 0.5
    if mask.sum() > 0:
        w = W_raw.copy(); w[~mask] = 0.0; s = w.sum()
        if s > 0: w /= s
        W[ci] = w
C_MIX = CM * W; curr_rets = lr @ C_MIX.T
BP_IDX = np.array([ALL_PAIRS.index(BEST_PAIR[c]) for c in ALL_CURRENCIES])
BP_SIGN = np.array([CM[ci, BP_IDX[ci]] for ci in range(NC)])
print(f"Loaded: {n} bars, {time()-t0:.1f}s")

PIP_SZ = np.array([0.01 if "JPY" in ALL_PAIRS[pj] else 0.0001 for pj in range(NP)])

class RingBuf:
    def __init__(self, size):
        self.size = size; self.buf = np.zeros(size)
        self.pos = 0; self.filled = 0
    def append(self, v):
        self.buf[self.pos] = v; self.pos = (self.pos+1) % self.size
        self.filled = min(self.filled+1, self.size)
    def last(self): return self.buf[(self.pos-1) % self.size]
    def stats(self):
        fn = self.filled
        if fn < 5: return 0, 0
        a = self.buf if fn >= self.size else self.buf[:fn]
        return float(np.mean(a)), float(np.std(a))

# ── BACKTEST CORE (reusable) ──
def run_bt_simple(z_window, zt, spread_mode, entry_fn=None, exit_fn=None):
    """Generic backtest with pluggable entry/exit.
    entry_fn(buf, ci, i, z, pj, sg): returns (ep, d) or None to skip.
    exit_fn(trade, i, P_cur, buf, ci): returns True to exit.
    """
    buf = [RingBuf(z_window) for _ in range(NC)]
    trades = []; opens = {}

    for i in range(1, n):
        for ci in range(NC):
            buf[ci].append(curr_rets[i-1, ci])

        for ci in list(opens.keys()):
            t = opens[ci]
            exit_flag = False
            if exit_fn:
                cz = buf[ci].last()
                exit_flag = exit_fn(t, i, PC[i, t['pj']], cz)
            else:
                if i - t['bi'] >= 5:
                    exit_flag = True
            if exit_flag:
                t = opens.pop(ci); pj = t['pj']
                ex = PC[i, pj]
                pnl = (ex - t['ep']) / PIP_SZ[pj] * 5.0 * t['d']
                if spread_mode == "full": pnl -= SPREAD_COST[pj]
                trades.append({**t, 'pnl': round(pnl, 2)})

        for ci in range(NC):
            if ci in opens: continue
            m, s = buf[ci].stats()
            if s < 1e-12: continue
            z = (buf[ci].last() - m) / s
            if abs(z) < zt: continue
            pj = BP_IDX[ci]; sg = BP_SIGN[ci]
            dir = 1 if z > 0 else -1

            if entry_fn:
                result = entry_fn(PC[i, pj], PHIGH[i, pj], PLOW[i, pj], z, dir, pj, sg)
                if result is None: continue
                ep, d_star = result
            else:
                ep = PC[i, pj]
                d_star = dir * sg

            opens[ci] = {'c': ci, 'pj': pj, 'd': int(d_star),
                         'z': round(float(z), 2), 'bi': i, 'ep': ep}

    for ci, t in opens.items():
        pj = t['pj']
        ex = PC[n-1, pj]
        pnl = (ex - t['ep']) / PIP_SZ[pj] * 5.0 * t['d']
        if spread_mode == "full": pnl -= SPREAD_COST[pj]
        trades.append({**t, 'pnl': round(pnl, 2)})

    days = n / 1440.0
    if not trades: return {"n": 0, "wr": 0, "avg": 0, "total": 0, "tpd": 0}
    wins = sum(1 for t in trades if t['pnl'] > 0)
    total = sum(t['pnl'] for t in trades)
    return {"n": len(trades), "wr": wins/len(trades)*100,
            "avg": total/len(trades), "total": total, "tpd": len(trades)/days}

# ── APPROACH 0: BASELINE ──
def fmt(r):
    return f"{r['n']:>6}  {r['wr']:>7.1f}%  ${r['avg']:>+7.2f}  ${r['total']:>+10.2f}  {r['tpd']:>5.1f}"

print(f"\n{'='*100}")
print("BASELINE")
b_mid = run_bt_simple(2000, 2.0, "mid")
b_full = run_bt_simple(2000, 2.0, "full")
print(f"  Mid: {fmt(b_mid)}")
print(f"  Full: {fmt(b_full)}")

# ── APPROACH 1: LIMIT ORDER EXECUTION ──
# Enter at LIMIT price (earn spread). Check if high/low crosses our limit within next bar.
print(f"\n{'='*100}")
print("APPROACH 1: LIMIT ORDER ENTRY")
print("Enter at BID (buy) or ASK (sell) — earn spread. Check high/low for fill next bar.")
print()

def limit_entry(close, high, low, z, direction, pj, sg):
    """Enter at limit. For sell (z>0): limit at bid. For buy (z<0): limit at ask.
    Check if bar high >= sell_limit or bar low <= buy_limit.
    If filled: ep = limit price, d = direction * sign."""
    half = HALF_SPREAD[pj] / 5.0 * PIP_SZ[pj]  # half-spread in price units
    if z > 0:  # sell — limit at bid (below mid)
        limit_px = close - half
        if high >= limit_px:
            return limit_px, -1 * sg  # sell at limit
    else:  # buy — limit at ask (above mid)
        limit_px = close + half
        if low <= limit_px:
            return limit_px, 1 * sg  # buy at limit
    return None

def limit_entry_nextbar(close, high, low, z, direction, pj, sg):
    """Same but check if limit would be filled on NEXT bar (deferred).
    We store the limit and check next bar's high/low."""
    return None  # Implemented via entry_delay below

# Test: limit entry at same bar
r = run_bt_simple(2000, 2.0, "mid", entry_fn=limit_entry)
print(f"  Limit entry mid:    {fmt(r)}")
print(f"  (Note: fill rate = {r['n']}/{b_mid['n']} = {r['n']/b_mid['n']*100:.1f}%)")
r = run_bt_simple(2000, 2.0, "full", entry_fn=limit_entry)
print(f"  Limit entry full:   {fmt(r)}")
print(f"  (Fill rate = {r['n']}/{b_full['n']} = {r['n']/b_full['n']*100:.1f}%)")

# ── APPROACH 2: DYNAMIC STOP LOSS / TAKE PROFIT ──
print(f"\n{'='*100}")
print("APPROACH 2: DYNAMIC EXIT (profit target / stop loss)")
print("Entry at market (pay spread). Exit at profit target ($20) or stop loss ($10).")
print("Max hold: 30 bars. Preserves ALL entries.")
print()

def dynamic_exit(trade, i, cur_price, curr_z):
    if i - trade['bi'] < 5: return False  # min hold
    if i - trade['bi'] >= 30: return True  # max hold
    # Track unrealized PnL
    pj = trade['pj']
    running_pnl = (cur_price - trade['ep']) / PIP_SZ[pj] * 5.0 * trade['d']
    if running_pnl >= 15.0: return True  # profit target
    if running_pnl <= -10.0: return True  # stop loss
    return False

r = run_bt_simple(2000, 2.0, "mid", exit_fn=dynamic_exit)
print(f"  Dynamic exit mid:   {fmt(r)}")
r = run_bt_simple(2000, 2.0, "full", exit_fn=dynamic_exit)
print(f"  Dynamic exit full:  {fmt(r)}")

# ── APPROACH 3: Z-BASED EXIT (exit when Z normalizes) ──
print(f"\n{'='*100}")
print("APPROACH 3: Z-BASED EXIT")
print("Exit when |Z| drops below 1.0 (pressure normalizing).")
print("Min hold: 3 bars. Max hold: 30 bars.")
print()

def z_exit(trade, i, cur_price, curr_z):
    if i - trade['bi'] < 3: return False
    if i - trade['bi'] >= 30: return True
    # If Z returned toward zero, reversion is happening
    entry_z = trade['z']
    if abs(curr_z) < 1.0: return True
    # If Z got MORE extreme in the SAME direction, our bet is wrong
    if abs(curr_z) > abs(entry_z) * 1.2: return True
    return False

r = run_bt_simple(2000, 2.0, "mid", exit_fn=z_exit)
print(f"  Z-exit mid:         {fmt(r)}")
r = run_bt_simple(2000, 2.0, "full", exit_fn=z_exit)
print(f"  Z-exit full:        {fmt(r)}")

# ── APPROACH 4: HYBRID — LIMIT ENTRY + DYNAMIC EXIT ──
print(f"\n{'='*100}")
print("APPROACH 4: HYBRID — LIMIT ENTRY (earn spread) + DYNAMIC EXIT")
print("Enter at limit (BID/ASK). Exit at PT/SL or max hold 30.")
print()

def limit_exit(trade, i, cur_price, curr_z):
    if i - trade['bi'] < 3: return False
    if i - trade['bi'] >= 30: return True
    pj = trade['pj']
    running_pnl = (cur_price - trade['ep']) / PIP_SZ[pj] * 5.0 * trade['d']
    if running_pnl >= 10.0: return True
    if running_pnl <= -10.0: return True
    return False

r = run_bt_simple(2000, 2.0, "mid", entry_fn=limit_entry, exit_fn=limit_exit)
print(f"  Limit+PT/SL mid:    {fmt(r)}")
print(f"  Fill rate: {r['n']}/{b_mid['n']} = {r['n']/b_mid['n']*100:.1f}%")
r = run_bt_simple(2000, 2.0, "full", entry_fn=limit_entry, exit_fn=limit_exit)
print(f"  Limit+PT/SL full:   {fmt(r)}")
print(f"  Fill rate: {r['n']}/{b_full['n']} = {r['n']/b_full['n']*100:.1f}%")

# ── APPROACH 5: VOLATILITY-BASED POSITION SIZING ──
print(f"\n{'='*100}")
print("APPROACH 5: VOLATILITY-ADJUSTED EXIT")
print("Exit when PnL < -2*ATR(5) OR PnL > +3*ATR(5) OR max 30 bars.")
print()

# Precompute ATR for each pair
ATR5 = np.zeros((n, NP))
for pj in range(NP):
    tr = np.zeros(n)
    for i in range(1, n):
        hl = PHIGH[i, pj] - PLOW[i, pj]
        hc = abs(PHIGH[i, pj] - PC[i-1, pj])
        lc = abs(PLOW[i, pj] - PC[i-1, pj])
        tr[i] = max(hl, hc, lc)
    for i in range(5, n):
        ATR5[i, pj] = np.mean(tr[i-4:i+1])

def vol_exit(trade, i, cur_price, curr_z):
    if i - trade['bi'] < 5: return False
    if i - trade['bi'] >= 30: return True
    pj = trade['pj']
    pnl = (cur_price - trade['ep']) / PIP_SZ[pj] * 5.0 * trade['d']
    atr_val = ATR5[i, pj]
    atr_dollars = atr_val / PIP_SZ[pj] * 5.0
    if atr_dollars == 0: return False
    if pnl >= 3 * atr_dollars: return True  # 3x ATR profit target
    if pnl <= -2 * atr_dollars: return True  # 2x ATR stop loss
    return False

r = run_bt_simple(2000, 2.0, "mid", exit_fn=vol_exit)
print(f"  Vol-based exit mid: {fmt(r)}")
r = run_bt_simple(2000, 2.0, "full", exit_fn=vol_exit)
print(f"  Vol-based exit full:{fmt(r)}")

# ── COMPARISON TABLE ──
print(f"\n{'='*100}")
print("COMPARISON SUMMARY")
print(f"{'Approach':<30} {'Mid WR':<10} {'Mid Avg':<10} {'Full WR':<10} {'Full Avg':<12} {'Trades':<8}")
print("-"*80)
for name, mid_r, full_r in [
    ("Baseline (market+fixhold)", b_mid, b_full),
    ("1-Limit entry", run_bt_simple(2000,2.0,"mid",entry_fn=limit_entry), 
     run_bt_simple(2000,2.0,"full",entry_fn=limit_entry)),
    ("2-Dynamic PT/SL", run_bt_simple(2000,2.0,"mid",exit_fn=dynamic_exit),
     run_bt_simple(2000,2.0,"full",exit_fn=dynamic_exit)),
    ("3-Z-based exit", run_bt_simple(2000,2.0,"mid",exit_fn=z_exit),
     run_bt_simple(2000,2.0,"full",exit_fn=z_exit)),
    ("4-Limit + PT/SL", run_bt_simple(2000,2.0,"mid",entry_fn=limit_entry,exit_fn=limit_exit),
     run_bt_simple(2000,2.0,"full",entry_fn=limit_entry,exit_fn=limit_exit)),
    ("5-Vol-based exit", run_bt_simple(2000,2.0,"mid",exit_fn=vol_exit),
     run_bt_simple(2000,2.0,"full",exit_fn=vol_exit)),
]:
    print(f"{name:<30} {mid_r['wr']:>6.1f}%  ${mid_r['avg']:>+6.2f}  {full_r['wr']:>6.1f}%  ${full_r['avg']:>+6.2f}    {full_r['n']:>5}")

mt5.shutdown()
print(f"\nTotal: {time()-t0:.1f}s")
