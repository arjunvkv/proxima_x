"""M5 backtest: does larger bar edge beat spread?"""
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

# Load M5 data (same count as M1 baseline)
N = 50000
PC = {}
print("Loading M5 data...")
for p in ALL_PAIRS:
    r = mt5.copy_rates_from_pos(p, mt5.TIMEFRAME_M5, 0, N)
    if r is not None: PC[p] = np.array([float(x[4]) for x in r])
n = min(len(v) for v in PC.values())
print(f"Loaded {n} M5 bars (covers {n/288:.0f} days) in {time()-t0:.1f}s")

# Build price matrix
P = np.zeros((n, NP))
for pj, p in enumerate(ALL_PAIRS):
    if p in PC:
        P[:n, pj] = PC[p][:n]

lr = np.diff(np.log(P + 1e-12), axis=0)
n_ret = lr.shape[0]

fv = np.std(lr[:VOL_W], axis=0) + 1e-10
W_raw = 1.0 / fv
W = np.zeros((NC, NP))
for ci in range(NC):
    mask = np.abs(CM[ci]) > 0.5
    if mask.sum() > 0:
        w = W_raw.copy(); w[~mask] = 0.0; s = w.sum()
        if s > 0: w /= s
        W[ci] = w
C_MIX = CM * W
curr_rets = lr @ C_MIX.T

BP_IDX = np.array([ALL_PAIRS.index(BEST_PAIR[c]) for c in ALL_CURRENCIES])
BP_SIGN = np.array([CM[ci, BP_IDX[ci]] for ci in range(NC)])
PIP_SZ = np.array([0.01 if "JPY" in ALL_PAIRS[pj] else 0.0001 for pj in range(NP)])

print(f"Precomputed {curr_rets.shape} currency returns in {time()-t0:.1f}s")

# Precompute ATR for dynamic exit
print("Computing ATR...")
ATR5 = np.zeros((n_ret, NP))
for pj in range(NP):
    tr = np.zeros(n_ret)
    for i in range(1, n_ret):
        hl = abs(P[i+1, pj] - P[i, pj])  # M5 high-low approximated by close range
        hc = abs(P[i+1, pj] - P[i, pj])
        lc = abs(P[i, pj] - P[i+1, pj])
        tr[i] = max(hl, hc, lc)
    for i in range(5, n_ret):
        ATR5[i, pj] = np.mean(tr[i-4:i+1])
print(f"ATR computed in {time()-t0:.1f}s")

class RingBuf:
    __slots__ = ('size','buf','pos','filled')
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

HOLD_BARS = 1  # 1 M5 bar = 5 minutes (matches 5 M1 bars)

def run_bt(z_window, zt, spread_mode, exit_fn=None):
    buf = [RingBuf(z_window) for _ in range(NC)]
    trades = []; opens = {}

    for i in range(1, n_ret):
        for ci in range(NC):
            buf[ci].append(curr_rets[i-1, ci])

        for ci in list(opens.keys()):
            t = opens[ci]
            exit_flag = False
            if exit_fn:
                cz = buf[ci].last()
                exit_flag = exit_fn(t, i, P[i+1, t['pj']], cz)
            else:
                if i - t['bi'] >= HOLD_BARS:
                    exit_flag = True
            if exit_flag:
                t = opens.pop(ci); pj = t['pj']
                ex = P[i+1, pj]  # exit at current bar's close
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
            ep = P[i+1, pj]  # enter at current bar close
            d_star = dir * sg
            opens[ci] = {'c': ci, 'pj': pj, 'd': int(d_star),
                         'z': round(float(z), 2), 'bi': i, 'ep': ep}

    for ci, t in opens.items():
        pj = t['pj']
        ex = P[n_ret, pj]
        pnl = (ex - t['ep']) / PIP_SZ[pj] * 5.0 * t['d']
        if spread_mode == "full": pnl -= SPREAD_COST[pj]
        trades.append({**t, 'pnl': round(pnl, 2)})

    days = n_ret / 288.0
    if not trades: return {"n": 0, "wr": 0, "avg": 0, "total": 0, "tpd": 0}
    wins = sum(1 for t in trades if t['pnl'] > 0)
    total = sum(t['pnl'] for t in trades)
    return {"n": len(trades), "wr": wins/len(trades)*100,
            "avg": total/len(trades), "total": total, "tpd": len(trades)/days}

def fmt(r):
    return f"{r['n']:>6}  {r['wr']:>7.1f}%  ${r['avg']:>+7.2f}  ${r['total']:>+10.2f}  {r['tpd']:>5.1f}"

# ── BASELINE ──
print(f"\n{'='*100}")
print("M5 BASELINE (2000 Z-window, |Z|>2.0, hold 1 bar = 5min)")
b_mid = run_bt(2000, 2.0, "mid")
b_full = run_bt(2000, 2.0, "full")
print(f"  Mid:  {fmt(b_mid)}")
print(f"  Full: {fmt(b_full)}")
print(f"  Edge/Spread: ${b_mid['avg']:.2f} / $~8.50 = {b_mid['avg']/8.5:.2f}x")

# ── DYNAMIC EXIT ──
def dynamic_exit(trade, i, cur_price, curr_z):
    if i - trade['bi'] < HOLD_BARS: return False
    if i - trade['bi'] >= 6: return True  # max 30min (6 M5 bars)
    pj = trade['pj']
    running_pnl = (cur_price - trade['ep']) / PIP_SZ[pj] * 5.0 * trade['d']
    if running_pnl >= 15.0: return True
    if running_pnl <= -10.0: return True
    return False

print(f"\n{'='*100}")
print("M5 DYNAMIC EXIT (PT$15 / SL$10, max 6 bars = 30min)")
d_mid = run_bt(2000, 2.0, "mid", exit_fn=dynamic_exit)
d_full = run_bt(2000, 2.0, "full", exit_fn=dynamic_exit)
print(f"  Mid:  {fmt(d_mid)}")
print(f"  Full: {fmt(d_full)}")

# ── Z-BASED EXIT ──
def z_exit(trade, i, cur_price, curr_z):
    if i - trade['bi'] < HOLD_BARS: return False
    if i - trade['bi'] >= 6: return True
    if abs(curr_z) < 1.0: return True
    if abs(curr_z) > abs(trade['z']) * 1.2: return True
    return False

print(f"\n{'='*100}")
print("M5 Z-BASED EXIT (|Z|<1.0 exit, max 6 bars)")
z_mid = run_bt(2000, 2.0, "mid", exit_fn=z_exit)
z_full = run_bt(2000, 2.0, "full", exit_fn=z_exit)
print(f"  Mid:  {fmt(z_mid)}")
print(f"  Full: {fmt(z_full)}")

# ── VOL-BASED EXIT ──
def vol_exit(trade, i, cur_price, curr_z):
    if i - trade['bi'] < HOLD_BARS: return False
    if i - trade['bi'] >= 6: return True
    pj = trade['pj']
    pnl = (cur_price - trade['ep']) / PIP_SZ[pj] * 5.0 * trade['d']
    atr_val = ATR5[i, pj]
    atr_dollars = atr_val / PIP_SZ[pj] * 5.0
    if atr_dollars == 0: return False
    if pnl >= 3 * atr_dollars: return True
    if pnl <= -2 * atr_dollars: return True
    return False

print(f"\n{'='*100}")
print("M5 VOL-BASED EXIT (3xATR PT / 2xATR SL, max 6 bars)")
v_mid = run_bt(2000, 2.0, "mid", exit_fn=vol_exit)
v_full = run_bt(2000, 2.0, "full", exit_fn=vol_exit)
print(f"  Mid:  {fmt(v_mid)}")
print(f"  Full: {fmt(v_full)}")

# ── SUMMARY ──
print(f"\n{'='*100}")
print("M5 SUMMARY (compare: M1 baseline was 53.1% mid, -$8.56 full)")
print(f"{'Approach':<25} {'Mid WR':<10} {'Mid Avg':<10} {'Full WR':<10} {'Full Avg':<12} {'Trades':<8} {'TPD':<6}")
print("-"*80)
for name, mid_r, full_r in [
    ("M5 Baseline", b_mid, b_full),
    ("M5 Dynamic PT/SL", d_mid, d_full),
    ("M5 Z-exit", z_mid, z_full),
    ("M5 Vol-exit", v_mid, v_full),
]:
    print(f"{name:<25} {mid_r['wr']:>6.1f}%  ${mid_r['avg']:>+6.2f}  {full_r['wr']:>6.1f}%  ${full_r['avg']:>+6.2f}  {full_r['n']:>5}  {full_r['tpd']:>5.1f}")

mt5.shutdown()
print(f"\nTotal time: {time()-t0:.1f}s")
