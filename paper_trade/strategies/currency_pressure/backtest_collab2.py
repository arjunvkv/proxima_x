"""Quick CP collab tests — optimized."""
import MetaTrader5 as mt5
import numpy as np
from time import time
import sys

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

CM = np.zeros((NC, NP))
for ci, c in enumerate(ALL_CURRENCIES):
    for pj, p in enumerate(ALL_PAIRS):
        b, q = base_quote(p)
        if b == c: CM[ci, pj] = 1.0
        elif q == c: CM[ci, pj] = -1.0

t0 = time()
if not mt5.initialize(): exit(1)
for p in ALL_PAIRS: mt5.symbol_select(p, True)

print("Loading...", file=sys.stderr)
PC = {}
for p in ALL_PAIRS:
    r = mt5.copy_rates_from_pos(p, mt5.TIMEFRAME_M1, 0, 50000)
    if r is not None: PC[p] = np.array([float(x[4]) for x in r])
n = min(len(v) for v in PC.values())
P = np.zeros((n, NP))
for pj, p in enumerate(ALL_PAIRS):
    if p in PC: P[:n, pj] = PC[p][:n]

lr = np.diff(np.log(P + 1e-12), axis=0)[:50000]
n = len(lr)
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
print(f"Loaded {n} bars in {time()-t0:.1f}s", file=sys.stderr)

# Precompute: for each BEST_PAIR, which other currency is involved
BEST_OTHER_CI = np.zeros(NC, dtype=np.int32)  # index of the "other" currency in the best pair
BEST_OTHER_SIGN = np.zeros(NC)  # CM sign of other currency in best pair
for ci, c in enumerate(ALL_CURRENCIES):
    pj = BP_IDX[ci]
    b, q = base_quote(ALL_PAIRS[pj])
    if b == c:
        BEST_OTHER_CI[ci] = ALL_CURRENCIES.index(q)
        BEST_OTHER_SIGN[ci] = CM[BEST_OTHER_CI[ci], pj]
    else:
        BEST_OTHER_CI[ci] = ALL_CURRENCIES.index(b)
        BEST_OTHER_SIGN[ci] = CM[BEST_OTHER_CI[ci], pj]

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

def run_bt(z_window, zt, spread_mode, mode="baseline"):
    buf = [RingBuf(z_window) for _ in range(NC)]
    trades = []; opens = {}
    days = n / 1440.0

    for i in range(1, n):
        for ci in range(NC):
            buf[ci].append(curr_rets[i-1, ci])

        for ci in list(opens.keys()):
            t = opens[ci]
            age = i - t['bi']
            exit_flag = (age >= 5)
            if not exit_flag and "_zexit" in mode:
                m, s = buf[ci].stats()
                if s > 1e-12:
                    cz = (buf[ci].last() - m) / s
                    if abs(cz) < 1.0: exit_flag = True
                    elif abs(cz) > abs(t['z']) * 1.2: exit_flag = True
            if "_flip" in mode and age == 2:
                pj = t['pj']
                pnl2 = (P[i, pj] - t['ep']) / PIP_SZ[pj] * 5.0 * t['d']
                if pnl2 < -3.0:
                    exit_flag = True
                    t['flip'] = True
            if exit_flag:
                t = opens.pop(ci); pj = t['pj']
                ex = P[i, pj]
                pnl = (ex - t['ep']) / PIP_SZ[pj] * 5.0 * t['d']
                if spread_mode == "full": pnl -= SPREAD_COST[pj]
                trades.append({**t, 'pnl': round(pnl, 2)})
                if t.get('flip'):
                    opens[ci] = {'c': ci, 'pj': pj, 'd': -t['d'],
                                 'z': t['z'], 'bi': i, 'ep': P[i, pj]}

        z_scores = np.zeros(NC)
        for ci in range(NC):
            m, s = buf[ci].stats()
            if s < 1e-12: continue
            z_scores[ci] = (buf[ci].last() - m) / s

        threshold = 3.0 if "extreme_z" in mode else zt

        for ci in range(NC):
            if ci in opens: continue
            z = z_scores[ci]
            if abs(z) < threshold: continue

            pj = BP_IDX[ci]; sg = BP_SIGN[ci]
            dir = 1 if z > 0 else -1
            d_star = dir * sg

            if "double" in mode:
                other_ci = BEST_OTHER_CI[ci]
                oz = z_scores[other_ci]
                osg = BEST_OTHER_SIGN[ci]
                odir = 1 if oz > 0 else -1
                o_dstar = odir * osg
                if d_star != o_dstar:
                    continue
            elif "dominant" in mode:
                abs_z = np.abs(z_scores)
                if abs_z[ci] < np.max(abs_z) - 0.3:
                    continue

            ep = P[i, pj]
            opens[ci] = {'c': ci, 'pj': pj, 'd': int(d_star),
                         'z': round(float(z), 2), 'bi': i, 'ep': ep}

    for ci, t in opens.items():
        pj = t['pj']; ex = P[n-1, pj]
        pnl = (ex - t['ep']) / PIP_SZ[pj] * 5.0 * t['d']
        if spread_mode == "full": pnl -= SPREAD_COST[pj]
        trades.append({**t, 'pnl': round(pnl, 2)})

    if not trades: return {"n":0,"wr":0,"avg":0,"total":0,"tpd":0}
    wins = sum(1 for t in trades if t['pnl'] > 0)
    total = sum(t['pnl'] for t in trades)
    return {"n":len(trades),"wr":wins/len(trades)*100,
            "avg":total/len(trades),"total":total,"tpd":len(trades)/days}

modes = [
    ("Baseline CP", "baseline"),
    ("Double Confirm", "double"),
    ("DC + Z-exit", "double_zexit"),
    ("Baseline + Z-exit", "baseline_zexit"),
    ("Extreme Z (|Z|>3)", "extreme_z"),
]
results = {}
for name, mode in modes:
    r = run_bt(2000, 2.0, "mid", mode=mode)
    rf = run_bt(2000, 2.0, "full", mode=mode)
    results[name] = (r, rf)

print(f"\n{'='*100}")
print(f"{'Mode':<22} {'Trades':<8} {'Mid WR':<8} {'Mid Avg':<10} {'Full WR':<8} {'Full Avg':<10} {'TPD':<6}")
print("-"*100)
for name in [m[0] for m in modes]:
    r, rf = results[name]
    print(f"{name:<22} {r['n']:<8} {r['wr']:>5.1f}%  ${r['avg']:>+5.2f}  {rf['wr']:>5.1f}%  ${rf['avg']:>+5.2f}  {r['tpd']:>4.1f}")

print(f"\n{'='*100}")
print("DELTA FROM BASELINE")
print(f"{'Mode':<22} {'Trades Δ':<10} {'Mid WR Δ':<8} {'Mid Avg Δ':<10} {'Full WR Δ':<8} {'Full Avg Δ':<10}")
print("-"*100)
b = results["Baseline CP"]
for name in [m[0] for m in modes[1:]]:
    r, rf = results[name]
    print(f"{name:<22} {r['n']-b[0]['n']:<+8}  {r['wr']-b[0]['wr']:>+5.1f}%  ${r['avg']-b[0]['avg']:>+5.2f}  {rf['wr']-b[0]['wr']:>+5.1f}%  ${rf['avg']-b[0]['avg']:>+5.2f}")

mt5.shutdown()
print(f"\nTime: {time()-t0:.1f}s")
