"""Blind Spot Alpha — Currency Divergence Momentum.

Core insight: The market's spread defense works pair-by-pair.
The currency NETWORK (strongest vs weakest relationship) is the blind spot.

When the strongest currency (max Z) and weakest currency (min Z) 
both exceed threshold in OPPOSITE directions, trade the pair between them.
Both currencies NATURALLY confirm the pair direction.
"""
import MetaTrader5 as mt5
import numpy as np
from time import time
import sys

ALL_CURRENCIES = ['USD','EUR','JPY','GBP','AUD','NZD','CAD','CHF']
ALL_PAIRS = ["EURUSD","GBPUSD","USDJPY","AUDUSD","NZDUSD","USDCAD","USDCHF",
    "EURJPY","GBPJPY","EURGBP","EURAUD","EURCHF","EURCAD","EURNZD",
    "GBPAUD","GBPCAD","GBPCHF","GBPNZD","AUDJPY","AUDCAD","AUDCHF","AUDNZD",
    "NZDJPY","NZDCAD","NZDCHF","CADJPY","CADCHF","CHFJPY"]
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
PIP_SZ = np.array([0.01 if "JPY" in ALL_PAIRS[pj] else 0.0001 for pj in range(NP)])

# Precompute: for each pair of currencies, find the pair
CURR_PAIR_MAP = {}  # (ci1, ci2) -> pj
for pj, p in enumerate(ALL_PAIRS):
    b, q = base_quote(p)
    ci1 = ALL_CURRENCIES.index(b)
    ci2 = ALL_CURRENCIES.index(q)
    CURR_PAIR_MAP[(ci1, ci2)] = pj
    CURR_PAIR_MAP[(ci2, ci1)] = pj
# Sign for each currency in each pair
CURR_PAIR_SIGN = {}  # (ci, pj) -> sign
for ci in range(NC):
    for pj in range(NP):
        CURR_PAIR_SIGN[(ci, pj)] = CM[ci, pj]

print(f"Loaded {n} bars in {time()-t0:.1f}s", file=sys.stderr)

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
    trade_id = 0

    hold = 10 if "hold10" in mode else 5
    zt_actual = 2.5 if "zt25" in mode else zt
    for i in range(1, n):
        for ci in range(NC):
            buf[ci].append(curr_rets[i-1, ci])

        for ci in list(opens.keys()):
            t = opens[ci]
            if i - t['bi'] >= hold:
                t = opens.pop(ci); pj = t['pj']
                ex = P[i, pj]
                pnl = (ex - t['ep']) / PIP_SZ[pj] * 5.0 * t['d']
                if spread_mode == "full": pnl -= SPREAD_COST[pj]
                trades.append({**t, 'pnl': round(pnl, 2)})

        z_scores = np.zeros(NC)
        for ci in range(NC):
            m, s = buf[ci].stats()
            if s < 1e-12: continue
            z_scores[ci] = (buf[ci].last() - m) / s

        abs_z = np.abs(z_scores)
        sorted_idx = np.argsort(z_scores)  # lowest to highest

        if mode.startswith("baseline"):
            for ci in range(NC):
                if ci in opens: continue
                if abs_z[ci] < zt: continue
                pj = _best_pair(ci)
                sg = CURR_PAIR_SIGN.get((ci, pj), 0)
                if sg == 0: continue
                dir = 1 if z_scores[ci] > 0 else -1
                d_star = dir * sg
                opens[ci] = {'id': trade_id, 'c': ci, 'pj': pj, 'd': int(d_star),
                             'z': round(float(z_scores[ci]), 2), 'bi': i, 'ep': P[i, pj]}
                trade_id += 1

        elif mode.startswith("divergence"):
            strongest_ci = sorted_idx[-1]
            weakest_ci = sorted_idx[0]
            sz = z_scores[strongest_ci]
            wz = z_scores[weakest_ci]
            if sz < zt_actual or wz > -zt_actual:
                continue
            if strongest_ci in opens or weakest_ci in opens:
                continue

            if abs(sz) >= abs(wz):
                trade_ci = strongest_ci; other_ci = weakest_ci
            else:
                trade_ci = weakest_ci; other_ci = strongest_ci

            pj = _best_pair(trade_ci)
            sg = CURR_PAIR_SIGN.get((trade_ci, pj), 0)
            if sg == 0: continue
            dir = 1 if z_scores[trade_ci] > 0 else -1
            d_star = dir * sg

            other_sg = CURR_PAIR_SIGN.get((other_ci, pj), 0)
            if other_sg == 0: continue
            other_dir = 1 if z_scores[other_ci] > 0 else -1
            other_dstar = other_dir * other_sg
            if d_star != other_dstar: continue

            opens[trade_ci] = {'id': trade_id, 'c': trade_ci, 'pj': pj, 'd': int(d_star),
                               'z': round(float(z_scores[trade_ci]), 2), 'bi': i, 'ep': P[i, pj],
                               'other_ci': other_ci, 'other_z': round(float(z_scores[other_ci]), 2)}
            trade_id += 1

        elif mode == "top_divergences":
            top3 = sorted_idx[-3:]   # 3 strongest (highest Z)
            bot3 = sorted_idx[:3]    # 3 weakest (lowest Z)
            entries = []
            for sci in top3:
                if z_scores[sci] < zt_actual: continue
                for wci in bot3:
                    if z_scores[wci] > -zt_actual: continue
                    if sci in opens or wci in opens: continue
                    if abs(z_scores[sci]) >= abs(z_scores[wci]):
                        tci, oci = sci, wci
                    else:
                        tci, oci = wci, sci
                    pj = _best_pair(tci)
                    sg = CURR_PAIR_SIGN.get((tci, pj), 0)
                    if sg == 0: continue
                    d_dir = 1 if z_scores[tci] > 0 else -1
                    dst = d_dir * sg
                    osg = CURR_PAIR_SIGN.get((oci, pj), 0)
                    if osg == 0: continue
                    odir = 1 if z_scores[oci] > 0 else -1
                    odst = odir * osg
                    if dst != odst: continue
                    entries.append((tci, pj, dst))
            for tci, pj, dst in entries:
                if tci in opens: continue
                opens[tci] = {'id': trade_id, 'c': tci, 'pj': pj, 'd': int(dst),
                              'z': round(float(z_scores[tci]), 2), 'bi': i, 'ep': P[i, pj]}
                trade_id += 1

        elif mode == "most_extreme":
            # Just trade the currency with highest |Z|
            max_ci = np.argmax(abs_z)
            if abs_z[max_ci] < zt: continue
            if max_ci in opens: continue
            pj = _best_pair(max_ci)
            sg = CURR_PAIR_SIGN.get((max_ci, pj), 0)
            if sg == 0: continue
            dir = 1 if z_scores[max_ci] > 0 else -1
            d_star = dir * sg
            opens[max_ci] = {'id': trade_id, 'c': max_ci, 'pj': pj, 'd': int(d_star),
                             'z': round(float(z_scores[max_ci]), 2), 'bi': i, 'ep': P[i, pj]}
            trade_id += 1

        elif mode == "top_bottom_baskets":
            # Top 2 strongest, bottom 2 weakest — trade all 4 pair combos that exist
            top2 = sorted_idx[-2:]
            bottom2 = sorted_idx[:2]
            entries = []
            for sci in top2:
                if sci in opens: continue
                if z_scores[sci] < zt: continue
                for wci in bottom2:
                    if wci in opens: continue
                    if z_scores[wci] > -zt: continue
                    pj = CURR_PAIR_MAP.get((sci, wci))
                    if pj is None: continue
                    s_sg = CURR_PAIR_SIGN.get((sci, pj), 0)
                    w_sg = CURR_PAIR_SIGN.get((wci, pj), 0)
                    s_dir = 1 if z_scores[sci] > 0 else -1
                    w_dir = 1 if z_scores[wci] > 0 else -1
                    s_dstar = s_dir * s_sg
                    w_dstar = w_dir * w_sg
                    if s_dstar == w_dstar:
                        entries.append((sci, wci, pj, s_dstar))
            for sci, wci, pj, d_star in entries[:2]:
                if sci in opens or wci in opens: continue
                opens[sci] = {'id': trade_id, 'c': sci, 'pj': pj, 'd': int(d_star),
                              'z': round(float(z_scores[sci]), 2), 'bi': i, 'ep': P[i, pj],
                              'other_ci': wci, 'other_z': round(float(z_scores[wci]), 2)}
                trade_id += 1

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

def _best_pair(ci):
    """Return best pair index for a currency (matches CP)"""
    BEST_PAIR = {"USD":"AUDUSD","EUR":"EURUSD","JPY":"NZDJPY","GBP":"GBPUSD",
        "AUD":"AUDUSD","NZD":"NZDUSD","CAD":"NZDCAD","CHF":"USDCHF"}
    return ALL_PAIRS.index(BEST_PAIR[ALL_CURRENCIES[ci]])

modes = [
    ("Baseline CP", "baseline"),
    ("Divergence (Z>2)", "divergence"),
    ("Divergence (Z>2, hold10)", "divergence_hold10"),
    ("Divergence (Z>2.5)", "divergence_zt25"),
    ("Top Divergences (top3bottom3)", "top_divergences"),
]
results = {}
for name, mode in modes:
    r = run_bt(2000, 2.0, "mid", mode=mode)
    rf = run_bt(2000, 2.0, "full", mode=mode)
    results[name] = (r, rf)

print(f"\n{'='*110}")
print(f"{'Mode':<32} {'Trades':<8} {'Mid WR':<8} {'Mid Avg':<12} {'Full WR':<8} {'Full Avg':<12} {'TPD':<6}")
print("-"*110)
for name in [m[0] for m in modes]:
    r, rf = results[name]
    print(f"{name:<32} {r['n']:<8} {r['wr']:>5.1f}%  ${r['avg']:>+6.2f}  {rf['wr']:>5.1f}%  ${rf['avg']:>+6.2f}  {r['tpd']:>4.1f}")

print(f"\n{'='*110}")
print("DELTA FROM BASELINE")
print(f"{'Mode':<32} {'Trades Δ':<10} {'Mid WR Δ':<10} {'Mid Avg Δ':<12} {'Full WR Δ':<10} {'Full Avg Δ':<12}")
print("-"*110)
b = results["Baseline CP"]
for name in [m[0] for m in modes[1:]]:
    r, rf = results[name]
    print(f"{name:<32} {r['n']-b[0]['n']:<+8}  {r['wr']-b[0]['wr']:>+5.1f}%  ${r['avg']-b[0]['avg']:>+5.2f}  {rf['wr']-b[0]['wr']:>+5.1f}%  ${rf['avg']-b[0]['avg']:>+5.2f}")

mt5.shutdown()
print(f"\nTime: {time()-t0:.1f}s")
