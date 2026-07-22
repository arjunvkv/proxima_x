"""Quick backtest: CP + NarrativeEngine filter.
Narrative filter: only trade the currency with the highest |Z| score (the "narrative leader").
Baseline: trade all currencies with |Z| > 2.0.
"""
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

CM = np.zeros((NC, NP))
for ci, c in enumerate(ALL_CURRENCIES):
    for pj, p in enumerate(ALL_PAIRS):
        b, q = base_quote(p)
        if b == c: CM[ci, pj] = 1.0
        elif q == c: CM[ci, pj] = -1.0

t0 = time()
if not mt5.initialize(): exit(1)
for p in ALL_PAIRS: mt5.symbol_select(p, True)

print("Loading M1 data...")
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
print(f"Loaded {n} bars in {time()-t0:.1f}s")

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

def run_bt(z_window, zt, spread_mode, narrative_filter=False):
    """narrative_filter=True: only trade when leader |Z| dominates (>= gap ahead of 2nd place)"""
    buf = [RingBuf(z_window) for _ in range(NC)]
    trades = []; opens = {}
    days = n / 1440.0

    for i in range(1, n):
        for ci in range(NC):
            buf[ci].append(curr_rets[i-1, ci])

        for ci in list(opens.keys()):
            t = opens[ci]
            if i - t['bi'] >= 5:
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

        if narrative_filter:
            abs_z = np.abs(z_scores)
            # Find top 2
            top2 = np.argsort(abs_z)[-2:]
            leader_ci = top2[1]
            second_ci = top2[0]
            leader_z = z_scores[leader_ci]
            gap = abs_z[leader_ci] - abs_z[second_ci]
            if abs(leader_z) < zt: continue
            if gap < 0.5: continue  # leader not dominant enough
            if leader_ci in opens: continue
            pj = BP_IDX[leader_ci]; sg = BP_SIGN[leader_ci]
            dir = 1 if leader_z > 0 else -1
            ep = P[i, pj]; d_star = dir * sg
            opens[leader_ci] = {'c': leader_ci, 'pj': pj, 'd': int(d_star),
                                'z': round(float(leader_z), 2), 'bi': i, 'ep': ep}
        else:
            for ci in range(NC):
                if ci in opens: continue
                z = z_scores[ci]
                if abs(z) < zt: continue
                pj = BP_IDX[ci]; sg = BP_SIGN[ci]
                dir = 1 if z > 0 else -1
                ep = P[i, pj]; d_star = dir * sg
                opens[ci] = {'c': ci, 'pj': pj, 'd': int(d_star),
                             'z': round(float(z), 2), 'bi': i, 'ep': ep}

    for ci, t in opens.items():
        pj = t['pj']; ex = P[n-1, pj]
        pnl = (ex - t['ep']) / PIP_SZ[pj] * 5.0 * t['d']
        if spread_mode == "full": pnl -= SPREAD_COST[pj]
        trades.append({**t, 'pnl': round(pnl, 2)})

    if not trades: return {"n": 0, "wr": 0, "avg": 0, "total": 0, "tpd": 0}
    wins = sum(1 for t in trades if t['pnl'] > 0)
    total = sum(t['pnl'] for t in trades)
    return {"n": len(trades), "wr": wins/len(trades)*100,
            "avg": total/len(trades), "total": total, "tpd": len(trades)/days}

print(f"\n{'='*100}")
print("BASELINE CP (all currencies with |Z|>2.0)")
b_mid = run_bt(2000, 2.0, "mid")
b_full = run_bt(2000, 2.0, "full")
print(f"  Mid:  {b_mid['n']:>6}  {b_mid['wr']:>6.1f}%  ${b_mid['avg']:>+7.2f}  {b_mid['tpd']:>5.1f}/d")
print(f"  Full: {b_full['n']:>6}  {b_full['wr']:>6.1f}%  ${b_full['avg']:>+7.2f}  {b_full['tpd']:>5.1f}/d")

print(f"\n{'='*100}")
print("NARRATIVE FILTER CP (only trade the currency with max |Z|)")
n_mid = run_bt(2000, 2.0, "mid", narrative_filter=True)
n_full = run_bt(2000, 2.0, "full", narrative_filter=True)
print(f"  Mid:  {n_mid['n']:>6}  {n_mid['wr']:>6.1f}%  ${n_mid['avg']:>+7.2f}  {n_mid['tpd']:>5.1f}/d")
print(f"  Full: {n_full['n']:>6}  {n_full['wr']:>6.1f}%  ${n_full['avg']:>+7.2f}  {n_full['tpd']:>5.1f}/d")

print(f"\n{'='*100}")
print("COMPARISON")
print(f"{'Mode':<30} {'Trades':<8} {'Mid WR':<10} {'Mid Avg':<12} {'Full WR':<10} {'Full Avg':<12}")
print("-"*80)
print(f"{'Baseline CP':<30} {b_mid['n']:<8} {b_mid['wr']:>6.1f}%  ${b_mid['avg']:>+6.2f}  {b_full['wr']:>6.1f}%  ${b_full['avg']:>+6.2f}")
print(f"{'Narrative Filter CP':<30} {n_mid['n']:<8} {n_mid['wr']:>6.1f}%  ${n_mid['avg']:>+6.2f}  {n_full['wr']:>6.1f}%  ${n_full['avg']:>+6.2f}")
print(f"{'Change':<30} {'-'*8} {n_mid['wr']-b_mid['wr']:>+6.1f}%  ${n_mid['avg']-b_mid['avg']:>+6.2f}  {n_full['wr']-b_full['wr']:>+6.1f}%  ${n_full['avg']-b_full['avg']:>+6.2f}")

mt5.shutdown()
print(f"\nTotal: {time()-t0:.1f}s")
