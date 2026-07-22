"""Check what distinguishes trades that recover at 5min from those that stay negative.
Only tests structural, non-fittable factors: depth of 1min loss, pair, direction."""
import MetaTrader5 as mt5
import numpy as np
from time import time
import sys
from collections import defaultdict

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

CURR_PAIR_SIGN = {}
for ci in range(NC):
    for pj in range(NP):
        CURR_PAIR_SIGN[(ci, pj)] = CM[ci, pj]

def _best_pair(ci):
    BEST_PAIR = {"USD":"AUDUSD","EUR":"EURUSD","JPY":"NZDJPY","GBP":"GBPUSD",
        "AUD":"AUDUSD","NZD":"NZDUSD","CAD":"NZDCAD","CHF":"USDCHF"}
    return ALL_PAIRS.index(BEST_PAIR[ALL_CURRENCIES[ci]])

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

print("Running divergence with full tracking...", file=sys.stderr)

buf = [RingBuf(2000) for _ in range(NC)]
trades = []
opens = {}
trade_id = 0

for i in range(1, n):
    for ci in range(NC):
        buf[ci].append(curr_rets[i-1, ci])

    for ci in list(opens.keys()):
        t = opens[ci]
        pj = t['pj']
        ex = P[i, pj]
        pnl = (ex - t['ep']) / PIP_SZ[pj] * 5.0 * t['d']
        t['intra_pnls'].append(round(pnl, 2))

    for ci in list(opens.keys()):
        t = opens[ci]
        if i - t['bi'] >= 5:
            t = opens.pop(ci)
            trades.append(t)

    z_scores = np.zeros(NC)
    for ci in range(NC):
        m, s = buf[ci].stats()
        if s < 1e-12: continue
        z_scores[ci] = (buf[ci].last() - m) / s

    sorted_idx = np.argsort(z_scores)
    strongest_ci = sorted_idx[-1]
    weakest_ci = sorted_idx[0]
    sz = z_scores[strongest_ci]
    wz = z_scores[weakest_ci]

    if sz < 2.0 or wz > -2.0: continue
    if strongest_ci in opens or weakest_ci in opens: continue

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

    opens[trade_ci] = {
        'id': trade_id, 'c': trade_ci, 'pj': pj, 'd': int(d_star),
        'bi': i, 'ep': P[i, pj],
        'sz': sz, 'wz': wz,
        'strongest': strongest_ci, 'weakest': weakest_ci,
        'intra_pnls': []
    }
    trade_id += 1

for ci, t in opens.items():
    trades.append(t)

print(f"Total trades: {len(trades)}", file=sys.stderr)

# Extract data
min1_pnl = np.array([t['intra_pnls'][0] if len(t.get('intra_pnls',[]))>0 else 0 for t in trades])
min5_pnl = np.array([t['intra_pnls'][4] if len(t.get('intra_pnls',[]))>4 else t.get('pnl',0) for t in trades])
directions = np.array([t['d'] for t in trades])
zs = np.array([t['sz'] for t in trades])
wzs = np.array([t['wz'] for t in trades])
gap = zs - wzs
pairs = np.array([t['pj'] for t in trades])

neg_at_1 = min1_pnl < 0
pos_at_1 = ~neg_at_1

# Which negative-at-1 recover by 5?
recovered = (min1_pnl < 0) & (min5_pnl > 0)
stayed_neg = (min1_pnl < 0) & (min5_pnl <= 0)

n_neg = neg_at_1.sum()
n_rec = recovered.sum()
n_stayed = stayed_neg.sum()

print(f"\n{'='*80}")
print(f"  RECOVERY ANALYSIS — trades negative at 1min")
print('='*80)
print(f"  Total negative at 1min: {n_neg}")
print(f"  Recovered by 5min: {n_rec} ({n_rec/n_neg*100:.1f}%)")
print(f"  Stayed negative:     {n_stayed} ({n_stayed/n_neg*100:.1f}%)")

# Factor 1: How deep was the 1-min loss?
print(f"\n{'='*80}")
print(f"  FACTOR 1: Depth of 1-min loss — does it predict recovery?")
print('='*80)
for threshold in [2, 5, 8, 10, 12, 15, 20, 25]:
    shallow = (min1_pnl < 0) & (min1_pnl > -threshold)
    deep = min1_pnl <= -threshold
    
    if shallow.any():
        rec_shallow = ((min1_pnl < 0) & (min1_pnl > -threshold) & (min5_pnl > 0)).sum()
        total_shallow = shallow.sum()
        rec_rate_s = rec_shallow / total_shallow * 100 if total_shallow > 0 else 0
        avg_shallow = min1_pnl[shallow]
    else:
        rec_rate_s = 0
        total_shallow = 0
    
    if deep.any():
        rec_deep = (deep & (min5_pnl > 0)).sum()
        total_deep = deep.sum()
        rec_rate_d = rec_deep / total_deep * 100 if total_deep > 0 else 0
    else:
        rec_rate_d = 0
        total_deep = 0
    
    print(f"  Loss >-${threshold:<2}: {total_shallow:3d} trades, {rec_rate_s:5.1f}% recover  |  Loss ≤-${threshold:<2}: {total_deep:3d} trades, {rec_rate_d:5.1f}% recover")

# Factor 2: Long vs Short
print(f"\n{'='*80}")
print(f"  FACTOR 2: Direction (Long=1, Short=-1)")
print('='*80)
for d in [1, -1]:
    mask = neg_at_1 & (directions == d)
    if mask.any():
        rec = (mask & (min5_pnl > 0)).sum()
        n = mask.sum()
        avg_5 = np.mean(min5_pnl[mask])
        print(f"  {'Longs' if d>0 else 'Shorts'} negative at 1min: {n} trades, {rec/n*100:.1f}% recover, avg min5=${avg_5:+.2f}")

# Factor 3: Z-score gap (strongest - weakest)
print(f"\n{'='*80}")
print(f"  FACTOR 3: Z-score gap between strongest & weakest")
print('='*80)
for g_thresh in [3, 4, 5, 6, 7, 8]:
    wide_gap = neg_at_1 & (gap >= g_thresh)
    narrow_gap = neg_at_1 & (gap < g_thresh)
    if wide_gap.any():
        rec_w = (wide_gap & (min5_pnl > 0)).sum()
        n_w = wide_gap.sum()
        avg_w = np.mean(min5_pnl[wide_gap])
        print(f"  Gap ≥{g_thresh}: {n_w:3d} trades, {rec_w/n_w*100:5.1f}% recover, avg min5=${avg_w:+.2f}")
    if narrow_gap.any():
        rec_n = (narrow_gap & (min5_pnl > 0)).sum()
        n_n = narrow_gap.sum()
        avg_n = np.mean(min5_pnl[narrow_gap])
        print(f"  Gap <{g_thresh}: {n_n:3d} trades, {rec_n/n_n*100:5.1f}% recover, avg min5=${avg_n:+.2f}")

# Factor 4: Per-pair recovery rate
print(f"\n{'='*80}")
print(f"  FACTOR 4: Per-pair recovery when negative at 1min")
print('='*80)
pair_neg = defaultdict(list)
for i in range(len(trades)):
    if neg_at_1[i]:
        pair_neg[pairs[i]].append(i)

for pj in sorted(pair_neg.keys()):
    idx = pair_neg[pj]
    n = len(idx)
    rec = sum(1 for i in idx if min5_pnl[i] > 0)
    avg_5 = np.mean([min5_pnl[i] for i in idx])
    print(f"  {ALL_PAIRS[pj]:<8}: {n:3d} neg at 1, {rec:3d} recover ({rec/n*100:5.1f}%), avg min5=${avg_5:+.2f}")

# Factor 5: Is the strongest currency the SAME as the trade currency?
print(f"\n{'='*80}")
print(f"  FACTOR 5: Does strongest=trader or weakest=trader matter?")
print('='*80)
strong_trader = np.array([t['c'] == t['strongest'] for t in trades])
weak_trader = np.array([t['c'] == t['weakest'] for t in trades])

for mask, label in [(strong_trader, "Strongest drives trade"), (weak_trader, "Weakest drives trade")]:
    neg_mask = neg_at_1 & mask
    if neg_mask.any():
        rec = (neg_mask & (min5_pnl > 0)).sum()
        n = neg_mask.sum()
        avg_5 = np.mean(min5_pnl[neg_mask])
        print(f"  {label:<30}: {n:3d} neg at 1, {rec:3d} recover ({rec/n*100:5.1f}%), avg min5=${avg_5:+.2f}")

# Factor 6: Which CURRENCY is the strongest when we enter?
print(f"\n{'='*80}")
print(f"  FACTOR 6: Strongest currency when negative at 1min — does strong USD behave differently?")
print('='*80)
strong_ccy_map = {ALL_CURRENCIES[t['strongest']]: i for i, t in enumerate(trades) if neg_at_1[i]}
# Actually let me just count per currency
strong_ccy_counts = defaultdict(list)
for i in range(len(trades)):
    if neg_at_1[i]:
        ccy = ALL_CURRENCIES[trades[i]['strongest']]
        strong_ccy_counts[ccy].append(i)

for ccy in sorted(strong_ccy_counts.keys()):
    idx = strong_ccy_counts[ccy]
    n = len(idx)
    rec = sum(1 for i in idx if min5_pnl[i] > 0)
    avg_5 = np.mean([min5_pnl[i] for i in idx])
    print(f"  Strongest={ccy:<4}: {n:3d} neg at 1, {rec:3d} recover ({rec/n*100:5.1f}%), avg min5=${avg_5:+.2f}")

mt5.shutdown()
print(f"\nTime: {time()-t0:.1f}s")
