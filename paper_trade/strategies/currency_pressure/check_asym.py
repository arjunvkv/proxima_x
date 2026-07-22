"""Minimal check: structural asymmetry only — close at 1min if negative, hold if positive."""
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

print("Running divergence...", file=sys.stderr)

# Run divergence once, tracking all intra-trade paths
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
        'intra_pnls': []
    }
    trade_id += 1

for ci, t in opens.items():
    trades.append(t)

print(f"Total trades: {len(trades)}", file=sys.stderr)

# --- Test structural asymmetry: close at 1min if negative ---
mid_pnls_5min = np.array([sum(t.get('intra_pnls', [])[4:5]) + (t['intra_pnls'][4] if len(t.get('intra_pnls',[]))>4 else 0) for t in trades])
# Actually simpler: just compute PnL at each minute
min1 = np.array([t['intra_pnls'][0] if len(t.get('intra_pnls',[]))>0 else t.get('pnl',0) for t in trades])
min5 = np.array([t['intra_pnls'][4] if len(t.get('intra_pnls',[]))>4 else t.get('pnl',0) for t in trades])

# Full spread versions
full1 = min1 - np.array([PAIR_SPREAD_PIPS.get(ALL_PAIRS[t['pj']], 2.0) * 5.0 for t in trades])
full5 = min5 - np.array([PAIR_SPREAD_PIPS.get(ALL_PAIRS[t['pj']], 2.0) * 5.0 for t in trades])

print(f"\n{'='*80}")
print(f"  STRUCTURAL ASYMMETRY TEST — only non-overfittable hypothesis")
print(f"  If trade is negative at 1min → close (proven hold decay)")
print(f"  If trade is positive at 1min → hold to 5min")
print('='*80)

# At minute 1, which trades are positive vs negative?
pos_at_1 = min1 > 0
neg_at_1 = ~pos_at_1

print(f"\n  Trades positive at 1min: {pos_at_1.sum()} ({np.mean(pos_at_1)*100:.1f}%)")
print(f"  Trades negative at 1min: {neg_at_1.sum()} ({np.mean(neg_at_1)*100:.1f}%)")

# If we close negative at 1min, what's the PnL?
asym_mid = np.where(pos_at_1, min5, min1)
asym_full = np.where(pos_at_1, full5, full1)

print(f"\n  {'Method':<25} {'Trades':<8} {'WR':<8} {'Avg':<10} {'Total':<12}")
print('-'*60)
print(f"  {'Hold 5min (baseline)':<25} {len(trades):<8} {np.mean(min5>0)*100:>5.1f}%  ${np.mean(min5):>+6.2f}  ${np.sum(min5):>+8.2f}")
print(f"  {'Close at 1min (all)':<25} {len(trades):<8} {np.mean(min1>0)*100:>5.1f}%  ${np.mean(min1):>+6.2f}  ${np.sum(min1):>+8.2f}")
print(f"  {'ASYMM: neg1→exit, pos1→5':<25} {len(trades):<8} {np.mean(asym_mid>0)*100:>5.1f}%  ${np.mean(asym_mid):>+6.2f}  ${np.sum(asym_mid):>+8.2f}")

# With spread
print(f"\n  FULL SPREAD:")
print(f"  {'Hold 5min':<25} {len(trades):<8} {np.mean(full5>0)*100:>5.1f}%  ${np.mean(full5):>+6.2f}  ${np.sum(full5):>+8.2f}")
print(f"  {'Close at 1min':<25} {len(trades):<8} {np.mean(full1>0)*100:>5.1f}%  ${np.mean(full1):>+6.2f}  ${np.sum(full1):>+8.2f}")
print(f"  {'ASYMM: neg1→exit, pos1→5':<25} {len(trades):<8} {np.mean(asym_full>0)*100:>5.1f}%  ${np.mean(asym_full):>+6.2f}  ${np.sum(asym_full):>+8.2f}")

# With ECN costs
ecn_cost = 1.75
asym_ecn = asym_mid - ecn_cost
full_ecn = min5 - ecn_cost
one_ecn = min1 - ecn_cost

print(f"\n  ECN ($1.75/trade):")
print(f"  {'Hold 5min':<25} {np.mean(full_ecn):>+6.2f}  ${np.sum(full_ecn):>+8.2f}")
print(f"  {'Close at 1min':<25} {np.mean(one_ecn):>+6.2f}  ${np.sum(one_ecn):>+8.2f}")
print(f"  {'ASYMM: neg1→exit, pos1→5':<25} {np.mean(asym_ecn):>+6.2f}  ${np.sum(asym_ecn):>+8.2f}")

# Per-minute breakdown
print(f"\n{'='*80}")
print(f"  PER-MINUTE BREAKDOWN (all mid)")
print('='*80)
for mi in range(5):
    arr = np.array([t.get('intra_pnls', [None])[mi] if len(t.get('intra_pnls',[])) > mi else None for t in trades])
    arr = arr[arr != np.array(None)]
    if len(arr) > 0:
        print(f"  Minute {mi+1}: avg=${np.mean(arr):+6.2f}  wr={np.mean(arr>0)*100:>5.1f}%  total=${np.sum(arr):+8.2f}")

# Conditional: given positive at min 1, what's min 5?
pos1_pnls = min5[pos_at_1]
neg1_pnls = min5[neg_at_1]
print(f"\n{'='*80}")
print(f"  CONDITIONAL: What happens to 1min winners/losers by 5min")
print('='*80)
print(f"  Positive at 1min → hold to 5min: avg=${np.mean(pos1_pnls):+6.2f}  wr={np.mean(pos1_pnls>0)*100:.1f}%")
print(f"  Negative at 1min → hold to 5min: avg=${np.mean(neg1_pnls):+6.2f}  wr={np.mean(neg1_pnls>0)*100:.1f}%")

# Of the negative at 1min, how many recover vs stay negative?
recovered = np.mean(neg1_pnls > 0) * 100
stayed_neg = np.mean(neg1_pnls <= 0) * 100
print(f"  Negative at 1min → recover by 5min: {recovered:.1f}%")
print(f"  Negative at 1min → stay negative: {stayed_neg:.1f}%")


# === SECOND TEST: Close at 1min if negative, extend winners to 10min ===
print(f"\n{'='*80}")
print(f"  EXTENDED ASYMM: neg at 1→close, pos at 5→hold to 10")
print('='*80)

# We need the 10-min PnL data
# Re-run with 10-min hold and track per-minute
buf2 = [RingBuf(2000) for _ in range(NC)]
trades10 = []
opens10 = {}
trade_id = 0

for i in range(1, n):
    for ci in range(NC):
        buf2[ci].append(curr_rets[i-1, ci])

    for ci in list(opens10.keys()):
        t = opens10[ci]
        pj = t['pj']
        ex = P[i, pj]
        pnl = (ex - t['ep']) / PIP_SZ[pj] * 5.0 * t['d']
        t['intra_pnls'].append(round(pnl, 2))

    for ci in list(opens10.keys()):
        t = opens10[ci]
        if i - t['bi'] >= 10:
            t = opens10.pop(ci)
            trades10.append(t)

    z_scores = np.zeros(NC)
    for ci in range(NC):
        m, s = buf2[ci].stats()
        if s < 1e-12: continue
        z_scores[ci] = (buf2[ci].last() - m) / s

    sorted_idx = np.argsort(z_scores)
    strongest_ci = sorted_idx[-1]
    weakest_ci = sorted_idx[0]
    sz = z_scores[strongest_ci]
    wz = z_scores[weakest_ci]

    if sz < 2.0 or wz > -2.0: continue
    if strongest_ci in opens10 or weakest_ci in opens10: continue

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

    opens10[trade_ci] = {
        'id': trade_id, 'c': trade_ci, 'pj': pj, 'd': int(d_star),
        'bi': i, 'ep': P[i, pj],
        'intra_pnls': []
    }
    trade_id += 1

for ci, t in opens10.items():
    trades10.append(t)

# Compute PnLs
hlen = len(trades10)
min1_10 = np.array([t['intra_pnls'][0] if len(t.get('intra_pnls',[]))>0 else t.get('pnl',0) for t in trades10])
min5_10 = np.array([t['intra_pnls'][4] if len(t.get('intra_pnls',[]))>4 else t.get('pnl',0) for t in trades10])
min10_10 = np.array([t.get('pnl', 0) for t in trades10])

# Asym: neg at 1 → close, pos at 5 → hold to 10
pos_at_1_10 = min1_10 > 0
pos_at_5_10 = min5_10 > 0

extended = np.where(~pos_at_1_10, min1_10, np.where(pos_at_5_10, min10_10, min5_10))

print(f"  {'Method':<35} {'Trades':<8} {'WR':<8} {'Avg':<10} {'Total':<12}")
print('-'*70)
print(f"  {'Hold 5min':<35} {hlen:<8} {np.mean(min5_10>0)*100:>5.1f}%  ${np.mean(min5_10):>+6.2f}  ${np.sum(min5_10):>+8.2f}")
print(f"  {'Hold 10min':<35} {hlen:<8} {np.mean(min10_10>0)*100:>5.1f}%  ${np.mean(min10_10):>+6.2f}  ${np.sum(min10_10):>+8.2f}")
print(f"  {'neg1→close, pos5→10':<35} {hlen:<8} {np.mean(extended>0)*100:>5.1f}%  ${np.mean(extended):>+6.2f}  ${np.sum(extended):>+8.2f}")

mt5.shutdown()
print(f"\nTime: {time()-t0:.1f}s")
