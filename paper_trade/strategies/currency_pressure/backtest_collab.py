"""Test CP collab exploit ideas.
1. Baseline CP
2. Double confirmation: both currencies in pair agree
3. Pair Z alignment: pair-level Z agrees with CP
4. Extreme Z: |Z| > 3.0 only
5. Leader dominance: trade only the highest |Z| currency
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

# Map: for each pair, which 2 currencies are involved
PAIR_CURR_CI = np.zeros((NP, 2), dtype=np.int32)
PAIR_CURR_SIGN = np.zeros((NP, 2))  # sign: +1 if base, -1 if quote
for pj, p in enumerate(ALL_PAIRS):
    b, q = base_quote(p)
    PAIR_CURR_CI[pj, 0] = ALL_CURRENCIES.index(b)
    PAIR_CURR_CI[pj, 1] = ALL_CURRENCIES.index(q)
    PAIR_CURR_SIGN[pj, 0] = 1.0  # base
    PAIR_CURR_SIGN[pj, 1] = -1.0  # quote

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
    """mode: baseline, double_confirm, pair_z_align, extreme_z, leader_dom"""
    buf = [RingBuf(z_window) for _ in range(NC)]
    buf_pair = [RingBuf(z_window) for _ in range(NP)]  # pair-level Z
    trades = []; opens = {}
    days = n / 1440.0

    for i in range(1, n):
        for ci in range(NC):
            buf[ci].append(curr_rets[i-1, ci])
        for pj in range(NP):
            buf_pair[pj].append(lr[i-1, pj])

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

        pair_z = np.zeros(NP)
        for pj in range(NP):
            m, s = buf_pair[pj].stats()
            if s < 1e-12: continue
            pair_z[pj] = (buf_pair[pj].last() - m) / s

        threshold = 3.0 if mode == "extreme_z" else zt

        for ci in range(NC):
            if ci in opens: continue
            z = z_scores[ci]
            if abs(z) < threshold: continue

            # Mode-specific entry filters
            pj = BP_IDX[ci]; sg = BP_SIGN[ci]
            dir = 1 if z > 0 else -1
            d_star = dir * sg

            if mode == "baseline":
                pass  # trade everything

            elif mode == "double_confirm":
                # Check the OTHER currency in the pair
                c0, c1 = PAIR_CURR_CI[pj]
                s0, s1 = PAIR_CURR_SIGN[pj]
                # Direction for base currency: if z > 0 (overbought) → short → -1 for base
                # Direction for quote currency: if z > 0 (overbought) → +1 for quote (since quote weakening = pair rising)
                # For direction to agree: BOTH imply same pair direction
                # For LONG: base oversold (z<0) OR quote overbought (z>0)
                # For SHORT: base overbought (z>0) OR quote oversold (z<0)
                z0 = z_scores[c0]
                z1 = z_scores[c1]
                long_confirm = (z0 < -zt) or (z1 > zt)  # ANY reason to be long
                short_confirm = (z0 > zt) or (z1 < -zt)  # ANY reason to be short
                if d_star > 0 and not long_confirm: continue
                if d_star < 0 and not short_confirm: continue
                # Now check the OTHER currency also agrees
                if d_star > 0:  # LONG: need base oversold OR quote overbought
                    if not (z0 < -zt or z1 > zt):
                        # Only one currency triggered. Check if the OTHER also has |Z| > zt in agreeing direction
                        other_agrees = (z0 < -zt and z1 > zt * 0.5) or (z1 > zt and z0 < -zt * 0.5)
                        if not other_agrees >= zt:
                            pass  # weaker confirm, but not required
                # Actually let's simplify: both currencies must have |Z| > zt in same pair direction
                both_long = z0 < -zt and z1 > zt
                both_short = z0 > zt and z1 < -zt
                if not (both_long or both_short):
                    continue

            elif mode == "pair_z_align":
                # Pair's own Z must agree with CP direction for this pair
                pz = pair_z[pj]
                # For LONG: pair Z should be negative (oversold, due to rise)
                # For SHORT: pair Z should be positive (overbought, due to fall)
                if d_star > 0 and pz > -0.5: continue  # LONG needs pair Z oversold
                if d_star < 0 and pz < 0.5: continue  # SHORT needs pair Z overbought

            elif mode == "leader_dom":
                abs_z = np.abs(z_scores)
                max_abs = np.max(abs_z)
                if abs_z[ci] < max_abs - 0.3: continue  # not dominant enough

            # entry
            ep = P[i, pj]
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

results = {}
for name, mode, zt in [
    ("Baseline CP", "baseline", 2.0),
    ("Double Confirm", "double_confirm", 2.0),
    ("Pair Z Align", "pair_z_align", 2.0),
    ("Extreme Z (|Z|>3)", "extreme_z", 2.0),
    ("Leader Dominant", "leader_dom", 2.0),
]:
    r = run_bt(2000, zt, "mid", mode=mode)
    r_full = run_bt(2000, zt, "full", mode=mode)
    results[name] = (r, r_full)

print(f"\n{'='*110}")
print(f"{'Mode':<22} {'Trades':<8} {'Mid WR':<10} {'Mid Avg':<12} {'Full WR':<10} {'Full Avg':<12} {'TPD':<6}")
print("-"*110)
baseline = results["Baseline CP"]
for name, (r, r_full) in results.items():
    print(f"{name:<22} {r['n']:<8} {r['wr']:>6.1f}%  ${r['avg']:>+6.2f}  {r_full['wr']:>6.1f}%  ${r_full['avg']:>+6.2f}  {r['tpd']:>5.1f}")

print(f"\n{'='*110}")
print("DELTA FROM BASELINE")
print(f"{'Mode':<22} {'Trades Δ':<10} {'Mid WR Δ':<10} {'Mid Avg Δ':<12} {'Full WR Δ':<10} {'Full Avg Δ':<12}")
print("-"*110)
b_r, b_full = baseline
for name, (r, r_full) in results.items():
    if name == "Baseline CP": continue
    print(f"{name:<22} {r['n']-b_r['n']:<+8}  {r['wr']-b_r['wr']:>+6.1f}%  ${r['avg']-b_r['avg']:>+6.2f}  {r_full['wr']-b_full['wr']:>+6.1f}%  ${r_full['avg']-b_full['avg']:>+6.2f}")

mt5.shutdown()
print(f"\nTotal: {time()-t0:.1f}s")
