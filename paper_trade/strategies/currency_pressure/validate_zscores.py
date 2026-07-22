"""Final definitive test: CP algorithm vs reference on identical current data."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import MetaTrader5 as mt5
import numpy as np
from collections import deque

import strategy as cp

ALL_CURRENCIES = cp._ALL_CURRENCIES
ALL_PAIRS = cp._ALL_PAIRS
VOL_W = cp._VOL_WINDOW
Z_W = cp.CONFIG["z_window"]

def base_quote(p):
    for c in ALL_CURRENCIES:
        if p.startswith(c): return c, p[len(c):]
    return None, None

def inv_vol_ret(pair_returns, fixed_vol):
    cr, cv = {}, {c:[] for c in ALL_CURRENCIES}
    for pair, ret in pair_returns.items():
        v = fixed_vol.get(pair, 1e-10)
        b, q = base_quote(pair)
        if b: cr.setdefault(b,[]).append(ret); cv.setdefault(b,[]).append(v)
        if q: cr.setdefault(q,[]).append(-ret); cv.setdefault(q,[]).append(v)
    r = {}
    for c in ALL_CURRENCIES:
        rets = cr.get(c,[]); vols = cv.get(c,[])
        if len(rets) < 2: continue
        w = np.array([1.0/v for v in vols]); w /= np.sum(w)
        r[c] = np.dot(rets, w)
    return r

if not mt5.initialize(): print("MT5 fail"); exit(1)
for p in ALL_PAIRS: mt5.symbol_select(p, True)

N = Z_W + VOL_W
prices = {}
for p in ALL_PAIRS:
    r = mt5.copy_rates_from_pos(p, mt5.TIMEFRAME_M1, 0, N)
    if r is not None: prices[p] = [(int(x[0]), float(x[4])) for x in r]

min_len = min(len(v) for v in prices.values())
print(f"Data: {min_len} bars, {prices[ALL_PAIRS[0]][0][0]} -> {prices[ALL_PAIRS[0]][-1][0]}")

# === REFERENCE ===
first = {p:[] for p in ALL_PAIRS}
for i in range(1, min_len):
    for p in ALL_PAIRS:
        if p in prices:
            pr, cr = prices[p][i-1][1], prices[p][i][1]
            if pr>0 and cr>0:
                r = np.log(cr/pr)
                if i <= VOL_W - 1: first[p].append(r)

ref_vol = {p: np.std(first[p])+1e-10 if first[p] and len(first[p])>1 else 1e-10 for p in ALL_PAIRS}

# Check: every pair has exactly 199 first returns
assert all(len(first[p]) == 199 for p in ALL_PAIRS), "Not all pairs have 199 first returns"

ref_h = {c:deque(maxlen=Z_W) for c in ALL_CURRENCIES}
for i in range(1, min_len):
    rets = {}
    for p in ALL_PAIRS:
        if p in prices:
            pr, cr = prices[p][i-1][1], prices[p][i][1]
            if pr>0 and cr>0: rets[p] = np.log(cr/pr)
    crs = inv_vol_ret(rets, ref_vol)
    for c, r in crs.items(): ref_h[c].append(r)

# === CP ALGORITHM ===
cp._pair_close_prev.clear(); cp._pair_return_buf.clear()
cp._curr_return_history.clear(); cp._pair_vol_fixed.clear(); cp._last_minute = None
for c in ALL_CURRENCIES: cp._curr_return_history[c] = deque(maxlen=Z_W)
for p in ALL_PAIRS:
    cp._pair_close_prev[p] = None
    cp._pair_return_buf[p] = deque(maxlen=VOL_W)

class MockFeed:
    mode = "live"
    def __init__(self): self._p = prices
    def copy_m1_history(self, pair, count=None):
        c = count or N; raw = self._p.get(pair, [])
        return [(t, pr) for t, pr in raw[-c:]]

cp.seed_history(MockFeed())

# === COMPARE ===
# 1. Fixed vol
vol_ok = all(abs(cp._pair_vol_fixed[p] - ref_vol[p]) < 1e-10 for p in ALL_PAIRS)
print(f"\nFixed vol (28 pairs): {'✅ IDENTICAL' if vol_ok else '❌ SOME DIFFER'}")
if not vol_ok:
    for p in ALL_PAIRS:
        if abs(cp._pair_vol_fixed[p] - ref_vol[p]) >= 1e-10:
            print(f"  {p}: CP={cp._pair_vol_fixed[p]:.10f} Ref={ref_vol[p]:.10f}")

# 2. Currency history
hist_ok = True
max_diff = 0
for c in ALL_CURRENCIES:
    ch = list(cp._curr_return_history[c]); rh = list(ref_h[c])
    if len(ch) != len(rh):
        print(f"  {c}: lengths differ {len(ch)} vs {len(rh)}"); hist_ok = False; continue
    for j, (a, b) in enumerate(zip(ch, rh)):
        d = abs(a-b)
        if d > 1e-10:
            max_diff = max(max_diff, d)
            if hist_ok: print(f"  FIRST DIFF: {c}[{j}] CP={a:.6e} Ref={b:.6e} d={d:.2e}")
            hist_ok = False
print(f"Currency hist ({len(ALL_CURRENCIES)} currs): {'✅ IDENTICAL' if hist_ok else f'❌ DIFF (max={max_diff:.2e})'}")

# 3. Z-scores (ALL bars, not just last)
z_ok = True
for c in ALL_CURRENCIES:
    ch = list(cp._curr_return_history[c]); rh = list(ref_h[c])
    if len(ch) < 5: continue
    for j in range(len(ch)):
        arr = np.array(ch[:j+1]); marr = np.array(rh[:j+1])
        cz = (arr[-1] - np.mean(arr)) / np.std(arr) if np.std(arr) > 1e-12 else 0
        rz = (marr[-1] - np.mean(marr)) / np.std(marr) if np.std(marr) > 1e-12 else 0
        if abs(cz - rz) > 1e-6:
            if z_ok: print(f"  FIRST Z DIFF: {c}[{j}] CP={cz:.6f} Ref={rz:.6f}")
            z_ok = False
print(f"Z-scores (ALL bars, ALL currencies): {'✅ IDENTICAL to 1e-6' if z_ok else '❌ DIFFERS'}")

# 4. Current Z-scores display
print(f"\nCurrent Z-scores:")
for c in ALL_CURRENCIES:
    ch = list(cp._curr_return_history[c]); rh = list(ref_h[c])
    if len(ch) < 5: continue
    ca, ra = np.array(ch), np.array(rh)
    cz = round((ca[-1]-np.mean(ca))/np.std(ca), 4) if np.std(ca) > 1e-12 else 0
    rz = round((ra[-1]-np.mean(ra))/np.std(ra), 4) if np.std(ra) > 1e-12 else 0
    d = abs(cz-rz)
    print(f"  {c}: CP={cz:+6.4f}  Ref={rz:+6.4f}  diff={d:.4f}")

mt5.shutdown()
print(f"\n═══ VERDICT: {'ALL IDENTICAL ✅' if (vol_ok and hist_ok and z_ok) else 'ISSUES FOUND ❌'} ═══")
