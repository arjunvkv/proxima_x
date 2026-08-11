"""s4_shift_test.py — is cached EURUSD exactly a k-trading-day shift of terminal EURUSD?

Tests shift k = 0..6 trading days: corr(cached[D], term[tradingday(D)-k]).
Also the reverse (terminal leads cached). Prints best-k table for EURUSD.
"""
import json
import numpy as np

C = json.load(open("s4_cached_closes.json"))
T = json.load(open(r"C:\Users\arjun\AppData\Local\Temp\s4_term_closes.json"))

cd = np.array(C["days"]); cc = np.array(C["ece"])
td = np.array(T["EURUSD"]["days"]); tc = np.array(T["EURUSD"]["close"])

# trading-day index: consecutive position among all days
def trading_idx(days):
    idx = {int(d): i for i, d in enumerate(days)}
    return idx

ci = trading_idx(cd); ti = trading_idx(td)
cpos = np.array([ci[d] for d in cd])   # position of each cached day in cached series
tpos_of_cached = np.array([ti[d] for d in cd if d in ti])  # term position of same day
valid = [d for d in cd if d in ti]

print("shift-k: cached[D] vs term[trading-pos(D) - k]  (k>0 => term lags cache)")
for k in range(0, 7):
    pairs = []
    for d in valid:
        tp = ti[d] - k
        if 0 <= tp < len(tc):
            pairs.append((cc[np.where(cd == d)[0][0]], tc[tp]))
    if len(pairs) < 10:
        print(f"  k={k}: n={len(pairs)} too few"); continue
    a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
    corr = np.corrcoef(a, b)[0, 1]
    md = np.abs(a - b).max()
    print(f"  k={k}: n={len(pairs)} corr={corr:.6f} max|d|={md:.5f}")

# also: same-day, no shift
pairs = [(cc[np.where(cd == d)[0][0]], tc[ti[d]]) for d in valid]
a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
print(f"same-day: n={len(pairs)} corr={np.corrcoef(a, b)[0,1]:.6f} max|d|={np.abs(a-b).max():.5f}")
