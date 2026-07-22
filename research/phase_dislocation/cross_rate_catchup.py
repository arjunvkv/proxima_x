"""Cross-rate laggard catch-up at M1 level.
When EURJPY moves, legs (EURUSD, USDJPY) must catch up due to market maker hedging.

The triangle: EURJPY = EURUSD × USDJPY
log(EURJPY) = log(EURUSD) + log(USDJPY)

If EURJPY returns exceed the sum of leg returns, the legs haven't caught up yet.
The legs should move in subsequent bars to close the gap.
Trade the leg with the most catching up to do.
"""
import numpy as np
import pandas as pd

# Load M1 data
df = pd.read_parquet("data/temp/mt5_m1_9day.parquet")
pairs_in_data = sorted(df["pair"].unique())
print(f"Pairs in data: {pairs_in_data}")

# Align all pairs
aligned = {}
for p in pairs_in_data:
    sub = df[df["pair"] == p].sort_values("time")
    aligned[p] = dict(zip(sub["time"], sub["close"]))
all_times = sorted(set.intersection(*[set(aligned[p].keys()) for p in pairs_in_data]))
price = np.column_stack([np.array([aligned[p][t] for t in all_times]) for p in pairs_in_data])
n, npairs = price.shape
times = all_times
print(f"Common bars: {n}, pairs: {npairs}")

lr = np.diff(np.log(price), axis=0)

# We need the triangle pairs specifically
pair_names = {p: i for i, p in enumerate(pairs_in_data)}
triangle_pairs = ["EURJPY", "EURUSD", "USDJPY"]
missing = [p for p in triangle_pairs if p not in pair_names]
if missing:
    print(f"Missing triangle pairs: {missing}")
    exit()

ej_idx = pair_names["EURJPY"]
eu_idx = pair_names["EURUSD"]
uj_idx = pair_names["USDJPY"]

print(f"\nTriangle: EURJPY(idx={ej_idx}), EURUSD(idx={eu_idx}), USDJPY(idx={uj_idx})")

# For each bar: compute leg contributions to EURJPY return
# log(price_EJ[t]/price_EJ[t-1]) ≈ log(price_EU[t]/price_EU[t-1]) + log(price_UJ[t]/price_UJ[t-1])
# Or equivalently: r_EJ ≈ r_EU + r_UJ (approximately, for small returns)

# The residual: r_EJ - (r_EU + r_UJ) = unexplained part
# If residual > 0: EURJPY moved more than legs → legs need to catch up
# If residual < 0: EURJPY moved less than legs → EURJPY needs to catch up

print("\n" + "=" * 60)
print("STRATEGY 1: Trade the LAGGARD leg when EURJPY outruns its legs")
print("=" * 60)

for res_thresh in [0.0003, 0.0005, 0.001, 0.002]:
    for fwd in [1, 2, 3, 5, 10]:
        results = []
        for i in range(0, len(lr) - fwd - 1):
            r_ej = lr[i, ej_idx]
            r_eu = lr[i, eu_idx]
            r_uj = lr[i, uj_idx]
            residual = r_ej - (r_eu + r_uj)

            if abs(residual) < res_thresh:
                continue

            # Which leg lags?
            # If EURJPY went up (r_ej > 0) but EURUSD barely moved: EURUSD is the laggard
            # The laggard is the leg with the smallest contribution relative to expected
            if r_ej > 0:
                # Both legs should go up to explain EURJPY's rise
                # Laggard = leg with smaller (more negative) contribution
                laggard_idx = eu_idx if r_eu < r_uj else uj_idx
            else:
                # Both legs should go down
                laggard_idx = eu_idx if r_eu > r_uj else uj_idx

            # Forward returns
            fwd_ej = np.sum(lr[i+1:i+1+fwd, ej_idx])
            fwd_eu = np.sum(lr[i+1:i+1+fwd, eu_idx])
            fwd_uj = np.sum(lr[i+1:i+1+fwd, uj_idx])

            # Does the laggard catch up? (move in the direction EURJPY pointed)
            if r_ej > 0:
                catchup_eu = fwd_eu > 0  # EURUSD should rise
                catchup_uj = fwd_uj > 0  # USDJPY should rise
                catchup_laggard = fwd_eu if laggard_idx == eu_idx else fwd_uj
            else:
                catchup_eu = fwd_eu < 0
                catchup_uj = fwd_uj < 0
                catchup_laggard = fwd_eu if laggard_idx == eu_idx else fwd_uj

            catchup_direction = catchup_laggard > 0 if r_ej > 0 else catchup_laggard < 0

            results.append({
                'residual': residual,
                'r_ej': r_ej,
                'laggard': 'EURUSD' if laggard_idx == eu_idx else 'USDJPY',
                'fwd_eu': fwd_eu,
                'fwd_uj': fwd_uj,
                'catchup_eu': catchup_eu,
                'catchup_uj': catchup_uj,
                'catchup_laggard': catchup_laggard,
                'laggard_caught_up': catchup_direction,
            })

        rdf = pd.DataFrame(results)
        if len(rdf) < 5:
            continue

        wr = rdf['laggard_caught_up'].mean() * 100
        wr_eu = rdf[rdf['laggard'] == 'EURUSD']['laggard_caught_up'].mean() * 100 if len(rdf[rdf['laggard'] == 'EURUSD']) > 0 else 0
        wr_uj = rdf[rdf['laggard'] == 'USDJPY']['laggard_caught_up'].mean() * 100 if len(rdf[rdf['laggard'] == 'USDJPY']) > 0 else 0
        n_eu = len(rdf[rdf['laggard'] == 'EURUSD'])
        n_uj = len(rdf[rdf['laggard'] == 'USDJPY'])

        print(f"  thresh={res_thresh:.4f} fwd={fwd:>2d}: n={len(rdf):>4d} WR={wr:.1f}% (EU={n_eu} wr={wr_eu:.0f}%, UJ={n_uj} wr={wr_uj:.0f}%)")

# Strategy 2: Trade EURJPY itself when it's outrunning its legs
print("\n" + "=" * 60)
print("STRATEGY 2: Trade EURJPY continuation when legs confirm")
print("=" * 60)

for res_thresh in [0.0003, 0.0005, 0.001]:
    for fwd in [1, 2, 3, 5, 10]:
        results = []
        for i in range(0, len(lr) - fwd - 1):
            r_ej = lr[i, ej_idx]
            r_eu = lr[i, eu_idx]
            r_uj = lr[i, uj_idx]
            residual = r_ej - (r_eu + r_uj)

            if abs(residual) < res_thresh:
                continue

            # Trade EURJPY in its direction
            fwd_ej = np.sum(lr[i+1:i+1+fwd, ej_idx])
            continued = (r_ej > 0 and fwd_ej > 0) or (r_ej < 0 and fwd_ej < 0)

            results.append({
                'residual': residual,
                'r_ej': r_ej,
                'fwd_ej': fwd_ej,
                'continued': continued,
            })

        rdf = pd.DataFrame(results)
        if len(rdf) < 5:
            continue
        wr = rdf['continued'].mean() * 100
        # Subset: strong residual only
        strong = rdf[abs(rdf['residual']) > res_thresh * 2]
        wr_strong = strong['continued'].mean() * 100 if len(strong) > 5 else 0
        print(f"  thresh={res_thresh:.4f} fwd={fwd:>2d}: n={len(rdf):>4d} WR={wr:.1f}% (strong={len(strong)} wr={wr_strong:.0f}%)")

# Strategy 3: Beyond the triangle — cross-pair hedging from ANY pair's move
print("\n" + "=" * 60)
print("STRATEGY 3: Which pair best predicts another pair's next bar?")
print("=" * 60)

for a_idx, a_name in enumerate(pairs_in_data):
    for b_idx, b_name in enumerate(pairs_in_data):
        if a_name == b_name:
            continue
        # Does A's return predict B's next-bar return?
        for fwd in [1, 3, 5]:
            if fwd >= len(lr) - 5:
                continue
            a_rets = lr[:-fwd, a_idx]
            b_fwd = np.array([np.sum(lr[i+1:i+1+fwd, b_idx]) for i in range(len(lr) - fwd - 1)])
            min_l = min(len(a_rets), len(b_fwd))
            a_rets = a_rets[:min_l]
            b_fwd = b_fwd[:min_l]
            same = np.mean((a_rets > 0) == (b_fwd > 0))
            if same > 0.56 or same < 0.44:
                print(f"  {a_name:>7s}→{b_name:>7s} fwd={fwd}: same={same*100:.1f}%")
