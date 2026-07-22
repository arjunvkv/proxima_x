"""Fast dark exploration of multi-pair M1 data."""
import numpy as np
import pandas as pd
import time

# Load and align
df = pd.read_parquet("data/temp/mt5_m1_9day.parquet")
pairs = sorted(df["pair"].unique())
print(f"Pairs: {pairs}")

# Pivot
aligned = {}
for p in pairs:
    sub = df[df["pair"] == p].sort_values("time")
    aligned[p] = dict(zip(sub["time"], sub["close"]))

all_times = sorted(set.intersection(*[set(aligned[p].keys()) for p in pairs]))
print(f"Common timestamps: {len(all_times)}")

price = np.column_stack([np.array([aligned[p][t] for t in all_times]) for p in pairs])
log_price = np.log(price)
n, npairs = price.shape
print(f"Array shape: {price.shape}")

# Precompute log returns at various lags
print("\nPrecomputing returns...")
max_lag = 120
ret = {}
for lag in [1, 2, 5, 10, 20, 30, 60]:
    ret[lag] = np.diff(log_price, n=1, axis=0)
    if lag > 1:
        # For longer lookback, diff directly
        pass
    # Actually let me just use raw array indexing for lookback

# ================================================================
# EXPLORATION 1: Does pair-level trend exhaustion predict reversal?
# When a pair has moved strongly for N bars, does it then revert?
# ================================================================
print("\n" + "=" * 60)
print("EXPLORATION 1: Trend Exhaustion Reversal")
print("=" * 60)

for lookback in [10, 20, 30, 60]:
    for p_idx in range(npairs):
        pair = pairs[p_idx]
        # Compute lookback return
        lookback_rets = np.full(n, np.nan)
        for i in range(lookback, n):
            lookback_rets[i] = price[i, p_idx] / price[i - lookback, p_idx] - 1

        # Find extreme moves (>1% or >2%)
        for pct_thresh in [0.005, 0.01, 0.02]:
            extreme_up = lookback_rets > pct_thresh
            extreme_dn = lookback_rets < -pct_thresh
            extreme = extreme_up | extreme_dn

            n_extreme = np.sum(extreme)
            if n_extreme < 5:
                continue

            # Forward returns after extreme
            for fwd in [5, 10, 20, 60]:
                n_possible = np.sum(extreme[:n - fwd - 1])
                if n_possible < 3:
                    continue

                fwd_rets = np.full(n, np.nan)
                for i in range(0, n - fwd):
                    fwd_rets[i] = price[i + fwd, p_idx] / price[i, p_idx] - 1

                mask = extreme[:n - fwd - 1]
                extreme_fwd = fwd_rets[:n - fwd - 1][mask]

                # We want REVERSAL after extreme move
                reversal = np.mean((extreme_fwd < 0) if np.nanmean(lookback_rets[mask]) > 0 else (extreme_fwd > 0))

                if n_possible >= 10:
                    print(f"  L={lookback:>2d} {pair:>7s} thresh={pct_thresh:.3f} Fwd={fwd:>2d}: {int(n_possible):>4d} events, reversal={reversal*100:.0f}%")

# ================================================================
# EXPLORATION 2: Can we predict direction from cross-pair returns?
# Does EURUSD's return predict USDJPY's return?
# ================================================================
print("\n" + "=" * 60)
print("EXPLORATION 2: Cross-pair Directional Prediction")
print("=" * 60)

log_px = np.log(price)
for lookback in [1, 5, 10, 30, 60]:
    for fwd in [1, 5, 10, 30]:
        if lookback + fwd >= n - 5:
            continue
        results = []
        for a_idx in range(npairs):
            a = pairs[a_idx]
            for b_idx in range(npairs):
                if a == b:
                    continue
                b = pairs[b_idx]

                # Return of A over lookback
                r_a = price[lookback:, a_idx] / price[:-lookback, a_idx] - 1
                # Return of B over fwd
                r_b_fwd = price[lookback + fwd:, b_idx] / price[lookback:-fwd, b_idx] - 1
                min_l = min(len(r_a), len(r_b_fwd))
                if min_l < 50:
                    continue
                r_a = r_a[:min_l]
                r_b_fwd = r_b_fwd[:min_l]

                # Same direction
                same_dir = np.mean((r_a > 0) == (r_b_fwd > 0))

                # Opposite
                opp_dir = np.mean((r_a > 0) != (r_b_fwd > 0))

                results.append((a, b, same_dir, opp_dir))

        if results:
            best = max(results, key=lambda x: max(x[2], x[3]))
            a, b, same, opp = best
            print(f"  L={lookback:>2d}→Fwd={fwd:>2d}: Best: {a:>7s}→{b:>7s} same={same*100:.0f}% opp={opp*100:.0f}%")

# ================================================================
# EXPLORATION 3: The "Lonely Pair" Effect
# When one pair moves opposite to all others, does it revert?
# ================================================================
print("\n" + "=" * 60)
print("EXPLORATION 3: Lonely Pair Reversion")
print("=" * 60)

for lookback in [5, 10, 20, 60]:
    n_events = 0
    n_reversals = 0
    for i in range(lookback, n - 20):
        rets = price[i] / price[i - lookback] - 1
        signs = np.sign(rets)
        n_pos = np.sum(signs > 0)
        n_neg = np.sum(signs < 0)
        n_nonzero = n_pos + n_neg
        if n_nonzero < 4:
            continue

        # Lonely pair: one pair moves opposite to consensus
        lonely = n_pos == 1 or n_neg == 1
        if not lonely:
            continue

        # Which is the lonely one?
        consensus_dir = 1 if n_pos > n_neg else -1
        lonely_idx = np.where(signs == -consensus_dir)[0][0]
        lonely_pair = pairs[lonely_idx]

        # Does the lonely pair revert in next 20 min?
        for fwd in [5, 10, 20]:
            if i + fwd >= n:
                continue
            fwd_ret = price[i + fwd, lonely_idx] / price[i, lonely_idx] - 1
            # Reversion = lonely pair moves back toward consensus direction
            reversed = np.sign(fwd_ret) == consensus_dir
            n_events += 1
            if reversed:
                n_reversals += 1

    if n_events > 0:
        print(f"  Lookback={lookback}min: {n_events} events, reversion={n_reversals/n_events*100:.0f}%")
