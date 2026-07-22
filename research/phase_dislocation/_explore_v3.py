"""Refined exploration — consensus exhaustion with stronger filters."""
import numpy as np
import pandas as pd
from collections import defaultdict

df = pd.read_parquet("data/temp/mt5_m1_9day.parquet")
pairs = sorted(df["pair"].unique())

# Align
aligned = {}
for p in pairs:
    sub = df[df["pair"] == p].sort_values("time")
    aligned[p] = dict(zip(sub["time"], sub["close"]))
all_times = sorted(set.intersection(*[set(aligned[p].keys()) for p in pairs]))
price = np.column_stack([np.array([aligned[p][t] for t in all_times]) for p in pairs])
n, npairs = price.shape
print(f"Bars: {n}, Pairs: {npairs}")

# ================================================================
# Consensus exhaustion — exact per-pair breakdown
# ================================================================
print("\n" + "=" * 60)
print("CONSENSUS EXHAUSTION — PER-PAIR BREAKDOWN")
print("=" * 60)

for agreement_req in [0.70, 0.80, 0.90]:
    for lookback in [20, 30, 60]:
        pair_stats = defaultdict(lambda: {"trades": 0, "wins": 0})
        for i in range(lookback, n - 60):
            rets = np.array([price[i, j] / price[i - lookback, j] - 1 for j in range(npairs)])
            signs = np.sign(rets)
            n_pos = np.sum(signs > 0)
            n_neg = np.sum(signs < 0)
            if n_pos + n_neg < 5:
                continue
            agreement = max(n_pos, n_neg) / (n_pos + n_neg)
            if agreement < agreement_req:
                continue

            consensus_dir = 1 if n_pos > n_neg else -1

            for j in range(npairs):
                if np.sign(rets[j]) != consensus_dir:
                    continue
                # Forward return — does this pair continue or reverse?
                for fwd in [5, 10, 20]:
                    if i + fwd >= n:
                        continue
                    fwd_ret = price[i + fwd, j] / price[i, j] - 1
                    reversed = (np.sign(fwd_ret) != consensus_dir)
                    pair_stats[(pairs[j], fwd)]["trades"] += 1
                    if reversed:
                        pair_stats[(pairs[j], fwd)]["wins"] += 1

        print(f"\n  Agreement>{agreement_req:.0f}% Lookback={lookback}min:")
        for fwd in [5, 10, 20]:
            # Best pair for reversal
            best_pair = max(pair_stats.keys(), key=lambda k: pair_stats[k]["wins"]/max(pair_stats[k]["trades"],1) if k[1]==fwd else 0)
            s = pair_stats[(best_pair[0], fwd)]
            if s["trades"] > 0:
                wr = s["wins"] / s["trades"] * 100
                print(f"    Fwd={fwd:>2d}min Best={best_pair[0]:>7s}: {s['trades']:>4d} trades, reversal WR={wr:.1f}%")

# ================================================================
# Session-specific consensus exhaustion (only during liquid hours)
# ================================================================
print("\n" + "=" * 60)
print("SESSION-SPECIFIC CONSENSUS EXHAUSTION")
print("=" * 60)

# Convert timestamps to hour of day
hours = np.array([int(pd.Timestamp(t, unit='s').hour + pd.Timestamp(t, unit='s').minute/60) for t in all_times])

for session_name, hr_start, hr_end in [("Asia", 0, 8), ("London", 7, 16), ("NY", 12, 21), ("London+NY", 12, 18)]:
    session_mask = (hours >= hr_start) & (hours < hr_end)
    session_indices = np.where(session_mask)[0]

    for lookback in [30, 60]:
        total = defaultdict(int)
        wins = defaultdict(int)
        for base_i in session_indices:
            i = int(base_i)
            if i < lookback or i > n - 30:
                continue
            rets = np.array([price[i, j] / price[i - lookback, j] - 1 for j in range(npairs)])
            signs = np.sign(rets)
            n_pos = np.sum(signs > 0)
            n_neg = np.sum(signs < 0)
            if n_pos + n_neg < 5:
                continue
            agreement = max(n_pos, n_neg) / (n_pos + n_neg)
            if agreement < 0.75:
                continue

            consensus_dir = 1 if n_pos > n_neg else -1
            for j in range(npairs):
                if np.sign(rets[j]) != consensus_dir:
                    continue
                for fwd in [5, 10, 20]:
                    if i + fwd >= n:
                        continue
                    fwd_ret = price[i + fwd, j] / price[i, j] - 1
                    reversed = (np.sign(fwd_ret) != consensus_dir)
                    k = (pairs[j], fwd)
                    total[k] += 1
                    if reversed:
                        wins[k] += 1

        if total:
            best = max(total.keys(), key=lambda k: wins[k]/max(total[k],1))
            wr = wins[best]/total[best]*100 if total[best] > 0 else 0
            print(f"  {session_name:>12s} L={lookback}: Best={best[0]:>7s} Fwd={best[1]:>2d}: {total[best]:>4d} trades, WR={wr:.1f}%")
