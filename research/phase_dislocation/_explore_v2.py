"""Fast dark exploration of multi-pair M1 data — fixed indexing."""
import numpy as np
import pandas as pd

df = pd.read_parquet("data/temp/mt5_m1_9day.parquet")
pairs = sorted(df["pair"].unique())
print(f"Pairs: {pairs}")

# Align to common timestamps
aligned = {}
for p in pairs:
    sub = df[df["pair"] == p].sort_values("time")
    aligned[p] = dict(zip(sub["time"], sub["close"]))

all_times = sorted(set.intersection(*[set(aligned[p].keys()) for p in pairs]))
price = np.column_stack([np.array([aligned[p][t] for t in all_times]) for p in pairs])
n, npairs = price.shape
print(f"Bars: {n}, Pairs: {npairs}")

# ================================================================
# EXPLORATION 1: Lonely Pair Reversion
# When 1 pair moves opposite to all others, does it revert?
# ================================================================
print("\n" + "=" * 60)
print("EXPLORATION 1: Lonely Pair Reversion")
print("=" * 60)

for lookback in [5, 10, 20, 30, 60]:
    total = 0
    correct = 0
    for i in range(lookback, n - 20):
        rets = np.array([price[i, j] / price[i - lookback, j] - 1 for j in range(npairs)])
        signs = np.sign(rets)
        n_pos = np.sum(signs > 0)
        n_neg = np.sum(signs < 0)

        if n_pos + n_neg < 4:
            continue
        if not (n_pos == 1 or n_neg == 1):
            continue  # not lonely

        consensus_dir = 1 if n_pos > n_neg else -1
        lonely_idx = int(np.where(signs == -consensus_dir)[0][0])

        for fwd in [5, 10, 20]:
            if i + fwd >= n:
                continue
            fwd_ret = price[i + fwd, lonely_idx] / price[i, lonely_idx] - 1
            reverted = (np.sign(fwd_ret) == consensus_dir)
            total += 1
            if reverted:
                correct += 1

    if total > 0:
        print(f"  Lookback={lookback:>2d}min: {total} events, reversion WR={correct/total*100:.1f}%")

# ================================================================
# EXPLORATION 2: Consensus Exhaustion
# When ALL pairs agree on direction for N bars, trade reversal of strongest
# ================================================================
print("\n" + "=" * 60)
print("EXPLORATION 2: Consensus Exhaustion Reversal")
print("=" * 60)

for lookback in [10, 20, 30, 60]:
    total = {5: 0, 10: 0, 20: 0}
    correct = {5: 0, 10: 0, 20: 0}
    for i in range(lookback, n - 60):
        rets = np.array([price[i, j] / price[i - lookback, j] - 1 for j in range(npairs)])
        signs = np.sign(rets)
        n_pos = np.sum(signs > 0)
        n_neg = np.sum(signs < 0)
        if n_pos + n_neg < 4:
            continue
        agreement = max(n_pos, n_neg) / (n_pos + n_neg)
        if agreement < 0.70:
            continue

        consensus_dir = 1 if n_pos > n_neg else -1
        # Strongest mover in consensus direction
        consensus_rets = [(j, rets[j]) for j in range(npairs) if np.sign(rets[j]) == consensus_dir]
        if not consensus_rets:
            continue
        best_j = max(consensus_rets, key=lambda x: abs(x[1]))[0]

        for fwd in [5, 10, 20]:
            if i + fwd >= n:
                continue
            fwd_ret = price[i + fwd, best_j] / price[i, best_j] - 1
            reversed = (np.sign(fwd_ret) != consensus_dir)
            total[fwd] += 1
            if reversed:
                correct[fwd] += 1

    for fwd in [5, 10, 20]:
        if total[fwd] > 0:
            wr = correct[fwd] / total[fwd] * 100
            print(f"  Lookback={lookback:>2d}min Fwd={fwd:>2d}min: {total[fwd]:>4d} trades, reversal WR={wr:.1f}%")

# ================================================================
# EXPLORATION 3: Cross-pair lead-lag directional accuracy
# ================================================================
print("\n" + "=" * 60)
print("EXPLORATION 3: Lead-Lag Directional Accuracy")
print("=" * 60)

r1 = np.diff(np.log(price), axis=0)
for lag in [1, 2, 5, 10]:
    total = 0
    correct = 0
    for a in range(npairs):
        for b in range(npairs):
            if a == b:
                continue
            n_obs = n - 1 - lag
            if n_obs < 100:
                continue
            a_signs = np.sign(r1[:n_obs, a])
            b_fwd = np.sign(r1[lag:lag + n_obs, b])
            mask = (a_signs != 0) & (b_fwd != 0)
            same = np.mean(a_signs[mask] == b_fwd[mask]) if np.sum(mask) > 0 else 0
            if np.sum(mask) > 50:
                total += 1
                if same > 0.5:
                    correct += 1
    if total > 0:
        print(f"  Lag={lag:>2d}min: {correct}/{total} pair combinations have >50% accuracy ({correct/total*100:.0f}%)")

# ================================================================
# EXPLORATION 4: The "Silent Opening" — after flat period, which
# pair leads the breakout?
# ================================================================
print("\n" + "=" * 60)
print("EXPLORATION 4: Post-Flat Breakout Leader")
print("=" * 60)

for quiet_len in [10, 20, 30]:
    for fwd in [1, 3, 5, 10]:
        total = 0
        leader_wins = {}
        for i in range(quiet_len + fwd, n - 5):
            # Check if ALL pairs were quiet
            quiet_rets = [abs(price[i, j] / price[i - quiet_len, j] - 1) for j in range(npairs)]
            max_q = max(quiet_rets)
            if max_q > 0.002:
                continue  # not quiet enough

            # Which pair moves first in the breakout?
            breakout_rets = [price[i + fwd, j] / price[i, j] - 1 for j in range(npairs)]
            leader = int(np.argmax(np.abs(breakout_rets)))

            # Does the leader's direction persist?
            if i + fwd + fwd >= n:
                continue
            cont_rets = [price[i + fwd + fwd, j] / price[i + fwd, j] - 1 for j in range(npairs)]
            leader_cont = np.sign(cont_rets[leader]) == np.sign(breakout_rets[leader])

            pair_name = pairs[leader]
            if pair_name not in leader_wins:
                leader_wins[pair_name] = {"total": 0, "cont": 0}
            leader_wins[pair_name]["total"] += 1
            if leader_cont:
                leader_wins[pair_name]["cont"] += 1
            total += 1

        if total > 0:
            best = max(leader_wins.items(), key=lambda x: x[1]["total"])
            print(f"  Quiet={quiet_len:>2d}min Fwd={fwd:>2d}min: {total} breakouts, best leader={best[0]} ({best[1]['cont']}/{best[1]['total']} continued)")
