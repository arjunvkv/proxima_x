"""Dark data exploration — mine M1 multi-pair data for non-obvious edges."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from collections import deque

# Load data
df = pd.read_parquet("data/temp/mt5_m1_9day.parquet")
pairs = sorted(df["pair"].unique())
print(f"Pairs: {pairs}")
print(f"Rows: {len(df):,}")
print(f"Time: {df['time'].min()} to {df['time'].max()}")

# Pivot: pair -> sorted rows
data = {}
for pair in pairs:
    sub = df[df["pair"] == pair].sort_values("time")
    data[pair] = sub.reset_index(drop=True)

# Align all pairs to common timestamps
all_times = set(data[pairs[0]]["time"])
for p in pairs[1:]:
    all_times = all_times & set(data[p]["time"])
all_times = sorted(all_times)
print(f"\nCommon timestamps: {len(all_times)}")

# Build aligned price array
prices = {}  # pair -> array of close prices at common times
for pair in pairs:
    ts_map = dict(zip(data[pair]["time"], data[pair]["close"]))
    prices[pair] = np.array([ts_map[t] for t in all_times])

n = len(all_times)
print(f"Aligned bars: {n}")

# ============================================================
# EXPLORATION 1: Cross-pair return predictability
# Does return in pair A at bar t predict return in pair B at t+1?
# ============================================================
print("\n" + "=" * 60)
print("EXPLORATION 1: Cross-pair lead-lag (M1)")
print("=" * 60)
for lag in [1, 2, 5, 10, 60]:
    results = []
    for a in pairs:
        for b in pairs:
            if a == b:
                continue
            r_a = np.diff(np.log(prices[a]))
            r_b = np.diff(np.log(prices[b]))
            min_len = min(len(r_a), len(r_b))
            r_a = r_a[:min_len]
            r_b = r_b[:min_len]

            # Lag correlation: does r_a[t] predict r_b[t+lag]?
            r_a_lagged = r_a[:-lag] if lag > 0 else r_a
            r_b_fwd = r_b[lag:] if lag > 0 else r_b
            min_l = min(len(r_a_lagged), len(r_b_fwd))
            if min_l < 100:
                continue

            corr = np.corrcoef(r_a_lagged[:min_l], r_b_fwd[:min_l])[0, 1]

            # Directional agreement
            same_sign = np.mean((r_a_lagged[:min_l] > 0) == (r_b_fwd[:min_l] > 0))

            results.append((a, b, lag, round(corr, 4), round(same_sign * 100, 1)))

    results.sort(key=lambda x: -abs(x[3]))
    print(f"\n  Lag={lag} min — Top 5 correlations:")
    for a, b, l, corr, ss in results[:5]:
        print(f"    {a:>7s} -> {b:>7s}: r={corr:>7.4f}, same_sign={ss:>5.1f}%")

# ============================================================
# EXPLORATION 2: Consensus exhaustion — does extended agreement
# predict reversal in the strongest mover?
# ============================================================
print("\n" + "=" * 60)
print("EXPLORATION 2: Consensus Exhaustion Reversal")
print("=" * 60)

for consensus_bars in [10, 20, 30, 60]:
    trades = []
    for i in range(consensus_bars + 5, n - 5):
        returns = {}
        for pair in pairs:
            prev = prices[pair][i - consensus_bars]
            cur = prices[pair][i]
            returns[pair] = (cur - prev) / prev

        # Check consensus: >60% agree on direction
        signs = [np.sign(r) for r in returns.values()]
        pos_count = sum(1 for s in signs if s > 0)
        neg_count = sum(1 for s in signs if s < 0)
        total_voting = pos_count + neg_count
        if total_voting < 4:
            continue
        consensus_dir = 1 if pos_count > neg_count else -1
        agreement_pct = max(pos_count, neg_count) / total_voting

        if agreement_pct < 0.65:
            continue

        # Pick the pair with the strongest move IN the consensus direction
        best_pair = None
        best_mag = 0
        for pair, r in returns.items():
            if np.sign(r) == consensus_dir and abs(r) > best_mag:
                best_mag = abs(r)
                best_pair = pair

        if best_pair is None:
            continue

        # Forward return over next 5-60 min
        for fwd_bars in [5, 10, 20, 60]:
            if i + fwd_bars >= n:
                continue
            fwd_ret = (prices[best_pair][i + fwd_bars] - prices[best_pair][i]) / prices[best_pair][i]
            is_win = (np.sign(fwd_ret) != consensus_dir)  # We want reversal
            trades.append({
                "pair": best_pair, "consensus_dir": consensus_dir,
                "agreement": agreement_pct, "entry_ret_bp": round(best_mag * 10000, 1),
                "fwd_bars": fwd_bars, "fwd_ret_bp": round(fwd_ret * 10000, 1),
                "is_reversal": is_win,
            })

    if trades:
        total = len(trades)
        reversals = sum(1 for t in trades if t["is_reversal"])
        wr = reversals / total * 100
        avg_fwd = np.mean([t["fwd_ret_bp"] for t in trades])
        print(f"\n  Consensus={consensus_bars}min agreement>65%:")
        for fb in [5, 10, 20, 60]:
            sub = [t for t in trades if t["fwd_bars"] == fb]
            if sub:
                rev = sum(1 for t in sub if t["is_reversal"])
                print(f"    Fwd {fb:>2d}min: {len(sub):>4d} trades, reversal WR={rev/len(sub)*100:>5.1f}%")
    else:
        print(f"\n  Consensus={consensus_bars}min: No trades")

# ============================================================
# EXPLORATION 3: Dispersion breakout — when pairs decouple
# after moving together, does the follower catch up?
# ============================================================
print("\n" + "=" * 60)
print("EXPLORATION 3: Dispersion Breakout Follower Catch-up")
print("=" * 60)

for lookback in [10, 20, 30, 60]:
    results = []
    for i in range(lookback + 1, n - 20):
        returns = {}
        for pair in pairs:
            returns[pair] = (prices[pair][i] / prices[pair][i - lookback]) - 1

        rets = np.array(list(returns.values()))
        dispersion = np.std(rets)
        prev_rets = []
        for j in range(max(0, i - lookback * 3), i):
            r = {}
            for pair in pairs:
                r[pair] = (prices[pair][j] / prices[pair][j - lookback]) - 1
            prev_rets.append(np.std(list(r.values())))
        avg_disp = np.mean(prev_rets)

        if avg_disp < 1e-10:
            continue
        disp_ratio = dispersion / avg_disp

        if disp_ratio < 2.0:
            continue  # Not a significant dispersion event

        # Find the leader (biggest mover) and followers
        sorted_pairs = sorted(returns.items(), key=lambda x: -abs(x[1]))
        leader = sorted_pairs[0]
        followers = [p for p, r in sorted_pairs[1:] if abs(r) > 0]

        if not followers:
            continue

        # Does any follower catch up in the next 10 bars?
        for fwd in [5, 10]:
            if i + fwd >= n:
                continue
            for foll in followers:
                fwd_ret_foll = (prices[foll][i + fwd] / prices[foll][i]) - 1
                fwd_ret_lead = (prices[leader[0]][i + fwd] / prices[leader[0]][i]) - 1

                # Follower catching up = moving more in leader's direction
                leader_sign = np.sign(leader[1])
                foll_catch = np.sign(fwd_ret_foll) == leader_sign
                lead_cont = np.sign(fwd_ret_lead) == leader_sign

                results.append({
                    "leader": leader[0], "follower": foll,
                    "leader_ret_bp": round(leader[1] * 10000, 1),
                    "foll_ret_bp": round(returns[foll] * 10000, 1),
                    "disp_ratio": round(disp_ratio, 1),
                    "fwd": fwd,
                    "foll_catch": foll_catch,
                    "lead_continued": lead_cont,
                })

    if results:
        for fb in [5, 10]:
            sub = [r for r in results if r["fwd"] == fb]
            if sub:
                catch = sum(1 for r in sub if r["foll_catch"])
                lead = sum(1 for r in sub if r["lead_continued"])
                print(f"\n  Lookback={lookback}min, dispersion>2x avg, Fwd={fb}min:")
                print(f"    Follower catch-up: {catch}/{len(sub)} = {catch/len(sub)*100:.1f}%")
                print(f"    Leader continued:  {lead}/{len(sub)} = {lead/len(sub)*100:.1f}%")
