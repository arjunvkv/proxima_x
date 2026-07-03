"""Diagnose STE — dump transition graph statistics."""
import sys; sys.path.insert(0, '.')
import time
import numpy as np
from collections import defaultdict

from research.replay_cache import ReplayCache
from signals.state_transition import RollingTransitionTracker, TransitionGraph

ticks = ReplayCache(["EURJPY"], "2026-04-21", "2026-05-10", tick_limit=50000, seed=42).compute()

# Warmup
ste = RollingTransitionTracker(window=500, entropy_threshold=1.0, min_direction=0.0, min_amplitude=0.0)

print(f"{'='*80}\n  TRANSITION GRAPH DIAGNOSTIC (50k ticks, EURJPY)\n{'='*80}")

signals = []
entropies = {b: [] for b in range(10)}
expecteds = {b: [] for b in range(10)}
for t in ticks:
    sig = ste.update(t["ecdf"], t["price"])
    signals.append(sig)

# After warmup, dump graph stats
g = ste._graph
print(f"\n  Total transitions recorded: {g._total_transitions}")
print(f"\n  Bucket | Count | MeanEntropy | MeanExpectedDir | TopTransition (count, amp)")
fb_buckets = sorted(g._graph.keys())
for b in fb_buckets:
    fb = g._graph[b]
    total = sum(v["count"] for v in fb.values())
    if total == 0:
        continue
    ent = g.transition_entropy(b)
    exp_dir = g.expected_direction(b)
    top = g.top_transitions(b, n=1)
    top_str = f"{top[0][0]}:{top[0][1]} ({top[0][2]:.4f})" if top else "none"
    print(f"  {b:>6}  | {total:>5} | {ent:>10.3f} | {exp_dir:>+12.4f} | {top_str}")

# Distribution of expected directions over time
print(f"\n  Signal distribution: {sum(1 for s in signals if s==1)} UP, "
      f"{sum(1 for s in signals if s==-1)} DOWN, "
      f"{sum(1 for s in signals if s==0)} FLAT")
print(f"  Signal fraction: {sum(1 for s in signals if s!=0)/len(signals)*100:.2f}%")
