"""State Transition Engine — Phase IV (v2).
Models ECDF bucket-to-bucket transitions for amplitude expansion.

Key insight: transitions between states may have larger absolute move
size than isolated directional signals, potentially surpassing spread.
"""
import numpy as np
from collections import defaultdict


class TransitionGraph:
    """Tracks bucket-to-bucket transition counts + amplitude.

    Graph[b][n] = {count, total_amplitude, mean_amplitude}
    where b = from_bucket (index 0-9), n = to_bucket (index 0-9)
    """
    NUM_BUCKETS = 10

    def __init__(self):
        self._graph = defaultdict(lambda: defaultdict(lambda: {"count": 0, "total_amp": 0.0}))
        self._total_transitions = 0

    def record(self, from_bucket, to_bucket, amplitude):
        """Record a transition from one ECDF decile to another."""
        if not (0 <= from_bucket < self.NUM_BUCKETS and 0 <= to_bucket < self.NUM_BUCKETS):
            return
        self._graph[from_bucket][to_bucket]["count"] += 1
        self._graph[from_bucket][to_bucket]["total_amp"] += abs(amplitude)
        self._total_transitions += 1

    def probability(self, from_bucket, to_bucket):
        """P(next_bucket | current_bucket)"""
        fb = self._graph[from_bucket]
        total = sum(v["count"] for v in fb.values())
        if total == 0:
            return 0.0
        return fb[to_bucket]["count"] / total

    def mean_amplitude(self, from_bucket, to_bucket):
        """E[|move| | from → to]"""
        fb = self._graph[from_bucket]
        if fb[to_bucket]["count"] == 0:
            return 0.0
        return fb[to_bucket]["total_amp"] / fb[to_bucket]["count"]

    def transition_entropy(self, from_bucket):
        """Shannon entropy of transition distribution from this bucket (normalized)."""
        fb = self._graph[from_bucket]
        total = sum(v["count"] for v in fb.values())
        if total == 0:
            return 1.0
        entropy = 0.0
        for v in fb.values():
            p = v["count"] / total
            if p > 0:
                entropy -= p * np.log2(p)
        max_entropy = np.log2(self.NUM_BUCKETS)
        return entropy / max_entropy if max_entropy > 0 else 1.0

    def expected_direction(self, from_bucket):
        """Direction-weighted expected outcome: weighted sum of to_bucket - from_bucket."""
        fb = self._graph[from_bucket]
        total = sum(v["count"] for v in fb.values())
        if total == 0:
            return 0.0
        expected = 0.0
        for to_b, v in fb.items():
            p = v["count"] / total
            expected += p * (to_b / (self.NUM_BUCKETS - 1) - from_bucket / (self.NUM_BUCKETS - 1))
        return expected

    def top_transitions(self, from_bucket, n=3):
        """Top-n transitions by mean amplitude from this bucket."""
        fb = self._graph[from_bucket]
        candidates = [(to_b, v["count"], v["total_amp"] / v["count"]) for to_b, v in fb.items()]
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[:n]

    def reset(self):
        self._graph.clear()
        self._total_transitions = 0


class RollingTransitionTracker:
    """Maintains transition graph over rolling window.

    Trades when:
    1. Current bucket has low transition entropy (< threshold)
    2. Expected direction is strong (abs(expected) > threshold)
    3. Expected amplitude > min_amplitude
    """
    def __init__(self, window=100, entropy_threshold=0.60, min_direction=0.15, min_amplitude=0.02):
        self.window = window
        self.entropy_threshold = entropy_threshold
        self.min_direction = min_direction
        self.min_amplitude = min_amplitude
        self._graph = TransitionGraph()
        self._history = []
        self._prev_bucket = None
        self._prev_price = None

    def update(self, ecdf, price):
        """Update with new ECDF value. Returns trade signal {-1, 0, +1}."""
        bucket = min(int(ecdf * 10), 9)

        if self._prev_bucket is not None and self._prev_price is not None:
            amplitude = price - self._prev_price
            self._graph.record(self._prev_bucket, bucket, amplitude)
            self._history.append((self._prev_bucket, bucket, amplitude))

            # Trim old transitions
            while len(self._history) > self.window:
                old_from, old_to, old_amp = self._history.pop(0)
                fb = self._graph._graph[old_from]
                if old_to in fb:
                    fb[old_to]["count"] -= 1
                    fb[old_to]["total_amp"] -= abs(old_amp)
                    if fb[old_to]["count"] <= 0:
                        del fb[old_to]
                self._graph._total_transitions -= 1

        self._prev_bucket = bucket
        self._prev_price = price

        return self._signal(bucket)

    def _signal(self, bucket):
        """Generate trade signal from current bucket."""
        entropy = self._graph.transition_entropy(bucket)
        if entropy > self.entropy_threshold or entropy == 1.0:
            return 0

        expected_dir = self._graph.expected_direction(bucket)
        if abs(expected_dir) < self.min_direction:
            return 0

        top = self._graph.top_transitions(bucket, n=1)
        if not top:
            return 0

        to_b, count, amp = top[0]
        if abs(amp) < self.min_amplitude:
            return 0

        return 1 if expected_dir > 0 else -1

    def entropy(self, bucket=None):
        if bucket is not None:
            return self._graph.transition_entropy(bucket)
        if self._prev_bucket is not None:
            return self._graph.transition_entropy(self._prev_bucket)
        return 1.0

    def transition_count(self):
        return self._graph._total_transitions

    def reset(self):
        self._graph.reset()
        self._history.clear()
        self._prev_bucket = None
        self._prev_price = None
