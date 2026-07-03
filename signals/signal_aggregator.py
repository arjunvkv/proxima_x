"""Rolling DOA accumulator.
Accumulates recent OSS signals with recency-weighted decay.
score = Σ(signal_i * confidence_i * decay^age_i)
Normalized to [-1, +1].
"""
from collections import deque


class RollingAccumulator:
    def __init__(self, window=20, decay=0.85):
        self.window = window
        self.decay = decay
        self._buffer = deque(maxlen=window)
        self._scores = deque(maxlen=window)  # cached for normalization

    def update(self, signal, confidence=1.0):
        self._buffer.append((signal, confidence))

    def score(self):
        if not self._buffer:
            return 0.0
        total = 0.0
        weight_sum = 0.0
        for i, (sig, conf) in enumerate(reversed(self._buffer)):
            w = conf * (self.decay ** i)
            total += sig * w
            weight_sum += w
        if weight_sum == 0:
            return 0.0
        return max(-1.0, min(1.0, total / weight_sum))

    def confidence(self):
        s = self.score()
        return abs(s)

    def direction(self):
        s = self.score()
        return 1 if s >= 0.65 else (-1 if s <= -0.65 else 0)

    def reset(self):
        self._buffer.clear()
        self._scores.clear()
