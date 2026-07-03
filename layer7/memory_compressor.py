"""
Memory Compressor — Wave 4: Entropy decay + long-term memory compression.

Tracks a rolling buffer of values and computes:
- Shannon entropy of the distribution
- Decay factor (1/(1+entropy)) for weighting down high-entropy regimes
- Prevents memory overconfidence collapse and regime overfitting
"""
import numpy as np
from typing import List, Optional


class MemoryCompressor:
    """Rolling entropy-based memory compression.

    Higher entropy → higher decay (more compression).
    Used to downweight signals from high-uncertainty regimes.
    """

    def __init__(self, max_buffer: int = 100):
        self.buffer: List[float] = []
        self.max_buffer = max_buffer

    def add(self, value: float) -> None:
        self.buffer.append(value)
        if len(self.buffer) > self.max_buffer:
            self.buffer.pop(0)

    def entropy(self) -> float:
        if len(self.buffer) < 5:
            return 0.0
        arr = np.array(self.buffer, dtype=float)
        p = np.abs(arr) / (np.sum(np.abs(arr)) + 1e-8)
        return float(-np.sum(p * np.log(p + 1e-8)))

    def decay_factor(self) -> float:
        return 1.0 / (1.0 + self.entropy())

    def state(self) -> dict:
        return {
            "buffer_size": len(self.buffer),
            "entropy": round(self.entropy(), 4),
            "decay_factor": round(self.decay_factor(), 4),
        }
