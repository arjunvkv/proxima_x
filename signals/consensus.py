"""Cross-symbol consensus.
Measures directional alignment across correlated assets.
consensus_strength = abs(mean(sign(scores)))
consensus_direction = sign(mean(scores))
"""
import numpy as np


class CrossSymbolConsensus:
    def __init__(self, min_strength=0.60):
        self.min_strength = min_strength

    def update(self, symbol_scores):
        self._scores = symbol_scores

    def consensus_strength(self):
        if not hasattr(self, "_scores") or not self._scores:
            return 1.0  # default to unanimous for single-symbol mode
        scores = list(self._scores.values())
        if len(scores) < 2:
            return 1.0
        signs = [np.sign(s) for s in scores if s != 0]
        if not signs:
            return 0.0
        return abs(np.mean(signs))

    def consensus_direction(self):
        if not hasattr(self, "_scores") or not self._scores:
            return 0
        scores = list(self._scores.values())
        if len(scores) < 2:
            return int(np.sign(np.mean(scores))) if np.mean(scores) != 0 else 0
        return int(np.sign(np.mean(scores))) if np.mean(scores) != 0 else 0

    def is_aligned(self):
        return self.consensus_strength() >= self.min_strength
