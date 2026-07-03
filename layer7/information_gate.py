"""
Information Gate — Wave 5 A1: Information strangulation layer.

Prevents over-conditioning and feature explosion by:
- Capping signal vector dimensionality
- Computing signal entropy for diagnostic logging
"""
import math


class InformationGate:
    def __init__(self, max_dim: int = 12):
        self.max_dim = max_dim

    def compress(self, signal_vector: list) -> list:
        if len(signal_vector) > self.max_dim:
            return signal_vector[:self.max_dim]
        return signal_vector

    def entropy(self, signal_vector: list) -> float:
        if not signal_vector:
            return 0.0
        total = sum(abs(x) for x in signal_vector) + 1e-8
        p = [abs(x) / total for x in signal_vector]
        return -sum(pi * math.log(pi) for pi in p if pi > 0)
