"""
Market Graph — Wave 4: Cross-symbol propagation (A7).

Builds a weighted directed graph of symbol interactions.
Each edge represents the influence of symbol_i on symbol_j,
updated via EMA. Used to propagate signals across symbols.
"""
from collections import defaultdict
from typing import Dict


class MarketGraph:
    """Cross-symbol propagation graph with EMA edge weights."""

    def __init__(self, alpha: float = 0.05, propagation_weight: float = 0.15):
        self._edges: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.alpha = alpha
        self.propagation_weight = propagation_weight

    def update_edge(self, a: str, b: str, influence: float) -> None:
        if a == b:
            return
        prev = self._edges[a].get(b, 0.0)
        self._edges[a][b] = (1.0 - self.alpha) * prev + self.alpha * influence

    def propagate(self, symbol: str, signal: float) -> float:
        propagated = 0.0
        for neighbor, w in self._edges[symbol].items():
            propagated += w * signal
        return propagated * self.propagation_weight

    def edge_weight(self, a: str, b: str) -> float:
        return self._edges.get(a, {}).get(b, 0.0)

    def state(self) -> dict:
        return {a: dict(edges) for a, edges in self._edges.items()}
