"""
GLOBAL RANK ENGINE — Phase 2 Deliverable

Computes cross-asset percentile ranking from per-symbol local ranks.

At each evaluation cycle:
1. Record local ES percentile for all assets via record_evaluation()
2. Call compute() to build cross-sectional ranking
3. Read global_rank, global_percentile per asset

Methods:
- record_evaluation(symbol, local_rank, raw_es): Call for every symbol at each eval cycle
- compute(): After all symbols recorded, compute global ranks
- get_global_percentile(sym): 0-100 scale
- get_global_rank(sym): 1-based rank (1 = best among all assets)
- get_qualified_assets(min_pct=80): Assets meeting global percentile threshold
"""

import numpy as np
from collections import OrderedDict


class GlobalRankEngine:
    def __init__(self):
        self._evaluations = OrderedDict()
        self._global_ranks = {}
        self._global_percentiles = {}
        self._n_assets = 0

    def record_evaluation(self, symbol: str, local_rank: float, raw_es: float = None):
        self._evaluations[symbol] = {
            "local_rank": local_rank,
            "raw_es": raw_es or 0.0,
        }

    def compute(self):
        if not self._evaluations:
            return {}
        symbols = list(self._evaluations.keys())
        self._n_assets = len(symbols)
        local_ranks = [self._evaluations[s]["local_rank"] for s in symbols]
        sorted_ranks = sorted(local_ranks)
        for sym in symbols:
            lr = self._evaluations[sym]["local_rank"]
            gr_pct = float(sum(1 for r in sorted_ranks if r <= lr)) / len(sorted_ranks) * 100.0
            gr_rank = len(sorted_ranks) - sum(1 for r in sorted_ranks if r <= lr) + 1
            self._global_percentiles[sym] = round(gr_pct, 1)
            self._global_ranks[sym] = gr_rank
        return self._global_percentiles

    def get_global_percentile(self, symbol: str) -> float:
        return self._global_percentiles.get(symbol, 0.0)

    def get_global_rank(self, symbol: str) -> int:
        return self._global_ranks.get(symbol, 0)

    def get_qualified_assets(self, min_global_pct: float = 80.0,
                              symbol_thresholds: dict = None):
        """Return assets meeting qualification criteria.

        If symbol_thresholds is provided, each symbol's local rank is compared
        against its per-symbol threshold (normalization bias fix).
        Otherwise, the global cross-asset percentile is compared against min_global_pct.
        """
        if symbol_thresholds:
            return sorted(
                [s for s in self._evaluations
                 if s in symbol_thresholds
                 and self._evaluations[s]["local_rank"] >= symbol_thresholds[s]],
                key=lambda s: -self._evaluations[s]["local_rank"],
            )
        return sorted(
            [s for s in self._evaluations if self._global_percentiles.get(s, 0) >= min_global_pct],
            key=lambda s: -self._global_percentiles.get(s, 0),
        )

    def get_qualified_by_local(self, symbol_thresholds: dict) -> list:
        """Return assets whose local ECDF rank meets per-symbol threshold."""
        return self.get_qualified_assets(symbol_thresholds=symbol_thresholds)

    def summary(self, symbol_thresholds: dict = None):
        lines = [f"{'Asset':<12} {'LocalRank':<12} {'GlobalRank':<14} {'GlobalPct':<12} {'Thresh':<8} {'Qual':<6}"]
        lines.append("-" * 66)
        for sym in self._evaluations:
            lr = self._evaluations[sym]["local_rank"]
            gr = self._global_ranks.get(sym, 0)
            gp = self._global_percentiles.get(sym, 0.0)
            th = symbol_thresholds.get(sym, 0.0) if symbol_thresholds else 80.0
            qual = "Y" if (symbol_thresholds and lr >= th) or (gp >= 80) else "N"
            lines.append(f"{sym:<12} {lr:<12.1f} {gr:<14} {gp:<12.1f} {th:<8.2f} {qual:<6}")
        return "\n".join(lines)

    def clear(self):
        self._evaluations.clear()

    @property
    def n_assets(self):
        return self._n_assets


def demo():
    """Run a demonstration with synthetic data matching the spec example."""
    engine = GlobalRankEngine()
    test_data = {
        "EURJPY": 92,
        "USDJPY": 95,
        "GBPJPY": 75,
        "XAUUSD": 88,
        "EURUSD": 91,
    }
    for sym, lr in test_data.items():
        engine.record_evaluation(sym, lr)
    engine.compute()
    print(engine.summary())
    print()
    print(f"Qualified assets (>=80%): {engine.get_qualified_assets()}")


if __name__ == "__main__":
    demo()
