import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("proxima_demo")


class CohortMemory:
    def __init__(self):
        self._cohort_events: Dict[str, list] = defaultdict(list)
        self._pairwise_fracture: Dict[str, int] = defaultdict(int)
        self._pairwise_total: Dict[str, int] = defaultdict(int)
        self._survival_clusters: Dict[str, List[Tuple[int, int, int]]] = defaultdict(list)

    def record(self, symbol: str, survival_vector: Tuple[int, int, int], fracture_score: float):
        self._cohort_events[symbol].append({
            "vector": survival_vector,
            "fracture": fracture_score,
        })
        self._survival_clusters[symbol].append(survival_vector)
        logger.info(f"[COHORT_MEMORY] record {symbol} vector={survival_vector} "
                    f"fracture={fracture_score:.2f} total_events={len(self._cohort_events[symbol])}")

    def update_pairwise(self, symbols_active: List[str]):
        for i in range(len(symbols_active)):
            for j in range(i + 1, len(symbols_active)):
                a, b = symbols_active[i], symbols_active[j]
                pair = tuple(sorted([a, b]))
                self._pairwise_total[pair] += 1
                events_a = self._cohort_events.get(a, [])
                events_b = self._cohort_events.get(b, [])
                if events_a and events_b:
                    fa = events_a[-1]["fracture"]
                    fb = events_b[-1]["fracture"]
                    if fa > 0.5 and fb > 0.5:
                        self._pairwise_fracture[pair] += 1

    def cohort_instability(self, symbol: str) -> float:
        sym_events = self._cohort_events.get(symbol, [])
        if not sym_events:
            return 0.0
        fractures = [e["fracture"] for e in sym_events if e["fracture"] is not None]
        if not fractures:
            return 0.0
        cohort_frac = []
        for pair, total in self._pairwise_total.items():
            if symbol in pair:
                frac_count = self._pairwise_fracture.get(pair, 0)
                if total > 0:
                    cohort_frac.append(frac_count / total)
        if not cohort_frac:
            return sum(fractures) / len(fractures)
        return (sum(fractures) / len(fractures) + sum(cohort_frac) / len(cohort_frac)) / 2.0

    def cluster_alignment(self, symbol: str) -> float:
        vectors = self._survival_clusters.get(symbol, [])
        if len(vectors) < 2:
            return 0.0
        n = len(vectors)
        agreements = 0
        total_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_pairs += 1
                agreements += sum(1 for a, b in zip(vectors[i], vectors[j]) if a == b)
        max_possible = total_pairs * 3
        return agreements / max_possible if max_possible > 0 else 0.0

    def co_fracture_matrix(self) -> Dict[str, Dict[str, float]]:
        symbols = set()
        for pair in self._pairwise_total:
            symbols.update(pair)
        matrix = {}
        for s in symbols:
            matrix[s] = {}
            for other in symbols:
                if s == other:
                    matrix[s][other] = 1.0
                    continue
                pair = tuple(sorted([s, other]))
                total = self._pairwise_total.get(pair, 0)
                frac = self._pairwise_fracture.get(pair, 0)
                matrix[s][other] = round(frac / max(total, 1), 3)
        return matrix

    def stats(self) -> dict:
        symbols = list(self._cohort_events.keys())
        total_events = sum(len(v) for v in self._cohort_events.values())
        frac_symbols = sum(1 for s in symbols
                          if self.cohort_instability(s) > 0.3)
        aligned = sum(1 for s in symbols
                     if self.cluster_alignment(s) > 0.5)
        return {
            "symbols": len(symbols),
            "total_events": total_events,
            "pairs_tracked": len(self._pairwise_total),
            "instability_symbols": frac_symbols,
            "aligned_symbols": aligned,
            "instability_asymmetry": max(
                [self.cohort_instability(s) for s in symbols] or [0.0]
            ) - min([self.cohort_instability(s) for s in symbols] or [0.0]),
        }
