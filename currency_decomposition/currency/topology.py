from collections import defaultdict
from config.settings import CURRENCY_LIST


class GraphTopology:

    def pair_weights(self, active_symbols: list[str]) -> dict[str, float]:
        graph = defaultdict(set)
        for symbol in active_symbols:
            base = symbol[:3]
            quote = symbol[3:6]
            if base in CURRENCY_LIST and quote in CURRENCY_LIST:
                graph[base].add(quote)
                graph[quote].add(base)

        weights = {}
        for symbol in active_symbols:
            base = symbol[:3]
            quote = symbol[3:6]
            base_degree = len(graph[base])
            quote_degree = len(graph[quote])
            weight = (min(base_degree / 7.0, 1.0)) * (min(quote_degree / 7.0, 1.0))
            weights[symbol] = weight

        return weights

    def currency_edge_balance(self, returns: dict[str, float]) -> dict[str, float]:
        counts = {c: 0 for c in CURRENCY_LIST}
        for symbol, value in returns.items():
            if value == 0:
                continue
            base = symbol[:3]
            quote = symbol[3:6]
            if base in CURRENCY_LIST:
                counts[base] += 1
            if quote in CURRENCY_LIST:
                counts[quote] += 1
        total = sum(counts.values()) or 1
        return {c: counts[c] / total for c in CURRENCY_LIST}
