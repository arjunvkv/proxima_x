from collections import defaultdict
from config.settings import CURRENCY_LIST, BASE_CURRENCY_MAP


class CurrencyObservability:

    def calculate(self, active_symbols: list[str]) -> dict[str, float]:
        graph = defaultdict(set)
        for symbol in active_symbols:
            base = symbol[:3]
            quote = symbol[3:6]
            if base in CURRENCY_LIST and quote in CURRENCY_LIST:
                graph[base].add(quote)
                graph[quote].add(base)

        scores = {}
        max_degree = len(CURRENCY_LIST) - 1
        for currency in CURRENCY_LIST:
            neighbors = graph.get(currency, set())
            degree_score = min(len(neighbors) / 7.0, 1.0)
            diversity_score = len(neighbors) / max(max_degree, 1)
            scores[currency] = degree_score * 0.6 + diversity_score * 0.4

        return scores
