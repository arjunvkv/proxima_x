from collections import Counter


class CurrencyConcentration:

    def calculate(self, positions: list) -> dict[str, float]:
        exposure = Counter()
        for p in positions:
            sym = p.symbol if hasattr(p, 'symbol') else (p.get('symbol', ''))
            base = sym[:3]
            quote = sym[3:6]
            direction = p.direction if hasattr(p, 'direction') else p.get('direction', 'BUY')
            mult = 1 if direction == "BUY" else -1
            exposure[base] += mult
            exposure[quote] -= mult

        total = sum(abs(v) for v in exposure.values()) or 1
        return {c: abs(v) / total for c, v in exposure.items()}
