import random
from typing import List


class SlippageModel:

    def compute_slippage(self, direction: int, spread: float, volatility: float, order_size: float = 1.0, liquidity: float = 1.0) -> float:
        base_slippage = spread * 0.5
        volatility_impact = volatility * order_size * 0.3
        liquidity_penalty = (1.0 - liquidity) * 0.2
        total_slippage = base_slippage + volatility_impact + liquidity_penalty
        noise = random.gauss(0, total_slippage * 0.2)
        slippage = direction * (total_slippage + noise)
        return slippage

    def compute_slippage_batch(self, orders: List[dict]) -> List[float]:
        results = []
        for order in orders:
            direction = order["direction"]
            spread = order["spread"]
            volatility = order["volatility"]
            order_size = order.get("order_size", 1.0)
            liquidity = order.get("liquidity", 1.0)
            slippage = self.compute_slippage(direction, spread, volatility, order_size, liquidity)
            results.append(slippage)
        return results
