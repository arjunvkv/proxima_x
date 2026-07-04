import random
from typing import Optional, Tuple

from .slippage_model import SlippageModel


class FillEngine:

    def __init__(self):
        self.slippage_model = SlippageModel()

    def compute_fill_probability(self, order_size: float, liquidity: float, volatility: float, spread: float) -> float:
        base_prob = 0.95
        size_penalty = min(1.0, order_size * 0.1)
        volatility_penalty = volatility * 0.1
        spread_penalty = min(0.2, spread * 0.01)
        liquidity_bonus = liquidity * 0.05
        prob = base_prob - size_penalty - volatility_penalty - spread_penalty + liquidity_bonus
        return max(0.0, min(1.0, prob))

    def simulate_fill(self, fill_prob: float, random_seed: Optional[float] = None) -> bool:
        if random_seed is not None:
            random.seed(random_seed)
        return random.random() < fill_prob

    def simulate_partial_fill(self, order_size: float, fill_prob: float) -> Tuple[float, float]:
        if fill_prob > 0.9:
            filled_quantity = order_size
        elif fill_prob > 0.7:
            filled_quantity = order_size * random.uniform(0.8, 1.0)
        elif fill_prob > 0.5:
            filled_quantity = order_size * random.uniform(0.5, 0.8)
        else:
            filled_quantity = order_size * random.uniform(0.1, 0.5)
        fill_price_adjustment = random.gauss(0, 0.01)
        return filled_quantity, fill_price_adjustment

    def execute_order(self, order: dict, market_state: dict) -> dict:
        direction = order["direction"]
        spread = order.get("spread", market_state.get("spread", 0.01))
        volatility = order.get("volatility", market_state.get("volatility", 0.5))
        order_size = order.get("order_size", 1.0)
        liquidity = order.get("liquidity", market_state.get("liquidity", 1.0))
        entry_price = market_state.get("entry_price", 100.0)

        slippage = self.slippage_model.compute_slippage(direction, spread, volatility, order_size, liquidity)
        fill_prob = self.compute_fill_probability(order_size, liquidity, volatility, spread)
        filled = self.simulate_fill(fill_prob)

        if filled:
            fill_quantity, fill_price_adjustment = self.simulate_partial_fill(order_size, fill_prob)
            partial = fill_quantity < order_size
            adjusted_price = entry_price + slippage + fill_price_adjustment
            fill_price = max(0.01, adjusted_price)
            execution_cost = abs(slippage) * fill_quantity
        else:
            fill_quantity = 0.0
            fill_price = entry_price
            partial = False
            execution_cost = 0.0

        return {
            "filled": filled,
            "fill_quantity": fill_quantity,
            "fill_price": fill_price,
            "slippage": slippage,
            "execution_cost": execution_cost,
            "partial_fill": partial,
            "fill_probability": fill_prob
        }
