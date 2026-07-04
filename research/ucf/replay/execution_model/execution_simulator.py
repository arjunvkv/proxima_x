from typing import List

from .slippage_model import SlippageModel
from .fill_engine import FillEngine


class ExecutionSimulator:

    def __init__(self):
        self.slippage_model = SlippageModel()
        self.fill_engine = FillEngine()

    def simulate_entry(self, entry_price: float, direction: int, spread: float, volatility: float, order_size: float = 1.0, liquidity: float = 1.0) -> dict:
        order = {
            "direction": direction,
            "spread": spread,
            "volatility": volatility,
            "order_size": order_size,
            "liquidity": liquidity
        }
        market_state = {
            "entry_price": entry_price,
            "spread": spread,
            "volatility": volatility,
            "liquidity": liquidity
        }
        result = self.fill_engine.execute_order(order, market_state)
        return result

    def simulate_exit(self, entry_price: float, exit_price: float, direction: int, spread: float, volatility: float) -> dict:
        exit_direction = -direction
        slippage = self.slippage_model.compute_slippage(exit_direction, spread, volatility)
        cost = abs(slippage)
        adjusted_exit_price = exit_price - slippage
        gross_pnl = (exit_price - entry_price) * direction
        net_pnl = gross_pnl - cost
        return {
            "exit_price": adjusted_exit_price,
            "slippage": slippage,
            "cost": cost,
            "net_pnl": net_pnl
        }

    def simulate_trade(self, entry: dict, exit: dict) -> dict:
        return {
            "symbol": entry.get("symbol", ""),
            "direction": entry.get("direction", 0),
            "entry_price": entry.get("entry_price", 0.0),
            "exit_price": exit.get("exit_price", 0.0),
            "filled": entry.get("filled", False),
            "quantity": entry.get("fill_quantity", 0.0),
            "gross_pnl": exit.get("net_pnl", 0.0),
            "execution_cost": entry.get("execution_cost", 0.0) + exit.get("cost", 0.0),
            "net_pnl": exit.get("net_pnl", 0.0) - entry.get("execution_cost", 0.0),
            "slippage_total": entry.get("slippage", 0.0) + exit.get("slippage", 0.0),
            "fill_probability": entry.get("fill_probability", 0.0),
            "partial_fill": entry.get("partial_fill", False)
        }

    def simulate_trade_batch(self, entries: List[dict], exits: List[dict]) -> List[dict]:
        results = []
        for entry, exit in zip(entries, exits):
            trade = self.simulate_trade(entry, exit)
            results.append(trade)
        return results
