from __future__ import annotations

from typing import Dict
import numpy as np
from mvs.continuation.synthetic_exit import SyntheticExitEngine


class ShadowContinuationEngine:
    __slots__ = ("synthetic",)

    def __init__(self) -> None:
        self.synthetic = SyntheticExitEngine()

    def continue_trade(self, trade_id: int, entry_price: float, direction: int, tick_prices: np.ndarray, avg_spread: float, actual_pnl: float) -> Dict:
        synthetic = self.synthetic.compute(entry_price=entry_price, direction=direction, tick_prices=tick_prices, avg_spread=avg_spread)
        continuation_alpha = synthetic["best_pnl"] / max(abs(actual_pnl), 1e-9)
        return {"shadow_exit_price": synthetic["best_exit_price"], "shadow_pnl": synthetic["best_pnl"], "shadow_duration_ticks": len(tick_prices), "shadow_path": tick_prices.tolist(), "continuation_alpha": float(continuation_alpha), "best_reason": synthetic["best_reason"]}
