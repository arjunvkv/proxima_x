from __future__ import annotations

from typing import Dict, List
import numpy as np


class SyntheticExitEngine:
    __slots__ = ()

    def compute(self, entry_price: float, direction: int, tick_prices: np.ndarray, avg_spread: float) -> Dict:
        if len(tick_prices) == 0:
            return {"best_exit_price": entry_price, "best_pnl": 0.0, "best_reason": "EMPTY", "all_exits": []}
        exits: List[Dict] = []
        rules = {
            "h20_exit": min(20, len(tick_prices) - 1),
            "entropy_inversion": int(len(tick_prices) * 0.60),
            "tpi_inversion": int(len(tick_prices) * 0.75),
            "age_expiry": min(500, len(tick_prices) - 1),
            "regime_break": int(len(tick_prices) * 0.40),
        }
        for reason, idx in rules.items():
            exit_price = float(tick_prices[min(idx, len(tick_prices) - 1)])
            pnl = (exit_price - entry_price) * direction
            exits.append({"reason": reason, "exit_price": exit_price, "pnl": float(pnl)})
        best = max(exits, key=lambda x: x["pnl"])
        return {"best_exit_price": best["exit_price"], "best_pnl": best["pnl"], "best_reason": best["reason"], "all_exits": exits}
