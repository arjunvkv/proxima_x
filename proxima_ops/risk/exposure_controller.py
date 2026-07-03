"""RHL-6: Exposure Controller — prevent concentration across asset classes."""

import logging
from typing import Optional

logger = logging.getLogger("proxima_ops.risk.exposure")

MAX_POSITIONS_TOTAL = 5
MAX_FX_POSITIONS = 3
MAX_GOLD_POSITIONS = 1
MAX_INDEX_POSITIONS = 1


class ExposureController:
    def __init__(self):
        pass

    def classify_asset(self, symbol: str) -> str:
        s = symbol.upper()
        if s in ("US100", "USTEC", "NQ100"):
            return "index"
        if s in ("XAUUSD", "GOLD", "XAGUSD"):
            return "gold"
        return "fx"

    def check(self, positions: list[dict], new_symbol: str = None) -> dict:
        if not positions:
            return {"allowed": True, "reason": "", "fx": 0, "gold": 0, "index": 0, "total": 0}

        total = len(positions)
        fx = sum(1 for p in positions if self.classify_asset(p.get("symbol", "")) == "fx")
        gold = sum(1 for p in positions if self.classify_asset(p.get("symbol", "")) == "gold")
        index = sum(1 for p in positions if self.classify_asset(p.get("symbol", "")) == "index")

        if new_symbol:
            cls = self.classify_asset(new_symbol)
            if total >= MAX_POSITIONS_TOTAL:
                return {"allowed": False, "reason": f"max_positions_total ({MAX_POSITIONS_TOTAL})", "fx": fx, "gold": gold, "index": index, "total": total}
            if cls == "fx" and fx >= MAX_FX_POSITIONS:
                return {"allowed": False, "reason": f"max_fx_positions ({MAX_FX_POSITIONS})", "fx": fx, "gold": gold, "index": index, "total": total}
            if cls == "gold" and gold >= MAX_GOLD_POSITIONS:
                return {"allowed": False, "reason": f"max_gold_positions ({MAX_GOLD_POSITIONS})", "fx": fx, "gold": gold, "index": index, "total": total}
            if cls == "index" and index >= MAX_INDEX_POSITIONS:
                return {"allowed": False, "reason": f"max_index_positions ({MAX_INDEX_POSITIONS})", "fx": fx, "gold": gold, "index": index, "total": total}

        return {"allowed": True, "reason": "", "fx": fx, "gold": gold, "index": index, "total": total}
