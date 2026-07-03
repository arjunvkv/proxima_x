import logging
from typing import Optional

logger = logging.getLogger("proxima_ops.drl.exec_quality")


class ExecutionQuality:
    def __init__(self):
        self._entries: list[dict] = []

    def record(self, signal_id: str, symbol: str,
               ideal_price: float, actual_price: float,
               point_value: float = 0.0001):
        slippage_pts = (actual_price - ideal_price) / point_value if point_value > 0 else 0.0
        rec = {
            "signal_id": signal_id,
            "symbol": symbol,
            "ideal_price": ideal_price,
            "actual_price": actual_price,
            "slippage_pts": round(abs(slippage_pts), 2),
            "slippage_bps": round(abs(slippage_pts) * point_value / ideal_price * 10000, 2) if ideal_price > 0 else 0}
        self._entries.append(rec)
        return rec

    def summary(self) -> dict:
        if not self._entries:
            return {"count": 0, "classification": "NO_DATA"}
        slippages = [e["slippage_pts"] for e in self._entries]
        mean_slip = sum(slippages) / len(slippages)
        max_slip = max(slippages)
        std_slip = (sum((s - mean_slip) ** 2 for s in slippages) / len(slippages)) ** 0.5
        if mean_slip < 0.5:
            cls = "EXCELLENT"
        elif mean_slip < 1.5:
            cls = "GOOD"
        elif mean_slip < 5.0:
            cls = "DEGRADED"
        else:
            cls = "CRITICAL"
        return {
            "count": len(self._entries),
            "mean_slippage_pts": round(mean_slip, 2),
            "max_slippage_pts": round(max_slip, 2),
            "slippage_std": round(std_slip, 2),
            "classification": cls}
