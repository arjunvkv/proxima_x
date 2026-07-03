import logging
from typing import Optional

logger = logging.getLogger("proxima_ops.drl.drawdown")


class DrawdownForensics:
    def __init__(self):
        self._losses: list[dict] = []

    def record(self, symbol: str, regime: str,
               es_rank: float, at_rank: float,
               persistence_forecast: str, duration: int,
               pnl_points: float, pnl_money: float):
        rec = {
            "symbol": symbol, "regime": regime,
            "es_rank": es_rank, "at_rank": at_rank,
            "persistence_forecast": persistence_forecast,
            "duration": duration,
            "pnl_points": pnl_points, "pnl_money": pnl_money}
        self._losses.append(rec)
        return rec

    def summary(self) -> dict:
        if not self._losses:
            return {"count": 0}
        regimes = {}
        for l in self._losses:
            r = l.get("regime", "UNKNOWN")
            regimes.setdefault(r, []).append(l)
        regime_breakdown = {}
        for r, losses in regimes.items():
            pnls = [x["pnl_money"] for x in losses]
            regime_breakdown[r] = {
                "count": len(losses),
                "total_loss": round(sum(pnls), 2),
                "mean_loss": round(sum(pnls) / len(pnls), 2),
                "mean_duration": round(sum(x["duration"] for x in losses) / len(losses), 1),
                "mean_es": round(sum(x["es_rank"] for x in losses) / len(losses), 4),
                "mean_at": round(sum(x["at_rank"] for x in losses) / len(losses), 4)}
        return {
            "count": len(self._losses),
            "total_loss": round(sum(l["pnl_money"] for l in self._losses), 2),
            "regime_breakdown": regime_breakdown}
