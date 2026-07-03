import logging
from typing import Optional

logger = logging.getLogger("proxima_ops.drl.regime")


class RegimeReality:
    def __init__(self):
        self._trades: list[dict] = []

    def record(self, symbol: str, regime: str,
               pnl_points: float, pnl_money: float,
               duration: int, es_rank: float, at_rank: float):
        rec = {
            "symbol": symbol, "regime": regime,
            "pnl_points": pnl_points, "pnl_money": pnl_money,
            "duration": duration, "es_rank": es_rank, "at_rank": at_rank}
        self._trades.append(rec)
        return rec

    def summary(self) -> dict:
        if not self._trades:
            return {"count": 0}
        regimes = {}
        for t in self._trades:
            r = t.get("regime", "UNKNOWN")
            regimes.setdefault(r, []).append(t)
        result = {}
        for r, trades in regimes.items():
            pnls = [t["pnl_money"] for t in trades]
            pts = [t["pnl_points"] for t in trades]
            mean_r = sum(pnls) / len(pnls) if pnls else 0
            pp = sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0
            sharpe = (mean_r / (sum((p - mean_r) ** 2 for p in pnls) / len(pnls)) ** 0.5
                      ) if len(pnls) > 1 else 0
            dd = min(0, min(pnls)) if pnls else 0
            result[r] = {
                "count": len(trades),
                "pp": round(pp, 4),
                "sharpe": round(sharpe, 4),
                "mean_return": round(sum(pnls) / len(pnls), 2),
                "mean_return_pts": round(sum(pts) / len(pts), 1) if pts else 0,
                "drawdown": round(abs(dd), 2),
                "mean_duration": round(sum(t["duration"] for t in trades) / len(trades), 1),
                "mean_es": round(sum(t["es_rank"] for t in trades) / len(trades), 4),
                "mean_at": round(sum(t["at_rank"] for t in trades) / len(trades), 4)}
        return {"regime_count": len(regimes), "regimes": result}
