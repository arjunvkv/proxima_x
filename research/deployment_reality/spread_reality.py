import logging
from typing import Optional

logger = logging.getLogger("proxima_ops.drl.spread")


class SpreadReality:
    def __init__(self):
        self._records: list[dict] = []

    def record(self, signal_id: str, symbol: str,
               spread_at_signal: int, spread_at_entry: Optional[int] = None,
               spread_1_bar: Optional[int] = None,
               spread_5_bar: Optional[int] = None,
               trade_pnl: Optional[float] = None,
               won: Optional[bool] = None):
        rec = {
            "signal_id": signal_id,
            "symbol": symbol,
            "spread_at_signal": spread_at_signal,
            "spread_at_entry": spread_at_entry,
            "spread_1_bar_later": spread_1_bar,
            "spread_5_bars_later": spread_5_bar,
            "trade_pnl": trade_pnl,
            "won": won}
        self._records.append(rec)
        return rec

    def summary(self) -> dict:
        if not self._records:
            return {"count": 0}
        wins = [r for r in self._records if r.get("won") is True]
        losses = [r for r in self._records if r.get("won") is False]
        w_spread = [r["spread_at_signal"] for r in wins if r.get("spread_at_signal") is not None]
        l_spread = [r["spread_at_signal"] for r in losses if r.get("spread_at_signal") is not None]
        return {
            "count": len(self._records),
            "wins": len(wins),
            "losses": len(losses),
            "mean_spread_wins": round(sum(w_spread) / len(w_spread), 1) if w_spread else None,
            "mean_spread_losses": round(sum(l_spread) / len(l_spread), 1) if l_spread else None,
            "spread_expansion_wins": sum(1 for r in wins if r.get("spread_1_bar_later") is not None and r["spread_1_bar_later"] > r.get("spread_at_signal", 0)),
            "spread_expansion_losses": sum(1 for r in losses if r.get("spread_1_bar_later") is not None and r["spread_1_bar_later"] > r.get("spread_at_signal", 0))}
