from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

BUY = 0
SELL = 1
ENTRY_IN = 0
ENTRY_OUT = 1


@dataclass
class ExitEvent:
    exit_time: datetime
    exit_price: float
    volume: float
    profit: float
    deal_id: int

    @property
    def is_profitable(self) -> bool:
        return self.profit > 0


@dataclass
class PositionLifecycle:
    position_id: int
    symbol: str
    direction: int
    entry_time: datetime
    entry_price: float
    total_volume: float
    exit_events: List[ExitEvent] = field(default_factory=list)
    total_pnl: float = 0.0
    total_swap: float = 0.0
    total_commission: float = 0.0
    magic: int = 0

    @property
    def final_exit_time(self) -> Optional[datetime]:
        return max((e.exit_time for e in self.exit_events), default=None)

    @property
    def weighted_exit_price(self) -> float:
        if not self.exit_events:
            return 0.0
        total_v = sum(e.volume for e in self.exit_events)
        if total_v == 0:
            return 0.0
        return sum(e.exit_price * e.volume for e in self.exit_events) / total_v

    @property
    def net_pnl(self) -> float:
        return self.total_pnl + self.total_swap + self.total_commission

    @property
    def partial_close_count(self) -> int:
        return len(self.exit_events)

    @property
    def has_full_exit(self) -> bool:
        return len(self.exit_events) > 0

    def to_dict(self) -> Dict:
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "direction": "BUY" if self.direction == BUY else "SELL",
            "entry_time": self.entry_time.isoformat(),
            "entry_price": self.entry_price,
            "total_volume": self.total_volume,
            "total_pnl": self.total_pnl,
            "total_swap": self.total_swap,
            "total_commission": self.total_commission,
            "net_pnl": self.net_pnl,
            "weighted_exit_price": self.weighted_exit_price,
            "final_exit_time": self.final_exit_time.isoformat() if self.final_exit_time else None,
            "partial_close_count": self.partial_close_count,
            "exit_events": [
                {
                    "exit_time": e.exit_time.isoformat(),
                    "exit_price": e.exit_price,
                    "volume": e.volume,
                    "profit": e.profit,
                    "deal_id": e.deal_id,
                }
                for e in self.exit_events
            ],
        }


class MT5HistoryLoader:
    def __init__(self) -> None:
        self._mt5 = None

    def _ensure_mt5(self):
        if self._mt5 is None:
            import MetaTrader5 as mt5
            self._mt5 = mt5
            self._mt5.initialize()

    def load_positions(self, days_back: int = 90) -> List[PositionLifecycle]:
        self._ensure_mt5()
        end = datetime.now()
        start = end - timedelta(days=days_back)
        deals = self._mt5.history_deals_get(start, end)
        if deals is None or len(deals) == 0:
            print(f"[MT5_HISTORY] No deals found in last {days_back} days")
            return []

        raw = [self._normalize_deal(d) for d in deals]
        raw = [d for d in raw if d is not None]

        return self._group_by_position(raw)

    def _normalize_deal(self, deal) -> Optional[Dict]:
        typ = getattr(deal, "type", -1)
        entry = getattr(deal, "entry", -1)
        if typ not in (BUY, SELL):
            return None
        if entry not in (ENTRY_IN, ENTRY_OUT):
            return None
        vol = getattr(deal, "volume", 0.0)
        if vol <= 0 and entry == ENTRY_IN:
            return None
        return {
            "deal_id": getattr(deal, "deal", 0),
            "position_id": getattr(deal, "position_id", 0),
            "symbol": getattr(deal, "symbol", ""),
            "type": typ,
            "entry": entry,
            "time": datetime.fromtimestamp(getattr(deal, "time", 0)),
            "price": float(getattr(deal, "price", 0.0)),
            "volume": float(vol),
            "profit": float(getattr(deal, "profit", 0.0)),
            "swap": float(getattr(deal, "swap", 0.0)),
            "commission": float(getattr(deal, "commission", 0.0)),
            "magic": getattr(deal, "magic", 0),
        }

    def _group_by_position(self, deals: List[Dict]) -> List[PositionLifecycle]:
        groups: Dict[int, Dict] = {}
        for d in deals:
            pid = d["position_id"]
            if pid == 0:
                continue
            if pid not in groups:
                groups[pid] = {"entries": [], "exits": []}
            if d["entry"] == ENTRY_IN:
                groups[pid]["entries"].append(d)
            else:
                groups[pid]["exits"].append(d)

        positions = []
        for pid, g in groups.items():
            if not g["entries"]:
                continue
            entry = g["entries"][0]
            if not g["exits"]:
                continue
            exits = sorted(g["exits"], key=lambda x: x["time"])
            total_vol = sum(e["volume"] for e in g["entries"])
            total_pnl = sum(e["profit"] for e in g["exits"])
            total_swap = sum(e["swap"] for e in g["entries"] + g["exits"])
            total_comm = sum(e["commission"] for e in g["entries"] + g["exits"])

            pl = PositionLifecycle(
                position_id=pid,
                symbol=entry["symbol"],
                direction=entry["type"],
                entry_time=entry["time"],
                entry_price=entry["price"],
                total_volume=total_vol,
                total_pnl=total_pnl,
                total_swap=total_swap,
                total_commission=total_comm,
                magic=entry["magic"],
            )
            for ex in exits:
                pl.exit_events.append(ExitEvent(
                    exit_time=ex["time"],
                    exit_price=ex["price"],
                    volume=ex["volume"],
                    profit=ex["profit"],
                    deal_id=ex["deal_id"],
                ))
            positions.append(pl)

        return sorted(positions, key=lambda p: p.entry_time)

    def load_tick_path(self, symbol: str, entry_time: datetime,
                       exit_time: datetime,
                       pre_buffer_minutes: int = 15,
                       post_buffer_minutes: int = 5) -> np.ndarray:
        self._ensure_mt5()
        if exit_time is None:
            return np.array([])
        from_ts = int((entry_time - timedelta(minutes=pre_buffer_minutes)).timestamp())
        to_ts = int((exit_time + timedelta(minutes=post_buffer_minutes)).timestamp())
        ticks = self._mt5.copy_ticks_range(symbol, from_ts, to_ts, self._mt5.COPY_TICKS_ALL)
        if ticks is None or len(ticks) == 0:
            return np.array([])
        return np.array(ticks.tolist(), dtype=np.float64)

    def summary(self, positions: List[PositionLifecycle]) -> Dict:
        if not positions:
            return {"total": 0}
        winners = [p for p in positions if p.net_pnl > 0]
        losers = [p for p in positions if p.net_pnl <= 0]
        return {
            "total": len(positions),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": len(winners) / len(positions) if positions else 0,
            "total_pnl": sum(p.net_pnl for p in positions),
            "avg_pnl": sum(p.net_pnl for p in positions) / len(positions),
            "total_volume": sum(p.total_volume for p in positions),
            "symbols": list(set(p.symbol for p in positions)),
            "partial_closes": sum(1 for p in positions if p.partial_close_count > 1),
        }

    def shutdown(self) -> None:
        if self._mt5:
            self._mt5.shutdown()
            self._mt5 = None
