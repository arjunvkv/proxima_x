from __future__ import annotations

from typing import Dict, List


class AgeRebuilder:
    __slots__ = ("trade_entry_ts", "tick_counters")

    def __init__(self) -> None:
        self.trade_entry_ts = {}
        self.tick_counters = {}

    def on_tick(self, tick_id: int, symbol: str, ts_ns: int, open_trades: List[dict]) -> Dict:
        if not open_trades:
            return {"age_ticks": 0, "age_seconds": 0.0}
        oldest = min(open_trades, key=lambda x: x.get("entry_ts_ns", x.get("ts_ns", ts_ns)))
        entry_ts = oldest.get("entry_ts_ns", oldest.get("ts_ns", ts_ns))
        age_ticks = tick_id - oldest.get("entry_tick_id", oldest.get("tick_id", tick_id))
        age_seconds = (ts_ns - entry_ts) / 1_000_000_000.0
        return {"age_ticks": int(age_ticks), "age_seconds": float(age_seconds)}
