import logging
from typing import Optional
from proxima_ops.execution.mt5_connector import MT5Connector

logger = logging.getLogger("proxima_ops.positions")


class PositionManager:
    def __init__(self, mt5: MT5Connector):
        self._mt5 = mt5
        self._local_positions: dict[int, dict] = {}

    def refresh(self):
        positions = self._mt5.get_positions()
        self._local_positions = {p["ticket"]: p for p in positions}

    @property
    def positions(self) -> list[dict]:
        self.refresh()
        return list(self._local_positions.values())

    def get(self, ticket: int) -> Optional[dict]:
        self.refresh()
        return self._local_positions.get(ticket)

    def get_by_symbol(self, symbol: str) -> Optional[dict]:
        self.refresh()
        for p in self._local_positions.values():
            if p["symbol"] == symbol:
                return p
        return None

    SIGNAL_EXIT_HORIZONS = {
        "OSS": 5,
        "TrOSS": 10,
        "SAL": 5,
    }

    def get_exit_horizon(self, signal_type: str, default: int = 10) -> int:
        return self.SIGNAL_EXIT_HORIZONS.get(signal_type, default)

    @property
    def total_profit(self) -> float:
        return sum(p["profit"] for p in self.positions)

    @property
    def active_count(self) -> int:
        return len(self.positions)

    @property
    def is_max_positions(self) -> bool:
        return self.active_count >= 5

    def has_symbol(self, symbol: str) -> bool:
        return self.get_by_symbol(symbol) is not None

    def hydrate_from_mt5(self, mt5_pos: dict) -> bool:
        """Hydrate internal state from a broker position discovered by watchdog."""
        try:
            ticket = mt5_pos.get("ticket")
            if ticket is None:
                return False
            self._local_positions[ticket] = dict(mt5_pos)
            self._local_positions[ticket]["source"] = "broker_recovery"
            return True
        except Exception:
            return False

    def close_ghost_position(self, ticket: int) -> bool:
        """Remove a ledger-only ghost position from internal state."""
        try:
            self._local_positions.pop(ticket, None)
            return True
        except Exception:
            return False

    def summary(self) -> dict:
        pos = self.positions
        return {
            "total": len(pos),
            "profit": sum(p["profit"] for p in pos),
            "positions": pos}
