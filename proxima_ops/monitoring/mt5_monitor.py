import logging
import time
from datetime import datetime
from typing import Optional
from proxima_ops.config.settings import SETTINGS
from proxima_ops.execution.mt5_connector import MT5Connector

logger = logging.getLogger("proxima_ops.monitor.mt5")


class MT5Monitor:
    def __init__(self, mt5: MT5Connector):
        self._mt5 = mt5
        self._last_check: Optional[float] = None
        self._last_healthy: Optional[float] = None
        self._consecutive_failures = 0
        self._status_history: list[dict] = []

    @property
    def status(self) -> str:
        if self._mt5.is_connected:
            return "CONNECTED"
        return "DISCONNECTED"

    @property
    def uptime_minutes(self) -> float:
        if self._last_healthy is None:
            return 0.0
        return (time.time() - self._last_healthy) / 60.0

    def check(self) -> dict:
        self._last_check = time.time()
        connected = self._mt5.is_connected
        account = self._mt5.get_account() if connected else None
        symbol_status = {}
        if connected:
            # Only check first 5 symbols to avoid 28x MT5 calls per cycle
            for sym in SETTINGS.symbols[:5]:
                info = self._mt5.verify_symbol(sym)
                spread_raw = self._mt5.get_tick(sym)
                spread = spread_raw["spread"] if spread_raw else info["spread"]
                spread_invalid = spread < 0 or spread >= 999
                spread_stale = spread == 0
                spread_ok = not spread_invalid and self._mt5.verify_spread(sym)
                symbol_status[sym] = {
                    "available": info["available"],
                    "spread": spread,
                    "spread_ok": spread_ok,
                    "trade_mode": info["trade_mode"],
                    "spread_invalid": spread_invalid,
                    "spread_stale": spread_stale}
        status = {
            "timestamp": datetime.now().isoformat(),
            "connected": connected,
            "account": account,
            "symbols": symbol_status,
            "consecutive_failures": self._consecutive_failures}
        self._status_history.append(status)
        if connected:
            self._last_healthy = time.time()
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1
        return status

    @property
    def health_summary(self) -> dict:
        status = self.status
        return {
            "mt5_status": status,
            "uptime_minutes": round(self.uptime_minutes, 1),
            "consecutive_failures": self._consecutive_failures,
            "symbols_ok": all(
                s.get("spread_ok", False)
                for s in self._status_history[-1].get("symbols", {}).values()
            ) if self._status_history else False}
