"""
ReplayMT5Connector - Drop-in replacement for MT5Connector.
Has the EXACT same API surface so run_proxima_demo.py needs zero changes to signal logic.
Delegates to PaperBroker, ReplayClock, and ReplayTickSource internally.
"""
import time
import logging
from typing import Optional
from datetime import datetime

from core.adapters.tick_source import TickSource, ReplayTickSource
from core.adapters.clock import ReplayClock
from core.adapters.broker import PaperBroker

logger = logging.getLogger("proxima.replay.connector")


class ReplayMT5Connector:
    def __init__(self, tick_source: ReplayTickSource, clock: ReplayClock, broker: PaperBroker):
        self._tick_source = tick_source
        self._clock = clock
        self._broker = broker
        self._connected = True
        self._account_info = None
        self._last_error = None

    @property
    def is_connected(self) -> bool:
        return True

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def connect(self) -> bool:
        self._connected = True
        logger.info("ReplayMT5Connector: Connected (virtual)")
        return True

    def disconnect(self):
        self._connected = False
        logger.info("ReplayMT5Connector: Disconnected")

    def ensure_connection(self) -> bool:
        return True

    def get_account(self) -> Optional[dict]:
        return self._broker.get_account()

    def get_positions(self) -> list[dict]:
        return self._broker.get_positions()

    def _get_broker_symbol(self, symbol: str) -> str:
        return self._broker._get_broker_symbol(symbol)

    def _point(self, symbol: str) -> float:
        sym = symbol.upper()
        if "XAU" in sym or "GOLD" in sym:
            return 0.01
        if "JPY" in sym:
            return 0.001
        if "XAG" in sym or "SILVER" in sym:
            return 0.0001
        return 0.00001

    def get_tick(self, symbol: str) -> Optional[dict]:
        tick = self._tick_source.get_tick(symbol)
        if tick is None:
            return None
        bid = float(tick.get("bid", 0.0))
        ask = float(tick.get("ask", 0.0))
        point = self._point(symbol)
        spread_pts = max(1, int((ask - bid) / point)) if point > 0 and ask > bid else 1
        ts = tick.get("time_sec", tick.get("timestamp", 0))
        result = {
            "symbol": symbol,
            "bid": bid,
            "ask": ask,
            "spread": spread_pts,
            "time": int(ts),
        }
        if hasattr(self._broker, '_feed_tick_for_bars'):
            self._broker._feed_tick_for_bars_manual(symbol, tick)
        return result

    def get_rates(self, symbol: str, count: int = 100, timeframe: int = 0) -> Optional[list]:
        return self._broker.get_rates(symbol, count=count, timeframe=timeframe)

    def verify_symbol(self, symbol: str) -> dict:
        return self._broker.verify_symbol(symbol)

    def verify_spread(self, symbol: str, es_rank: float = None) -> bool:
        return self._broker.verify_spread(symbol, es_rank=es_rank)

    def _get_filling_mode(self, symbol: str) -> int:
        return 1

    def place_order(self, symbol: str, order_type: str, volume: float,
                    price: float, sl: float = 0.0, tp: float = 0.0,
                    comment: str = "PROXIMA_V2") -> Optional[dict]:
        return self._broker.place_order(symbol, order_type, volume, price, sl=sl, tp=tp, comment=comment)

    def close_order(self, ticket: int) -> bool:
        return self._broker.close_order(ticket)

    def close_all(self) -> list[dict]:
        return self._broker.close_all()

    def get_deal_history(self, ticket: int) -> Optional[list]:
        return None
