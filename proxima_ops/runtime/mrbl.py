"""
MRBL — MT5 Runtime Binding Layer

Converts MT5 execution into a persistent streaming connection.
Replaces function-call execution with event-based order emission.
"""

import time
from typing import Optional


class MT5RuntimeBinding:
    """Binds MT5 execution into a persistent streaming runtime layer.

    Provides connect/disconnect lifecycle, order emission with retry
    and latency tracking, tick subscription, and runtime status.
    """

    def __init__(self, mt5_connector=None, max_retries: int = 3):
        self._mt5_connector = mt5_connector
        self._max_retries = max_retries
        self._connected = False
        self._simulation_mode = mt5_connector is None
        self._symbols_subscribed = []
        self._last_tick_time = None
        self._orders_emitted_total = 0
        self._orders_failed_total = 0
        self._start_time = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Initialize MT5 connection.

        If a connector was provided, delegates to its initializer.
        Otherwise runs in simulation mode and returns True.
        """
        try:
            if self._mt5_connector is not None:
                self._connected = self._mt5_connector.initialize()
            else:
                self._connected = True
                self._simulation_mode = True

            if self._connected:
                self._start_time = time.time()

            return self._connected
        except Exception as exc:
            self._connected = False
            return False

    def disconnect(self) -> bool:
        """Close the MT5 connection gracefully."""
        try:
            if self._mt5_connector is not None:
                self._mt5_connector.shutdown()
            self._connected = False
            self._symbols_subscribed = []
            return True
        except Exception:
            self._connected = False
            return False

    # ------------------------------------------------------------------
    # Order emission
    # ------------------------------------------------------------------

    def emit_order(self, order_params: dict) -> dict:
        """Send an order to MT5 with retry and latency tracking.

        Parameters
        ----------
        order_params : dict
            Parameters for the order (symbol, volume, type, etc.).

        Returns
        -------
        dict
            Result with keys: emitted, ticket, error, retry_count, latency_ms.
        """
        start = time.perf_counter()
        last_error = None
        ticket = None
        retries = 0

        for attempt in range(self._max_retries + 1):
            try:
                if self._mt5_connector is not None:
                    result = self._mt5_connector.place_order(order_params)
                    # result could be a dict or a scalar ticket number
                    if isinstance(result, dict):
                        ticket = result.get("ticket")
                        if ticket is None and result.get("retcode", -1) != 0:
                            last_error = str(result.get("comment", "order failed"))
                            retries = attempt + 1
                            continue
                    else:
                        ticket = result
                else:
                    # Simulation mode: fake a ticket
                    ticket = 1000000 + int(time.time() * 1000) % 100000

                # Success
                self._orders_emitted_total += 1
                elapsed_ms = (time.perf_counter() - start) * 1000
                return {
                    "emitted": True,
                    "ticket": ticket,
                    "error": None,
                    "retry_count": attempt,
                    "latency_ms": round(elapsed_ms, 3),
                }

            except Exception as exc:
                last_error = str(exc)
                retries = attempt + 1
                continue

        # All attempts exhausted
        self._orders_failed_total += 1
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "emitted": False,
            "ticket": None,
            "error": last_error or "unknown error",
            "retry_count": retries,
            "latency_ms": round(elapsed_ms, 3),
        }

    # ------------------------------------------------------------------
    # Tick subscription
    # ------------------------------------------------------------------

    def subscribe_ticks(self, symbols: list) -> dict:
        """Subscribe to a real-time tick stream for the given symbols.

        Parameters
        ----------
        symbols : list
            List of symbol strings (e.g. ["EURUSD", "GBPUSD"]).

        Returns
        -------
        dict
            Result with keys: subscribed, symbols, failed_symbols.
        """
        if not symbols:
            return {"subscribed": False, "symbols": [], "failed_symbols": []}

        succeeded = []
        failed = []

        for symbol in symbols:
            try:
                if self._mt5_connector is not None:
                    ok = self._mt5_connector.symbol_select(symbol, True)
                else:
                    ok = True

                if ok:
                    succeeded.append(symbol)
                    if symbol not in self._symbols_subscribed:
                        self._symbols_subscribed.append(symbol)
                else:
                    failed.append(symbol)
            except Exception:
                failed.append(symbol)

        subscribed = len(failed) == 0 and len(succeeded) > 0
        return {
            "subscribed": subscribed,
            "symbols": succeeded,
            "failed_symbols": failed,
        }

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def close_position(self, ticket: int) -> bool:
        """Close an MT5 position.

        Parameters
        ----------
        ticket : int
            The ticket ID of the position to close.

        Returns
        -------
        bool
            True if successfully closed.
        """
        try:
            if self._mt5_connector is not None:
                return self._mt5_connector.close_order(ticket)
            else:
                return True
        except Exception:
            return False

    def get_status(self) -> dict:
        """Return runtime status snapshot."""
        uptime = 0.0
        if self._connected and self._start_time is not None:
            uptime = time.time() - self._start_time

        return {
            "connected": self._connected,
            "symbols_subscribed": list(self._symbols_subscribed),
            "last_tick_time": self._last_tick_time,
            "orders_emitted_total": self._orders_emitted_total,
            "orders_failed_total": self._orders_failed_total,
            "uptime_seconds": round(uptime, 3),
        }
