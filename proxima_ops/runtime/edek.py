"""EDEK — Event-Driven Execution Kernel.

Replaces cycle loop with on_tick/event listener approach.
Execution is triggered only by live market events.
"""

import time
from typing import Optional


class EventDrivenExecutionKernel:
    """Event-Driven Execution Kernel.

    Triggered solely by live market ticks.  The kernel evaluates every
    *tick_interval_ms* at most, returning immediately (non-blocking) on
    every call.

    trigger_state values
    --------------------
    "IDLE"
        Waiting for ticks.
    "EVALUATING"
        Tick triggered, decision evaluation in progress.
    "COMMITTED"
        SES committed, waiting for execution window.
    "EXECUTING"
        EFK executing order.
    """

    def __init__(
        self,
        mt5_connector: Optional[object] = None,
        tick_interval_ms: int = 100,
    ) -> None:
        self._mt5_connector = mt5_connector
        self._tick_interval_ms = max(1, int(tick_interval_ms))
        self._running: bool = False
        self._trigger_state: str = "IDLE"
        self._last_evaluation_time: float = 0.0
        self._last_event_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_tick(self, symbol: str, tick: dict) -> dict:
        """Process a single tick event (non-blocking).

        Parameters
        ----------
        symbol : str
            Market symbol (e.g. "EURUSD").
        tick : dict
            Tick data dict with at least ``"bid"`` and ``"ask"`` keys.

        Returns
        -------
        dict
            Result summary (see class docstring for schema).
        """
        now = time.time()
        latency = (now - self._last_event_time) * 1000.0 if self._last_event_time > 0 else 0.0
        self._last_event_time = now

        result: dict = {
            "event_processed": False,
            "trigger_state": self._trigger_state,
            "symbol": symbol,
            "bid": 0.0,
            "ask": 0.0,
            "triggered_evaluation": False,
            "latency_since_last_event_ms": latency,
        }

        try:
            bid = float(tick.get("bid", 0.0))
            ask = float(tick.get("ask", 0.0))
            result["bid"] = bid
            result["ask"] = ask

            elapsed = (now - self._last_evaluation_time) * 1000.0

            if self._running and elapsed >= self._tick_interval_ms:
                self._trigger_state = "EVALUATING"
                result["triggered_evaluation"] = True
                result["trigger_state"] = self._trigger_state
                # --- evaluation placeholder ---
                # Future hook: call SES / strategy evaluator here
                # -----------------------------------------------
                self._last_evaluation_time = now
                result["event_processed"] = True
            elif self._running:
                result["event_processed"] = True

        except Exception:
            # Swallow exceptions — the kernel must never crash on a tick.
            result["event_processed"] = False

        return result

    def start(self) -> None:
        """Start the kernel (allow evaluation on incoming ticks)."""
        self._running = True
        self._trigger_state = "IDLE"

    def stop(self) -> None:
        """Stop the kernel (ignore further ticks)."""
        self._running = False
        self._trigger_state = "IDLE"

    def is_running(self) -> bool:
        """Return ``True`` if the kernel is currently active."""
        return self._running

    def reset_trigger_state(self) -> None:
        """Reset trigger state back to IDLE after evaluation completes."""
        if self._running:
            self._trigger_state = "IDLE"

