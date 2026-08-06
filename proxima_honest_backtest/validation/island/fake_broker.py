"""FakeBroker — deterministic broker double for the offline island.

Emits the SAME event schema as RealDemoExecutor (ORDER_SENT / BROKER_FILL /
BROKER_REJECT / POSITION_SYNC) through the shared EventEmitter, so the entire
validator pipeline (LiveRunner -> stream.jsonl -> ReconMonitor -> signoff) is
proven offline before any MT5 connection. The FTMO phase swaps ONLY this seam.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from proxima_honest_backtest.engine.types import Trade
from proxima_honest_backtest.live.executor import ExecutionReport


class FakeBroker:
    """Behavior-parameterizable broker double. Deterministic when seeded.

    scenarios (per symbol, optional):
        "instant"     -> fill at requested price (default)
        "requote"     -> reject once (REQUOTE) then fill
        "reject"      -> reject forever (INVALID_VOLUME)
        "silent"      -> never ack (triggers UNKNOWN in the state machine)
        "late_fill"   -> fill but at requested_price + offset_pips (slippage probe)
        "delay_s"     -> simulate broker ack delay in seconds (latency probe)
    """

    def __init__(self, emitter=None, seed: int = 0, scenarios: Optional[Dict[str, Any]] = None,
                 requote_attempts: int = 1) -> None:
        self.emitter = emitter
        self._rng = random.Random(seed)
        self.scenarios = scenarios or {}
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.acks: List[Dict[str, Any]] = []
        self._requote_left = requote_attempts
        self._run_id = getattr(emitter, "run_id", "fake")
        self.run_id = self._run_id
        self._strategy = getattr(emitter, "strategy", "unknown")

    def _scenario(self, symbol: str) -> str:
        s = self.scenarios.get(symbol)
        if isinstance(s, str):
            return s
        if isinstance(s, dict):
            return s.get("mode", "instant")
        return "instant"

    def _emit(self, event_type: str, symbol: str, decision_id: str, **payload) -> None:
        if self.emitter is not None:
            self.emitter.emit(event_type, symbol=symbol, decision_id=decision_id, **payload)

    # ------------------------------------------------------------------
    def execute_order(self, side: str, quantity: float, symbol: str, price: float,
                      volatility: float, hour_utc: int, timestamp=None,
                      decision_id: Optional[str] = None) -> ExecutionReport:
        scenario = self._scenario(symbol)
        spec = self.scenarios.get(symbol, {})
        delay = spec.get("delay_s", 0.0) if isinstance(spec, dict) else 0.0
        dec_id = decision_id or f"{self._run_id}|{symbol}|{timestamp or 0}"
        self._emit("ORDER_SENT", symbol, dec_id, send_time_utc=str(timestamp),
                   requested_price=price, quantity=quantity)

        if scenario == "silent":
            # never respond — the state machine will mark UNKNOWN
            return ExecutionReport(None, None, False, "timeout")

        if scenario == "reject":
            self._emit("BROKER_REJECT", symbol, dec_id, reject_reason="INVALID_VOLUME", broker_code="10014")
            return ExecutionReport(None, None, False, "INVALID_VOLUME")

        if scenario == "requote":
            if self._requote_left > 0:
                self._requote_left -= 1
                self._emit("BROKER_REJECT", symbol, dec_id, reject_reason="REQUOTE", broker_code="10004")
                return ExecutionReport(None, None, False, "REQUOTE")
            # retry (after requote consumed) fills
            self._emit("ORDER_SENT", symbol, dec_id, send_time_utc=str(timestamp),
                       requested_price=price, quantity=quantity)

        if scenario == "late_fill":
            offset = spec.get("offset_pips", 2.0) if isinstance(spec, dict) else 2.0
            pip = 0.01 if "JPY" in symbol else 0.0001
            fill = price + (offset * pip if side.upper() == "BUY" else -offset * pip)
        else:
            fill = price

        qty = quantity
        self.positions[symbol] = {"side": side, "entry": fill, "quantity": qty}
        self.acks.append({"symbol": symbol, "decision_id": dec_id, "fill": fill})
        self._emit("BROKER_FILL", symbol, dec_id,
                   broker_ticket=f"{symbol}-{len(self.acks)}", fill_price=fill,
                   fill_time_utc=str(timestamp), filled_quantity=qty,
                   slippage_pips=round(abs(fill - price) * (100 if "JPY" in symbol else 10000), 3))
        trade = Trade(timestamp=timestamp, symbol=symbol, side=side, quantity=qty, price=fill)
        return ExecutionReport(trade, fill, True, broker_profile="fake")

    def close_position(self, symbol: str, price: float, timestamp=None,
                       decision_id: Optional[str] = None) -> ExecutionReport:
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return ExecutionReport(None, None, False, "no_position")
        dec_id = decision_id or f"{self._run_id}|{symbol}|{timestamp or 0}|X"
        qty = pos["quantity"]
        self.acks.append({"symbol": symbol, "decision_id": dec_id, "fill": price})
        self._emit("ORDER_SENT", symbol, dec_id, send_time_utc=str(timestamp),
                   requested_price=price, quantity=qty)
        self._emit("BROKER_FILL", symbol, dec_id, broker_ticket=f"X{len(self.acks)}",
                   fill_price=price, fill_time_utc=str(timestamp),
                   filled_quantity=qty, slippage_pips=0.0)
        return ExecutionReport(
            Trade(timestamp=timestamp, symbol=symbol, side="SELL" if pos["side"] == "BUY" else "BUY",
                  quantity=pos["quantity"], price=price),
            price, True, broker_profile="fake")

    def positions_get(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.positions)

    def reset(self) -> None:
        self.positions.clear()
        self.acks.clear()