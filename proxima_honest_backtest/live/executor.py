"""Live / paper executor — ExecutionSimulator-compatible interface over MT5.

- paper mode: deterministic fill at requested +/- half-spread (NO RNG) so
  replay parity is reproducible.
- live mode: real mt5.order_send with per-pair magic (MAGIC_BASE + pair_index),
  SL/TP crash guards, position recovery from the account on startup.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from proxima_honest_backtest.engine.types import Trade
from proxima_honest_backtest.live.events.emitter import EventEmitter
from proxima_honest_backtest.live.events.schema import normalize_side
from proxima_honest_backtest.live.execution_state import ExecState, ExecutionStateMachine


@dataclass
class ExecutionReport:
    trade: Optional[Trade]
    fill_price: Optional[float]
    filled: bool
    reject_reason: Optional[str] = None
    broker_profile: str = "live"


class LiveExecutor:
    """Executes validated strategy decisions against MT5 (paper or live)."""

    def __init__(
        self,
        pairs: List[str],
        magic_base: int = 400000,
        base_lot: float = 0.15,
        mode: str = "paper",          # "paper" | "live"
        mt5: Optional[Any] = None,    # required for live mode
        hard_sl_pips: float = 50.0,
        hard_tp_pips: float = 0.0,
        slippage_pips: float = 1.0,
        spread_model_half: Optional[float] = None,  # pips; paper default = 0.5
        emitter: Optional[EventEmitter] = None,
        timeout_sec: float = 5.0,
    ) -> None:
        self.pairs = pairs
        self.magic_base = magic_base
        self.base_lot = base_lot
        self.mode = mode
        self.mt5 = mt5
        self.hard_sl = hard_sl_pips
        self.hard_tp = hard_tp_pips
        self.slippage_pips = slippage_pips
        self.spread_half = spread_model_half if spread_model_half is not None else 0.5
        self.positions: Dict[str, Dict[str, Any]] = {}
        self._pair_idx = {p: i for i, p in enumerate(pairs)}
        self._ticket_seq = 0
        self._seen_tickets = set()
        self.state_machine = ExecutionStateMachine(timeout_sec=timeout_sec)
        self.emitter = emitter
        self.run_id = getattr(emitter, "run_id", "local")
        self._strategy = getattr(emitter, "strategy", "unknown") if emitter else "unknown"

    # ------------------------------------------------------------------
    # Position recovery — MUST run before the runner starts
    # ------------------------------------------------------------------
    def recover_positions(self) -> Dict[str, Dict[str, Any]]:
        """Rebuild in-memory positions from MT5 account state (by magic range).

        Also rebuilds the ExecutionStateMachine so one-in-flight gates stay
        consistent across restarts (a broker-open position == OPEN state).
        """
        self.positions = {}
        self.state_machine.reset()
        if self.mode != "live" or self.mt5 is None:
            return self.positions
        try:
            opened = self.mt5.positions_get() or []
            for pos in opened:
                magic = int(getattr(pos, "magic", 0))
                if self.magic_base <= magic < self.magic_base + len(self.pairs):
                    sym = pos.symbol
                    side = "LONG" if pos.type == 0 else "SHORT"
                    qty = float(pos.volume)
                    self.positions[sym] = {
                        "side": side,
                        "entry_price": float(pos.price_open),
                        "quantity": qty,
                        "ticket": int(pos.ticket),
                    }
                    self._seen_tickets.add(int(pos.ticket))
                    self.state_machine.mark_sent_enter(
                        sym, f"recover|{sym}|{pos.ticket}", qty, normalize_side(side), str(pos.ticket))
                    self.state_machine.mark_fill(sym, str(pos.ticket), qty)
        except Exception:
            pass
        return self.positions

    def magic_for(self, symbol: str) -> int:
        return self.magic_base + self._pair_idx.get(symbol, 0)

    def symbol_lot_size(self, symbol: str, contract_size: Optional[float] = None) -> float:
        """MT5 contract size for a symbol (paper fallback = 100000)."""
        if self.mt5 is not None:
            try:
                info = self.mt5.symbol_info(symbol)
                cs = getattr(info, "trade_contract_size", None)
                if cs:
                    return float(cs)
            except Exception:
                pass
        return float(contract_size if contract_size is not None else 100000.0)

    def position_size(self, symbol: str, backtest_units: float = 10000.0,
                      contract_size: Optional[float] = None) -> float:
        """Convert backtest notional units into an MT5 lot volume.

        apples-to-apples: 1 backtest trade == 1 MT5 position of equal exposure.
        Live uses real symbol contract; paper falls back to 100000 default.
        """
        cs = self.symbol_lot_size(symbol, contract_size)
        lots = backtest_units / cs
        vols = self._volume_bounds(symbol)
        if vols is not None:
            vmin, vmax, step = vols
            lots = min(max(lots, vmin), vmax)
            lots = round(lots / step) * step
        return float(lots)

    def _volume_bounds(self, symbol: str):
        if self.mt5 is None:
            return None
        try:
            info = self.mt5.symbol_info(symbol)
            if info is None:
                return None
            vmin = getattr(info, "volume_min", None)
            vmax = getattr(info, "volume_max", None)
            step = getattr(info, "volume_step", None)
            if vmin is None or vmax is None or step is None:
                return None
            return (float(vmin), float(vmax), float(step))
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Event helpers (only when an emitter is attached)
    # ------------------------------------------------------------------
    def _emit(self, event_type: str, symbol: Optional[str], decision_id: Optional[str], **payload) -> None:
        if self.emitter is not None:
            self.emitter.emit(event_type, symbol=symbol, decision_id=decision_id, **payload)

    def emit_position_sync(self, recon_status: str, broker_positions: Optional[Dict] = None) -> None:
        self._emit(
            "POSITION_SYNC", None, None,
            engine_positions=self.positions,
            broker_positions=broker_positions or {},
            reconciliation_status=recon_status,
        )

    def engine_open_symbols(self) -> List[str]:
        return [sym for sym, e in self.state_machine._entries.items() if e.state == ExecState.OPEN]

    def pending_symbols(self) -> List[str]:
        return [sym for sym, e in self.state_machine._entries.items() if e.state in
                (ExecState.ENTRY_PENDING, ExecState.EXIT_PENDING)]

    # ------------------------------------------------------------------
    # Interface (matches ExecutionSimulator.execute_order)
    # ------------------------------------------------------------------
    def execute_order(
        self,
        side: str,
        quantity: float,
        symbol: str,
        price: float,
        volatility: float,
        hour_utc: int,
        timestamp: Optional[datetime] = None,
        decision_id: Optional[str] = None,
    ) -> ExecutionReport:
        qty = quantity or self.base_lot
        norm_side = normalize_side(side)
        buy = norm_side == "L"
        # ---- live volume contract: must be a valid lot volume ----
        if self.mode == "live" and self.mt5 is not None:
            from proxima_honest_backtest.live.sizing import validate_volume
            try:
                info = self.mt5.symbol_info(symbol)
            except Exception:
                info = None
            if info is not None:
                ok, reason = validate_volume(info, qty, symbol)
                if not ok:
                    return ExecutionReport(None, None, False, f"invalid_volume:{reason}")
        # ---- one-in-flight ENTRY gate ----
        if not self.state_machine.can_enter(symbol):
            return ExecutionReport(None, None, False, "one_in_flight:entry")
        if decision_id is None:
            decision_id = f"{self.run_id}|{symbol}|{timestamp or datetime.utcnow()}|E"
        self.state_machine.mark_sent_enter(symbol, decision_id, qty, norm_side, None)
        self._emit("ORDER_SENT", symbol, decision_id,
                   send_time_utc=str(timestamp or datetime.utcnow()),
                   requested_price=price, quantity=qty)
        try:
            if self.mode == "live" and self.mt5 is not None:
                report = self._live_order(side, qty, symbol, price)
            else:
                report = self._paper_order(side, qty, symbol, price, timestamp)
        except Exception as exc:
            report = ExecutionReport(None, None, False, f"exec_error:{exc}")
        # ---- apply broker outcome to state machine ----
        if report.filled:
            self.state_machine.mark_fill(symbol, getattr(report.trade, "price", None), qty)
            self._ticket_seq += 1
            self._emit("BROKER_FILL", symbol, decision_id,
                       broker_ticket=f"E{self._ticket_seq}",
                       fill_price=report.fill_price, fill_time_utc=str(datetime.utcnow()),
                       filled_quantity=qty,
                       slippage_pips=round(abs(float(report.fill_price or price) - price) * self._pip_scale(symbol), 3))
        else:
            self.state_machine.mark_reject(symbol, report.reject_reason or "unknown")
            self._emit("BROKER_REJECT", symbol, decision_id,
                       reject_reason=report.reject_reason or "unknown", broker_code="")
        return report

    @staticmethod
    def _pip_scale(symbol: str) -> float:
        return 100.0 if "JPY" in symbol else 10000.0

    def _paper_order(
        self, side: str, quantity: float, symbol: str,
        price: float, timestamp: Optional[datetime] = None,
    ) -> ExecutionReport:
        slip = self._pip_to_price(symbol, self.spread_half)
        buy = normalize_side(side) == "L"
        fill = price + slip if buy else price - slip
        trade = Trade(
            timestamp=timestamp or datetime.utcnow(),
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=fill,
        )
        self.positions[symbol] = {
            "side": side,
            "entry_price": fill,
            "quantity": quantity,
        }
        return ExecutionReport(
            trade=trade, fill_price=fill, filled=True, broker_profile="paper",
        )

    def _live_order(self, side: str, quantity: float, symbol: str, price: float) -> ExecutionReport:
        if self.mt5 is None:
            return ExecutionReport(None, None, False, "no_mt5")
        buy = normalize_side(side) == "L"
        digits = int(self.mt5.symbol_info(symbol).digits)
        order_type = self.mt5.ORDER_TYPE_BUY if buy else self.mt5.ORDER_TYPE_SELL
        sl = None
        tp = None
        if self.hard_sl > 0 or self.hard_tp > 0:
            pip = 0.01 if "JPY" in symbol else 0.0001
            if self.hard_sl > 0:
                sl = price - (self.hard_sl * pip) if buy else price + (self.hard_sl * pip)
            if self.hard_tp > 0:
                tp = price + (self.hard_tp * pip) if buy else price - (self.hard_tp * pip)
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": quantity,
            "type": order_type,
            "price": price,
            "sl": round(float(sl), digits) if sl is not None else 0.0,
            "tp": round(float(tp), digits) if tp is not None else 0.0,
            "deviation": int(self.slippage_pips * 10),
            "magic": self.magic_for(symbol),
            "comment": "proxima_live_v1",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }
        try:
            result = self.mt5.order_send(request)
        except Exception as exc:
            return ExecutionReport(None, None, False, f"order_send_error:{exc}")
        if result is None or getattr(result, "retcode", -1) != self.mt5.TRADE_RETCODE_DONE:
            code = getattr(result, "retcode", -1)
            return ExecutionReport(None, None, False, f"retcode:{code}")
        fill = float(getattr(result, "price", price) or price)
        trade = Trade(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=fill,
        )
        ticket = None
        try:
            for pos in (self.mt5.positions_get(symbol=symbol) or []):
                if pos.magic == self.magic_for(symbol) and pos.ticket not in self._seen_tickets:
                    ticket = int(pos.ticket)
                    self._seen_tickets.add(ticket)
                    break
        except Exception:
            ticket = None
        self.positions[symbol] = {
            "side": side, "entry_price": fill, "quantity": quantity,
            "ticket": ticket or int(getattr(result, "order", 0)),
        }
        return ExecutionReport(trade, fill, True, broker_profile="live")

    # ------------------------------------------------------------------
    # Exits
    # ------------------------------------------------------------------
    def close_position(
        self, symbol: str, price: float, timestamp: Optional[datetime] = None,
        decision_id: Optional[str] = None,
    ) -> ExecutionReport:
        pos = self.positions.get(symbol)
        if pos is None:
            return ExecutionReport(None, None, False, "no_position")
        # ---- one-in-flight EXIT gate ----
        if not self.state_machine.can_exit(symbol):
            return ExecutionReport(None, None, False, "one_in_flight:exit")
        qty = pos["quantity"]
        norm_side = "S" if pos["side"].upper() in ("LONG", "BUY", "L") else "L"
        if decision_id is None:
            decision_id = f"{self.run_id}|{symbol}|{timestamp or datetime.utcnow()}|X"
        self.state_machine.mark_sent_exit(symbol, qty, norm_side)
        self._emit("ORDER_SENT", symbol, decision_id, send_time_utc=str(timestamp or datetime.utcnow()),
                   requested_price=price, quantity=qty)
        if self.mode == "live" and self.mt5 is not None:
            ticket = pos.get("ticket")
            if ticket:
                res = self._live_close(ticket, symbol, qty, pos["side"], price)
            else:
                res = ExecutionReport(None, None, False, "no_ticket")
            if not res.filled:
                # Close rejected: keep position OPEN for recovery, emit REJECT.
                self.state_machine.mark_exit_reject(symbol, res.reject_reason or "close_failed")
                self._emit("BROKER_REJECT", symbol, decision_id,
                           reject_reason=res.reject_reason or "close_failed", broker_code="")
                return res
            self._ticket_seq += 1
            self.positions.pop(symbol, None)
            self.state_machine.mark_closed(symbol)
            self._emit("BROKER_FILL", symbol, decision_id,
                       broker_ticket=f"X{self._ticket_seq}", fill_price=res.fill_price,
                       fill_time_utc=str(timestamp or datetime.utcnow()),
                       filled_quantity=qty, slippage_pips=0.0)
            return ExecutionReport(
                Trade(timestamp=timestamp or datetime.utcnow(), symbol=symbol,
                      side="SELL" if pos["side"].upper() in ("LONG", "BUY", "L") else "BUY",
                      quantity=qty, price=res.fill_price or price),
                res.fill_price or price, True, broker_profile=self.mode)
        self.positions.pop(symbol, None)
        self.state_machine.mark_closed(symbol)
        self._ticket_seq += 1
        close_side = "SELL" if pos["side"].upper() in ("LONG", "BUY", "L") else "BUY"
        trade = Trade(
            timestamp=timestamp or datetime.utcnow(),
            symbol=symbol,
            side=close_side,
            quantity=pos["quantity"],
            price=price,
        )
        self._emit("BROKER_FILL", symbol, decision_id,
                   broker_ticket=f"X{self._ticket_seq}", fill_price=price, fill_time_utc=str(timestamp or datetime.utcnow()),
                   filled_quantity=qty, slippage_pips=0.0)
        return ExecutionReport(trade, price, True, broker_profile=self.mode)

    # ------------------------------------------------------------------
    # PnL helper
    # ------------------------------------------------------------------
    def _live_close(self, ticket: int, symbol: str, qty: float,
                    side: str, price: float) -> ExecutionReport:
        """Close an open MT5 position by ticket via order_send DEAL."""
        if self.mt5 is None:
            return ExecutionReport(None, None, False, "no_mt5")
        close_side = self.mt5.ORDER_TYPE_SELL if side.upper() in ("LONG", "BUY", "L") \
            else self.mt5.ORDER_TYPE_BUY
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "position": int(ticket),
            "symbol": symbol,
            "volume": float(qty),
            "type": close_side,
            "price": float(price),
            "deviation": int(self.slippage_pips * 10),
            "magic": self.magic_for(symbol),
            "comment": "proxima_live_v1",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }
        try:
            result = self.mt5.order_send(request)
        except Exception as exc:
            return ExecutionReport(None, None, False, f"order_send_error:{exc}")
        if result is None or getattr(result, "retcode", -1) != self.mt5.TRADE_RETCODE_DONE:
            code = getattr(result, "retcode", -1)
            return ExecutionReport(None, None, False, f"retcode:{code}")
        fill = float(getattr(result, "price", price) or price)
        trade = Trade(
            timestamp=datetime.utcnow(), symbol=symbol, side=close_side,
            quantity=qty, price=fill,
        )
        return ExecutionReport(trade, fill, True, broker_profile="live")

    def calculate_pnl(
        self, entry_price: float, exit_price: float, quantity: float,
        side: str, symbol: str,
    ) -> float:
        buy = normalize_side(side) == "L"
        raw = (exit_price - entry_price) * quantity if buy else (entry_price - exit_price) * quantity
        return raw  # account-currency; keep sign for bookkeeping

    @staticmethod
    def _pip_to_price(symbol: str, pips: float) -> float:
        pip = 0.01 if "JPY" in symbol else 0.0001
        return pips * pip
