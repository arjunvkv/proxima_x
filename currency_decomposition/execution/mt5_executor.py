"""MT5 execution — all terminal calls routed through MT5Adapter single-owner thread."""
import time
from typing import Optional
import MetaTrader5 as mt5
from config.settings import LOT_SIZE, MAX_TOTAL_LOTS, STOP_LOSS_PIPS, TAKE_PROFIT_PIPS
from data.models import DirectionHypothesis, ExecutionResult, PaperPosition

CD_MAGIC = 236000
STRATEGY_NAME = "CD_v1"


class MT5Executor:
    def __init__(self, mt5_adapter):
        self.mt5 = mt5_adapter
        self.positions: list[PaperPosition] = []
        self._position_meta: dict[int, dict] = {}
        self.sync_failed = False
        self.execution_ledger: list[dict] = []
        self._ledger_maxlen = 200

    def _resolve_symbol(self, symbol: str) -> str:
        if self.mt5.call_mt5(mt5.symbol_info, symbol) is not None:
            return symbol
        suffixes = ["", ".", "m", "ecn", ".m"]
        for sfx in suffixes:
            candidate = f"{symbol}{sfx}"
            if self.mt5.call_mt5(mt5.symbol_info, candidate) is not None:
                import sys
                print(f"[SYMBOL RESOLVE] {symbol} -> {candidate}", file=sys.stderr)
                return candidate
        return symbol

    def position_count(self) -> int:
        if not self.positions:
            self.sync()
        return len(self.positions)

    def sync(self) -> None:
        try:
            positions = self.mt5.call_mt5(mt5.positions_get)
            if positions is None:
                self.sync_failed = True
                import sys
                print("[SYNC] MT5 positions unavailable", file=sys.stderr)
                return

            self.sync_failed = False
            cd_positions = [p for p in positions if CD_MAGIC <= p.magic < CD_MAGIC + 200]
            new_list = []
            for p in cd_positions:
                direction = "BUY" if p.type == 0 else "SELL"
                meta = self._position_meta.get(p.ticket, {})
                pos = PaperPosition(
                    id=str(p.ticket),
                    symbol=p.symbol,
                    direction=direction,
                    entry_price=p.price_open,
                    current_price=p.price_current,
                    entry_time=p.time,
                    lots=p.volume,
                    stop_loss=p.sl or 0.0,
                    take_profit=p.tp or 0.0,
                    drs_entry=meta.get("drs_entry", 0.0),
                    currency_strengths_entry=meta.get("currency_strengths", {}),
                    pnl=p.profit + (getattr(p, 'swap', 0) or 0) + (getattr(p, 'commission', 0) or 0),
                )
                new_list.append(pos)
            if len(self.positions) != len(new_list):
                import sys
                print(f"[SYNC] {len(positions)} MT5 total -> {len(cd_positions)} CD -> {len(new_list)} synced", file=sys.stderr)
                for p in new_list:
                    print(f"[SYNC]   {p.symbol} {p.direction} pnl={p.pnl:.2f}", file=sys.stderr)
            self.positions = new_list
        except Exception as exc:
            self.sync_failed = True
            import sys
            print(f"[SYNC] ERROR: {exc}", file=sys.stderr)

    def total_pnl(self) -> float:
        return sum(p.pnl or 0 for p in self.positions)

    def total_lots(self) -> float:
        return sum(p.lots or 0 for p in self.positions)

    def update_prices(self, ticks) -> None:
        tick_map = {t.symbol: t for t in ticks}
        for p in self.positions:
            tick = tick_map.get(p.symbol)
            if tick:
                p.current_price = tick.bid if p.direction == "BUY" else tick.ask
        mt5_positions = self.mt5.call_mt5(mt5.positions_get)
        if mt5_positions is None:
            return
        mt5_by_ticket = {str(pt.ticket): pt for pt in mt5_positions if CD_MAGIC <= pt.magic < CD_MAGIC + 200}
        for p in self.positions:
            mp = mt5_by_ticket.get(p.id)
            if mp:
                p.pnl = mp.profit + (getattr(mp, 'swap', 0) or 0) + (getattr(mp, 'commission', 0) or 0)

    def execute(self, hypothesis: DirectionHypothesis, tick=None) -> ExecutionResult:
        symbol = self._resolve_symbol(hypothesis.symbol)
        if not self.mt5.call_mt5(mt5.symbol_select, symbol, True):
            import sys
            print(f"[SYMBOL SELECT FAILED] {symbol}", file=sys.stderr)
            return ExecutionResult(success=False, reason="SYMBOL_SELECT_FAILED")
        tick_info = self.mt5.call_mt5(mt5.symbol_info_tick, symbol)
        if tick_info is None:
            return ExecutionResult(success=False, reason="NO_TICK")
        direction_type = mt5.ORDER_TYPE_BUY if hypothesis.direction > 0 else mt5.ORDER_TYPE_SELL
        price = tick_info.ask if hypothesis.direction > 0 else tick_info.bid
        sym_info = self.mt5.call_mt5(mt5.symbol_info, symbol)
        if sym_info is None:
            return ExecutionResult(success=False, reason="NO_SYMBOL_INFO")
        point = sym_info.point
        sl_dist = STOP_LOSS_PIPS * 10 * point
        tp_dist = TAKE_PROFIT_PIPS * 10 * point
        if hypothesis.direction > 0:
            sl = price - sl_dist
            tp = price + tp_dist
        else:
            sl = price + sl_dist
            tp = price - tp_dist
        filling_modes = [mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN]
        last_error = None
        result = None
        used_filling = None
        current_lots = self.total_lots()
        if current_lots + LOT_SIZE > MAX_TOTAL_LOTS:
            return ExecutionResult(success=False, reason=f"EXCEEDS_MAX_TOTAL_LOTS ({current_lots}+{LOT_SIZE}>{MAX_TOTAL_LOTS})")
        for filling in filling_modes:
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": LOT_SIZE,
                "type": int(direction_type),
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 10,
                "magic": CD_MAGIC,
                "comment": STRATEGY_NAME,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": int(filling),
            }
            self._log_ledger("order_send", hypothesis.symbol, hypothesis.direction, ticket=None, filling=filling)
            result = self.mt5.call_mt5(mt5.order_send, request)
            if result is None:
                last_error = "ORDER_SEND_NONE"
                continue
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                used_filling = filling
                break
            last_error = f"MT5_{result.retcode}"
            self._log_ledger("order_fail", hypothesis.symbol, hypothesis.direction, reason=last_error, filling=filling)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            return ExecutionResult(success=False, reason=last_error or "ORDER_SEND_NONE")
        self._position_meta[result.order] = {
            "drs_entry": hypothesis.drs_score,
            "currency_strengths": {
                "base": hypothesis.base_strength,
                "quote": hypothesis.quote_strength
            },
            "confidence": hypothesis.confidence,
        }
        self.sync()
        sync_ok = any(str(result.order) == p.id for p in self.positions)
        self._log_ledger("sync_confirm" if sync_ok else "sync_missing",
                         hypothesis.symbol, hypothesis.direction,
                         ticket=result.order, sync_confirmed=sync_ok)
        return ExecutionResult(success=True, position_id=str(result.order), price=price)

    def close_position(self, position_id: str, exit_price: float, reason: str = "") -> ExecutionResult:
        ticket = int(position_id)
        positions = self.mt5.call_mt5(mt5.positions_get, ticket=ticket)
        if positions is None or len(positions) == 0:
            return ExecutionResult(success=False, reason="NOT_FOUND")
        pos = positions[0]
        close_type = mt5.ORDER_TYPE_BUY if pos.type == 1 else mt5.ORDER_TYPE_SELL
        tick_info = self.mt5.call_mt5(mt5.symbol_info_tick, pos.symbol)
        if tick_info is None:
            return ExecutionResult(success=False, reason="NO_TICK")
        close_price = tick_info.bid if close_type == mt5.ORDER_TYPE_BUY else tick_info.ask
        close_comment = f"CD_close_{reason}" if reason else "CD_close"
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": int(close_type),
            "position": ticket,
            "price": close_price,
            "deviation": 10,
            "magic": CD_MAGIC + 100,
            "comment": close_comment,
            "type_time": mt5.ORDER_TIME_GTC,
        }
        last_error = None
        result = None
        for fill in [None, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN]:
            if fill is not None:
                request["type_filling"] = int(fill)
            elif "type_filling" in request:
                del request["type_filling"]
            fill_label = "DEFAULT" if fill is None else f"FILL_{fill}"
            self._log_ledger("close_send", pos.symbol, "BUY" if close_type == mt5.ORDER_TYPE_BUY else "SELL", ticket=ticket, filling=fill_label)
            result = self.mt5.call_mt5(mt5.order_send, request)
            if result is None:
                last_error = "CLOSE_NONE"
                continue
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                break
            last_error = f"CLOSE_{result.retcode}"
            self._log_ledger("close_fail", pos.symbol, "", ticket=ticket, reason=last_error, filling=fill_label)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            self._log_ledger("close_fail", pos.symbol, "", ticket=ticket, reason=last_error or "CLOSE_NONE")
            return ExecutionResult(success=False, reason=last_error or "CLOSE_NONE")
        self._position_meta.pop(ticket, None)
        self.sync()
        close_confirmed = not any(str(ticket) == p.id for p in self.positions)
        self._log_ledger("close_confirm" if close_confirmed else "close_still_open",
                         pos.symbol, "", ticket=ticket, close_confirmed=close_confirmed)
        return ExecutionResult(success=True, position_id=position_id, price=close_price, reason=reason)

    def close_all(self, prices: dict[str, float], reason: str = "MANUAL") -> list[ExecutionResult]:
        results = []
        for p in list(self.positions):
            r = self.close_position(p.id, 0.0, reason)
            results.append(r)
        return results

    def recent_failures(self, n: int = 3) -> list[dict]:
        return [e for e in self.execution_ledger[-20:] if "fail" in e.get("event", "")][-n:]

    def _log_ledger(self, event: str, symbol: str, direction, ticket=None, **extra) -> None:
        self.execution_ledger.append({
            "event": event,
            "symbol": symbol,
            "direction": direction,
            "ticket": ticket,
            "time": time.time(),
            **extra,
        })
        if len(self.execution_ledger) > self._ledger_maxlen:
            self.execution_ledger = self.execution_ledger[-self._ledger_maxlen:]

    def positions_summary(self) -> list[dict]:
        self.sync()
        return [
            {
                "id": p.id, "symbol": p.symbol, "direction": p.direction,
                "entry": p.entry_price, "current": p.current_price,
                "pnl": p.pnl or 0.0,
                "age_s": max(0, time.time() - p.entry_time),
            }
            for p in self.positions
        ]
