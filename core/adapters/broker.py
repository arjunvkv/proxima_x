import logging
from abc import ABC, abstractmethod
from typing import Optional

from data.execution_cost import ExecutionCost
from core.execution.execution_event import ExecutionEvent

logger = logging.getLogger("proxima.adapters.broker")


class Broker(ABC):

    @abstractmethod
    def get_tick(self, symbol: str) -> Optional[dict]:
        pass

    @abstractmethod
    def get_rates(self, symbol: str, count: int = 100, timeframe: int = 0) -> Optional[list]:
        pass

    @abstractmethod
    def get_account(self) -> Optional[dict]:
        pass

    @abstractmethod
    def get_positions(self) -> list[dict]:
        pass

    @abstractmethod
    def place_order(self, symbol: str, order_type: str, volume: float,
                    price: float, sl: float = 0.0, tp: float = 0.0,
                    comment: str = "PROXIMA") -> Optional[dict]:
        pass

    @abstractmethod
    def close_order(self, ticket: int) -> bool:
        pass

    @abstractmethod
    def close_all(self) -> list[dict]:
        pass

    @abstractmethod
    def verify_spread(self, symbol: str, es_rank: float = None) -> bool:
        pass

    @abstractmethod
    def verify_symbol(self, symbol: str) -> dict:
        pass

    @abstractmethod
    def _get_broker_symbol(self, symbol: str) -> str:
        pass


class MT5Broker(Broker):
    def __init__(self, mt5_connector):
        self._mt5 = mt5_connector

    def get_tick(self, symbol: str) -> Optional[dict]:
        return self._mt5.get_tick(symbol)

    def get_rates(self, symbol: str, count: int = 100, timeframe: int = 0) -> Optional[list]:
        return self._mt5.get_rates(symbol, count=count, timeframe=timeframe)

    def get_account(self) -> Optional[dict]:
        return self._mt5.get_account()

    def get_positions(self) -> list[dict]:
        return self._mt5.get_positions()

    def place_order(self, symbol: str, order_type: str, volume: float,
                    price: float, sl: float = 0.0, tp: float = 0.0,
                    comment: str = "PROXIMA") -> Optional[dict]:
        return self._mt5.place_order(symbol, order_type, volume, price, sl=sl, tp=tp, comment=comment)

    def close_order(self, ticket: int) -> bool:
        return self._mt5.close_order(ticket)

    def close_all(self) -> list[dict]:
        return self._mt5.close_all()

    def verify_spread(self, symbol: str, es_rank: float = None) -> bool:
        return self._mt5.verify_spread(symbol, es_rank=es_rank)

    def verify_symbol(self, symbol: str) -> dict:
        return self._mt5.verify_symbol(symbol)

    def _get_broker_symbol(self, symbol: str) -> str:
        return self._mt5._get_broker_symbol(symbol)


class PaperBroker(Broker):
    def __init__(self, tick_source=None, clock=None, execution_model=None,
                     initial_balance: float = 100000.0, execution_cost=None,
                     tick_value_map: Optional[dict] = None,
                     execution_event_sink=None):
        self._tick_source = tick_source
        self._clock = clock
        self._execution = execution_model
        self._execution_cost = execution_cost or ExecutionCost()
        # tick_value_usd: symbol -> broker-authoritative USD value of ONE POINT
        # (the symbol's machine point, e.g. 0.001 for EURJPY) per 1.0 lot.
        # Mirrors live MT5 symbol_info.trade_tick_value. When present, per-trade
        # PnL is computed EXACTLY as live deals would — the backtest<->live
        # alignment contract. When absent (legacy), the old pip-based formula
        # is preserved for backward compatibility.
        self._tick_value_usd = dict(tick_value_map or {})
        self._ledger = None
        self._next_ticket = 1000000
        self._positions: dict[int, dict] = {}
        self._history: list[dict] = []
        self._balance = initial_balance
        self._equity = initial_balance
        self._closed_pnl = 0.0
        self._symbol_map = {
            "XAUUSD": "XAUUSD", "EURJPY": "EURJPY",
            "USDJPY": "USDJPY", "GBPJPY": "GBPJPY", "EURUSD": "EURUSD",
        }
        self._spreads: dict[str, int] = {}
        self._slippage_points: dict[str, float] = {}
        self._bar_buffers: dict[str, list[dict]] = {}
        self._tick_buffers_for_bars: dict[str, list[dict]] = {}
        # optional ExecutionEvent sink: when set, OPEN/CLOSE events are
        # emitted so a live-shadow comparer can diff paper vs MT5 fills 1:1.
        # None (default) = identical legacy behaviour.
        self._execution_event_sink = execution_event_sink

    def _get_broker_symbol(self, symbol: str) -> str:
        return self._symbol_map.get(symbol.upper(), symbol)

    def set_ledger(self, ledger):
        self._ledger = ledger

    def get_tick(self, symbol: str) -> Optional[dict]:
        if self._tick_source:
            return self._tick_source.get_tick(symbol)
        return None

    def _feed_tick_for_bars(self, symbol: str):
        tick = self.get_tick(symbol)
        if tick is None:
            return
        self._feed_tick_for_bars_manual(symbol, tick)

    def _feed_tick_for_bars_manual(self, symbol: str, tick: dict):
        ts = tick.get("time_sec", tick.get("timestamp", 0))
        if ts <= 0:
            return
        buf = self._tick_buffers_for_bars.setdefault(symbol, [])
        buf.append({"ts": ts, "bid": tick.get("bid", 0), "ask": tick.get("ask", 0)})
        if len(buf) > 100000:
            buf[:50000] = []

    def _build_h1_bars(self, symbol: str) -> list[dict]:
        buf = self._tick_buffers_for_bars.get(symbol, [])
        if not buf:
            return []
        bars: dict[int, dict] = {}
        for t in buf:
            ts = t["ts"]
            hour = int(ts // 3600) * 3600
            mid = (t["bid"] + t["ask"]) / 2.0
            if hour not in bars:
                bars[hour] = {"time": hour, "open": mid, "high": mid, "low": mid, "close": mid, "tick_volume": 0, "volume": 0}
            b = bars[hour]
            b["high"] = max(b["high"], mid)
            b["low"] = min(b["low"], mid)
            b["close"] = mid
            b["tick_volume"] += 1
            b["volume"] += 1
        result = sorted(bars.values(), key=lambda x: x["time"])
        return result

    def get_rates(self, symbol: str, count: int = 100, timeframe: int = 0) -> Optional[list]:
        bars = self._build_h1_bars(symbol)
        if not bars:
            return None
        return bars[-count:]

    def get_account(self) -> Optional[dict]:
        now = self._clock.time() if self._clock else 0
        return {
            "login": 999999, "balance": self._balance,
            "equity": self._equity, "margin": 0.0,
            "margin_free": self._equity, "margin_level": 0.0,
            "leverage": 100, "currency": "USD",
            "server": "PAPER", "name": "PaperBroker",
            "time": int(now),
        }

    def get_positions(self) -> list[dict]:
        now = self._clock.time() if self._clock else 0
        result = []
        for ticket, pos in list(self._positions.items()):
            if pos.get("status") != "OPEN":
                continue
            tick = self.get_tick(pos["symbol"])
            current = tick["bid"] if tick and pos["side"] == "SELL" else (tick["ask"] if tick else pos["entry"])
            if pos["side"] == "BUY":
                profit = (current - pos["entry"]) * pos["volume"] * self._pip_value(pos["symbol"])
            else:
                profit = (pos["entry"] - current) * pos["volume"] * self._pip_value(pos["symbol"])
            result.append({
                "ticket": ticket,
                "symbol": pos["symbol"],
                "type": pos["side"],
                "volume": pos["volume"],
                "price_open": pos["entry"],
                "price_current": current,
                "sl": pos.get("sl", 0.0),
                "tp": pos.get("tp", 0.0),
                "profit": round(profit, 2),
                "swap": 0.0,
                "commission": 0.0,
                "time": int(pos.get("opened_at", now)),
                "magic": 202406,
                "comment": pos.get("comment", ""),
            })
        return result

    def _pip_value(self, symbol: str) -> float:
        return 0.01 if "JPY" in symbol else 0.0001

    def _point(self, symbol: str) -> float:
        return 0.01 if "JPY" in symbol else 0.0001

    def _emit_execution(self, event):
        """Forward an ExecutionEvent to the optional sink (no-op if unset)."""
        if self._execution_event_sink is not None:
            try:
                self._execution_event_sink(event)
            except Exception as e:
                logger.warning(f"PaperBroker: execution sink raised: {e}")
    def place_order(self, symbol: str, order_type: str, volume: float,
                    price: float, sl: float = 0.0, tp: float = 0.0,
                    comment: str = "PROXIMA") -> Optional[dict]:
        tick = self.get_tick(symbol)
        if tick is None:
            logger.warning(f"PaperBroker: No tick for {symbol}")
            return None

        broker_sym = self._get_broker_symbol(symbol)
        fill_price = tick["ask"] if order_type.upper() == "BUY" else tick["bid"]

        latency_ms = 0.0
        if self._execution:
            fill_price = self._execution.apply_slippage(symbol, fill_price, order_type)
            latency_ms = self._execution.sample_latency() or 0.0
            if self._clock and latency_ms > 0:
                self._clock.sleep(latency_ms / 1000.0)
        else:
            # No ExecutionModel provided — apply shared cost-model slippage so
            # friction parity holds even without a replay execution model.
            fill_price = self._execution_cost.slippage_price(symbol, order_type, fill_price)

        commission = self._execution_cost.commission(volume)

        ticket = self._next_ticket
        self._next_ticket += 1
        now = self._clock.time() if self._clock else 0

        self._positions[ticket] = {
            "ticket": ticket,
            "symbol": symbol,
            "broker_symbol": broker_sym,
            "side": order_type.upper(),
            "entry": fill_price,
            "volume": volume,
            "sl": sl,
            "tp": tp,
            "comment": comment,
            "opened_at": now,
            "status": "OPEN",
            "commission": commission,
            "point": tick.get("point", self._point(symbol)),
            "slippage": fill_price - (tick["ask"] if order_type.upper() == "BUY" else tick["bid"]),
        }

        # emit OPEN execution event to the sink (optional).
        self._emit_execution(ExecutionEvent(
            event_type="OPEN", ticket=ticket, symbol=symbol, side=order_type.upper(),
            volume=volume, timestamp=now, bid=tick.get("bid", 0.0), ask=tick.get("ask", 0.0),
            requested_price=price, fill_price=fill_price, latency_ms=latency_ms,
            slippage_points=fill_price - (tick["ask"] if order_type.upper() == "BUY" else tick["bid"]),
            status="FILLED",
            commission=(-commission),  # signed, MT5-shaped (neg = cost)
        ))
        logger.info(f"PaperBroker: {order_type} {volume} {symbol} @ {fill_price} ticket={ticket}")
        if self._ledger is not None:
            self._ledger.add_trade({
                "ticket": ticket,
                "symbol": symbol,
                "side": order_type.upper(),
                "entry": fill_price,
                "sl": sl,
                "tp": tp,
                "ts": now,
                "volume": volume,
            })
        return {"ticket": ticket, "price": fill_price, "volume": volume, "type": order_type, "symbol": symbol}

    def close_order(self, ticket: int) -> bool:
        pos = self._positions.get(ticket)
        if not pos or pos.get("status") != "OPEN":
            logger.warning(f"PaperBroker: Position {ticket} not found or not open")
            return False

        tick = self.get_tick(pos["symbol"])
        if tick is None:
            logger.warning(f"PaperBroker: No tick for {pos['symbol']}")
            return False

        close_price = tick["bid"] if pos["side"] == "BUY" else tick["ask"]
        pt = self._point(pos["symbol"])
        if pos["side"] == "BUY":
            profit_points = (close_price - pos["entry"]) / pt
        else:
            profit_points = (pos["entry"] - close_price) / pt
        tv = self._tick_value_usd.get(pos["symbol"].upper())
        if tv is not None:
            # Broker-authoritative USD per machine POINT per 1.0 lot (live MT5
            # trade_tick_value). profit_points is already in machine points
            # (self._point returns 0.001 for JPY / 0.00001 direct / 0.01 gold).
            # Backtest per-trade PnL then equals live deals exactly.
            profit_money = profit_points * pos["volume"] * tv
        else:
            # Legacy per-pip conversion (backward-compatible default).
            profit_money = profit_points * pos["volume"] * (10 if "JPY" not in pos["symbol"] else 1000 / pos["entry"])
        # Both legs' commission are borne by the trade (open was charged at
        # fill; charge the close leg now) — mirror of live MT5 per-side fees.
        close_commission = self._execution_cost.commission(pos["volume"])
        open_commission = pos.get("commission", 0.0)
        total_commission = open_commission + close_commission
        gross_profit_money = profit_money  # raw price PnL before both-leg fees
        profit_money -= total_commission

        now = self._clock.time() if self._clock else 0
        pos["status"] = "CLOSED"
        pos["closed_at"] = now
        pos["close_price"] = close_price
        pos["profit"] = profit_money
        pos["commission"] = total_commission

        self._closed_pnl += profit_money
        self._balance += profit_money
        self._equity = self._balance

        self._history.append({
            # legacy / compatibility keys (semantics UNCHANGED:
            # profit = NET per-trade PnL; entry/close = filled prices)
            "ticket": ticket,
            "symbol": pos["symbol"],
            "side": pos["side"],
            "entry": pos["entry"],
            "close": close_price,
            "volume": pos["volume"],
            "profit": profit_money,
            "opened_at": pos["opened_at"],
            "closed_at": now,
            # new canonical MT5-shaped fields (added alongside; nothing
            # removed) mapping 1:1 onto MT5 history_deals. commission is
            # SIGNED like MT5 deal.commission (negative = cost), so
            # net_profit == gross_profit + commission + swap exactly.
            "price_open": pos["entry"],
            "price_close": close_price,
            "gross_profit": gross_profit_money,
            "commission": -total_commission,
            "swap": 0.0,
            "net_profit": profit_money,
        })

        if self._ledger is not None:
            self._ledger.add_trade({
                "ticket": ticket,
                "symbol": pos["symbol"],
                "side": pos["side"],
                "entry": pos["entry"],
                "close": close_price,
                "volume": pos["volume"],
                "profit": profit_money,
                "opened_at": pos["opened_at"],
                "closed_at": now,
                "phase": "close",
            })

        # emit CLOSE execution event to the sink (optional).
        self._emit_execution(ExecutionEvent(
            event_type="CLOSE", ticket=ticket, symbol=pos["symbol"], side=pos["side"],
            volume=pos["volume"], timestamp=now, bid=tick.get("bid", 0.0), ask=tick.get("ask", 0.0),
            requested_price=pos["entry"], fill_price=close_price,
            gross_profit=gross_profit_money, commission=(-total_commission), swap=0.0,
            net_profit=profit_money,
        ))
        logger.info(f"PaperBroker: Close {ticket} {pos['symbol']} profit={profit_money:.2f}")
        return True

    def close_all(self) -> list[dict]:
        results = []
        for ticket in list(self._positions.keys()):
            if self._positions[ticket].get("status") == "OPEN":
                ok = self.close_order(ticket)
                results.append({"ticket": ticket, "symbol": self._positions[ticket]["symbol"], "closed": ok})
        return results

    def _point(self, symbol: str) -> float:
        sym = symbol.upper()
        if "XAU" in sym or "GOLD" in sym:
            return 0.01
        if "JPY" in sym:
            return 0.001
        return 0.00001

    def _compute_spread(self, symbol: str, bid: float, ask: float) -> int:
        pt = self._point(symbol)
        if pt > 0 and ask > bid:
            return max(1, int((ask - bid) / pt))
        return 1

    def verify_spread(self, symbol: str, es_rank: float = None) -> bool:
        tick = self.get_tick(symbol)
        if tick is None:
            return False
        bid = tick.get("bid", 0.0)
        ask = tick.get("ask", 0.0)
        if bid <= 0 or ask <= 0:
            return False
        spread = tick.get("spread", self._compute_spread(symbol, bid, ask))
        if spread <= 0 or spread >= 999:
            return False
        BASE_SPREAD = {"EURJPY": 15, "USDJPY": 15, "GBPJPY": 20, "XAUUSD": 50, "EURUSD": 10}
        base = BASE_SPREAD.get(symbol, 20)
        limit = base * 3
        return spread <= limit

    def verify_symbol(self, symbol: str) -> dict:
        return {
            "symbol": symbol,
            "available": True,
            "spread": 10,
            "trade_mode": "ENABLED",
            "digits": 5 if "JPY" not in symbol else 3,
            "point": self._point(symbol),
            "lot_min": 0.01,
            "lot_max": 10.0,
            "lot_step": 0.01,
        }

    def check_sl_tp(self, symbol: str):
        tick = self.get_tick(symbol)
        if tick is None:
            return
        bid, ask = tick["bid"], tick["ask"]
        for ticket, pos in list(self._positions.items()):
            if pos.get("status") != "OPEN" or pos["symbol"] != symbol:
                continue
            sl = pos.get("sl", 0.0)
            tp = pos.get("tp", 0.0)
            if sl > 0 or tp > 0:
                if pos["side"] == "BUY":
                    if sl > 0 and bid <= sl:
                        logger.info(f"PaperBroker: SL hit {ticket} {symbol} @ {bid}")
                        self._positions[ticket]["sl_hit"] = True
                        self.close_order(ticket)
                    elif tp > 0 and ask >= tp:
                        logger.info(f"PaperBroker: TP hit {ticket} {symbol} @ {ask}")
                        self.close_order(ticket)
                else:
                    if sl > 0 and ask >= sl:
                        logger.info(f"PaperBroker: SL hit {ticket} {symbol} @ {ask}")
                        self.close_order(ticket)
                    elif tp > 0 and bid <= tp:
                        logger.info(f"PaperBroker: TP hit {ticket} {symbol} @ {bid}")
                        self.close_order(ticket)

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    @property
    def total_pnl(self) -> float:
        return self._closed_pnl
