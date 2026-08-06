import time
import logging
from typing import Optional
from datetime import datetime
import time as _time
from proxima_ops.config.settings import SETTINGS
from proxima_ops.execution.magic_resolver import generate_magic, infer_strategy_from_comment

logger = logging.getLogger("proxima_ops.mt5")

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 not installed. Install with: pip install MetaTrader5")


class MT5Connector:
    def __init__(self):
        self._connected = False
        self._account_info = None
        self._last_error = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
        self._reconnect_delay_s = 5

    @property
    def is_connected(self) -> bool:
        if not MT5_AVAILABLE:
            return False
        return self._connected and mt5.terminal_info() is not None

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def initialize(self) -> bool:
        """Alias for connect() — used by MRBL and other runtime modules."""
        return self.connect()

    def symbol_select(self, symbol: str, enable: bool = True) -> bool:
        """Enable/disable a symbol in MT5 MarketWatch."""
        if not MT5_AVAILABLE:
            return False
        try:
            return mt5.symbol_select(symbol, enable)
        except Exception:
            return False

    def connect(self) -> bool:
        if not MT5_AVAILABLE:
            self._last_error = "MetaTrader5 package not installed"
            return False
        if self._connected:
            if mt5.terminal_info() is not None:
                return True
            self._connected = False
        if not mt5.initialize(path=SETTINGS.mt5_path, timeout=SETTINGS.mt5_timeout_ms):
            err = mt5.last_error()
            self._last_error = f"MT5 initialize failed: {err}"
            logger.error(self._last_error)
            return False
        if SETTINGS.mt5_account and SETTINGS.mt5_password:
            authorized = mt5.login(
                SETTINGS.mt5_account,
                password=SETTINGS.mt5_password,
                server=SETTINGS.mt5_server)
            if not authorized:
                err = mt5.last_error()
                self._last_error = f"MT5 login failed: {err}"
                logger.error(self._last_error)
                mt5.shutdown()
                return False
        self._account_info = mt5.account_info()
        self._connected = True
        self._reconnect_attempts = 0
        logger.info(f"Connected to MT5. Account: {self._account_info.login if self._account_info else 'N/A'}, "
                    f"Balance: {self._account_info.balance if self._account_info else 'N/A'}")
        return True

    def disconnect(self):
        if MT5_AVAILABLE:
            try:
                mt5.shutdown()
            except Exception:
                pass
        self._connected = False
        self._account_info = None
        logger.info("Disconnected from MT5")

    def ensure_connection(self) -> bool:
        if self.is_connected:
            return True
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            self._last_error = f"Max reconnection attempts ({self._max_reconnect_attempts}) reached"
            return False
        self._reconnect_attempts += 1
        logger.info(f"Reconnection attempt {self._reconnect_attempts}/{self._max_reconnect_attempts}")
        time.sleep(self._reconnect_delay_s)
        return self.connect()

    def get_account(self) -> Optional[dict]:
        if not self.ensure_connection():
            return None
        info = mt5.account_info()
        if info is None:
            return None
        return {
            "login": info.login, "balance": info.balance,
            "equity": info.equity, "margin": info.margin,
            "margin_free": info.margin_free, "margin_level": info.margin_level,
            "leverage": info.leverage, "currency": info.currency,
            "server": info.server, "name": info.name}

    def get_positions(self) -> list[dict]:
        if not self.ensure_connection():
            return []
        positions = mt5.positions_get()
        if positions is None:
            return []
        result = []
        for p in positions:
            result.append({
                "ticket": p.ticket, "symbol": p.symbol,
                "type": "BUY" if p.type == 0 else "SELL",
                "volume": p.volume, "price_open": p.price_open,
                "price_current": p.price_current, "sl": p.sl,
                "tp": p.tp, "profit": p.profit,
                "swap": p.swap, "commission": getattr(p, "commission", 0.0),
                "time": p.time, "magic": p.magic, "comment": p.comment})
        return result

    def _get_broker_symbol(self, symbol: str) -> str:
        if not MT5_AVAILABLE:
            return symbol
        # Check if the symbol is already visible or exists
        if mt5.symbol_info(symbol) is not None:
            return symbol

        # Map common symbols to broker alternatives
        mappings = {
            "XAUUSD": ["GOLD", "XAUUSD.", "XAUUSDm"],
            "EURJPY": ["EURJPY.", "EURJPYecn", "EURJPYm"],
            "USDJPY": ["USDJPY.", "USDJPYecn", "USDJPYm"],
            "GBPJPY": ["GBPJPY.", "GBPJPYecn", "GBPJPYm"],
            "EURUSD": ["EURUSD.", "EURUSDm", "EURUSDecn"],
            "GBPUSD": ["GBPUSD.", "GBPUSDm", "GBPUSDecn"],
            "AUDUSD": ["AUDUSD.", "AUDUSDm"],
            "USDCAD": ["USDCAD.", "USDCADm"],
            "USDCHF": ["USDCHF.", "USDCHFm"],
            "NZDUSD": ["NZDUSD.", "NZDUSDm"],
        }
        
        alternatives = mappings.get(symbol.upper(), [])
        for alt in alternatives:
            if mt5.symbol_info(alt) is not None:
                logger.info(f"Mapping symbol {symbol} to broker symbol {alt}")
                return alt
        return symbol

    def get_tick(self, symbol: str) -> Optional[dict]:
        if getattr(self, '_tick_cache', None):
            cached = self._tick_cache.get_tick(symbol)
            if cached is not None:
                return cached
        if not self.ensure_connection():
            return None

        broker_symbol = self._get_broker_symbol(symbol)
        original_symbol = symbol

        # Force symbol into MarketWatch and verify
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            select_ok = mt5.symbol_select(broker_symbol, True)

            # Check if the symbol exists in MarketWatch
            sym_info = mt5.symbol_info(broker_symbol)
            if sym_info is None:
                logger.warning(
                    f"[TICK_DIAG] symbol_info is None for symbol={broker_symbol} "
                    f"(original={original_symbol}) attempt={attempt}/{max_attempts} "
                    f"symbol_select_ok={select_ok} last_error={mt5.last_error()}"
                )
                if attempt < max_attempts:
                    time.sleep(0.1)
                    continue
                return None

            tick = mt5.symbol_info_tick(broker_symbol)
            if tick is not None:
                # Success — log full details at DEBUG level
                point = sym_info.point if sym_info.point and sym_info.point > 0 else 1e-5
                spread_pts = max(0, round((tick.ask - tick.bid) / point)) if point > 0 else 0
                logger.debug(
                    f"[TICK_OK] symbol={broker_symbol} (original={original_symbol}) "
                    f"bid={tick.bid} ask={tick.ask} spread={spread_pts}pts "
                    f"time={tick.time} point={point} select_ok={select_ok}"
                )
                return {"symbol": broker_symbol, "bid": tick.bid, "ask": tick.ask,
                        "spread": spread_pts, "time": tick.time,
                        "point": point, "digits": int(sym_info.digits)}

            # Tick is None — log comprehensive diagnostics
            last_err = mt5.last_error()
            logger.warning(
                f"[TICK_DIAG] tick is None for symbol={broker_symbol} "
                f"(original={original_symbol}) attempt={attempt}/{max_attempts} "
                f"symbol_select_ok={select_ok} "
                f"symbol_info={'OK' if sym_info else 'NONE'} "
                f"symbol_info.spread={sym_info.spread if sym_info else 'N/A'} "
                f"symbol_info.trade_mode={sym_info.trade_mode if sym_info else 'N/A'} "
                f"last_error={last_err}"
            )

            if attempt < max_attempts:
                time.sleep(0.1)

        logger.error(
            f"[TICK_DIAG] All {max_attempts} attempts exhausted for "
            f"symbol={broker_symbol} (original={original_symbol})"
        )
        return None

    def get_rates(self, symbol: str, count: int = 100,
                  timeframe: int = 0) -> Optional[list]:
        if not hasattr(self, '_rates_cache'):
            self._rates_cache = {}
        key = (symbol, count, timeframe if isinstance(timeframe, str) else str(timeframe))
        cached = self._rates_cache.get(key)
        now = _time.time()
        if cached and (now - cached['ts']) < 300.0:
            return cached['data']
        if not self.ensure_connection():
            return None
        symbol = self._get_broker_symbol(symbol)
        mt5.symbol_select(symbol, True)
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1
        }
        if isinstance(timeframe, str):
            timeframe = tf_map.get(timeframe.upper(), mt5.TIMEFRAME_H1)
        elif timeframe == 0:
            timeframe = mt5.TIMEFRAME_H1
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None:
            err = mt5.last_error()
            logger.error(f"copy_rates_from_pos failed for {symbol} (timeframe={timeframe}, count={count}): {err}")
            return None
        result = [{
            "time": int(r['time']), "open": float(r['open']), "high": float(r['high']),
            "low": float(r['low']), "close": float(r['close']), "volume": float(r['tick_volume'])}
            for r in rates]
        self._rates_cache[key] = {'data': result, 'ts': now}
        return result

    def verify_symbol(self, symbol: str) -> dict:
        symbol = self._get_broker_symbol(symbol)
        result = {"symbol": symbol, "available": False,
                  "spread": 999, "trade_mode": "UNKNOWN"}
        if not self.ensure_connection():
            return result
        mt5.symbol_select(symbol, True)
        info = mt5.symbol_info(symbol)
        if info is None:
            return result
        result["available"] = True
        result["spread"] = info.spread
        result["trade_mode"] = ["DISABLED", "ENABLED", "CLOSE_ONLY"][info.trade_mode] if info.trade_mode < 3 else "UNKNOWN"
        result["digits"] = info.digits
        result["point"] = info.point
        result["stops_level"] = info.trade_stops_level
        result["lot_min"] = info.volume_min
        result["lot_max"] = info.volume_max
        result["lot_step"] = info.volume_step
        return result

    def verify_spread(self, symbol: str, es_rank: float = None) -> bool:
        orig_symbol = symbol
        broker_symbol = self._get_broker_symbol(symbol)
        tick = self.get_tick(broker_symbol)
        if tick is None:
            return False
        spread = tick["spread"]
        if spread < 0 or spread >= 999:
            return False
        if spread == 0:
            return True

        BASE_SPREAD = {"EURJPY": 15, "USDJPY": 15, "GBPJPY": 20, "XAUUSD": 50, "EURUSD": 10}

        if es_rank is not None and es_rank > 0:
            es = min(es_rank / 100.0, 1.0)
            gamma = 1.5
            base = BASE_SPREAD.get(orig_symbol, 20)
            elastic_limit = int(base * (1.0 + pow(es, gamma)))
            passed = spread <= elastic_limit
            logger.debug(f"Elastic spread[{orig_symbol}]: raw={spread} base={base} es={es:.3f} "
                         f"gamma={gamma} elastic_limit={elastic_limit} passed={passed}")
        else:
            limit = SETTINGS.max_spread_points.get(orig_symbol, 50)
            if limit == 50:
                limit = SETTINGS.max_spread_points.get(broker_symbol, 50)
            elastic_limit = limit
            passed = spread <= limit
            logger.debug(f"Static spread[{orig_symbol}]: raw={spread} limit={elastic_limit} passed={passed}")

        return passed

    def _get_filling_mode(self, symbol: str) -> int:
        if not MT5_AVAILABLE:
            return mt5.ORDER_FILLING_IOC
        symbol = self._get_broker_symbol(symbol)
        mt5.symbol_select(symbol, True)
        info = mt5.symbol_info(symbol)
        if info is None:
            return mt5.ORDER_FILLING_IOC
        fm = info.filling_mode
        if fm & 2:
            return mt5.ORDER_FILLING_IOC
        if fm & 1:
            return mt5.ORDER_FILLING_FOK
        # Most forex brokers default to IOC; RETURN causes retcode 10018
        return mt5.ORDER_FILLING_IOC

    def place_order(self, symbol: str, order_type: str, volume: float,
                    price: float, sl: float = 0.0, tp: float = 0.0,
                    comment: str = "PROXIMA_V2") -> Optional[dict]:
        if not self.ensure_connection():
            return None
        symbol = self._get_broker_symbol(symbol)
        mt5.symbol_select(symbol, True)
        mt5_type = 0 if order_type.upper() == "BUY" else 1
        strategy_id = infer_strategy_from_comment(comment)
        magic = generate_magic(strategy_id, order_type.upper())
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": SETTINGS.max_slippage_points,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._get_filling_mode(symbol)}
        result = mt5.order_send(request)
        if result is None:
            self._last_error = f"Order send returned None for {symbol}"
            return None
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self._last_error = f"Order failed for {symbol}: retcode={result.retcode}, comment={result.comment}"
            logger.error(self._last_error)
            return None
        return {"ticket": result.order, "price": price,
                "volume": volume, "type": order_type, "symbol": symbol}

    def close_order(self, ticket: int) -> bool:
        if not self.ensure_connection():
            return False
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            self._last_error = f"Position {ticket} not found"
            return False
        pos = position[0]
        symbol = self._get_broker_symbol(pos.symbol)
        mt5.symbol_select(symbol, True)
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            self._last_error = f"Symbol info not available for {symbol}"
            return False
        mt5_type = 1 if pos.type == 0 else 0
        price = tick.bid if mt5_type == 1 else tick.ask
        if price is None or price <= 0:
            self._last_error = f"Invalid price {price} for {symbol}"
            return False
        if pos.volume is None or pos.volume <= 0:
            self._last_error = f"Invalid volume {pos.volume} for ticket {ticket}"
            return False
        close_type = "SELL" if mt5_type == 1 else "BUY"
        close_magic = generate_magic("PROXIMA_V2", close_type, instance=99)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": pos.volume,
            "type": mt5_type,
            "position": ticket,
            "price": price,
            "deviation": 50,
            "magic": close_magic,
            "comment": "PROXIMA_V2_CLOSE",
            "type_filling": self._get_filling_mode(symbol)}
        logger.info(f"[CLOSE_REQ] ticket={ticket} sym={symbol} vol={pos.volume} type={mt5_type} price={price}")
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            retcode = result.retcode if result is not None else "NONE"
            rcomment = result.comment if result is not None else "NONE"
            err_detail = mt5.last_error() if hasattr(mt5, 'last_error') else '?'
            self._last_error = f"Close order failed for ticket {ticket}: retcode={retcode} comment={rcomment} last_error={err_detail}"
            logger.error(f"[CLOSE_FAIL] ticket={ticket} sym={symbol} vol={pos.volume} type={mt5_type} price={price} deviation=50 retcode={retcode} comment={rcomment} req={request}")
            return False
        return True

    def modify_position_sl_tp(self, ticket: int, sl: float, tp: float) -> bool:
        """Modify SL/TP on an existing position."""
        if not self.ensure_connection():
            return False
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            self._last_error = f"Position {ticket} not found"
            return False
        pos = position[0]
        symbol = self._get_broker_symbol(pos.symbol)
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": symbol,
            "sl": sl,
            "tp": tp,
            "magic": 202406,
            "comment": "PROXIMA_V2_SLTP_INJECT",
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            self._last_error = f"Modify SL/TP failed for ticket {ticket}: retcode={result.retcode if result else 'NONE'}"
            logger.error("[SLTP_MOD_FAIL] ticket=%s sl=%s tp=%s retcode=%s", ticket, sl, tp, result.retcode if result else 'NONE')
            return False
        logger.info("[SLTP_MOD_OK] ticket=%s sl=%s tp=%s", ticket, sl, tp)
        return True

    def get_deal_history(self, position_id: Optional[int] = None,
                         hours_back: int = 48) -> list[dict]:
        """Fetch recent closed deals from MT5 history.
        
        Args:
            position_id: If provided, only deals for this position.
            hours_back: How many hours of history to fetch.
        
        Returns:
            List of deal dicts with keys: deal, position_id, symbol, type,
            entry, time, price, volume, profit, swap, commission, reason, magic, comment.
        """
        if not self.ensure_connection():
            return []
        from datetime import datetime, timedelta
        end = datetime.now()
        start = end - timedelta(hours=hours_back)
        if position_id is not None:
            deals = mt5.history_deals_get(position=position_id)
        else:
            deals = mt5.history_deals_get(start, end)
        if deals is None:
            return []
        result = []
        for d in deals:
            result.append({
                "deal": getattr(d, "deal", 0),
                "position_id": getattr(d, "position_id", 0),
                "symbol": getattr(d, "symbol", ""),
                "type": getattr(d, "type", -1),          # 0=BUY, 1=SELL
                "entry": getattr(d, "entry", -1),         # 0=in, 1=out
                "time": getattr(d, "time", 0),
                "price": float(getattr(d, "price", 0.0)),
                "volume": float(getattr(d, "volume", 0.0)),
                "profit": float(getattr(d, "profit", 0.0)),
                "swap": float(getattr(d, "swap", 0.0)),
                "commission": float(getattr(d, "commission", 0.0)),
                "reason": getattr(d, "reason", -1),
                "magic": getattr(d, "magic", 0),
                "comment": getattr(d, "comment", ""),
            })
        return result

    def get_historical_ticks(self, symbol: str, count: int = 2000) -> Optional[list[dict]]:
        """Fetch recent historical ticks from MT5 for pre-seeding the RF gate buffer.
        Returns list of {bid, ask, time} dicts, oldest-first, up to `count` ticks.
        Falls back to wider time windows if fewer ticks are available."""
        if not self.ensure_connection():
            return None
        broker_sym = self._get_broker_symbol(symbol)
        mt5.symbol_select(broker_sym, True)
        from datetime import timedelta
        now = datetime.now()
        # Try progressively wider windows
        windows = [
            ("2 hours", now - timedelta(hours=2), 5_000),
            ("6 hours", now - timedelta(hours=6), 10_000),
            ("1 day",   now - timedelta(days=1), 30_000),
            ("7 days",  now - timedelta(days=7), 100_000),
        ]
        for label, date_from, req_count in windows:
            try:
                ticks = mt5.copy_ticks_from(broker_sym, date_from, req_count, mt5.COPY_TICKS_INFO)
                if ticks is not None and len(ticks) >= count:
                    # Take the most recent `count` ticks
                    recent = ticks[-count:]
                    result = []
                    for t in recent:
                        bid = getattr(t, 'bid', 0.0)
                        ask = getattr(t, 'ask', 0.0)
                        ts = int(getattr(t, 'time', 0))
                        result.append({"bid": bid, "ask": ask, "time": ts})
                    logger.info(f"Fetched {len(result)} historical ticks for {symbol} (window={label})")
                    return result
                elif ticks is not None and len(ticks) > 0:
                    logger.info(f"Partial ticks for {symbol}: {len(ticks)}/{count} (window={label}), widening…")
            except Exception as e:
                logger.warning(f"get_historical_ticks({symbol}, window={label}): {e}")
        # Last resort: return whatever we can get from the widest window
        try:
            ticks = mt5.copy_ticks_from(broker_sym, now - timedelta(days=30), 200_000, mt5.COPY_TICKS_INFO)
            if ticks is not None and len(ticks) > 0:
                recent = ticks[-min(count, len(ticks)):]
                result = [{"bid": getattr(t, 'bid', 0.0), "ask": getattr(t, 'ask', 0.0), "time": int(getattr(t, 'time', 0))} for t in recent]
                logger.info(f"Fetched {len(result)} historical ticks for {symbol} (30-day fallback)")
                return result
        except Exception as e:
            logger.warning(f"get_historical_ticks({symbol}, 30-day fallback): {e}")
        logger.warning(f"No historical ticks available for {symbol}")
        return None

    def close_all(self) -> list[dict]:
        results = []
        for pos in self.get_positions():
            ok = self.close_order(pos["ticket"])
            results.append({"ticket": pos["ticket"], "symbol": pos["symbol"],
                            "closed": ok})
        return results
