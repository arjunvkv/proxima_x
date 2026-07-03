import logging
import math
from typing import Optional
from proxima_ops.config.settings import SETTINGS
from proxima_ops.execution.mt5_connector import MT5Connector
from proxima_ops.risk.catastrophic_stop import (
    get_risk_stop_distance,
    catastrophic_sl,
    catastrophic_tp,
)


logger = logging.getLogger("proxima_ops.orders")


class OrderManager:
    MAX_CLOSE_RETRIES = 3

    def __init__(self, mt5: MT5Connector):
        self._mt5 = mt5
        self._pending: list[dict] = []
        self._close_attempts: dict[int, int] = {}

    def calculate_volume(self, symbol: str, price: float,
                         account_balance: float, risk_pct: float = None) -> float:
        risk = risk_pct if risk_pct is not None else SETTINGS.risk_per_trade
        if price is None or price <= 0 or account_balance is None or account_balance <= 0:
            return 0.01
        risk_amount = float(account_balance) * float(risk)

        if "JPY" in symbol:
            point_value_per_lot = max(float(price), 1.0)
            point_value_per_lot = 1000.0 / point_value_per_lot
        else:
            point_value_per_lot = 10.0

        assumed_sl_points = get_risk_stop_distance(symbol)["stop_pips"]
        lots = risk_amount / max(assumed_sl_points * point_value_per_lot, 1.0)
        lots = max(0.01, round(lots, 2))

        # MT5 volume normalization: enforce volume_step, volume_min, volume_max
        try:
            import MetaTrader5 as mt5
            info = mt5.symbol_info(symbol)
            if info:
                step = info.volume_step
                vmin = info.volume_min
                vmax = info.volume_max
                lots = math.floor((lots / step) + 0.5) * step
                lots = max(vmin, min(vmax, lots))
                lots = round(lots, int(abs(math.log10(step))) if step >= 0.01 else 2)
                if lots < vmin:
                    logger.warning(f"Volume {lots} < min {vmin} for {symbol}, returning 0")
                    return 0.0
        except Exception as e:
            logger.warning(f"[VOL_NORM_FAIL] {symbol}: {e}")
            return 0.0

        return min(lots, 1.0)

    def _compute_atr(self, symbol: str, period: int = 14) -> float:
        """Compute ATR from M5 data for wider volatility frame. Returns 0.0 on failure."""
        try:
            rates = self._mt5.get_rates(symbol, count=period + 1, timeframe="M5")
            if not rates or len(rates) < period + 1:
                logger.warning(f"[ATR_FAIL] {symbol}: {len(rates) if rates else 0} bars for period={period}")
                return 0.0
            trs = []
            for i in range(1, len(rates)):
                high = rates[i]["high"]
                low = rates[i]["low"]
                prev_close = rates[i - 1]["close"]
                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                trs.append(tr)
            if len(trs) < period:
                return 0.0
            atr = sum(trs[-period:]) / period
            logger.info(f"[ATR] {symbol} M5 period={period} atr={atr:.5f}")
            return atr
        except Exception as e:
            logger.warning(f"[ATR_FAIL] {symbol}: {e}")
            return 0.0

    def _compute_mandatory_sl_tp(self, symbol: str, price: float,
                                  direction: str) -> tuple[float, float]:
        """Compute ATR-based SL/TP. Returns (0.0, 0.0) → HARD REJECT."""
        atr = self._compute_atr(symbol)
        if atr <= 0:
            logger.error(f"[SLTP_FAIL] {symbol}: ATR=0, cannot compute mandatory SL/TP")
            return 0.0, 0.0
        info = self._mt5.verify_symbol(symbol)
        point = info.get("point", 1e-5)
        stops_level = info.get("stops_level", 0)
        broker_min_stop = stops_level * point
        sl_distance = max(2.8 * atr, broker_min_stop)
        tp_distance = 1.6 * sl_distance
        digits = info.get("digits", 5)
        if direction == "BUY":
            sl = round(price - sl_distance, digits)
            tp = round(price + tp_distance, digits)
        else:
            sl = round(price + sl_distance, digits)
            tp = round(price - tp_distance, digits)
        if sl <= 0.0 or tp <= 0.0:
            logger.error(f"[HARD_REJECT] {symbol}: SL={sl} TP={tp} invalid — blocking order")
            return 0.0, 0.0
        logger.info(f"[MANDATORY_SL_TP] {symbol} {direction} price={price} "
                    f"ATR={atr:.5f} sl_dist={sl_distance:.5f} SL={sl} TP={tp}")
        return sl, tp

    def _resolve_sl_tp(self, symbol: str, price: float, direction: str,
                        sl: float, tp: float) -> tuple[float, float]:
        """Return (sl, tp). First tries ATR-based, then catastrophic fallback, then HARD REJECT."""
        if sl == 0.0 or tp == 0.0:
            atr_sl, atr_tp = self._compute_mandatory_sl_tp(symbol, price, direction)
            if atr_sl != 0.0 and atr_tp != 0.0:
                if sl == 0.0:
                    sl = atr_sl
                if tp == 0.0:
                    tp = atr_tp
            else:
                if sl == 0.0:
                    sl = catastrophic_sl(symbol, price, direction)
                if tp == 0.0:
                    if sl != 0.0:
                        sl_dist = abs(price - sl)
                        tp_dist = 1.5 * sl_dist
                        decimals = 3 if ("JPY" in symbol or "XAU" in symbol) else 5
                        tp = round(price + tp_dist, decimals) if direction == "BUY" else round(price - tp_dist, decimals)
                    else:
                        tp = 0.0
        if sl <= 0.0 or tp <= 0.0:
            logger.error(f"[HARD_REJECT] {symbol}: SL={sl} TP={tp} after resolution — blocking order")
        return sl, tp

    def execute_buy(self, symbol: str, price: float,
                    account_balance: float, risk_pct: float = None,
                    sl: float = 0.0, tp: float = 0.0,
                    volume: float = 0.0, comment: str = "PROXIMA_V2") -> Optional[dict]:
        if not self._mt5.verify_spread(symbol):
            logger.warning(f"Spread too high for {symbol}, skipping")
            return None
        if not self._mt5.verify_symbol(symbol)["available"]:
            logger.warning(f"Symbol {symbol} not available")
            return None
        if volume <= 0:
            volume = self.calculate_volume(symbol, price, account_balance, risk_pct=risk_pct)
        if volume <= 0:
            return None
        sl, tp = self._resolve_sl_tp(symbol, price, "BUY", sl, tp)
        if sl <= 0.0 or tp <= 0.0:
            logger.error(f"[HARD_REJECT] {symbol} BUY: SL={sl} TP={tp} — order blocked")
            return None
        logger.info(f"[BUY_PATH_HIT] {symbol} vol={volume} price={price} sl={sl} tp={tp}")
        result = self._mt5.place_order(symbol, "BUY", volume, price, sl=sl, tp=tp, comment=comment)
        if result:
            logger.info(f"BUY {symbol} {volume} @ {price} sl={sl} tp={tp} — ticket={result.get('ticket')}")
        else:
            logger.error(f"[BUY_FAIL] {symbol} vol={volume} error={self._mt5.last_error}")
        return result

    def execute_sell(self, symbol: str, price: float,
                     account_balance: float, risk_pct: float = None,
                     sl: float = 0.0, tp: float = 0.0,
                     volume: float = 0.0, comment: str = "PROXIMA_V2") -> Optional[dict]:
        if not self._mt5.verify_spread(symbol):
            logger.warning(f"Spread too high for {symbol}, skipping")
            return None
        if not self._mt5.verify_symbol(symbol)["available"]:
            logger.warning(f"Symbol {symbol} not available")
            return None
        if volume <= 0:
            volume = self.calculate_volume(symbol, price, account_balance, risk_pct=risk_pct)
        if volume <= 0:
            return None
        sl, tp = self._resolve_sl_tp(symbol, price, "SELL", sl, tp)
        if sl <= 0.0 or tp <= 0.0:
            logger.error(f"[HARD_REJECT] {symbol} SELL: SL={sl} TP={tp} — order blocked")
            return None
        logger.info(f"[SELL_PATH_HIT] {symbol} vol={volume} price={price} sl={sl} tp={tp}")
        result = self._mt5.place_order(symbol, "SELL", volume, price, sl=sl, tp=tp, comment=comment)
        if result:
            logger.info(f"SELL {symbol} {volume} @ {price} sl={sl} tp={tp} — ticket={result.get('ticket')}")
        else:
            logger.error(f"[SELL_FAIL] {symbol} vol={volume} error={self._mt5.last_error}")
        return result

    def place_order(self, symbol: str, order_type: str, volume: float, price: float,
                     sl: float = 0.0, tp: float = 0.0,
                     comment: str = "PROXIMA_V2") -> Optional[dict]:
        if not self._mt5.verify_symbol(symbol)["available"]:
            logger.warning(f"Symbol {symbol} not available")
            return None
        mt5_type = "BUY" if order_type.lower() == "buy" else "SELL"
        sl, tp = self._resolve_sl_tp(symbol, price, mt5_type, sl, tp)
        if sl <= 0.0 or tp <= 0.0:
            logger.error(f"[HARD_REJECT] {symbol} {mt5_type}: SL={sl} TP={tp} — order blocked")
            return None
        result = self._mt5.place_order(symbol, mt5_type, volume, price, sl=sl, tp=tp, comment=comment)
        if result:
            logger.info(f"[{mt5_type}_ROUTER] {symbol} vol={volume} @ {price} sl={sl} tp={tp} ticket={result.get('ticket')}")
        else:
            logger.error(f"[{mt5_type}_ROUTER_FAIL] {symbol} vol={volume} error={self._mt5.last_error}")
        return result

    def close(self, ticket: int) -> bool:
        attempts = self._close_attempts.setdefault(ticket, 0)
        if attempts >= self.MAX_CLOSE_RETRIES:
            logger.warning(f"[CLOSE_ABORTED] ticket={ticket} max retries ({self.MAX_CLOSE_RETRIES}) reached")
            return False
        result = self._mt5.close_order(ticket)
        self._close_attempts[ticket] = attempts + 1
        if result:
            logger.info(f"Closed ticket {ticket}")
            self._close_attempts.pop(ticket, None)
        else:
            logger.error(f"Failed to close ticket {ticket} (attempt {attempts + 1}/{self.MAX_CLOSE_RETRIES}): {self._mt5.last_error}")
        return result

    def close_all(self) -> list[dict]:
        return self._mt5.close_all()
