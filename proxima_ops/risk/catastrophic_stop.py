"""RHL-1: Broker Catastrophic Stop — disaster-protection SL on every order."""
import logging
logger = logging.getLogger("proxima_ops.risk.catastrophic_stop")

CATASTROPHIC_STOP_PIPS = {
    "EURUSD": 50,
    "EURJPY": 50,
    "USDJPY": 50,
    "GBPJPY": 70,
    "XAUUSD": 500,
}
"""Max pip loss per trade. Exists only for terminal/VPS/MT5 crash survival."""


def get_risk_stop_distance(symbol: str) -> dict:
    """Single source of truth for stop distance and pip size.

    Used by calculate_volume(), TradeRiskVerifier, and catastrophic_sl().
    Returns:
      stop_pips:  number of pips for the risk stop
      pip_size:   price increment per pip (0.01 for JPY/XAU, 0.0001 for others)
    """
    pips = CATASTROPHIC_STOP_PIPS.get(symbol, 50)
    if "JPY" in symbol or "XAU" in symbol or "XAG" in symbol:
        pip_size = 0.01
    else:
        pip_size = 0.0001
    return {"stop_pips": pips, "pip_size": pip_size}


def catastrophic_sl(symbol: str, entry_price: float, order_type: str) -> float:
    """Return broker SL price based on the catastrophic stop in pips.
    
    For SELL orders, enforces broker minimum stop distance to avoid
    TRADE_RETCODE_INVALID_STOPS (retcode 10016).
    """
    sd = get_risk_stop_distance(symbol)
    p = sd["stop_pips"] * sd["pip_size"]
    sl_price = entry_price - p if order_type == "BUY" else entry_price + p
    decimals = 3 if ("JPY" in symbol or "XAU" in symbol or "XAG" in symbol) else 5
    # P2-A: Broker stop distance enforcement for SELL orders
    # Prevents retcode 10016 invalid stops when SL is too close to market
    if order_type == "SELL":
        try:
            import MetaTrader5 as _mt5
            sym_info = _mt5.symbol_info(symbol)
            tick = _mt5.symbol_info_tick(symbol)
            if sym_info:
                point = sym_info.point
                stop_level = sym_info.trade_stops_level * point
                freeze_level = sym_info.trade_freeze_level * point
                anchor = tick.ask if tick else entry_price
                min_sl = anchor + stop_level + freeze_level
                if sl_price < min_sl:
                    logger.warning(
                        f"[SL_NORM] {symbol} SELL sl={sl_price} "
                        f"< min={min_sl:.5f} (anchor={anchor:.5f} stop_level={stop_level:.5f} freeze={freeze_level:.5f})")
                    sl_price = min_sl
        except Exception:
            logger.debug(f"[SL_NORM] {symbol} could not query broker levels")
    return round(sl_price, decimals)


def catastrophic_tp(symbol: str, entry_price: float, order_type: str) -> float:
    """Return broker TP price at 1.5x the catastrophic SL distance."""
    sd = get_risk_stop_distance(symbol)
    p = sd["stop_pips"] * sd["pip_size"] * 1.5
    tp_price = entry_price + p if order_type == "BUY" else entry_price - p
    return tp_price


def pip_distance(symbol: str, entry: float, current: float) -> float:
    diff = abs(entry - current)
    if "JPY" in symbol:
        return diff / 0.01
    elif "XAU" in symbol or "XAG" in symbol:
        return diff / 0.01
    return diff / 0.0001
