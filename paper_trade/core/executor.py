"""MT5 order execution. Market orders only. Logs every fill."""
import time
from paper_trade.components import pip_value_usd

_PIP_SIZE = {"JPY": 0.01}
_PIP_DEFAULT = 0.0001

def _calc_pnl(pair, entry, exit, lot_size, direction):
    """Pair-aware USD PnL. Handles both USD-quoted and non-USD pairs."""
    pip_size = _PIP_SIZE.get(pair[-3:], _PIP_DEFAULT)
    pips = (exit - entry) / pip_size
    pv = pip_value_usd(pair, exit)
    return round(pips * pv * lot_size * direction, 2)

def _get_filling_type(symbol, mt5):
    filling = getattr(symbol, "filling_mode", 0)
    if filling & 2:
        return mt5.ORDER_FILLING_IOC
    elif filling & 1:
        return mt5.ORDER_FILLING_FOK
    else:
        return mt5.ORDER_FILLING_RETURN

class Executor:
    """Handles order placement and position management."""

    def __init__(self, feed, logger, max_spread_mult=1.5, magic=202407):
        self.feed = feed
        self.logger = logger
        self.max_spread_mult = max_spread_mult
        self.magic = magic
        self.positions = []
        self._use_mt5 = feed.mode == "live"

    def submit_market(self, pair, direction, lot_size=1.0, timestamp=None, signal_meta=None):
        """Submit market order. Returns fill dict or None."""
        if self._use_mt5:
            return self._live_submit(pair, direction, lot_size, timestamp, signal_meta)
        else:
            return self._paper_submit(pair, direction, lot_size, timestamp, signal_meta)

    def _live_submit(self, pair, direction, lot_size, ts, signal_meta=None):
        mt5 = self.feed.mt5
        symbol = mt5.symbol_info(pair)
        if symbol is None:
            self.logger.log("REJECT", pair, "symbol_not_found")
            return None

        tick = mt5.symbol_info_tick(pair)
        if tick is None:
            self.logger.log("REJECT", pair, "no_tick")
            return None

        spread = tick.ask - tick.bid
        normal_spread = symbol.spread
        if spread > normal_spread * self.max_spread_mult:
            self.logger.log("REJECT", pair, f"spread_widen:{spread:.1f}")
            return None

        order_type = mt5.ORDER_TYPE_BUY if direction > 0 else mt5.ORDER_TYPE_SELL
        price = tick.ask if direction > 0 else tick.bid
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pair,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "deviation": 10,
            "magic": self.magic,
            "comment": "paper_trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": _get_filling_type(symbol, mt5),
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.log("REJECT", pair, f"retcode:{result.retcode}")
            return None

        fill = {
            "pair": pair, "direction": direction, "lot_size": lot_size,
            "entry_price": price, "entry_time": int(time.time()),
            "spread": spread, "slip": abs(result.price - price),
        }
        if signal_meta:
            fill["z_score"] = signal_meta.get("z_score")
            fill["trigger_currency"] = signal_meta.get("currency")
        self.logger.log("FILL", pair, fill)
        self.positions.append(fill)
        return fill

    def _paper_submit(self, pair, direction, lot_size, ts, signal_meta=None):
        """Simulated fill using current feed price."""
        bar = self.feed.current_bar()
        if bar is None or pair not in bar:
            return None
        p = bar[pair]
        entry_price = p.get("ask", 0) if direction > 0 else p.get("bid", 0)
        if entry_price == 0:
            return None
        fill = {
            "pair": pair, "direction": direction, "lot_size": lot_size,
            "entry_price": entry_price, "entry_time": ts or int(time.time()),
            "spread": p.get("spread", 0), "slip": 0.0,
        }
        if signal_meta:
            fill["z_score"] = signal_meta.get("z_score")
            fill["trigger_currency"] = signal_meta.get("currency")
        self.logger.log("FILL", pair, fill)
        self.positions.append(fill)
        return fill

    def close_position(self, fill, exit_price=None, exit_time=None):
        """Close a position. Returns PnL dict."""
        if self._use_mt5:
            return self._live_close(fill)
        else:
            return self._paper_close(fill, exit_price, exit_time)

    def _live_close(self, fill):
        mt5 = self.feed.mt5
        symbol_name = fill["pair"]
        symbol = mt5.symbol_info(symbol_name)
        if symbol is None:
            return None

        # Find the open MT5 position by symbol + magic + direction
        pos_type = 0 if fill["direction"] > 0 else 1
        positions = mt5.positions_get(symbol=symbol_name)
        pos = None
        if positions:
            for p in positions:
                if p.magic == self.magic and p.type == pos_type:
                    pos = p
                    break
        if pos is None:
            self.logger.log("REJECT_CLOSE", symbol_name, "position_not_found")
            return None

        order_type = mt5.ORDER_TYPE_SELL if fill["direction"] > 0 else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(symbol_name)
        if tick is None:
            return None
        price = tick.bid if fill["direction"] > 0 else tick.ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol_name,
            "volume": fill["lot_size"],
            "type": order_type,
            "position": pos.ticket,
            "price": price,
            "deviation": 10,
            "magic": self.magic,
            "comment": "paper_trade_close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": _get_filling_type(symbol, mt5),
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.log("REJECT_CLOSE", symbol_name, f"retcode:{result.retcode}")
            return None
        tick2 = mt5.symbol_info_tick(symbol_name)
        exit_price = tick2.bid if fill["direction"] > 0 else tick2.ask
        gross = _calc_pnl(fill["pair"], fill["entry_price"], exit_price, fill["lot_size"], fill["direction"])
        self.logger.log("CLOSE", symbol_name, {"entry": fill["entry_price"], "exit": exit_price, "gross_pnl": gross})
        return {"exit_price": exit_price, "gross_pnl": gross}

    def _paper_close(self, fill, exit_price, exit_time):
        bar = self.feed.current_bar()
        if bar is None or fill["pair"] not in bar:
            return None
        p = bar[fill["pair"]]
        price = exit_price or (p.get("bid") if fill["direction"] > 0 else p.get("ask"))
        gross = _calc_pnl(fill["pair"], fill["entry_price"], price, fill["lot_size"], fill["direction"])
        self.logger.log("CLOSE", fill["pair"], {"entry": fill["entry_price"], "exit": price, "gross_pnl": gross})
        return {"exit_price": price, "gross_pnl": gross}

    def check_open_positions(self, current_time, hold_seconds=180):
        """Close positions older than hold_seconds."""
        closed = []
        remaining = []
        for pos in self.positions:
            age = current_time - pos["entry_time"]
            if age >= hold_seconds:
                result = self.close_position(pos, exit_time=current_time)
                if result:
                    closed.append({**pos, **result})
                else:
                    remaining.append(pos)
            else:
                remaining.append(pos)
        self.positions = remaining
        return closed
