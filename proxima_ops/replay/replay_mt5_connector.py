"""ReplayMT5Connector — a drop-in replacement for MT5Connector that replays historical data.
Uses M1 data for signal generation, derives M5 from M1 for ATR/SL-TP."""
import csv, os, math, logging
from typing import Optional

logger = logging.getLogger("proxima_ops.replay.mt5_connector")

SYMBOLS = ["EURUSD", "GBPUSD", "EURJPY", "USDJPY"]
DATA_DIR = "data"

class ReplayMT5Connector:
    """Duck-type compatible with MT5Connector, driven by historical M1 data."""

    def __init__(self, data_dir: str = DATA_DIR, initial_balance: float = 24988.47):
        self._data_dir = data_dir
        self._balance = initial_balance
        self._equity = initial_balance
        self._connected = True
        self._last_error = None

        # Load M1 data: {symbol: [{"time": int, "open": float, "high": float, "low": float, "close": float}, ...]}
        self._m1_data: dict[str, list[dict]] = {}
        self._m1_count = 0  # total bars per symbol (all symbols same length after alignment)
        for sym in SYMBOLS:
            bars = self._load_csv(f"{sym}_M1.csv")
            if bars:
                self._m1_data[sym] = bars
            else:
                self._m1_data[sym] = []

        # Align all symbols to same count (min length)
        min_len = min(len(v) for v in self._m1_data.values()) if self._m1_data else 0
        if min_len > 0:
            for sym in SYMBOLS:
                self._m1_data[sym] = self._m1_data[sym][-min_len:]
        self._m1_count = min_len

        # Current cursor: index into M1 arrays (0 = oldest bar, progresses forward)
        self._cursor = 0  # starts BEFORE first bar; advance(1) moves to index 0

        # Simulated positions: ticket -> dict
        self._positions: dict[int, dict] = {}
        self._deals: list[dict] = []
        self._next_ticket = 1_000_000
        self._next_position_id = 1_000_000
        self._max_reconnect_attempts = 3
        self._reconnect_attempts = 0
        self._reconnect_delay_s = 1

        logger.info(f"ReplayMT5Connector: {self._m1_count} M1 bars per symbol")

    # ── Data Loading ────────────────────────────────────────────────────

    def _load_csv(self, filename: str) -> list[dict]:
        path = os.path.join(self._data_dir, filename)
        if not os.path.exists(path):
            logger.warning(f"Data file not found: {path}")
            return []
        bars = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                bars.append({
                    "time": int(row["time"]),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row.get("volume", 0)),
                })
        logger.info(f"Loaded {len(bars)} bars from {filename}")
        return bars

    # ── Cursor Control ──────────────────────────────────────────────────

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def total_bars(self) -> int:
        return self._m1_count

    @property
    def is_at_end(self) -> bool:
        return self._cursor >= self._m1_count - 1

    def advance(self, n_bars: int = 1):
        """Move cursor forward by n_bars. No auto-close — executor handles SL/TP."""
        self._cursor = min(self._cursor + n_bars, self._m1_count - 1)

    def seek(self, index: int):
        """Jump to a specific index."""
        self._cursor = max(0, min(index, self._m1_count - 1))

    def reset(self):
        """Reset to start of data."""
        self._cursor = 0
        self._positions.clear()
        self._deals.clear()
        self._next_ticket = 1_000_000
        self._next_position_id = 1_000_000

    # ── SL/TP Evaluation ────────────────────────────────────────────────

    def _get_bar(self, symbol: str, index: int) -> Optional[dict]:
        if 0 <= index < len(self._m1_data.get(symbol, [])):
            return self._m1_data[symbol][index]
        return None

    def _calc_pnl(self, pos: dict, exit_price: float) -> float:
        entry = pos["price_open"]
        vol = pos["volume"]
        symbol = pos["symbol"]
        direction = pos["type"]
        pts = (exit_price - entry) if direction == "BUY" else (entry - exit_price)
        if "JPY" in symbol:
            return round(pts * 1000 * vol, 2)
        else:
            return round(pts * 100000 * vol, 2)

    def _sl_tp_hit_price(self, pos: dict, bar: dict) -> Optional[float]:
        """Return the price at which SL or TP would be hit on this bar, or None."""
        direction = pos["type"]
        sl = pos.get("sl", 0)
        tp = pos.get("tp", 0)
        o, h, l, c = bar["open"], bar["high"], bar["low"], bar["close"]

        if direction == "BUY":
            if sl > 0 and l <= sl:
                return sl
            if tp > 0 and h >= tp:
                return tp
        else:
            if sl > 0 and h >= sl:
                return sl
            if tp > 0 and l <= tp:
                return tp
        return None

    # ── Derived M5 from M1 ──────────────────────────────────────────────

    def _build_m5_from_m1(self, symbol: str, n_bars: int) -> list[dict]:
        """Derive the last n M5 bars by grouping 5 consecutive M1 bars.
        Returns list of {time, open, high, low, close} ending at current cursor."""
        bars = self._m1_data.get(symbol, [])
        if not bars or self._cursor < 0:
            return []
        # Calculate how many complete M5 groups we have up to current cursor
        end_idx = min(self._cursor, len(bars) - 1)
        # Start from 0 (oldest) to build clean M5 groups
        all_groups = []
        i = 0
        while i <= end_idx:
            if i > len(bars) - 1:
                break
            group = bars[i:min(i + 5, len(bars))]
            if len(group) < 3:  # incomplete group, skip
                break
            all_groups.append({
                "time": group[0]["time"],
                "open": group[0]["open"],
                "high": max(b["high"] for b in group),
                "low": min(b["low"] for b in group),
                "close": group[-1]["close"],
            })
            i += 5
        return all_groups[-n_bars:] if len(all_groups) >= n_bars else all_groups

    # ── MT5Connector Interface ──────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def connect(self) -> bool:
        self._connected = True
        self._reconnect_attempts = 0
        return True

    def disconnect(self):
        self._connected = False

    def ensure_connection(self) -> bool:
        return True

    def get_account(self) -> Optional[dict]:
        return {
            "login": 5051788806,
            "balance": round(self._balance, 2),
            "equity": round(self._equity, 2),
            "margin": 0.0,
            "margin_free": round(self._balance, 2),
            "margin_level": 0.0,
            "leverage": 500,
            "currency": "USD",
            "server": "ICMarkets-Demo",
            "name": "Replay Account",
        }

    def get_positions(self) -> list[dict]:
        cur = self._cursor
        result = []
        for ticket, pos in self._positions.items():
            bar = self._get_bar(pos["symbol"], cur)
            if bar is None:
                continue

            # Reflect SL/TP hit in price_current so executor detects it
            hit = self._sl_tp_hit_price(pos, bar)
            if hit is not None:
                cur_price = hit
            else:
                cur_price = bar["close"]

            pts = (cur_price - pos["price_open"]) if pos["type"] == "BUY" else (pos["price_open"] - cur_price)
            if "JPY" in pos["symbol"]:
                unrealized = round(pts * 1000 * pos["volume"], 2)
            else:
                unrealized = round(pts * 100000 * pos["volume"], 2)
            p = {
                "ticket": ticket,
                "symbol": pos["symbol"],
                "type": pos["type"],
                "volume": pos["volume"],
                "price_open": pos["price_open"],
                "price_current": cur_price,
                "sl": pos.get("sl", 0),
                "tp": pos.get("tp", 0),
                "profit": unrealized,
                "swap": 0.0,
                "commission": 0.0,
                "time": bar["time"] if bar else int(cur),
                "magic": 202406,
                "comment": pos.get("comment", "REPLAY"),
            }
            result.append(p)
        return result

    def get_tick(self, symbol: str) -> Optional[dict]:
        bar = self._get_bar(symbol, self._cursor)
        if bar is None:
            bar = self._get_bar(symbol, self._cursor - 1)
        if bar is None:
            return {"bid": 1.0, "ask": 1.0001, "spread": 1, "time": int(time.time()) if hasattr(time, 'time') else 0}
        if "JPY" in symbol:
            spread = 0.02
        else:
            spread = 0.0001
        return {
            "symbol": symbol,
            "bid": bar["close"] - spread / 2,
            "ask": bar["close"] + spread / 2,
            "spread": 1,
            "time": bar["time"],
        }

    def get_rates(self, symbol: str, count: int = 100,
                  timeframe: str = "M1") -> Optional[list]:
        bars = self._m1_data.get(symbol, [])
        if not bars:
            return None

        if str(timeframe).upper() == "M5":
            return self._build_m5_from_m1(symbol, count)

        # Default: M1 — return up to `count` bars ending at cursor
        end_idx = min(self._cursor, len(bars) - 1)
        start_idx = max(0, end_idx - count + 1)
        result = bars[start_idx:end_idx + 1]
        return [{
            "time": b["time"],
            "open": b["open"],
            "high": b["high"],
            "low": b["low"],
            "close": b["close"],
            "volume": b["volume"],
        } for b in result]

    def verify_symbol(self, symbol: str) -> dict:
        available = symbol in self._m1_data and len(self._m1_data[symbol]) > 0
        return {
            "symbol": symbol,
            "available": available,
            "spread": 1,
            "trade_mode": "ENABLED",
            "digits": 3 if "JPY" in symbol else 5,
            "point": 0.001 if "JPY" in symbol else 1e-5,
            "stops_level": 0,
            "lot_min": 0.01,
            "lot_max": 100,
            "lot_step": 0.01,
        }

    def verify_spread(self, symbol: str, es_rank: float = None) -> bool:
        return True

    def place_order(self, symbol: str, order_type: str, volume: float,
                    price: float, sl: float = 0.0, tp: float = 0.0,
                    comment: str = "REPLAY") -> Optional[dict]:
        ticket = self._next_ticket
        self._next_ticket += 1
        pos_id = self._next_position_id
        self._next_position_id += 1
        bar = self._get_bar(symbol, self._cursor) or {"time": int(self._cursor)}

        self._positions[ticket] = {
            "ticket": ticket,
            "_position_id": pos_id,
            "symbol": symbol,
            "type": order_type,
            "volume": volume,
            "price_open": price,
            "price_current": price,
            "sl": sl,
            "tp": tp,
            "profit": 0.0,
            "time": bar["time"],
            "comment": comment,
        }

        logger.info(
            f"[REPLAY_OPEN] ticket={ticket} {symbol} {order_type} "
            f"vol={volume} price={price:.5f} sl={sl:.5f} tp={tp:.5f}"
        )
        return {"ticket": ticket, "price": price, "volume": volume,
                "type": order_type, "symbol": symbol}

    def close_order(self, ticket: int) -> bool:
        pos = self._positions.pop(ticket, None)
        if not pos:
            self._last_error = f"Position {ticket} not found"
            return False

        # Use the SL/TP-aware price from the current bar
        bar = self._get_bar(pos["symbol"], self._cursor)
        if bar:
            hit = self._sl_tp_hit_price(pos, bar)
            exit_price = hit if hit is not None else bar["close"]
            exit_time = bar["time"]
        else:
            exit_price = pos["price_open"]
            exit_time = int(self._cursor)

        pnl = self._calc_pnl(pos, exit_price)
        self._balance += pnl
        self._equity = self._balance

        deal = {
            "deal": ticket,
            "position_id": pos.get("_position_id", ticket),
            "symbol": pos["symbol"],
            "type": 0 if pos["type"] == "BUY" else 1,
            "entry": 1,
            "time": exit_time,
            "price": exit_price,
            "volume": pos["volume"],
            "profit": pnl,
            "swap": 0.0,
            "commission": 0.0,
            "reason": "CLOSE_ORDER",
            "magic": 202406,
            "comment": "REPLAY",
        }
        self._deals.append(deal)
        logger.info(f"[REPLAY_CLOSE] ticket={ticket} {pos['symbol']} {pos['type']} "
                     f"entry={pos['price_open']:.5f} exit={exit_price:.5f} "
                     f"pnl={pnl:.2f}")
        return True

    def close_all(self) -> list[dict]:
        results = []
        for ticket in list(self._positions.keys()):
            ok = self.close_order(ticket)
            results.append({"ticket": ticket, "closed": ok, "symbol": self._positions.get(ticket, {}).get("symbol", "?")})
        return results

    def get_deal_history(self, position_id: Optional[int] = None,
                         hours_back: int = 999999) -> list[dict]:
        if position_id is not None:
            return [d for d in self._deals if d.get("position_id") == position_id]
        return self._deals

    def get_historical_ticks(self, symbol: str, count: int = 2000) -> Optional[list[dict]]:
        return None  # Not needed for replay
