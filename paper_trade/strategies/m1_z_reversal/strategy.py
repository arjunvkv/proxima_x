"""M1 Z-Reversal strategy — mean reversion on extreme z-score bars with trailing stop.

ARCHITECTURE
  BarBuilder accumulates bid ticks into M1 OHLC bars (open/high/low/close per minute).
  When a new bar completes, PairState.update() evaluates |z|>2.0 & ATR>25pctl gate.
  Signals are emitted at the bar boundary, entry is delayed 3-5s (randomized).
  TrailingStopManager handles all open positions with randomized stops.

No lookahead: all rolling calcs use only PAST bar data (true shift(1) semantics).
"""
import time, random
from paper_trade.core.config import register

STRATEGY_NAME = "m1_z_reversal"

CONFIG = {
    "name": STRATEGY_NAME,
    "mt5_account": 0,
    "mt5_path": None,
    "pairs": ["EURUSD", "EURJPY", "GBPJPY"],
    "hold_bars": 0,
    "session_start": 0,
    "session_end": 24,
    "max_concurrent": 6,
    "max_spread_mult": 2.0,
    "max_daily_loss": 1000,
    "lot_size": 0.10,
    "z_thresh": 2.0,
    "atr_pctl": 0.25,
    "stop_a": 0.15,
    "trig_a": 0.20,
    "gap_a": 0.10,
    "max_hold_min": 54,
    "z_window": 50,
    "atr_window": 20,
    "atr_gate_window": 100,
    "entry_offset_s": 3,
    "entry_offset_jitter": 2,
    "min_stop_pips": 1.5,
}

register(STRATEGY_NAME, CONFIG)

# ─── Fixed-size ring buffers ──────────────────────────────────────

class RollingStats:
    def __init__(self, window=100):
        self.window = window; self.buffer = []

    def add(self, v):
        self.buffer.append(v)
        if len(self.buffer) > self.window:
            self.buffer.pop(0)

    def quantile(self, q):
        if len(self.buffer) < self.window * 0.5:
            return None
        s = sorted(self.buffer)
        return s[int(len(s) * q)]


class ZBuffer:
    def __init__(self, window=50):
        self.window = window; self.returns = []

    def add(self, ret):
        self.returns.append(ret)
        if len(self.returns) > self.window:
            self.returns.pop(0)

    def z_score(self, value):
        n = len(self.returns)
        if n < self.window * 0.8:
            return None
        m = sum(self.returns) / n
        var = sum((x - m) ** 2 for x in self.returns) / (n - 1)
        return (value - m) / (var ** 0.5 if var > 1e-10 else 1e-10)


class ATRBuffer:
    def __init__(self, window=20):
        self.window = window; self.ranges = []

    def add(self, r):
        self.ranges.append(r)
        if len(self.ranges) > self.window:
            self.ranges.pop(0)

    def value(self):
        if len(self.ranges) < self.window * 0.5:
            return None
        return sum(self.ranges) / len(self.ranges)


# ─── Bar builder: ticks → M1 OHLC ────────────────────────────────

class BarBuilder:
    """Accumulates bid ticks into M1 bars. Emits completed bar on minute rollover."""

    def __init__(self):
        self.minute = None
        self.open = None
        self.high = None
        self.low = None
        self.close = None

    def update(self, bid, tick_time):
        """Feed a tick. Returns (bar_complete, bar_dict) on minute rollover, else (False, None)."""
        tick_min = tick_time // 60

        if self.minute is None:
            self.minute = tick_min
            self.open = bid
            self.high = bid
            self.low = bid
            self.close = bid
            return False, None

        if tick_min == self.minute:
            self.high = max(self.high, bid)
            self.low = min(self.low, bid)
            self.close = bid
            return False, None

        bar = {"open": self.open, "high": self.high,
               "low": self.low, "close": self.close,
               "time": self.minute * 60}

        self.minute = tick_min
        self.open = bid
        self.high = bid
        self.low = bid
        self.close = bid
        return True, bar

    def current_close(self):
        return self.close


# ─── Per-pair signal state ────────────────────────────────────────

class PairState:
    def __init__(self, pair, config):
        self.pair = pair
        self.cfg = config
        self.z_buf = ZBuffer(window=config.get("z_window", 50))
        self.atr_buf = ATRBuffer(window=config.get("atr_window", 20))
        self.gate_buf = RollingStats(window=config.get("atr_gate_window", 100))
        self.last_close = None
        self.last_bar_time = None
        self.bar_builder = BarBuilder()

    def seed_bar(self, bar):
        """Seed one historical bar (pre-populate buffers, no signal)."""
        if self.last_close is not None:
            ret = bar["close"] - self.last_close
            self.z_buf.add(ret)
        atr_v = self.atr_buf.value()
        r = bar["high"] - bar["low"]
        self.atr_buf.add(r)
        if atr_v is not None:
            self.gate_buf.add(atr_v)
        self.last_close = bar["close"]
        self.last_bar_time = bar["time"]

    def update(self, bid, tick_time):
        """Feed tick → check if bar completed → evaluate signal.

        All checks use buffer state from PRIOR bars only (shift(1) semantics).
        Buffer updates happen AFTER signal evaluation.
        """
        completed, bar = self.bar_builder.update(bid, tick_time)
        if not completed:
            return None

        signal = None
        if self.last_close is not None:
            ret = bar["close"] - self.last_close
            z_v = self.z_buf.z_score(ret)

            atr_v = self.atr_buf.value()
            gate_v = self.gate_buf.quantile(self.cfg.get("atr_pctl", 0.25))

            if z_v is not None and atr_v is not None and gate_v is not None:
                if abs(z_v) > self.cfg.get("z_thresh", 2.0) and atr_v > gate_v:
                    direction = -1 if z_v > 0 else 1
                    signal = {
                        "pair": self.pair,
                        "direction": direction,
                        "confidence": min(1.0, abs(z_v) / 5.0),
                        "atr": atr_v,
                        "z_score": z_v,
                        "bar_time": bar["time"],
                    }

            # Update buffers AFTER signal check (shift(1) semantics)
            self.z_buf.add(ret)
            if atr_v is not None:
                self.gate_buf.add(atr_v)

        r = bar["high"] - bar["low"]
        self.atr_buf.add(r)
        self.last_close = bar["close"]
        self.last_bar_time = bar["time"]
        return signal


_states = {}


def seed_history(feed):
    """Pre-seed all PairState buffers by fetching M1 history from MT5.

    Uses full OHLC from copy_m1_history so ATR buffers get real bar ranges.
    """
    cfg = CONFIG
    for pair in cfg["pairs"]:
        ps = PairState(pair, cfg)
        bars = feed.copy_m1_history(pair, count=100)
        if bars:
            for b in bars:
                ps.seed_bar(b)
        _states[pair] = ps


def generate_signal(data, current_time=None):
    """Evaluate all pairs for z-reversal signals.

    Args:
        data: {pair: {bid, ask, time, ...}} — tick-level from feed.current_bar()
        current_time: optional int unix time

    Returns:
        dict | None
    """
    cfg = CONFIG
    ts = current_time or int(time.time())

    for pair, p in data.items():
        if pair not in _states:
            _states[pair] = PairState(pair, cfg)

        bid = p.get("bid", 0)
        ask = p.get("ask", 0)
        mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else bid
        tick_time = p.get("time", ts)
        if mid <= 0:
            continue

        signal = _states[pair].update(mid, tick_time)
        if signal is not None:
            delay = cfg.get("entry_offset_s", 3) + random.randint(0, cfg.get("entry_offset_jitter", 2))
            signal["delay_s"] = delay
            return signal

    return None


# ─── Trailing stop manager ────────────────────────────────────────

class TrailingStopManager:
    def __init__(self, config):
        self.positions = {}
        self.config = config
        self._ticket = 1000

    def hydrate_from_mt5(self, mt5, magic, feed):
        """Load open MT5 positions with given magic into internal tracker.

        Uses current M1 bars to estimate ATR for stop distances.
        Returns count of hydrated positions.
        """
        import MetaTrader5 as mt5_module
        positions = mt5.positions_get()
        if not positions:
            return 0

        hydrated = 0
        for pos in positions:
            if pos.magic != magic:
                continue
            pair = pos.symbol
            direction = 1 if pos.type == mt5_module.ORDER_TYPE_BUY else -1

            bars = feed.copy_m1_history(pair, count=self.config.get("atr_window", 20))
            atr_v = 0.001
            if bars and len(bars) > 1:
                ranges = [b["high"] - b["low"] for b in bars]
                atr_v = sum(ranges) / len(ranges)

            ticket = self._ticket
            self._ticket += 1
            pip_size = 0.01 if "JPY" in pair else 0.0001
            min_s = max(0, self.config.get("min_stop_pips", 0) * pip_size)
            s = max(self.config.get("stop_a", 0.15) * atr_v * random.uniform(0.9, 1.1), min_s)
            tg = self.config.get("trig_a", 0.20) * atr_v * random.uniform(0.9, 1.1)
            gp = self.config.get("gap_a", 0.10) * atr_v * random.uniform(0.9, 1.1)
            stop = pos.price_open - s if direction == 1 else pos.price_open + s

            self.positions[ticket] = {
                "ticket": ticket,
                "pair": pair,
                "direction": direction,
                "entry": pos.price_open,
                "best": pos.price_open,
                "stop": stop,
                "s": s, "tg": tg, "gp": gp,
                "lot_size": pos.volume,
                "entry_time": int(pos.time),
                "timestamp": int(pos.time),
                "skip_until": int(time.time()) + 5,
                "hydrated": True,
                "mt5_ticket": pos.ticket,
            }
            hydrated += 1
        return hydrated

    def add(self, pair, direction, entry_price, atr_v, lot_size=0.1, spread=0, timestamp=0):
        ticket = self._ticket; self._ticket += 1
        pip_size = 0.01 if "JPY" in pair else 0.0001
        min_s = max(spread * 1.0, self.config.get("min_stop_pips", 0) * pip_size)
        s = max(self.config.get("stop_a", 0.15) * atr_v * random.uniform(0.9, 1.1), min_s)
        tg = self.config.get("trig_a", 0.20) * atr_v * random.uniform(0.9, 1.1)
        gp = self.config.get("gap_a", 0.10) * atr_v * random.uniform(0.9, 1.1)
        stop = entry_price - s if direction == 1 else entry_price + s
        now = timestamp or int(time.time())
        self.positions[ticket] = {
            "ticket": ticket, "pair": pair, "direction": direction,
            "entry": entry_price, "best": entry_price, "stop": stop,
            "s": s, "tg": tg, "gp": gp, "lot_size": lot_size,
            "entry_time": now, "timestamp": now,
            "skip_until": now + 5,
        }
        return ticket

    def update(self, bid, ask, timestamp=0):
        closed = []
        now = timestamp or int(time.time())
        for t in list(self.positions.keys()):
            p = self.positions[t]
            if now < p.get("skip_until", 0):
                continue
            price = bid if p["direction"] == 1 else ask
            if p["direction"] == 1:
                if price > p["best"]:
                    p["best"] = price
                    if p["best"] - p["entry"] > p["tg"]:
                        p["stop"] = p["best"] - p["gp"]
                if price <= p["stop"]:
                    closed.append(self.positions.pop(t))
            else:
                if price < p["best"]:
                    p["best"] = price
                    if p["entry"] - p["best"] > p["tg"]:
                        p["stop"] = p["best"] + p["gp"]
                if price >= p["stop"]:
                    closed.append(self.positions.pop(t))
        return closed

    def check_expiry(self, current_time):
        closed = []
        max_hold = self.config.get("max_hold_min", 54) * 60
        for t in list(self.positions.keys()):
            if current_time - self.positions[t]["timestamp"] >= max_hold:
                closed.append(self.positions.pop(t))
        return closed

    def total_count(self):
        return len(self.positions)

    def pair_count(self, pair):
        return sum(1 for p in self.positions.values() if p["pair"] == pair)

    def pair_positions(self, pair):
        return [p for p in self.positions.values() if p["pair"] == pair]
