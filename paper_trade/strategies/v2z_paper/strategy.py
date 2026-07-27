"""V2+z Paper — z-threshold filtered V2 (fade every M1 bar) with trailing stop.
Validated across 26 pairs, 2 data sources, 3 time splits.
"""
import time, random
from paper_trade.core.config import register
from collections import deque

STRATEGY_NAME = "v2z_paper"

CONFIG = {
    "name": STRATEGY_NAME,
    "mt5_account": None,
    "mt5_path": None,
    "pairs": ["GBPNZD", "EURNZD", "GBPAUD", "EURAUD", "GBPCAD", "AUDNZD"],
    "hold_bars": 0,
    "session_start": 0,
    "session_end": 24,
    "max_concurrent": 3,
    "max_spread_mult": 2.0,
    "max_spread_pips": 2.0,
    "max_daily_loss": 1250,
    "lot_size": 1.0,
    "z_thresh": 2.5,
    "stop_a": 1.5,
    "trig_a": 0.5,
    "gap_a": 0.03,
    "max_hold_min": 54,
    "tp_mult": 0.5,
    "min_tp_pips": 5.0,
    "z_window": 50,
    "atr_window": 20,
    "entry_offset_s": 0,
    "entry_offset_jitter": 0,
    "min_stop_pips": 0.0,
    "randomize_stops": False,
}
register(STRATEGY_NAME, CONFIG)

# ─── Rolling computation buffers ──────────────────────────────

class ZBuffer:
    def __init__(self, window=50):
        self.w = window
        self.buf = []

    def add(self, v):
        self.buf.append(v)
        if len(self.buf) > self.w:
            self.buf.pop(0)

    def z_score(self, value):
        n = len(self.buf)
        if n < self.w * 0.8:
            return None
        m = sum(self.buf) / n
        var = sum((x - m) ** 2 for x in self.buf) / (n - 1)
        if var < 1e-10:
            return None
        return (value - m) / (var ** 0.5)

    def std(self):
        n = len(self.buf)
        if n < self.w * 0.5:
            return None
        m = sum(self.buf) / n
        var = sum((x - m) ** 2 for x in self.buf) / (n - 1)
        return var ** 0.5 if var > 1e-10 else 0.0


class ATRBuffer:
    def __init__(self, window=20):
        self.w = window
        self.buf = []

    def add(self, r):
        self.buf.append(r)
        if len(self.buf) > self.w:
            self.buf.pop(0)

    def value(self):
        if len(self.buf) < self.w * 0.5:
            return None
        return sum(self.buf) / len(self.buf)


# ─── Bar builder: ticks → M1 bars ─────────────────────────────

class BarBuilder:
    def __init__(self):
        self.minute = None
        self.open = None
        self.high = None
        self.low = None
        self.close = None

    def update(self, price, tick_time):
        tick_min = tick_time // 60
        if self.minute is None:
            self.minute = tick_min
            self.open = self.high = self.low = self.close = price
            return False, None
        if tick_min == self.minute:
            self.high = max(self.high, price)
            self.low = min(self.low, price)
            self.close = price
            return False, None
        bar = {"open": self.open, "high": self.high, "low": self.low,
               "close": self.close, "time": self.minute * 60}
        self.minute = tick_min
        self.open = self.high = self.low = self.close = price
        return True, bar


# ─── Per-pair state ───────────────────────────────────────────

class PairState:
    def __init__(self, pair, config):
        self.pair = pair
        self.cfg = config
        self.z_buf = ZBuffer(window=config.get("z_window", 50))
        self.atr_buf = ATRBuffer(window=config.get("atr_window", 20))
        self.last_close = None
        self.last_bar_time = None
        self.bar_builder = BarBuilder()
        self.seed_count = 0

    def seed_bar(self, bar):
        if self.last_close is not None:
            ret = bar["close"] - self.last_close
            self.z_buf.add(ret)
        r = bar["high"] - bar["low"]
        self.atr_buf.add(r)
        self.last_close = bar["close"]
        self.last_bar_time = bar["time"]
        self.seed_count += 1

    def update(self, price, tick_time):
        completed, bar = self.bar_builder.update(price, tick_time)
        if not completed:
            return None, None
        if self.last_close is None:
            r = bar["high"] - bar["low"]
            self.atr_buf.add(r)
            self.last_close = bar["close"]
            return bar, None
        ret = bar["close"] - self.last_close
        z_v = self.z_buf.z_score(ret)
        atr_v = self.atr_buf.value()
        signal = None
        if z_v is not None and atr_v is not None:
            zt = self.cfg.get("z_thresh", 1.0)
            if abs(z_v) >= zt:
                direction = -1 if z_v > 0 else 1
                return_std = self.z_buf.std()
                tp_mult = self.cfg.get("tp_mult", 0.5)
                min_tp = self.cfg.get("min_tp_pips", 5.0) * 0.0001
                tp_price = None
                if return_std is not None and return_std > 0:
                    tp_distance = max(tp_mult * abs(z_v) * return_std, min_tp)
                    tp_price = bar["close"] + direction * tp_distance
                signal = {
                    "pair": self.pair, "direction": direction,
                    "confidence": min(1.0, abs(z_v) / 5.0),
                    "atr": atr_v, "z_score": z_v, "bar_time": bar["time"],
                    "tp_price": tp_price,
                }
        self.z_buf.add(ret)
        r = bar["high"] - bar["low"]
        self.atr_buf.add(r)
        self.last_close = bar["close"]
        return bar, signal


_states = {}
_pair_atr = {}

def seed_history(feed):
    for pair in CONFIG["pairs"]:
        ps = PairState(pair, CONFIG)
        bars = feed.copy_m1_history(pair, count=100)
        if bars:
            for b in bars:
                ps.seed_bar(b)
        _states[pair] = ps


def generate_signal(data, current_time=None):
    ts = current_time or int(time.time())
    completed_bars = {}
    signals = []
    for pair, p in data.items():
        if pair not in _states:
            _states[pair] = PairState(pair, CONFIG)
        bid = p.get("bid", 0)
        if bid <= 0:
            continue
        tick_time = p.get("time", ts)
        bar, signal = _states[pair].update(bid, tick_time)
        if bar is not None:
            completed_bars[pair] = bar
        if signal is not None:
            delay = (CONFIG.get("entry_offset_s", 0)
                     + random.randint(0, CONFIG.get("entry_offset_jitter", 0)))
            signal["delay_s"] = delay
            signals.append(signal)
    if signals:
        signals.sort(key=lambda s: abs(s.get("z_score", 0)), reverse=True)
    return signals, completed_bars


# ─── Trailing stop manager ────────────────────────────────────

class TrailingStopManager:
    def __init__(self, config):
        self.positions = {}
        self.config = config
        self._ticket = 1000

    def add(self, pair, direction, entry_price, atr_v, lot_size=1.0, spread=0, timestamp=0, tp_price=None, mt5_ticket=None):
        ticket = self._ticket; self._ticket += 1
        rand = lambda: random.uniform(0.9, 1.1) if self.config.get("randomize_stops", False) else 1.0
        s = self.config.get("stop_a", 0.15) * atr_v * rand()
        tg = self.config.get("trig_a", 0.20) * atr_v * rand()
        gp = self.config.get("gap_a", 0.10) * atr_v * rand()
        min_s = self.config.get("min_stop_pips", 0.0)
        if min_s > 0:
            pip_size = 0.01 if "JPY" in pair else 0.0001
            s = max(s, min_s * pip_size)
        stop = entry_price - s if direction == 1 else entry_price + s
        now = timestamp or int(time.time())
        self.positions[ticket] = {
            "ticket": ticket, "pair": pair, "direction": direction,
            "entry": entry_price, "best": entry_price, "stop": stop,
            "s": s, "tg": tg, "gp": gp, "lot_size": lot_size,
            "entry_time": now, "timestamp": now, "skip_until": now + 1,
            "tp_price": tp_price, "mt5_ticket": mt5_ticket,
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

    def check_bars(self, pair, bar, timestamp=0):
        """Trailing stop check using completed bar OHLC (matches backtest)."""
        closed = []
        now = timestamp or int(time.time())
        for t in list(self.positions.keys()):
            p = self.positions[t]
            if p["pair"] != pair:
                continue
            if now < p.get("skip_until", 0):
                continue
            # Skip entry bar — matches backtest check_stops
            if bar["time"] <= p["entry_time"]:
                continue
            h, l_, entry = bar["high"], bar["low"], p["entry"]
            if p["direction"] == 1:
                if h > p["best"]:
                    p["best"] = h
                if p["best"] - entry > p["tg"]:
                    p["stop"] = p["best"] - p["gp"]
                if l_ <= p["stop"]:
                    closed.append({**self.positions.pop(t), "exit": p["stop"]})
            else:
                if l_ < p["best"]:
                    p["best"] = l_
                if entry - p["best"] > p["tg"]:
                    p["stop"] = p["best"] + p["gp"]
                if h >= p["stop"]:
                    closed.append({**self.positions.pop(t), "exit": p["stop"]})
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
