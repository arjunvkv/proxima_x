"""V2+z Bar — bar-level paper trade matching hfdf_m1 backtest exactly.
Uses M1 bar completions (via copy_rates_from from history server) for entries and trailing stop checks.
No tick-level PairState/TSM — eliminates intra-bar noise gap (35-41% WR on ticks → 76-79% on bars).
"""
from paper_trade.core.config import register
import MetaTrader5 as _mt5
import os

_Z_LOG_FILE = os.environ.get("V2Z_ZLOG", "")

def _zlog(line):
    if _Z_LOG_FILE:
        with open(_Z_LOG_FILE, "a") as f:
            f.write(line + "\n")

STRATEGY_NAME = "v2z_bar"

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
    "max_daily_loss": 1250,
    "lot_size": 1.0,
    "z_thresh": 2.5,
    "stop_a": 3.0,
    "trig_a": 1.0,
    "gap_a": 0.05,
    "max_hold_min": 54,
    "z_window": 50,
    "atr_window": 20,
    "settle_seconds": 1,
}
register(STRATEGY_NAME, CONFIG)


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


class PerPairState:
    def __init__(self, pair, config):
        self.pair = pair
        self.cfg = config
        self.z_buf = ZBuffer(window=config.get("z_window", 50))
        self.atr_buf = ATRBuffer(window=config.get("atr_window", 20))
        self.last_close = None

    def seed_bar(self, bar):
        if self.last_close is not None:
            self.z_buf.add(bar["close"] - self.last_close)
        self.atr_buf.add(bar["high"] - bar["low"])
        self.last_close = bar["close"]

    def on_bar(self, bar):
        if self.last_close is None:
            self.atr_buf.add(bar["high"] - bar["low"])
            self.last_close = bar["close"]
            return None
        ret = bar["close"] - self.last_close
        z_v = self.z_buf.z_score(ret)
        atr_v = self.atr_buf.value()
        _zlog(f"Z,{self.pair},{bar['time']},{z_v},{atr_v},{ret}")
        signal = None
        if z_v is not None and atr_v is not None:
            zt = self.cfg.get("z_thresh", 2.5)
            if abs(z_v) >= zt:
                direction = -1 if z_v > 0 else 1
                signal = {
                    "pair": self.pair, "direction": direction,
                    "confidence": min(1.0, abs(z_v) / 5.0),
                    "atr": atr_v, "z_score": z_v, "bar_time": bar["time"],
                    "entry_price": bar["close"],
                }
        self.z_buf.add(ret)
        self.atr_buf.add(bar["high"] - bar["low"])
        self.last_close = bar["close"]
        return signal


_states = {}
_last_minute = None
_last_bars = {}


def seed_history(feed):
    for pair in CONFIG["pairs"]:
        ps = PerPairState(pair, CONFIG)
        bars = feed.copy_m1_history(pair, count=100)
        if bars:
            for b in bars:
                ps.seed_bar(b)
        _states[pair] = ps


def generate_signal(data, current_time=None):
    """Returns (signals_list, bar_data). signals_list is empty [] on no signal.

    Processes ALL pairs per call — each pair gets on_bar called exactly once.
    """
    global _last_minute, _last_bars
    import time as _time
    now = current_time or int(_time.time())
    current_minute = now // 60

    if _last_minute is None:
        _last_minute = current_minute
        _last_bars = {}
        return [], {}

    if current_minute <= _last_minute:
        return [], {}

    gap_minutes = current_minute - _last_minute
    _last_minute = current_minute

    bars = {}
    bar_time = (now // 60) * 60 - 60  # last completed bar in UTC

    # Gap > 1 minute (weekend break, data gap): reset per-pair last_close
    # so on_bar skips z-score for the first post-gap bar — prevents false
    # signals from Friday → Sunday price gap.
    if gap_minutes > 1:
        for ps in _states.values():
            ps.last_close = None
    for pair in CONFIG["pairs"]:
        if pair not in _states:
            _states[pair] = PerPairState(pair, CONFIG)
        rates = _mt5.copy_rates_from(pair, _mt5.TIMEFRAME_M1, bar_time, 1)
        if rates is not None and len(rates) > 0:
            bar = {
                "open": float(rates[0][1]),
                "high": float(rates[0][2]),
                "low": float(rates[0][3]),
                "close": float(rates[0][4]),
                "time": int(rates[0][0]),
            }
            bars[pair] = bar

    _last_bars = bars

    signals = []
    for pair in CONFIG["pairs"]:
        bar = bars.get(pair)
        if bar is None:
            continue
        signal = _states[pair].on_bar(bar)
        if signal is not None:
            signals.append(signal)
    return signals, bars

    return None, bars


def get_last_bars():
    return _last_bars


class BarStopManager:
    def __init__(self, config):
        self.cfg = config
        self.positions = {}

    def add(self, pair, direction, entry_price, atr_v, entry_time):
        s = self.cfg.get("stop_a", 0.15) * atr_v
        tg = self.cfg.get("trig_a", 0.20) * atr_v
        gp = self.cfg.get("gap_a", 0.10) * atr_v
        stop = entry_price - s if direction == 1 else entry_price + s
        self.positions[pair] = {
            "pair": pair, "direction": direction,
            "entry": entry_price, "best": entry_price, "stop": stop,
            "atr": atr_v, "s": s, "tg": tg, "gp": gp,
            "entry_time": entry_time, "bars_held": 0,
        }

    def remove(self, pair):
        return self.positions.pop(pair, None)

    def check_stops(self, bar_data, current_time):
        closed = []
        max_bars = self.cfg.get("max_hold_min", 54)

        for pair in list(self.positions.keys()):
            pos = self.positions.get(pair)
            if pos is None:
                continue
            bar = bar_data.get(pair)
            if bar is None:
                continue

            # Skip the entry bar — backtest starts stop checks from the
            # bar AFTER entry. Checking the entry bar's OHLC would let
            # intra-bar swings trigger immediate stops on bar-level logic.
            if bar["time"] <= pos["entry_time"]:
                continue

            pos["bars_held"] += 1
            direction = pos["direction"]
            entry = pos["entry"]
            h = bar["high"]
            l_ = bar["low"]

            stop_hit = False
            if direction == 1:
                if h > pos["best"]:
                    pos["best"] = h
                if pos["best"] - entry > pos["tg"]:
                    pos["stop"] = pos["best"] - pos["gp"]
                if l_ <= pos["stop"]:
                    closed.append({**pos, "exit": pos["stop"],
                                   "exit_time": bar["time"], "exit_reason": "stop"})
                    stop_hit = True
            else:
                if l_ < pos["best"]:
                    pos["best"] = l_
                if entry - pos["best"] > pos["tg"]:
                    pos["stop"] = pos["best"] + pos["gp"]
                if h >= pos["stop"]:
                    closed.append({**pos, "exit": pos["stop"],
                                   "exit_time": bar["time"], "exit_reason": "stop"})
                    stop_hit = True

            if stop_hit:
                del self.positions[pair]
            elif pos["bars_held"] >= max_bars:
                closed.append({**pos, "exit": bar["close"],
                               "exit_time": bar["time"], "exit_reason": "expiry"})
                del self.positions[pair]

        return closed

    def total_count(self):
        return len(self.positions)

    def pair_count(self, pair):
        return 1 if pair in self.positions else 0
