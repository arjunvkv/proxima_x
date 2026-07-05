"""Tick Buffer: rolling 200-tick ring buffer with get_tpi().
Uses MT5 live ticks when connected, falls back to parquet for offline.

Usage:
    from data.tick_buffer import TickBuffer
    buf = TickBuffer()
    tpi = buf.get_tpi("EURUSD")  # returns TPI value for last 200 ticks
"""
import os
import time as _time
import logging
import numpy as np
from collections import defaultdict

# TPI computation window
TPI_WINDOW = 200

def _session_hours(symbol):
    """Return (start_hour, end_hour) UTC trading session for a symbol.
    Based on quote currency:
      XXXJPY → Tokyo (0-8)
      XXXUSD → NY (12-21)
      XXXCAD → NY (12-21)
      Everything else → London (7-16)
    """
    s = symbol.upper()
    if s.endswith("JPY"):
        return (0, 8)
    if s.endswith("USD") or s.endswith("CAD"):
        return (12, 21)
    return (7, 16)

# Session hours (UTC) — dynamically derived for all symbols
SESSION = {}  # populated on first access via _get_session

class TickBuffer:
    """Maintains rolling tick ring buffers per symbol."""

    def __init__(self, max_ticks=TPI_WINDOW + 50):
        self.max_ticks = max_ticks
        self._buffers = {}  # symbol -> { "bid": [], "ask": [], "time": [] }

    def _ensure(self, symbol):
        if symbol not in self._buffers:
            self._buffers[symbol] = {"bid": [], "ask": [], "time": []}

    def append(self, symbol, bid, ask, timestamp):
        if bid is None or ask is None:
            return
        if bid != bid or ask != ask:
            return
        if abs(bid) > 1e12 or abs(ask) > 1e12:
            return
        self._ensure(symbol)
        buf = self._buffers[symbol]
        if len(buf["time"]) > 0 and timestamp <= buf["time"][-1]:
            return
        buf["bid"].append(bid)
        buf["ask"].append(ask)
        buf["time"].append(timestamp)
        if len(buf["bid"]) > self.max_ticks:
            buf["bid"].pop(0)
            buf["ask"].pop(0)
            buf["time"].pop(0)

    def reset(self, symbol=None):
        if symbol is not None:
            self._buffers.pop(symbol, None)
        else:
            self._buffers.clear()

    def get_mid_prices(self, symbol):
        buf = self._buffers.get(symbol)
        if not buf or len(buf["bid"]) < 2:
            return None
        bid = np.array(buf["bid"], dtype=np.float64)
        ask = np.array(buf["ask"], dtype=np.float64)
        return (bid + ask) / 2.0

    def compute_tpi(self, symbol):
        """Compute winsorized magnitude-weighted TPI over last N ticks in buffer.
        Deltas clipped at P5/P95 to prevent single-outlier dominance."""
        mid = self.get_mid_prices(symbol)
        if mid is None or len(mid) < 3:
            return None
        n = len(mid)
        # Use trailing 200 ticks (or all available if less)
        window = min(n, TPI_WINDOW)
        recent = mid[-window:]
        delta_w = np.diff(recent)
        # Winsorize: clip outliers at P5/P95 to prevent single spike dominance
        if len(delta_w) >= 10:
            p5, p95 = np.percentile(delta_w, [5, 95])
            delta_w = np.clip(delta_w, p5, p95)
        sum_up = np.sum(delta_w[delta_w > 1e-8])
        sum_down = np.abs(np.sum(delta_w[delta_w < -1e-8]))
        total_mag = sum_up + sum_down
        if total_mag < 1e-10:
            return 0.0
        return float((sum_up - sum_down) / total_mag)

    def get_tpi(self, symbol):
        """Returns {'tpi': float, 'direction': int, 'confidence': float, 'n_ticks': int} or None."""
        tpi = self.compute_tpi(symbol)
        if tpi is None:
            return None
        return {
            "tpi": tpi,
            "direction": 1 if tpi > 0 else (-1 if tpi < 0 else 0),
            "confidence": abs(tpi),
            "n_ticks": min(len(self._buffers.get(symbol, {}).get("bid", [])), TPI_WINDOW),
        }

    def load_from_mt5(self, symbol, num_ticks=TPI_WINDOW + 50):
        """Batch-initialize buffer from MT5 latest ticks.
        Call once at startup, then use append() per tick.

        Returns number of ticks loaded, or 0 on failure.
        """
        try:
            import MetaTrader5 as mt5
            ticks = mt5.copy_ticks_from(symbol, int(_time.time()) - 7200, num_ticks, mt5.COPY_TICKS_ALL)
            if ticks is None or len(ticks) < 2:
                return 0
            self._ensure(symbol)
            buf = self._buffers[symbol]
            bid = list(ticks["bid"])
            ask = list(ticks["ask"])
            ts = list(ticks["time"])
            n = len(bid)
            self._ensure(symbol)
            for i in range(n):
                buf["bid"].append(bid[i])
                buf["ask"].append(ask[i])
                buf["time"].append(ts[i])
            if len(buf["bid"]) > self.max_ticks:
                buf["bid"] = buf["bid"][-self.max_ticks:]
                buf["ask"] = buf["ask"][-self.max_ticks:]
                buf["time"] = buf["time"][-self.max_ticks:]
            return n
        except Exception:
            return 0

    def feed_from_mt5_tick(self, symbol):
        """Fetch and append the latest MT5 tick for a symbol.
        Call periodically (every 1-10s) to keep buffer current.
        """
        try:
            import MetaTrader5 as mt5
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return False
            self.append(symbol, tick.bid, tick.ask, tick.time)
            return True
        except Exception:
            return False


# ============================================================
# Offline / Research Mode: Load from parquet files
# ============================================================

_PARQUET_CACHE = {}

def _get_tick_data(symbol, stride=5):
    """Load tick data from parquet, cache it. Returns mid prices array."""
    if symbol not in _PARQUET_CACHE:
        import polars as pl
        path = f"C:/Trading/Agentic_Trading/data/ticks/{symbol}_ticks.parquet"
        if not os.path.exists(path):
            logging.error(f"Missing TPI tick file: {path} — TPI will return None for {symbol}")
            return None
        df = pl.read_parquet(path)
        arr = df.to_numpy()
        if stride > 1:
            arr = arr[::stride]
        bid = arr[:, 1].astype(np.float64)
        ask = arr[:, 2].astype(np.float64)
        ts = arr[:, 0].astype(np.int64)
        mid = (bid + ask) / 2.0
        _PARQUET_CACHE[symbol] = {"mid": mid, "ts": ts, "bid": bid, "ask": ask}
    return _PARQUET_CACHE[symbol]


def get_offline_tpi(symbol, tick_idx=-1):
    """Compute winsorized magnitude-weighted TPI over last 200 ticks at a given index.
    tick_idx=-1 means the latest available tick.
    """
    data = _get_tick_data(symbol)
    if data is None:
        return None
    mid = data["mid"]
    n = len(mid)
    if n < TPI_WINDOW:
        return None
    if tick_idx < 0:
        tick_idx = n - 1
    if tick_idx < TPI_WINDOW:
        return None

    window = mid[tick_idx - TPI_WINDOW + 1: tick_idx + 1]
    delta = np.diff(window)
    if len(delta) >= 10:
        p5, p95 = np.percentile(delta, [5, 95])
        delta = np.clip(delta, p5, p95)
    sum_up = np.sum(delta[delta > 1e-8])
    sum_down = np.abs(np.sum(delta[delta < -1e-8]))
    total_mag = sum_up + sum_down
    if total_mag < 1e-10:
        return 0.0
    return float((sum_up - sum_down) / total_mag)


# Cache for historical TPI percentile distribution (computed once, cached to disk)
_TPI_PERCENTILE_CACHE = {}
_DIST_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", ".tpi_cache")

def _ensure_percentile_distribution(symbol, percentile):
    """Compute and cache the full historical |TPI| distribution once per symbol.
    Caches to .npy on disk for fast reload on subsequent runs.
    """
    cache_key = f"{symbol}_{percentile}"
    if cache_key in _TPI_PERCENTILE_CACHE:
        return _TPI_PERCENTILE_CACHE[cache_key]

    cache_path = os.path.join(_DIST_CACHE_DIR, f"{cache_key}.npy")

    # Try loading from disk cache first
    if os.path.exists(cache_path):
        try:
            non_zero = np.load(cache_path)
            _TPI_PERCENTILE_CACHE[cache_key] = non_zero
            return non_zero
        except Exception:
            pass

    data = _get_tick_data(symbol)
    if data is None:
        return None
    mid = data["mid"]
    n = len(mid)
    if n <= TPI_WINDOW:
        return None

    steps = n - TPI_WINDOW
    all_conf = np.zeros(steps, dtype=np.float64)
    for i in range(steps):
        window = mid[i: i + TPI_WINDOW]
        dw = np.diff(window)
        if len(dw) >= 10:
            p5, p95 = np.percentile(dw, [5, 95])
            dw = np.clip(dw, p5, p95)
        su = np.sum(dw[dw > 1e-8])
        sd = np.abs(np.sum(dw[dw < -1e-8]))
        tm = su + sd
        if tm > 1e-10:
            all_conf[i] = abs((su - sd) / tm)

    non_zero = all_conf[all_conf > 0]
    if len(non_zero) == 0:
        _TPI_PERCENTILE_CACHE[cache_key] = None
        return None

    non_zero.sort()
    _TPI_PERCENTILE_CACHE[cache_key] = non_zero

    # Save to disk for future runs
    try:
        os.makedirs(_DIST_CACHE_DIR, exist_ok=True)
        np.save(cache_path, non_zero)
    except Exception:
        pass

    return non_zero


def get_offline_tpi_signal(symbol, tick_idx=-1, percentile=None):
    """Full TPI signal from offline data.
    If percentile is provided, checks if |TPI| >= that percentile of historical values.
    Uses cached percentile distribution (computed once).
    """
    data = _get_tick_data(symbol)
    if data is None:
        return None
    mid = data["mid"]
    n = len(mid)
    if n < TPI_WINDOW:
        return None

    tpi = get_offline_tpi(symbol, tick_idx)
    if tpi is None:
        return None

    confidence = abs(tpi)

    pct = None
    eligible = True
    if percentile is not None and percentile > 0:
        dist = _ensure_percentile_distribution(symbol, percentile)
        if dist is not None and len(dist) > 0:
            pct = float(np.searchsorted(dist, confidence) / len(dist) * 100)
            eligible = pct >= percentile

    # Session check
    ts_sec = float(data["ts"][tick_idx]) / 1_000_000
    hour = (ts_sec // 3600) % 24
    lo, hi = _session_hours(symbol)
    session_ok = lo <= hour < hi

    return {
        "tpi": tpi,
        "direction": 1 if tpi > 0 else (-1 if tpi < 0 else 0),
        "confidence": confidence,
        "percentile": pct,
        "eligible": eligible and session_ok,
        "n_ticks": TPI_WINDOW,
        "session_hour": int(hour),
        "session_ok": session_ok,
    }
