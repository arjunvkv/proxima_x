"""NOVA feed — load the bars cache (audit_7_eas/market/<SYM>.pqt) into numpy.

Schema in cache: time:Int64 (bar-open epoch s), open/high/low/close:Float64,
sorted by time. NOVA keeps one dict per symbol of aligned numpy arrays.
"""
import os

import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, "audit_7_eas", "market")


def load_one(symbol: str) -> dict:
    """Load a single symbol -> {ts, open, high, low, close} numpy arrays."""
    path = os.path.join(CACHE, f"{symbol}.pqt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no cached bars for {symbol}: {path}")
    df = pl.read_parquet(path).sort("time")
    return {
        "ts": df["time"].to_numpy().astype(np.int64),
        "open": df["open"].to_numpy().astype(np.float64),
        "high": df["high"].to_numpy().astype(np.float64),
        "low": df["low"].to_numpy().astype(np.float64),
        "close": df["close"].to_numpy().astype(np.float64),
    }


def load_many(symbols: list[str]) -> dict[str, dict]:
    """Load several symbols -> {sym: bars dict}. Missing symbols raise."""
    return {s: load_one(s) for s in symbols}


def bars_list_to_arrays(bars: list[dict]) -> dict:
    """Convert legacy list-of-dict bars (ts/open/high/low/close) to arrays."""
    n = len(bars)
    out = {}
    for key in ("ts", "open", "high", "low", "close"):
        out[key] = np.array([b[key] for b in bars], dtype=np.float64 if key != "ts" else np.int64)
    return out
