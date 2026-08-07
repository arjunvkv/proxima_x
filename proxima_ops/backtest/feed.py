"""Feed — data oracle for the generalized backtest engine.

Two grains, one canonical bar stream:
  * BAR feed: MT5 M5 bars read from the audit cache (audit_7_eas/market/<SYM>.pqt)
    — the exact 200-day tape Tokyo_H0 was validated on.
  * TICK feed: tick-level from the Phase-4 tick archive (data/<sym>/ archive parquet)
    via the ReplayFeed, so a tick-base strategy consumes byte-identical canonical
    ticks. Bar extraction from ticks (session-form) is provided for bar-equivalent
    comparison.

Canonical bar = {ts (epoch s, from bar open), open, high, low, close}.
Applies to the CURRENT WORKING COPY (server truth) or the replay cache.
"""
from __future__ import annotations
import os

ROOT = r"C:\Trading\Proxima_X"
BAR_CACHE = os.path.join(ROOT, "audit_7_eas", "market")


def load_bars_cached(symbol: str) -> list[dict]:
    import polars as pl
    df = pl.read_parquet(os.path.join(BAR_CACHE, f"{symbol}.pqt")).sort("time")
    return [{"ts": int(r["time"]), "open": r["open"], "high": r["high"],
             "low": r["low"], "close": r["close"]} for r in df.iter_rows(named=True)]


def build_bars_map(universe: list[str]) -> dict[str, list[dict]]:
    """BAR feed for a whole universe (offline; no terminal)."""
    return {s: load_bars_cached(s) for s in universe}


def build_tick_feed_from_archive(symbol: str, archive_dir: str | None = None):
    """TICK feed for one symbol from a tick archive (Phase 4 ReplayFeed path).

    Returns (symbol, ReplayFeed) — the live engine's canonical tick producer.
    Prefers an existing archive; raises FileNotFoundError if none exists yet.
    """
    import importlib
    replay = importlib.import_module("replay")
    tick_archive = replay.tick_archive
    arch = tick_archive.TickArchive.load(os.path.join(archive_dir or os.path.join(ROOT, "data"), f"{symbol}.parquet"))
    feed = replay.ReplayFeed(archive=arch)
    return symbol, feed


def bars_from_ticks(symbol, feed, timeframe_s: int = 300):
    """Derive canonical M5 bars from a tick feed (for engine parity when needed)."""
    raise NotImplementedError("tick->bar resampling lives in the tick path of the engine")