import hashlib
import logging
import heapq
from collections import defaultdict
from typing import Iterator, Optional

import polars as pl

from replay.replay_clock import ReplayClock

logger = logging.getLogger("proxima.replay.feed")


class ReplayFeed:
    MODE_REALTIME = "REALTIME"
    MODE_ACCELERATED = "ACCELERATED"
    MODE_BURST = "BURST"

    _symbol_rank = {
        "EURJPY": 0,
        "USDJPY": 1,
        "GBPJPY": 2,
        "XAUUSD": 3,
        "EURUSD": 4,
    }

    def __init__(self, clock: ReplayClock, speed: float = 1.0, mode: str = "ACCELERATED"):
        self._clock = clock
        self._speed = speed
        self._mode = mode
        self._symbols: list[str] = []
        self._tick_buffers: dict[str, list[dict]] = {}
        self._tick_cursors: dict[str, int] = {}
        self._merged: list[dict] = []
        self._cursor: int = 0
        self._current_ticks: dict[str, dict] = {}
        self._preloaded: bool = False
        self._total_loaded: int = 0
        self._branches: dict[str, tuple] = {}
        self._ledger = None

    def set_ledger(self, ledger):
        self._ledger = ledger

    def load_symbol(self, symbol: str, df: pl.LazyFrame):
        self._symbols.append(symbol)
        try:
            collected = df.collect()
            ticks = collected.to_dicts()
            for idx, row in enumerate(ticks):
                row["_local_idx"] = idx
            self._tick_buffers[symbol] = ticks
            self._tick_cursors[symbol] = 0
            self._total_loaded += len(ticks)
            logger.info(f"ReplayFeed: Loaded {len(ticks)} ticks for {symbol}")
        except Exception as e:
            logger.error(f"ReplayFeed: Failed to load {symbol}: {e}")
            self._tick_buffers[symbol] = []
            self._tick_cursors[symbol] = 0

    def load_symbol_from_archive(self, symbol: str, archive, start, end):
        df = archive.load_range(symbol, start, end)
        if df is not None:
            self.load_symbol(symbol, df)

    def _merge_sort(self):
        heap = []
        for sym in sorted(self._symbols, key=lambda s: self._symbol_rank.get(s, 99)):
            buf = self._tick_buffers.get(sym, [])
            cur = self._tick_cursors.get(sym, 0)
            rank = self._symbol_rank.get(sym, 99)
            if cur < len(buf):
                tick = buf[cur]
                ts = tick.get("time_sec", tick.get("timestamp", 0))
                tms = tick.get("time_msc")
                if not tms:
                    tms = int(ts * 1000) if ts else 0
                local_idx = tick.get("_local_idx", cur)
                heapq.heappush(heap, (ts, tms, rank, local_idx, sym, tick))
        merged = []
        while heap:
            ts, tms, rank, local_idx, sym, tick = heapq.heappop(heap)
            tick["_event_id"] = hashlib.sha256(
                f"{sym}:{ts}:{tms}:{local_idx}".encode()
            ).hexdigest()[:16]
            merged.append(tick)
            self._tick_cursors[sym] = self._tick_cursors.get(sym, 0) + 1
            buf = self._tick_buffers.get(sym, [])
            next_idx = self._tick_cursors[sym]
            if next_idx < len(buf):
                next_tick = buf[next_idx]
                next_ts = next_tick.get("time_sec", next_tick.get("timestamp", 0))
                next_tms = next_tick.get("time_msc")
                if not next_tms:
                    next_tms = int(next_ts * 1000) if next_ts else 0
                next_local = next_tick.get("_local_idx", next_idx)
                heapq.heappush(heap, (next_ts, next_tms, rank, next_local, sym, next_tick))
        self._merged = merged
        self._cursor = 0
        self._preloaded = True
        logger.info(f"ReplayFeed: Merged {len(merged)} ticks across {len(self._symbols)} symbols")

    def prepare(self):
        if not self._preloaded:
            self._merge_sort()

    def next(self) -> Optional[dict]:
        self.prepare()
        if self._cursor >= len(self._merged):
            return None
        tick = self._merged[self._cursor]
        self._cursor += 1
        sym = tick.get("symbol", "")
        self._current_ticks[sym] = tick
        ts = tick.get("time_sec", tick.get("timestamp", 0))
        if self._mode == self.MODE_BURST:
            self._clock.advance_to(ts)
        elif self._mode in (self.MODE_ACCELERATED, self.MODE_REALTIME):
            self._clock.advance_to(ts)
        if self._ledger is not None:
            self._ledger.add_tick(tick)
        return tick

    def peek(self) -> Optional[dict]:
        self.prepare()
        if self._cursor >= len(self._merged):
            return None
        return self._merged[self._cursor]

    def seek(self, index: int):
        self.prepare()
        self._cursor = max(0, min(index, len(self._merged) - 1))

    def seek_time(self, timestamp: float):
        self.prepare()
        for i, tick in enumerate(self._merged):
            ts = tick.get("time_sec", tick.get("timestamp", 0))
            if ts >= timestamp:
                self._cursor = i
                return
        self._cursor = len(self._merged) - 1

    def branch(self, name: str) -> int:
        self._branches[name] = (self._cursor, dict(self._current_ticks))
        return self._cursor

    def restore_branch(self, name: str) -> bool:
        if name not in self._branches:
            return False
        cursor, ticks = self._branches[name]
        self._cursor = cursor
        self._current_ticks = dict(ticks)
        return True

    def get_by_index(self, index: int) -> Optional[dict]:
        if 0 <= index < len(self._merged):
            return self._merged[index]
        return None

    def current_tick(self, symbol: str) -> Optional[dict]:
        return self._current_ticks.get(symbol)

    def get_range(self, symbol: str, start_ts: int, end_ts: int) -> list[dict]:
        buf = self._tick_buffers.get(symbol, [])
        return [t for t in buf if start_ts <= t.get("time_sec", t.get("timestamp", 0)) <= end_ts]

    def stream(self, symbols: list[str] = None) -> Iterator[dict]:
        while True:
            tick = self.next()
            if tick is None:
                break
            if symbols and tick.get("symbol", "") not in symbols:
                continue
            yield tick

    @property
    def done(self) -> bool:
        return self._cursor >= len(self._merged)

    @property
    def progress(self) -> float:
        if not self._merged:
            return 0.0
        return self._cursor / len(self._merged)

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def total(self) -> int:
        return len(self._merged)

    @property
    def speed(self) -> float:
        return self._speed

    @speed.setter
    def speed(self, value: float):
        self._speed = max(0.1, value)
        self._clock.speed = self._speed

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str):
        if value.upper() in (self.MODE_REALTIME, self.MODE_ACCELERATED, self.MODE_BURST):
            self._mode = value.upper()
