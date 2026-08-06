"""data.ftmo_tick_ingester — pull real FTMO (MT5) ticks into the replay archive.

Phase 4 (replay from the live tape): a strategy must be backtested on the
EXACT tick feed the FTMO account will trade. This module downloads real
broker ticks via MT5 copy_ticks_range and stores them in the replay
TickArchive (parquet) so ReplayFeed -> ReplayTickSource -> engine replays
the true live tape with identical bid/ask/spread.

Contract: every stored tick keeps the raw MT5 fields (time_msc, bid, ask,
last, volume, volume_real, flags) plus canonical spread = ask - bid in price
units, and time_sec/timestamp_ns derived from time_msc. The archive schema
(TICK_SCHEMA) already matches this shape.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import List, Optional

from replay.tick_archive import TickArchive

logger = logging.getLogger("proxima.data.ftmo_ingest")

# Tick flags (MT5): TICK_FLAG_BID=1, TICK_FLAG_ASK=2, TICK_FLAG_LAST=4
FLAG_BID, FLAG_ASK, FLAG_LAST = 1, 2, 4


class FTMOTickIngester:
    """Downloads real broker ticks and stores them in the replay archive."""

    def __init__(self, archive: Optional[TickArchive] = None):
        self._mt5 = None
        self._archive = archive or TickArchive()

    def _ensure_mt5(self):
        if self._mt5 is None:
            import MetaTrader5 as mt5
            self._mt5 = mt5
            if not self._mt5.initialize():
                raise RuntimeError("MT5 initialize() failed — is the terminal running?")
            logger.info(f"MT5 initialized: {self._mt5.account_info().login if self._mt5.account_info() else 'unknown'}")

    @property
    def connected(self) -> bool:
        try:
            self._ensure_mt5()
            return bool(self._mt5.account_info())
        except Exception:
            return False

    def fetch_ticks(self, symbol: str, start: datetime, end: datetime,
                    copy_all: bool = True) -> List[dict]:
        """Fetch raw ticks from the broker (real FTMO tape)."""
        self._ensure_mt5()
        flag = self._mt5.COPY_TICKS_ALL if copy_all else self._mt5.COPY_TICKS_INFO
        ticks = self._mt5.copy_ticks_range(symbol, int(start.timestamp()),
                                           int(end.timestamp()), flag)
        if ticks is None or len(ticks) == 0:
            logger.warning(f"[FTMO_INGEST] No ticks for {symbol} {start}..{end}")
            return []
        out = []
        for t in ticks:
            time_msc = int(getattr(t, "time_msc", 0))
            bid = float(getattr(t, "bid", 0.0))
            ask = float(getattr(t, "ask", 0.0))
            out.append({
                "timestamp_ns": time_msc * 1_000_000,   # ms -> ns
                "time_sec": int(time_msc // 1000),
                "time_msc": time_msc,
                "bid": bid,
                "ask": ask,
                "spread": round(ask - bid, 8),          # canonical: price units
                "last": float(getattr(t, "last", 0.0)),
                "volume": float(getattr(t, "volume", 0.0)),
                "volume_real": float(getattr(t, "volume_real", 0.0)),
                "flags": int(getattr(t, "flags", 0)),
                "symbol": symbol,
            })
        return out

    def ingest_range(self, symbol: str, start: datetime, end: datetime,
                     copy_all: bool = True) -> int:
        """Download ticks for a range and store them in the archive.

        Returns the number of ticks stored (0 on empty).
        """
        ticks = self.fetch_ticks(symbol, start, end, copy_all=copy_all)
        if not ticks:
            return 0
        # Deduplicate: copy_ticks_range can return the same tick twice at
        # range boundaries; the archive already dedups on (timestamp_ns, symbol).
        seen = set()
        uniq = []
        for t in ticks:
            key = (t["timestamp_ns"], t["symbol"])
            if key in seen:
                continue
            seen.add(key)
            uniq.append(t)
        self._archive.store_ticks(symbol, uniq)
        logger.info(f"[FTMO_INGEST] Stored {len(uniq)} ticks for {symbol} "
                    f"{start.date()}..{end.date()}")
        return len(uniq)

    def ingest_days(self, symbol: str, days: int = 30, copy_all: bool = True,
                    chunk_hours: int = 48) -> int:
        """Ingest the last N days of real ticks, chunked to bound memory."""
        end = datetime.now()
        start = end - timedelta(days=days)
        total = 0
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + timedelta(hours=chunk_hours), end)
            total += self.ingest_range(symbol, cursor, chunk_end, copy_all=copy_all)
            cursor = chunk_end
        return total

    def close(self) -> None:
        if self._mt5 is not None:
            try:
                self._mt5.shutdown()
            except Exception:
                pass
            self._mt5 = None
