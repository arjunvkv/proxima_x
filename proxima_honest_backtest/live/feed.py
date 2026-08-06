"""Market data feeds for the live engine.

LiveM5Feed — polls MT5 for all pairs at a 1s cadence, detects new M5 bars by
broker candle timestamp (never modulo), and emits one ALIGNED bar-set per ts.

ReplayFeed — replays parquet bars through the same feed interface so the live
runner can be verified WITHOUT a terminal (decision parity vs backtest).
"""
from __future__ import annotations

import time
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class BaseFeed:
    def wait_for_new_bar(self, timeout: Optional[float] = None) -> Optional[Dict[str, Dict]]:
        raise NotImplementedError

    def current_bars(self) -> Dict[str, Dict]:
        raise NotImplementedError


class ReplayFeed(BaseFeed):
    """Feed over pre-aligned bar records (same shape the backtest engine uses).

    `records` is a list of dicts (from MultiPairBacktestEngine._align_bars) each
    with "time" and per-pair columns. Yields one aligned bar-set per record,
    exactly matching the backtest's per-ts market snapshot.
    """

    def __init__(self, records: List[Dict[str, Any]]) -> None:
        self._records = records
        self._idx = 0

    def wait_for_new_bar(self, timeout: Optional[float] = None) -> Optional[Dict[str, Dict]]:
        if self._idx >= len(self._records):
            return None
        record = self._records[self._idx]
        self._idx += 1
        ts = record["time"]
        bars: Dict[str, Dict] = {}
        for key, val in record.items():
            if key == "time":
                continue
            if key.endswith(("_open", "_high", "_low", "_volume", "_spread")):
                continue
            if val is None or (isinstance(val, float) and math.isnan(val)):
                continue
            bars[key] = {
                "time": ts,
                "open": float(record.get(f"{key}_open", val)),
                "high": float(record.get(f"{key}_high", val)),
                "low": float(record.get(f"{key}_low", val)),
                "close": float(val),
                "spread": float(record.get(f"{key}_spread", 0)),
            }
        return bars if bars else None

    def current_bars(self) -> Dict[str, Dict]:
        return {}


class LiveM5Feed(BaseFeed):
    """Single-threaded 18-pair M5 bar coordinator via the MT5 Python API.

    Polls all pairs once per pass; fires ONE decision per bar timestamp when a
    super-majority of pairs has rolled to a new candle (>= required_pairs).
    Missing pairs are excluded from the emitted bar-set (matching backtest).
    """

    def __init__(
        self,
        pairs: List[str],
        mt5: Any,
        poll_interval_sec: float = 1.0,
        readiness_ratio: float = 0.8,
        max_bar_age_sec: float = 8.0,
    ) -> None:
        self.pairs = pairs
        self.mt5 = mt5
        self.poll_interval = poll_interval_sec
        self.required = max(1, int(math.ceil(readiness_ratio * len(pairs))))
        self.max_bar_age = max_bar_age_sec
        self._last_processed: Optional[datetime] = None
        self._last_poll: Dict[str, Dict] = {}

    @staticmethod
    def _bar_open(bar: Dict[str, Any]) -> datetime:
        ts = bar["time"]
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(int(ts), tz=timezone.utc)
        if isinstance(ts, datetime):
            return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts
        return datetime.fromisoformat(str(ts))

    def _poll(self) -> Dict[str, Dict]:
        bars: Dict[str, Dict] = {}
        for p in self.pairs:
            try:
                rates = self.mt5.copy_rates_from_pos(p, self.mt5.TIMEFRAME_M5, 0, 2)
                if rates is None or len(rates) < 2:
                    continue
                newest = rates[-1]
                bars[p] = {
                    "time": self._bar_open(newest),
                    "open": float(newest["open"]),
                    "high": float(newest["high"]),
                    "low": float(newest["low"]),
                    "close": float(newest["close"]),
                    "spread": float(getattr(newest, "spread", 0) or 0),
                }
            except Exception:
                continue
        return bars

    def wait_for_new_bar(self, timeout: Optional[float] = None) -> Optional[Dict[str, Dict]]:
        deadline = (time.time() + timeout) if timeout else None
        while True:
            bars = self._poll()
            # candidate = the most common bar-open time across available pairs
            times: Dict[str, int] = {}
            for p, b in bars.items():
                t = self._bar_open(b).isoformat()
                times[t] = times.get(t, 0) + 1
            if times:
                candidate_iso = max(times, key=times.get)
                if times[candidate_iso] >= self.required:
                    candidate_ts = datetime.fromisoformat(candidate_iso)
                    if self._last_processed is None or candidate_ts > self._last_processed:
                        aligned = {p: b for p, b in bars.items()
                                   if self._bar_open(b).isoformat() == candidate_iso}
                        self._last_processed = candidate_ts
                        self._last_poll = aligned
                        return aligned
            if deadline is not None and time.time() >= deadline:
                return None
            time.sleep(self.poll_interval)

    def current_bars(self) -> Dict[str, Dict]:
        return dict(self._last_poll)

    @property
    def last_processed(self) -> Optional[datetime]:
        return self._last_processed
