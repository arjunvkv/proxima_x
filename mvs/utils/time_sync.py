"""mvs.utils.time_sync — server/local clock offset tracking.

Reconciles MT5 server timestamps with local monotonic time so tick-derived
velocities and latencies stay meaningful when the local clock drifts.
"""
from __future__ import annotations

from typing import Optional


class TimeSync:
    __slots__ = ("_offset_ns", "_samples", "_max_samples", "_synced")

    def __init__(self, max_samples: int = 100) -> None:
        self._offset_ns = 0
        self._samples: list = []
        self._max_samples = max(1, max_samples)
        self._synced = False

    def observe(self, server_ts_ns: int, local_ts_ns: int) -> None:
        """Record a server/local timestamp pair and update the offset."""
        offset = int(server_ts_ns) - int(local_ts_ns)
        self._samples.append(offset)
        if len(self._samples) > self._max_samples:
            self._samples.pop(0)
        self._offset_ns = sum(self._samples) // len(self._samples)
        self._synced = True

    def to_local(self, server_ts_ns: int) -> int:
        return int(server_ts_ns) - self._offset_ns

    def to_server(self, local_ts_ns: int) -> int:
        return int(local_ts_ns) + self._offset_ns

    @property
    def offset_ns(self) -> int:
        return self._offset_ns

    @property
    def synced(self) -> bool:
        return self._synced

    def reset(self) -> None:
        self._offset_ns = 0
        self._samples.clear()
        self._synced = False