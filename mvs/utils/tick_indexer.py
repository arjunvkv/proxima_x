"""mvs.utils.tick_indexer — monotonic tick sequence bookkeeping."""
from __future__ import annotations

from typing import Dict, List, Tuple


class TickIndexer:
    """Tracks the ts_ns -> tick_id mapping and per-symbol tick ordering.

    Used by the MVS engine to keep a consistent, queryable tick timeline.
    """

    __slots__ = ("_index", "_tick_ids", "_count")

    def __init__(self) -> None:
        self._index: Dict[int, int] = {}   # ts_ns -> tick_id
        self._tick_ids: List[int] = []
        self._count = 0

    def add(self, ts_ns: int, tick_id: int) -> None:
        self._index[int(ts_ns)] = int(tick_id)
        self._tick_ids.append(int(tick_id))
        self._count += 1

    def get(self, ts_ns: int) -> int:
        return self._index.get(int(ts_ns), -1)

    def tick_ids(self) -> List[int]:
        return list(self._tick_ids)

    def range(self, start_ts_ns: int, end_ts_ns: int) -> List[Tuple[int, int]]:
        out = []
        for ts_ns, tick_id in self._index.items():
            if start_ts_ns <= ts_ns <= end_ts_ns:
                out.append((ts_ns, tick_id))
        out.sort(key=lambda x: x[0])
        return out

    @property
    def count(self) -> int:
        return self._count

    def reset(self) -> None:
        self._index.clear()
        self._tick_ids.clear()
        self._count = 0