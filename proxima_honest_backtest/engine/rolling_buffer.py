from __future__ import annotations

from collections import deque
from typing import Dict, List

import numpy as np

from proxima_honest_backtest.engine.types import ReadOnlyView


class RollingBuffer:
    def __init__(self, maxlen: int, columns: List[str]) -> None:
        if maxlen < 1:
            raise ValueError("maxlen must be at least 1")
        if not columns:
            raise ValueError("columns must not be empty")
        self._maxlen = maxlen
        self._columns = list(columns)
        self._col_index: Dict[str, int] = {name: i for i, name in enumerate(self._columns)}
        self._buffer: deque[np.ndarray] = deque(maxlen=maxlen)

    @property
    def maxlen(self) -> int:
        return self._maxlen

    @property
    def columns(self) -> List[str]:
        return list(self._columns)

    def append(self, values: Dict[str, float]) -> None:
        row = np.empty(len(self._columns), dtype=np.float64)
        for name, value in values.items():
            idx = self._col_index.get(name)
            if idx is None:
                raise KeyError(f"Unknown column: {name}")
            row[idx] = float(value)
        self._buffer.append(row)

    def get_window(self, end_idx: int, length: int) -> ReadOnlyView:
        if end_idx < 0 or end_idx >= len(self._buffer):
            raise IndexError(f"end_idx {end_idx} out of range for buffer of length {len(self._buffer)}")
        if length < 1:
            raise ValueError("length must be at least 1")
        start_idx = end_idx - length + 1
        if start_idx < 0:
            raise IndexError(
                f"requested window of length {length} at end_idx {end_idx} "
                f"exceeds available data (only {end_idx + 1} rows)"
            )
        window_data: Dict[str, List[float]] = {col: [] for col in self._columns}
        for i in range(start_idx, end_idx + 1):
            row = self._buffer[i]
            for j, col in enumerate(self._columns):
                window_data[col].append(float(row[j]))
        return ReadOnlyView(window_data)

    def get_column(self, name: str) -> tuple[float, ...]:
        if name not in self._col_index:
            raise KeyError(f"Unknown column: {name}")
        idx = self._col_index[name]
        return tuple(float(row[idx]) for row in self._buffer)

    def latest(self, name: str) -> float:
        if name not in self._col_index:
            raise KeyError(f"Unknown column: {name}")
        if not self._buffer:
            raise IndexError("buffer is empty")
        idx = self._col_index[name]
        return float(self._buffer[-1][idx])

    def __len__(self) -> int:
        return len(self._buffer)

    def is_full(self) -> bool:
        return len(self._buffer) == self._maxlen

    def clear(self) -> None:
        self._buffer.clear()
