"""RingMemoryPool — fixed-capacity circular buffer over structured numpy arrays.

Backs the MVS truth planes (market / perception / action / outcome). Because
this module lives under `mvs/utils/` it was excluded from the repo by the broad
`utils/` gitignore rule, so it must be re-added here for the engine to import.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class RingMemoryPool:
    __slots__ = ("_buf", "_size", "_write", "_count", "_dtype")

    def __init__(self, size: int = 1024, dtype: Optional[np.dtype] = None) -> None:
        self._dtype = np.dtype(dtype) if dtype is not None else np.dtype(float)
        self._size = max(1, int(size))
        self._buf = np.zeros(self._size, dtype=self._dtype)
        self._write = 0
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    @property
    def capacity(self) -> int:
        return self._size

    def append(self, row: np.ndarray) -> int:
        arr = np.asarray(row, dtype=self._dtype)
        self._buf[self._write % self._size] = arr
        self._write += 1
        self._count = min(self._count + 1, self._size)
        return self._write - 1

    def latest(self) -> np.ndarray:
        if self._count == 0:
            return self._buf[self._write % self._size]
        return self._buf[(self._write - 1) % self._size]

    def window(self, n: int) -> np.ndarray:
        n = max(0, min(int(n), self._count))
        if n == 0:
            return self._buf[:0].copy()
        start = (self._write - n) % self._size
        if start + n <= self._size:
            return self._buf[start:start + n].copy()
        return np.concatenate([self._buf[start:], self._buf[:n - (self._size - start)]])

    def view(self, n: Optional[int] = None) -> np.ndarray:
        if n is None:
            n = self._count
        return self.window(n)

    def clear(self) -> None:
        self._write = 0
        self._count = 0