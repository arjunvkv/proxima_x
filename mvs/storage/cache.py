from __future__ import annotations

import os
import time
import threading
import pickle
from collections import OrderedDict
from typing import Optional


class LRUCache:
    __slots__ = ("_cache", "_max_size", "_ttl", "_lock", "_mmap_path")

    def __init__(self, max_size: int = 100000, ttl_seconds: int = 3600, mmap_path: Optional[str] = None) -> None:
        self._cache: OrderedDict[int, tuple[int, float]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._lock = threading.RLock()
        self._mmap_path = mmap_path

    @property
    def size(self) -> int:
        return len(self._cache)

    def _evict(self) -> None:
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def _expire(self) -> None:
        now = time.time()
        expired = [k for k, (_, ts) in self._cache.items() if now - ts > self._ttl]
        for k in expired:
            del self._cache[k]

    def get(self, key: int) -> Optional[int]:
        with self._lock:
            self._expire()
            if key not in self._cache:
                return None
            value, ts = self._cache.pop(key)
            self._cache[key] = (value, ts)
            return value

    def put(self, key: int, value: int) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.pop(key)
            self._cache[key] = (value, time.time())
            self._evict()

    def delete(self, key: int) -> None:
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def persist(self) -> None:
        if self._mmap_path is None:
            return
        with self._lock:
            with open(self._mmap_path, "wb") as f:
                pickle.dump(self._cache, f)

    def load(self) -> None:
        if self._mmap_path is None:
            return
        if not os.path.exists(self._mmap_path):
            return
        with self._lock:
            with open(self._mmap_path, "rb") as f:
                self._cache = pickle.load(f)

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: int) -> bool:
        return key in self._cache
