import logging
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Set
import math

logger = logging.getLogger("proxima_demo")

EVENT_TYPES = ["SUCCESS", "FAILURE", "RUPTURE"]


class PathMemoryEngine:
    def __init__(self, window: int = 5):
        self._window = window
        self._paths: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._path_counts: Dict[tuple, int] = defaultdict(int)
        self._path_events: Dict[tuple, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._path_latency: Dict[tuple, List[int]] = defaultdict(list)
        self._total_events: int = 0

    def update(self, symbol: str, state: tuple):
        self._paths[symbol].append(state)

    def current_path(self, symbol: str) -> tuple:
        p = tuple(self._paths.get(symbol, []))
        return p

    def record_event(self, symbol: str, event_type: str):
        path = self.current_path(symbol)
        if not path:
            return
        self._path_counts[path] += 1
        self._path_events[path][event_type] += 1
        self._total_events += 1
        latency = len(path)
        self._path_latency[path].append(latency)
        logger.info(f"[PATH_EVENT] {symbol} path={path} "
                    f"event={event_type} occurrences={self._path_counts[path]}")

    def path_probability(self, path: tuple) -> float:
        if self._total_events == 0:
            return 0.0
        return self._path_counts.get(path, 0) / self._total_events

    def event_probability(self, path: tuple, event_type: str) -> float:
        path_events = self._path_events.get(path, {})
        total = sum(path_events.values())
        if total == 0:
            return 0.0
        return path_events.get(event_type, 0) / total

    def path_similarity(self, path_a: tuple, path_b: tuple) -> float:
        if not path_a and not path_b:
            return 1.0
        if not path_a or not path_b:
            return 0.0
        max_len = max(len(path_a), len(path_b))
        i = len(path_a) - 1
        j = len(path_b) - 1
        suffix = 0
        while i >= 0 and j >= 0 and path_a[i] == path_b[j]:
            suffix += 1
            i -= 1
            j -= 1
        return round(suffix / max_len, 4)

    def rare_paths(self, threshold: int = 3) -> List[tuple]:
        return [p for p, c in self._path_counts.items() if c < threshold]

    def repeated_paths(self, threshold: int = 5) -> List[tuple]:
        return [p for p, c in self._path_counts.items() if c >= threshold]

    def unique_paths(self) -> int:
        return len(self._path_counts)

    def all_paths(self) -> Dict[tuple, int]:
        return dict(self._path_counts)

    def stats(self) -> dict:
        all_paths = list(self._path_counts.keys())
        unique_p = len(all_paths)
        repeated = self.repeated_paths()
        rare = self.rare_paths()
        mean_len = sum(len(p) for p in all_paths) / max(unique_p, 1)
        return {
            "unique_paths": unique_p,
            "events": self._total_events,
            "mean_length": round(mean_len, 2),
            "repeated_paths": len(repeated),
            "rare_paths": len(rare),
        }
