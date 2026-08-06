"""utils/profiler.py — lightweight wall-clock profiler (restored).

``run_all.py`` imports ``Profiler`` from the package root, so expose it here
and re-export from ``utils/__init__.py``.
"""
import time
from typing import Optional


class Profiler:
    """Stopwatch-style profiler keyed by name."""

    def __init__(self):
        self._starts: dict[str, float] = {}
        self._results: dict[str, float] = {}

    def start(self, name: str) -> None:
        self._starts[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        started = self._starts.pop(name, None)
        elapsed = (time.perf_counter() - started) if started is not None else 0.0
        self._results[name] = elapsed
        return elapsed

    def result(self, name: str) -> Optional[float]:
        return self._results.get(name)

    def summary(self) -> dict:
        return dict(self._results)