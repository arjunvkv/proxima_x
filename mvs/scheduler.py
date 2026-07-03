from __future__ import annotations

from typing import List, Dict
from mvs.orchestrator import MVSEngine


class MVSScheduler:
    __slots__ = ("engines", "_running")

    def __init__(self, symbols: List[str], db_path: str = "mvs.duckdb") -> None:
        self.engines: Dict[str, MVSEngine] = {sym: MVSEngine(sym, db_path) for sym in symbols}
        self._running = True

    def run_tick(self) -> Dict[str, dict]:
        results = {}
        for sym, engine in self.engines.items():
            try:
                results[sym] = engine.run_tick()
            except Exception as e:
                results[sym] = {"error": str(e)}
        return results

    def run_cycle(self, n_ticks: int, report_interval: int = 100) -> None:
        for i in range(n_ticks):
            results = self.run_tick()
            if (i + 1) % report_interval == 0:
                print(f"Tick {i+1}/{n_ticks}: {len(results)} engines")

    def close_all(self) -> None:
        for engine in self.engines.values():
            engine.close()
