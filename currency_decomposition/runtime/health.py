import time
import psutil
import os
from data.models import HealthStatus

class HealthMonitor:
    def __init__(self):
        self._process = psutil.Process(os.getpid())
        self._solve_times = []
        self._last_check = 0.0
        self.state = HealthStatus()

    def record_solve(self, latency_ms: float) -> None:
        self._solve_times.append(latency_ms)
        if len(self._solve_times) > 50:
            self._solve_times.pop(0)

    def check(self, mt5_ok: bool, tick_freshness: dict, graph_quality: float,
              snapshot_ok: bool) -> HealthStatus:
        now = time.time()

        fresh_ratio = sum(1 for v in tick_freshness.values() if v > 0.5) / max(len(tick_freshness), 1)
        tick_quality = fresh_ratio

        solve_latency = sum(self._solve_times[-10:]) / max(len(self._solve_times[-10:]), 1) if self._solve_times else 0.0
        memory_mb = self._process.memory_info().rss / (1024 * 1024)

        failures = 0
        if not mt5_ok:
            failures += 1
        if tick_quality < 0.3:
            failures += 1
        if graph_quality < 0.4:
            failures += 1
        if not snapshot_ok:
            failures += 1

        if failures >= 2:
            state = "FAILED"
        elif failures >= 1:
            state = "DEGRADED"
        else:
            state = "OK"

        self.state = HealthStatus(
            state=state,
            mt5_ok=mt5_ok,
            tick_quality=tick_quality,
            graph_quality=graph_quality,
            last_snapshot_ok=snapshot_ok,
            solve_latency_ms=solve_latency,
            memory_mb=memory_mb
        )
        return self.state

    def get_status(self) -> HealthStatus:
        return self.state

