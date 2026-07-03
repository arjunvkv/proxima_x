import time as time_module
import logging
from datetime import datetime

logger = logging.getLogger("proxima.replay.clock")


class ReplayClock:
    def __init__(self, speed_factor: float = 1.0, start_ts: float = 0.0):
        self._speed = speed_factor
        self._virtual_ts: float = start_ts
        self._last_tick_ts: float = start_ts
        self._paused: bool = False
        self._start_wall: float = time_module.time()
        self._start_virtual: float = start_ts

    def reset(self, start_ts: float = 0.0):
        self._virtual_ts = start_ts
        self._last_tick_ts = start_ts
        self._start_wall = time_module.time()
        self._start_virtual = start_ts
        self._paused = False

    def now(self) -> datetime:
        return datetime.fromtimestamp(self._virtual_ts)

    def time(self) -> float:
        return self._virtual_ts

    def sleep(self, seconds: float):
        if seconds <= 0 or self._paused or self._speed <= 0:
            return
        if self._speed < 1000:
            wall = seconds / self._speed
            if wall > 0.001:
                time_module.sleep(min(wall, 60.0))

    def advance_to(self, ts: float):
        if ts > self._virtual_ts:
            self._virtual_ts = ts
            self._last_tick_ts = ts

    def advance_delta(self, delta_sec: float):
        if delta_sec > 0 and not self._paused:
            self._virtual_ts += delta_sec

    @property
    def speed(self) -> float:
        return self._speed

    @speed.setter
    def speed(self, value: float):
        self._speed = max(0.0, value)

    @property
    def elapsed_virtual(self) -> float:
        return self._virtual_ts - self._start_virtual

    @property
    def elapsed_wall(self) -> float:
        return time_module.time() - self._start_wall

    @property
    def is_paused(self) -> bool:
        return self._paused

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    @property
    def date(self) -> str:
        return datetime.fromtimestamp(self._virtual_ts).strftime("%Y-%m-%d %H:%M:%S")
