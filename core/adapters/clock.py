import time as time_module
import logging
from abc import ABC, abstractmethod
from datetime import datetime

logger = logging.getLogger("proxima.adapters.clock")


class Clock(ABC):

    @abstractmethod
    def now(self) -> datetime:
        pass

    @abstractmethod
    def time(self) -> float:
        pass

    @abstractmethod
    def sleep(self, seconds: float):
        pass

    @abstractmethod
    def advance_to(self, ts: float):
        pass


class RealClock(Clock):
    def now(self) -> datetime:
        return datetime.now()

    def time(self) -> float:
        return time_module.time()

    def sleep(self, seconds: float):
        if seconds > 0:
            time_module.sleep(seconds)

    def advance_to(self, ts: float):
        pass


class ReplayClock(Clock):
    def __init__(self, speed_factor: float = 1.0):
        self._speed = speed_factor
        self._virtual_ts: float = 0.0
        self._last_wall: float = 0.0
        self._paused: bool = False
        self._start_wall: float = 0.0
        self._start_virtual: float = 0.0

    def reset(self, start_ts: float = 0.0):
        self._virtual_ts = start_ts
        self._last_wall = time_module.time()
        self._start_wall = time_module.time()
        self._start_virtual = start_ts
        self._paused = False

    def now(self) -> datetime:
        return datetime.fromtimestamp(self._virtual_ts)

    def time(self) -> float:
        return self._virtual_ts

    def sleep(self, seconds: float):
        if seconds <= 0 or self._paused:
            return
        if self._speed <= 0:
            return
        wall_sleep = seconds / self._speed
        if wall_sleep > 0:
            time_module.sleep(wall_sleep)

    def advance_to(self, ts: float):
        if ts > self._virtual_ts:
            self._virtual_ts = ts
            self._last_wall = time_module.time()

    @property
    def speed(self) -> float:
        return self._speed

    @speed.setter
    def speed(self, value: float):
        self._speed = max(0.1, value)

    @property
    def elapsed_virtual(self) -> float:
        return self._virtual_ts - self._start_virtual

    @property
    def elapsed_wall(self) -> float:
        return time_module.time() - self._start_wall
