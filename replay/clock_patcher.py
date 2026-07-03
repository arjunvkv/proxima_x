"""
Global clock patcher for replay mode.
Monkey-patches time.time and time.sleep to use ReplayClock.
datetime.now is not patched (Python 3.11+ type is immutable).
"""
import time as time_module
import logging

logger = logging.getLogger("proxima.replay.clock_patcher")

_original_time = time_module.time
_original_sleep = time_module.sleep
_original_monotonic = time_module.monotonic

_patched_clock = None


def _patched_time() -> float:
    if _patched_clock is not None:
        return _patched_clock.time()
    return _original_time()


def _patched_sleep(seconds: float):
    if _patched_clock is not None and seconds > 0:
        _patched_clock.sleep(seconds)
        return
    _original_sleep(seconds)


def patch_clock(clock):
    global _patched_clock
    _patched_clock = clock
    time_module.time = _patched_time
    time_module.sleep = _patched_sleep
    logger.info(f"Clock patched: time(), sleep() -> ReplayClock (speed={clock.speed}x)")


def unpatch_clock():
    global _patched_clock
    _patched_clock = None
    time_module.time = _original_time
    time_module.sleep = _original_sleep
    logger.info("Clock unpatched: restored original time functions")


class ClockPatcher:
    def __init__(self, clock):
        self._clock = clock

    def __enter__(self):
        patch_clock(self._clock)
        return self

    def __exit__(self, *args):
        unpatch_clock()
