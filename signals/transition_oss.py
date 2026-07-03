"""TransitionOSS — production TrOSS module.
Only trades when ECDF crosses bucket boundaries.
cross_threshold=2 means crossing >2 bucket boundaries triggers a trade.
"""
from .outcome_surface_signal import OutcomeSurfaceSignal


class TransitionOSS:
    """Transition-triggered OSS signal generator.

    Returns OSS signal only when ECDF crosses a bucket boundary
    by at least `cross_threshold` buckets.
    """
    def __init__(self, oss: OutcomeSurfaceSignal, cross_threshold: int = 2):
        self._oss = oss
        self.cross_threshold = cross_threshold
        self._prev_bucket = {}

    def update(self, sym: str, ecdf: float) -> int:
        bucket = min(int(ecdf * 10), 9)
        prev = self._prev_bucket.get(sym)
        self._prev_bucket[sym] = bucket
        if prev is None:
            return 0
        diff = abs(bucket - prev)
        if diff >= self.cross_threshold:
            return self._oss.predict(ecdf)
        return 0

    def reset(self, sym: str = None):
        if sym:
            self._prev_bucket.pop(sym, None)
        else:
            self._prev_bucket.clear()
