import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger("proxima.replay.sampler")

SESSION_HOURS = {
    "ASIA": (0, 9),
    "LONDON": (8, 17),
    "NY": (13, 22),
    "OVERLAP": (13, 17),
    "DEAD": (22, 24),
}


class Sampler:
    def __init__(self, archive, seed: int = 42):
        self._archive = archive
        self._rng = np.random.default_rng(seed)

    def exact_window(self, symbol: str, start: datetime, end: datetime):
        return self._archive.load_range(symbol, start, end)

    def random_window(self, symbol: str, days: int):
        return self._archive.load_random_window(symbol, days, seed=self._next_seed())

    def session_window(self, symbol: str, session: str):
        return self._archive.load_random_session(symbol, session, seed=self._next_seed())

    def regime_window(self, symbol: str, regime: str):
        return self._archive.load_random_regime(symbol, regime, seed=self._next_seed())

    def volatility_bucket(self, symbol: str, bucket: str):
        return self._archive.load_random_volatility(symbol, bucket, seed=self._next_seed())

    def _next_seed(self) -> int:
        return int(self._rng.integers(0, 2 ** 31))

    def reseed(self, seed: int):
        self._rng = np.random.default_rng(seed)
