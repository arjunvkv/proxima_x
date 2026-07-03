import logging
import numpy as np

logger = logging.getLogger("proxima.replay.execution")


class ExecutionModel:
    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)
        self._decision_latency_ms = (50, 200)
        self._network_latency_ms = (10, 50)
        self._broker_latency_ms = (20, 100)
        self._slippage_bps_range = (0.0, 2.0)
        self._latency_enabled = True
        self._slippage_enabled = True
        self._queue_delay_enabled = False

    def sample_latency(self) -> float:
        if not self._latency_enabled:
            return 0.0
        d = self._rng.uniform(*self._decision_latency_ms)
        n = self._rng.uniform(*self._network_latency_ms)
        b = self._rng.uniform(*self._broker_latency_ms)
        return d + n + b

    def apply_slippage(self, symbol: str, price: float, side: str, urgency: float = 0.5) -> float:
        if not self._slippage_enabled:
            return price
        spread_bps = self._slippage_bps_range[1] * urgency
        slippage_bps = self._rng.uniform(0, spread_bps)
        slippage = price * slippage_bps / 10000.0
        if side.upper() == "BUY":
            return price + slippage
        else:
            return price - slippage

    def set_latency_range(self, decision: tuple, network: tuple, broker: tuple):
        self._decision_latency_ms = decision
        self._network_latency_ms = network
        self._broker_latency_ms = broker

    def set_slippage_bps(self, low: float, high: float):
        self._slippage_bps_range = (low, high)

    @property
    def latency_enabled(self) -> bool:
        return self._latency_enabled

    @latency_enabled.setter
    def latency_enabled(self, value: bool):
        self._latency_enabled = value

    @property
    def slippage_enabled(self) -> bool:
        return self._slippage_enabled

    @slippage_enabled.setter
    def slippage_enabled(self, value: bool):
        self._slippage_enabled = value

    def reseed(self, seed: int):
        self._rng = np.random.default_rng(seed)
