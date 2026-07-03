import numpy as np
from collections import defaultdict, deque


class PerSymbolECDF:
    """
    Streaming volatility-normalized ECDF per symbol.
    Converts raw price values into z-score residuals, then ranks [0,1].
    Uses exponentially-weighted statistics to handle regime shifts.
    """

    def __init__(self, window_size: int = 5000, alpha: float = 0.001):
        self.window_size = window_size
        self.alpha = alpha  # EWMA decay for mean/var
        self._frozen = True
        self.buffers = defaultdict(lambda: deque(maxlen=window_size))
        self._ewma_mean: dict[str, float] = {}
        self._ewma_var: dict[str, float] = {}

    def hydrate(self, symbol: str, prices: list[float]):
        """Bulk-load prices into buffer with EWMA initialization."""
        buf = self.buffers[symbol]
        buf.clear()
        valid = [float(p) for p in prices if p > 0]
        for p in valid:
            buf.append(p)
        if len(valid) > 10:
            arr = np.array(valid)
            self._ewma_mean[symbol] = float(np.mean(arr))
            self._ewma_var[symbol] = float(np.var(arr)) + 1e-10

    def _normalize(self, symbol: str, value: float) -> float:
        """Convert value to volatility-normalized z-score."""
        mean = self._ewma_mean.get(symbol, value)
        std = np.sqrt(self._ewma_var.get(symbol, 1.0))
        return (value - mean) / max(std, 1e-10)

    def _update_ewma(self, symbol: str, value: float):
        """Update EWMA mean and variance with new observation."""
        if symbol not in self._ewma_mean:
            self._ewma_mean[symbol] = value
            self._ewma_var[symbol] = 1.0
            return
        delta = value - self._ewma_mean[symbol]
        self._ewma_mean[symbol] += self.alpha * delta
        self._ewma_var[symbol] = (1 - self.alpha) * (self._ewma_var[symbol] + self.alpha * delta ** 2)

    def compute(self, symbol: str, value: float) -> float:
        """Rank volatility-normalized value against buffer without modifying."""
        buf = self.buffers.get(symbol)
        if buf is None or len(buf) < 10:
            return 0.5
        z = self._normalize(symbol, value)
        arr = np.array(buf, dtype=np.float64)
        arr_z = (arr - self._ewma_mean.get(symbol, np.mean(arr))) / max(np.sqrt(self._ewma_var.get(symbol, np.var(arr))), 1e-10)
        rank = np.sum(arr_z <= z) / len(arr_z)
        return float(rank)

    def update(self, symbol: str, value: float) -> float:
        """Rank and append value, updating EWMA statistics."""
        buf = self.buffers[symbol]
        buf.append(value)
        self._update_ewma(symbol, value)

        if len(buf) < 10:
            return 0.5

        z = self._normalize(symbol, value)
        arr = np.array(buf, dtype=np.float64)
        arr_z = (arr - self._ewma_mean.get(symbol, np.mean(arr))) / max(np.sqrt(self._ewma_var.get(symbol, np.var(arr))), 1e-10)
        rank = np.sum(arr_z <= z) / len(arr_z)

        return float(rank)

    def compute_and_update(self, symbol: str, value: float) -> float:
        """Rank against existing buffer, then append value."""
        buf = self.buffers[symbol]
        if len(buf) < 10:
            buf.append(float(value))
            self._update_ewma(symbol, value)
            return 0.5
        z = self._normalize(symbol, value)
        arr = np.array(buf, dtype=np.float64)
        arr_z = (arr - self._ewma_mean.get(symbol, np.mean(arr))) / max(np.sqrt(self._ewma_var.get(symbol, np.var(arr))), 1e-10)
        rank = np.sum(arr_z <= z) / len(arr_z)
        buf.append(float(value))
        self._update_ewma(symbol, value)
        return float(rank)

    def transform_batch(self, ticks: list[dict]) -> list[dict]:
        out = []
        for t in ticks:
            r = self.update(t["symbol"], t["price"])
            t2 = dict(t)
            t2["ecdf_rank"] = r
            out.append(t2)
        return out
