import numpy as np


class MarketManifold:
    def __init__(self, alpha: float = 0.15):
        self.alpha = alpha
        self.z_prev = None

    def update(self, z: np.ndarray) -> np.ndarray:
        if self.z_prev is None:
            self.z_prev = z.copy()
        else:
            self.z_prev = self.alpha * z + (1.0 - self.alpha) * self.z_prev
        return self.z_prev.copy()

    def update_batch(self, z_series: np.ndarray) -> np.ndarray:
        smoothed = np.zeros_like(z_series)
        self.z_prev = None
        for i in range(len(z_series)):
            smoothed[i] = self.update(z_series[i])
        return smoothed

    def reset(self):
        self.z_prev = None
