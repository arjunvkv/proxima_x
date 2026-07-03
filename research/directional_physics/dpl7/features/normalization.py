import numpy as np


class OnlineNormalizer:
    def __init__(self, dim: int = 8):
        self.mu = np.zeros(dim)
        self.sigma = np.ones(dim)
        self.n = 0

    def normalize(self, x: np.ndarray) -> np.ndarray:
        if self.n == 0:
            self.mu = x.copy()
            self.sigma = np.where(np.abs(x) < 1e-8, 1.0, np.abs(x))
            self.n = 1
            return np.zeros_like(x)
        self.n += 1
        delta = x - self.mu
        self.mu += delta / self.n
        delta2 = x - self.mu
        self.sigma = np.sqrt(((self.n - 1) / self.n) * self.sigma ** 2 + (delta * delta2) / self.n)
        self.sigma = np.where(self.sigma < 1e-8, 1.0, self.sigma)
        return (x - self.mu) / self.sigma

    def reset(self):
        self.mu = np.zeros_like(self.mu)
        self.sigma = np.ones_like(self.sigma)
        self.n = 0


def ecdf_rank(arr: np.ndarray, value: float) -> float:
    if len(arr) == 0:
        return 0.5
    return float(np.sum(arr <= value)) / len(arr)
