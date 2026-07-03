import numpy as np


class StateEncoder:
    def __init__(self, dim: int = 8):
        self.dim = dim
        self._feature_keys = [
            "es_rank", "es_slope", "energy_balance", "energy_creation",
            "energy_release", "energy_efficiency", "time_density", "returns_vol"
        ]

    def encode(self, features: dict) -> np.ndarray:
        raw = np.array([features.get(k, 0.0) for k in self._feature_keys], dtype=np.float64)
        raw = np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=-1.0)
        norm = np.linalg.norm(raw) + 1e-8
        return raw / norm

    @property
    def feature_keys(self):
        return list(self._feature_keys)
