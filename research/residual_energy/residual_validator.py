from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from research.energy_reality.energy_validator import EnergyValidator, VOL_METRICS, HORIZONS, TARGET_ASSETS, _future_returns


@dataclass
class REPResult:
    rq_name: str
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)


RESIDUAL_TYPES = ["linear", "random_forest", "xgboost"]
BENCHMARKS = ["ES", "ATR Breakout", "Donchian", "Momentum", "Vol Expansion"]
TIME_WINDOWS = [
    ("2018-01-01", "2020-01-01", "2018-2020"),
    ("2020-01-01", "2022-01-01", "2020-2022"),
    ("2022-01-01", "2024-01-01", "2022-2024"),
    ("2024-01-01", "2027-01-01", "2024-2026"),
]


class ResidualEnergyValidator:
    def __init__(self, asset: str = "EURJPY"):
        self.asset = asset
        self.energy = EnergyValidator(asset)
        self.data: dict | None = None
        self.signals: dict | None = None
        self.price: np.ndarray | None = None
        self.es: np.ndarray | None = None
        self.fut_ret: np.ndarray | None = None
        self.residuals: dict[str, np.ndarray] = {}
        self.models: dict[str, Any] = {}
        self._loaded = False

    def load(self, asset: str | None = None) -> dict:
        a = asset or self.asset
        self.signals = self.energy.load(a)
        self.price = self.energy.price
        self.es = self.energy.es_signal()
        self.fut_ret = self.energy.fut_ret
        self.data = self.energy.data
        self._loaded = True
        return self.signals

    def build_residuals(self, force: bool = False):
        if self.residuals and not force:
            return
        self.load()
        vol_metrics = self.energy.vol_metrics
        n = len(self.es)
        X_list = []
        valid = np.ones(n, dtype=bool)
        for name in VOL_METRICS:
            arr = vol_metrics[name]
            X_list.append(arr)
            valid = valid & ~np.isnan(arr)
        valid = valid & ~np.isnan(self.es)
        X = np.column_stack([v[valid] for v in X_list])
        y = self.es[valid]

        configs = [
            ("linear", LinearRegression()),
            ("random_forest", RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)),
            ("xgboost", XGBRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1, verbosity=0)),
        ]

        for name, model in configs:
            model.fit(X, y)
            y_pred = model.predict(X)
            residual_full = np.full(n, np.nan, dtype=np.float64)
            residual_full[valid] = y - y_pred
            self.residuals[name] = residual_full
            self.models[name] = {"model": model, "r2": float(model.score(X, y)), "n_valid": int(np.sum(valid))}

    def get_residual(self, residual_type: str = "xgboost") -> np.ndarray:
        self.build_residuals()
        return self.residuals.get(residual_type, np.full(len(self.es), np.nan))

    def eval_alpha(self, signal: np.ndarray, horizon_idx: int = 2) -> dict:
        return self.energy.eval_alpha(signal, horizon_idx)

    def decile_alpha(self, signal: np.ndarray, horizon_idx: int = 2) -> dict:
        return self.energy.decile_alpha(signal, horizon_idx)

    def es_alpha(self, horizon_idx: int = 2) -> dict:
        return self.eval_alpha(self.es, horizon_idx)

    def residual_alpha(self, residual_type: str = "xgboost", horizon_idx: int = 2) -> dict:
        res = self.get_residual(residual_type)
        return self.eval_alpha(res, horizon_idx)

    def multi_horizon_alpha(self, signal: np.ndarray) -> dict[str, dict]:
        return {str(h): self.eval_alpha(signal, i) for i, h in enumerate(HORIZONS)}

    def correlation(self, a: np.ndarray, b: np.ndarray) -> float:
        valid = ~(np.isnan(a) | np.isnan(b))
        if np.sum(valid) < 10:
            return 0.0
        return float(np.corrcoef(a[valid], b[valid])[0, 1])

    def mutual_info(self, a: np.ndarray, b: np.ndarray, n_neighbors: int = 5) -> float:
        from sklearn.feature_selection import mutual_info_regression
        valid = ~(np.isnan(a) | np.isnan(b))
        if np.sum(valid) < 10:
            return 0.0
        return float(mutual_info_regression(a[valid, np.newaxis], b[valid], n_neighbors=n_neighbors, random_state=42)[0])

    def incremental_info(self, features: list[np.ndarray], target: np.ndarray) -> float:
        from sklearn.feature_selection import mutual_info_regression
        valid = np.ones(len(target), dtype=bool)
        for f in features:
            valid = valid & ~np.isnan(f)
        valid = valid & ~np.isnan(target)
        if np.sum(valid) < 10:
            return 0.0
        X = np.column_stack([f[valid] for f in features])
        y = target[valid]
        return float(mutual_info_regression(X, y, n_neighbors=5, random_state=42).sum())

    def _clean(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): self._clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._clean(v) for v in obj]
        if isinstance(obj, tuple):
            return list(self._clean(v) for v in obj)
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist() if hasattr(obj, 'tolist') else float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj
