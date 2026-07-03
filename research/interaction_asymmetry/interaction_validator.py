from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from research.adaptive_alpha_engine.aae_validator import AAEValidator, HORIZONS, TARGET_ASSETS, TIME_WINDOWS, _zscore, _future_returns


SYNC_STATES = ["FULLY_SYNCHRONIZED", "PARTIALLY_SYNCHRONIZED", "ASYMMETRIC", "FULLY_DIVERGENT"]
PAIRS = [("memory_density", "energy_storage"), ("memory_density", "adaptive_time"), ("energy_storage", "adaptive_time")]


@dataclass
class IAEResult:
    rq_name: str
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)


class InteractionValidator:
    def __init__(self, asset: str = "EURJPY"):
        self.asset = asset
        self.aae = AAEValidator()
        self.data: dict | None = None
        self.signals: dict | None = None
        self.md: NDArray[np.float64] | None = None
        self.es: NDArray[np.float64] | None = None
        self.at: NDArray[np.float64] | None = None
        self.price: NDArray[np.float64] | None = None
        self.fut_ret: NDArray[np.float64] | None = None
        self._zcache: dict[str, NDArray[np.float64]] = {}

    def load(self, asset: str | None = None) -> dict:
        a = asset or self.asset
        self.data = self.aae.load_asset_data(a)
        self.signals = self.aae.compute_signals(self.data)
        self.price = self.signals["price"]
        self.md = np.asarray(self.signals["memory_density"], dtype=np.float64)
        self.es = np.asarray(self.signals["energy_storage"], dtype=np.float64)
        self.at = np.asarray(self.signals["adaptive_time"], dtype=np.float64)
        self._zcache = {}
        horizons_arr = np.array(HORIZONS, dtype=np.int32)
        self.fut_ret = _future_returns(self.price, horizons_arr)
        return self.signals

    def z(self, arr: NDArray[np.float64]) -> NDArray[np.float64]:
        return _zscore(arr.copy())

    def z_var(self, key: str, arr: NDArray[np.float64]) -> NDArray[np.float64]:
        if key not in self._zcache:
            self._zcache[key] = self.z(arr)
        return self._zcache[key]

    def md_z(self) -> NDArray[np.float64]:
        return self.z_var("md", self.md)

    def es_z(self) -> NDArray[np.float64]:
        return self.z_var("es", self.es)

    def at_z(self) -> NDArray[np.float64]:
        return self.z_var("at", self.at)

    def divergence(self, a: NDArray[np.float64], b: NDArray[np.float64], method: str = "difference") -> NDArray[np.float64]:
        if method == "difference":
            return a - b
        elif method == "zscore_difference":
            return self.z(a) - self.z(b)
        elif method == "acceleration_difference":
            da = np.diff(a, prepend=a[0])
            db = np.diff(b, prepend=b[0])
            return self.z(da) - self.z(db)
        raise ValueError(f"Unknown method: {method}")

    def velocity(self, arr: NDArray[np.float64], window: int = 1) -> NDArray[np.float64]:
        if window <= 1:
            return np.diff(arr, prepend=arr[0])
        padded = np.pad(arr, (window, 0), mode="edge")
        return arr - padded[:len(arr)]

    def acceleration(self, arr: NDArray[np.float64]) -> NDArray[np.float64]:
        vel = self.velocity(arr)
        return self.velocity(vel)

    def classify_synchronization(self, md_z: NDArray[np.float64], es_z: NDArray[np.float64], at_z: NDArray[np.float64]) -> NDArray[np.int64]:
        n = len(md_z)
        states = np.zeros(n, dtype=np.int64)
        for i in range(n):
            signs = np.array([np.sign(md_z[i]), np.sign(es_z[i]), np.sign(at_z[i])])
            n_plus = int(np.sum(signs > 0))
            n_minus = int(np.sum(signs < 0))
            if n_plus == 3 or n_minus == 3:
                states[i] = 0
            elif n_plus == 2 or n_minus == 2:
                states[i] = 1
            elif n_plus == 1 or n_minus == 1:
                states[i] = 2
            else:
                states[i] = 3
        return states

    def eval_alpha(self, signal: NDArray[np.float64], horizon_idx: int = 2) -> dict:
        return self.aae.eval_alpha(signal, self.fut_ret, horizon_idx)

    def decile_alpha(self, signal: NDArray[np.float64], horizon_idx: int = 2) -> dict:
        return self.aae.eval_alpha(signal, self.fut_ret, horizon_idx)

    def benchmark_es_alpha(self) -> dict:
        n = min(len(self.es), len(self.fut_ret))
        sig = self.es[:n]
        return self.eval_alpha(sig, 2)

    def detect_leader(self, md_z: NDArray[np.float64], es_z: NDArray[np.float64], at_z: NDArray[np.float64]) -> NDArray[np.int64]:
        n = len(md_z)
        leader = np.zeros(n, dtype=np.int64)
        abs_md = np.abs(md_z)
        abs_es = np.abs(es_z)
        abs_at = np.abs(at_z)
        for i in range(n):
            vals = [(abs_md[i], 0), (abs_es[i], 1), (abs_at[i], 2)]
            leader[i] = max(vals, key=lambda x: x[0])[1]
        return leader

    def detect_contradictions(self, md_z: NDArray[np.float64], es_z: NDArray[np.float64], at_z: NDArray[np.float64]) -> NDArray[np.int64]:
        n = len(md_z)
        ctype = np.zeros(n, dtype=np.int64)
        for i in range(n):
            smd = np.sign(md_z[i])
            ses = np.sign(es_z[i])
            sat = np.sign(at_z[i])
            disagree = 0
            if smd != ses:
                disagree += 1
            if smd != sat:
                disagree += 1
            if ses != sat:
                disagree += 1
            ctype[i] = disagree
        return ctype

    def tension_index(self, md_z: NDArray[np.float64], es_z: NDArray[np.float64], at_z: NDArray[np.float64], window: int = 20) -> NDArray[np.float64]:
        n = len(md_z)
        tension = np.zeros(n, dtype=np.float64)
        for i in range(window, n):
            w_md = md_z[i - window:i]
            w_es = es_z[i - window:i]
            w_at = at_z[i - window:i]
            cov_sum = 0.0
            for a, b in [(w_md, w_es), (w_md, w_at), (w_es, w_at)]:
                cov_sum += abs(float(np.cov(a, b)[0, 1]))
            tension[i] = cov_sum / 3.0
        return tension

    def interaction_pressure(self, md_z: NDArray[np.float64], es_z: NDArray[np.float64], at_z: NDArray[np.float64], window: int = 20) -> NDArray[np.float64]:
        n = len(md_z)
        pressure = np.zeros(n, dtype=np.float64)
        for i in range(window, n):
            d1 = abs(np.nanmean(md_z[i - window:i]) - np.nanmean(es_z[i - window:i]))
            d2 = abs(np.nanmean(md_z[i - window:i]) - np.nanmean(at_z[i - window:i]))
            d3 = abs(np.nanmean(es_z[i - window:i]) - np.nanmean(at_z[i - window:i]))
            pressure[i] = d1 + d2 + d3
        return pressure
