from __future__ import annotations

import numpy as np
import numba
from numpy.typing import NDArray

from config.settings import settings


@numba.jit(nopython=True, cache=True)
def _revisit_freq(price: NDArray[np.float64], window: int, bins: int) -> NDArray[np.float32]:
    n = len(price)
    result = np.zeros(n, dtype=np.float32)
    for i in range(window - 1, n):
        seg = price[i - window + 1 : i + 1]
        lo = seg.min()
        hi = seg.max()
        if hi - lo < 1e-10:
            result[i] = 1.0
            continue
        bin_edges = np.linspace(lo, hi, bins + 1)
        current = price[i]
        count = 0
        for b in range(bins):
            if bin_edges[b] <= current < bin_edges[b + 1]:
                for j in range(window - 1):
                    if bin_edges[b] <= seg[j] < bin_edges[b + 1]:
                        count += 1
                break
        result[i] = count / (window - 1) if window > 1 else 0.0
    return result


@numba.jit(nopython=True, cache=True)
def _dwell_time(price: NDArray[np.float64], window: int, threshold: float) -> NDArray[np.float32]:
    n = len(price)
    result = np.zeros(n, dtype=np.float32)
    for i in range(window - 1, n):
        seg = price[i - window + 1 : i + 1]
        cur = price[i]
        dwell = 0
        for j in range(window - 2, -1, -1):
            if abs(seg[j] - cur) / (abs(cur) + 1e-10) < threshold:
                dwell += 1
            else:
                break
        result[i] = dwell
    return result


@numba.jit(nopython=True, cache=True)
def _escape_velocity(price: NDArray[np.float64], window: int) -> NDArray[np.float32]:
    n = len(price)
    result = np.zeros(n, dtype=np.float32)
    for i in range(window, n):
        zone_mean = np.mean(price[i - window : i])
        cur = price[i]
        diff = cur - zone_mean
        zone_std = np.std(price[i - window : i]) + 1e-10
        result[i] = abs(diff) / zone_std
    return result


@numba.jit(nopython=True, cache=True)
def _return_prob(price: NDArray[np.float64], window: int, threshold: float) -> NDArray[np.float32]:
    n = len(price)
    result = np.full(n, 0.5, dtype=np.float32)
    for i in range(window, n):
        seg = price[i - window : i]
        lo = seg.min()
        hi = seg.max()
        mid = (lo + hi) / 2.0
        cur = price[i]
        if abs(cur - mid) / (abs(mid) + 1e-10) > threshold:
            ahead = min(i + window // 2, n)
            returns = 0
            for j in range(i + 1, ahead):
                if abs(price[j] - mid) / (abs(mid) + 1e-10) < threshold:
                    returns += 1
            result[i] = returns / max(1, ahead - i - 1)
    return result


@numba.jit(nopython=True, cache=True)
def _mass_entropy(mass: NDArray[np.float32], window: int) -> NDArray[np.float32]:
    n = len(mass)
    result = np.zeros(n, dtype=np.float32)
    eps = 1e-10
    for i in range(window - 1, n):
        seg = mass[i - window + 1 : i + 1]
        s = np.sum(seg) + eps
        ent = 0.0
        for j in range(window):
            p = seg[j] / s
            if p > 0:
                ent -= p * np.log(p + eps)
        result[i] = ent / np.log(window + 1)
    return result


class LiquidityMigrationResearch:
    def __init__(self):
        self.window = settings.research.liquidity_window

    def compute_revisit_frequency(self, price: NDArray[np.float64], window: int = settings.research.liquidity_window, bins: int = 30) -> NDArray[np.float32]:
        return _revisit_freq(price.astype(np.float64), min(window, len(price)), bins)

    def compute_dwell_time(self, price: NDArray[np.float64], window: int = settings.research.liquidity_window, threshold_pct: float = 0.001) -> NDArray[np.float32]:
        return _dwell_time(price.astype(np.float64), min(window, len(price)), threshold_pct)

    def compute_escape_velocity(self, price: NDArray[np.float64], window: int = settings.research.liquidity_window) -> NDArray[np.float32]:
        return _escape_velocity(price.astype(np.float64), min(window, len(price)))

    def compute_return_probability(self, price: NDArray[np.float64], window: int = settings.research.liquidity_window) -> NDArray[np.float32]:
        return _return_prob(price.astype(np.float64), min(window, len(price)), 0.01)

    def compute_liquidity_mass(self, revisit: NDArray[np.float32], dwell: NDArray[np.float32]) -> NDArray[np.float32]:
        n = min(len(revisit), len(dwell))
        return (revisit[:n] * dwell[:n]).astype(np.float32)

    def compute_liquidity_flow(self, liquidity_mass: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.diff(liquidity_mass, prepend=liquidity_mass[0:1]).astype(np.float32)

    def compute_migration_vector(self, liquidity_flow: NDArray[np.float32], window: int = 10) -> NDArray[np.float32]:
        n = len(liquidity_flow)
        result = np.zeros(n, dtype=np.float32)
        w = min(window, n)
        for i in range(w - 1, n):
            result[i] = np.mean(liquidity_flow[i - w + 1 : i + 1])
        return result

    def compute_liquidity_entropy(self, liquidity_mass: NDArray[np.float32], window: int = 20) -> NDArray[np.float32]:
        return _mass_entropy(liquidity_mass, min(window, len(liquidity_mass)))

    def compute_all(self, price: NDArray[np.float64]) -> dict:
        revisit = self.compute_revisit_frequency(price)
        dwell = self.compute_dwell_time(price)
        mass = self.compute_liquidity_mass(revisit, dwell)
        flow = self.compute_liquidity_flow(mass)
        return {
            "revisit_frequency": revisit,
            "dwell_time": dwell,
            "escape_velocity": self.compute_escape_velocity(price),
            "return_probability": self.compute_return_probability(price),
            "liquidity_mass": mass,
            "liquidity_flow": flow,
            "migration_vector": self.compute_migration_vector(flow),
            "liquidity_entropy": self.compute_liquidity_entropy(mass),
        }
