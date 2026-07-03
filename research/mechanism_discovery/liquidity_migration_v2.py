from __future__ import annotations

from typing import Any, Optional

import numpy as np
import numba
from numpy.typing import NDArray

from research.mechanism_discovery.base import BaseMechanism, MechanismScore


@numba.jit(nopython=True, cache=True)
def _numba_rolling_entropy(migration_flow: NDArray[np.float64], window: int, entropy_bins: int) -> NDArray[np.float64]:
    n = len(migration_flow)
    roll_entropy = np.zeros(n)
    
    for i in range(window - 1, n):
        seg = migration_flow[i - window + 1 : i + 1]
        
        smin = np.min(seg)
        smax = np.max(seg)
        srange = smax - smin
        if srange < 1e-12:
            roll_entropy[i] = 0.0
            continue
            
        hist = np.zeros(entropy_bins, dtype=np.float64)
        for val in seg:
            bin_idx = int((val - smin) / srange * entropy_bins)
            if bin_idx >= entropy_bins:
                bin_idx = entropy_bins - 1
            elif bin_idx < 0:
                bin_idx = 0
            hist[bin_idx] += 1.0
            
        total = float(window)
        ent = 0.0
        for b in range(entropy_bins):
            p = hist[b] / total
            if p > 0.0:
                ent -= p * np.log(p)
        roll_entropy[i] = ent
        
    return roll_entropy


@numba.jit(nopython=True, cache=True)
def _numba_herfindahl_index(x: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    N = len(x)
    result = np.zeros(N, dtype=np.float64)
    for i in range(window - 1, N):
        seg = x[i - window + 1 : i + 1]
        total = float(np.sum(seg)) + 1e-12
        shares = seg / total
        result[i] = float(np.sum(shares ** 2))
    return result


@numba.jit(nopython=True, cache=True)
def _numba_rolling_mean(x: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    N = len(x)
    result = np.zeros(N, dtype=np.float64)
    if N == 0:
        return result
    cum = np.cumsum(x)
    result[window - 1] = cum[window - 1] / float(window)
    result[window:] = (cum[window:] - cum[:-window]) / float(window)
    return result


@numba.jit(nopython=True, cache=True)
def _numba_rolling_std(x: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    N = len(x)
    result = np.zeros(N, dtype=np.float64)
    if N < 2:
        return result
    cum = np.cumsum(x)
    cum_sq = np.cumsum(x ** 2)
    for i in range(window - 1, N):
        s = cum[i] - (cum[i - window] if i >= window else 0.0)
        sq = cum_sq[i] - (cum_sq[i - window] if i >= window else 0.0)
        var = sq / float(window) - (s / float(window)) ** 2
        result[i] = float(np.sqrt(max(0.0, var)))
    return result


class LiquidityMigrationSystem(BaseMechanism):
    STD_THRESHOLD: float = 1.0
    ENTROPY_BINS: int = 10

    def __init__(self, name: str = "liquidity_migration", category: str = "liquidity_dynamics", window: int = 50) -> None:
        super().__init__(name, category)
        self.window = window
        self._state_contribution: NDArray = np.array([], dtype=np.float64)

    def compute(self, data: dict[str, NDArray], states: Optional[NDArray] = None) -> dict[str, Any]:
        price = np.asarray(data.get("price", []), dtype=np.float64)
        volume = np.asarray(data.get("volume", []), dtype=np.float64)
        high = np.asarray(data.get("high", []), dtype=np.float64)
        low = np.asarray(data.get("low", []), dtype=np.float64)
        returns = np.asarray(data.get("returns", []), dtype=np.float64)

        N = len(price)
        if N < self.window + 1:
            return self._empty_result(N)

        if len(returns) < N:
            returns = np.diff(price, prepend=price[0])
        if len(volume) < N:
            volume = np.ones(N, dtype=np.float64)
        if len(high) < N:
            high = price.copy()
        if len(low) < N:
            low = price.copy()

        liquidity_proxy = volume / (high - low + 1e-10)
        liquidity_efficiency = np.abs(returns) / (volume + 1e-10)
        slippage_proxy = (high - low) / (volume + 1e-10)

        roll_mean = self._rolling_mean(liquidity_proxy, self.window)
        roll_std = self._rolling_std(liquidity_proxy, self.window)

        leaving = (roll_mean - liquidity_proxy) > self.STD_THRESHOLD * roll_std
        arriving = (liquidity_proxy - roll_mean) > self.STD_THRESHOLD * roll_std

        migration_flow = np.gradient(liquidity_proxy)
        migration_velocity = migration_flow.copy()
        migration_acceleration = np.gradient(migration_velocity)

        accumulation_zones = (liquidity_proxy > roll_mean) & (migration_flow > 0)
        disappearance_zones = (liquidity_proxy < roll_mean) & (migration_flow < 0)

        liquidity_concentration = self._herfindahl_index(liquidity_proxy, self.window)
        hotspot_mask = np.zeros(N, dtype=bool)
        if N > self.window:
            conc_mean = float(np.mean(liquidity_concentration[self.window - 1:]))
            conc_std = float(np.std(liquidity_concentration[self.window - 1:])) + 1e-12
            hotspot_mask[self.window - 1:] = liquidity_concentration[self.window - 1:] > conc_mean + self.STD_THRESHOLD * conc_std
        liquidity_hotspots = np.where(hotspot_mask)[0]

        # Calculate overall migration entropy manually without SciPy
        binned = np.digitize(migration_flow, np.linspace(
            float(np.min(migration_flow)), float(np.max(migration_flow)) + 1e-12, self.ENTROPY_BINS))
        hist = np.zeros(self.ENTROPY_BINS, dtype=np.float64)
        for b in range(self.ENTROPY_BINS):
            hist[b] = float(np.sum(binned == b + 1))
        hist = hist / max(1e-12, float(np.sum(hist)))
        
        # Entropy formula: -sum(p * log(p))
        migration_entropy_overall = 0.0
        for b in range(self.ENTROPY_BINS):
            p = hist[b]
            if p > 0.0:
                migration_entropy_overall -= p * np.log(p)

        roll_entropy = _numba_rolling_entropy(migration_flow, self.window, self.ENTROPY_BINS)

        migration_direction = np.sign(migration_flow)
        migration_direction = migration_direction.astype(np.float64)
        recent = migration_direction[max(0, N - self.window):]
        counts = np.bincount(np.clip((recent + 1).astype(np.int64), 0, 2).astype(np.int64), minlength=3)
        dominant_direction = float(np.argmax(counts) - 1)

        migration_regime = np.ones(N, dtype=np.int64) * 2
        inflow_mask = migration_flow > roll_std
        outflow_mask = migration_flow < -roll_std
        migration_regime[inflow_mask] = 0
        migration_regime[outflow_mask] = 1

        net_migration = np.cumsum(migration_flow)

        self._state_contribution = net_migration / max(1e-12, float(np.max(np.abs(net_migration))))

        result = {
            "liquidity_proxy": liquidity_proxy,
            "liquidity_efficiency": liquidity_efficiency,
            "slippage_proxy": slippage_proxy,
            "migration_flow": migration_flow,
            "migration_velocity": migration_velocity,
            "migration_acceleration": migration_acceleration,
            "migration_entropy": migration_entropy_overall,
            "migration_entropy_rolling": roll_entropy,
            "migration_direction": migration_direction,
            "liquidity_concentration": liquidity_concentration,
            "liquidity_hotspots": liquidity_hotspots,
            "accumulation_zones": accumulation_zones,
            "disappearance_zones": disappearance_zones,
            "accumulation_regime": accumulation_zones,
            "distribution_regime": disappearance_zones,
            "migration_regime": migration_regime,
            "dominant_migration_direction": dominant_direction,
            "net_migration": net_migration,
            "leaving_zones": leaving,
            "arriving_zones": arriving,
        }
        self._state.update(result)
        return result

    def get_state_contribution(self) -> NDArray:
        return self._state_contribution

    def _herfindahl_index(self, x: NDArray, window: int) -> NDArray:
        return _numba_herfindahl_index(x.astype(np.float64), window)

    @staticmethod
    def _rolling_mean(x: NDArray, window: int) -> NDArray:
        return _numba_rolling_mean(x.astype(np.float64), window)

    @staticmethod
    def _rolling_std(x: NDArray, window: int) -> NDArray:
        return _numba_rolling_std(x.astype(np.float64), window)

    def _empty_result(self, N: int) -> dict[str, Any]:
        size = max(1, N)
        self._state_contribution = np.zeros(size, dtype=np.float64)
        return {
            "liquidity_proxy": np.zeros(size, dtype=np.float64),
            "liquidity_efficiency": np.zeros(size, dtype=np.float64),
            "slippage_proxy": np.zeros(size, dtype=np.float64),
            "migration_flow": np.zeros(size, dtype=np.float64),
            "migration_velocity": np.zeros(size, dtype=np.float64),
            "migration_acceleration": np.zeros(size, dtype=np.float64),
            "migration_entropy": 0.0,
            "migration_entropy_rolling": np.zeros(size, dtype=np.float64),
            "migration_direction": np.zeros(size, dtype=np.float64),
            "liquidity_concentration": np.zeros(size, dtype=np.float64),
            "liquidity_hotspots": np.array([], dtype=np.int64),
            "accumulation_zones": np.zeros(size, dtype=bool),
            "disappearance_zones": np.zeros(size, dtype=bool),
            "accumulation_regime": np.zeros(size, dtype=bool),
            "distribution_regime": np.zeros(size, dtype=bool),
            "migration_regime": np.ones(size, dtype=np.int64) * 2,
            "dominant_migration_direction": 0.0,
            "net_migration": np.zeros(size, dtype=np.float64),
            "leaving_zones": np.zeros(size, dtype=bool),
            "arriving_zones": np.zeros(size, dtype=bool),
        }
