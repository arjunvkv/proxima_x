from __future__ import annotations

from typing import Any, Optional

import numpy as np
import numba
from numpy.typing import NDArray

from research.mechanism_discovery.base import BaseMechanism, MechanismScore


@numba.jit(nopython=True, cache=True)
def _numba_memory_interference(returns: NDArray[np.float64], context_window: int, outcome_window: int, ret_std: float) -> NDArray[np.float64]:
    n = len(returns)
    interference = np.full(n, 0.5, dtype=np.float64)
    if ret_std < 1e-12:
        return interference
        
    for t in range(context_window, n - outcome_window):
        ctx = returns[t - context_window : t]
        
        max_outcomes = t - context_window
        outcomes = np.zeros(max_outcomes)
        outcomes_count = 0
        
        for s in range(0, t - context_window):
            past_ctx = returns[s : s + context_window]
            
            sq_sum = 0.0
            valid_count = 0
            for k in range(context_window):
                v1 = ctx[k]
                v2 = past_ctx[k]
                if not np.isnan(v1) and not np.isnan(v2):
                    sq_sum += (v1 - v2) ** 2
                    valid_count += 1
            if valid_count == 0:
                continue
            rmse = np.sqrt(sq_sum / valid_count)
            
            if rmse < ret_std * 0.5:
                val_sum = 0.0
                val_cnt = 0
                for k in range(outcome_window):
                    val = returns[s + context_window + k]
                    if not np.isnan(val):
                        val_sum += val
                        val_cnt += 1
                if val_cnt > 0:
                    fut = val_sum / val_cnt
                    outcomes[outcomes_count] = fut
                    outcomes_count += 1
                    
        if outcomes_count >= 3:
            valid_outcomes = outcomes[:outcomes_count]
            mean_out = np.mean(valid_outcomes)
            var_out = 0.0
            for i in range(outcomes_count):
                var_out += (valid_outcomes[i] - mean_out) ** 2
            std_out = np.sqrt(var_out / outcomes_count)
            
            consistency = 1.0 - std_out / (ret_std + 1e-12)
            if consistency < 0.0:
                consistency = 0.0
            elif consistency > 1.0:
                consistency = 1.0
            interference[t] = 1.0 - consistency
            
    return interference


class MemoryLandscape(BaseMechanism):
    def __init__(
        self,
        name: str = "memory_landscape",
        category: str = "memory_fields",
        memory_decay_half_life: int = 50,
    ) -> None:
        super().__init__(name, category)
        self.memory_decay_half_life = memory_decay_half_life
        self._state_contribution: NDArray = np.array([], dtype=np.float64)

    def compute(
        self, data: dict[str, NDArray], states: Optional[NDArray] = None
    ) -> dict[str, Any]:
        price = np.asarray(data.get("price", []), dtype=np.float64)
        returns = np.asarray(data.get("returns", []), dtype=np.float64)
        volume = np.asarray(data.get("volume", []), dtype=np.float64)
        high = np.asarray(data.get("high", []), dtype=np.float64)
        low = np.asarray(data.get("low", []), dtype=np.float64)

        N = len(price)
        if N == 0:
            return self._empty_result()

        returns = returns if len(returns) == N else np.zeros(N, dtype=np.float64)
        volume = volume if len(volume) == N else np.zeros(N, dtype=np.float64)
        high = high if len(high) == N else np.zeros(N, dtype=np.float64)
        low = low if len(low) == N else np.zeros(N, dtype=np.float64)

        memory_density = self._compute_memory_density(returns, volume, N)
        memory_gradient = np.gradient(memory_density)
        memory_decay_rate, memory_half_life = self._compute_memory_decay(memory_density)
        memory_interference = self._compute_memory_interference(returns, N)
        memory_landscape = memory_density + memory_gradient * memory_interference

        memory_regime = np.zeros(N, dtype=np.int32)
        memory_regime[(memory_gradient < 0) & (memory_interference < 0.5)] = 1
        memory_regime[memory_interference >= 0.5] = 2

        grad_std = float(np.nanstd(memory_gradient))
        if grad_std > 0:
            attraction_mask = (memory_gradient > grad_std) & (memory_interference < 0.5)
            repulsion_mask = (memory_gradient < -grad_std) & (memory_interference < 0.5)
        else:
            attraction_mask = np.zeros(N, dtype=bool)
            repulsion_mask = np.zeros(N, dtype=bool)
        attraction_events = np.where(attraction_mask)[0].tolist()
        repulsion_events = np.where(repulsion_mask)[0].tolist()

        peak = float(np.max(np.abs(memory_landscape)))
        self._state_contribution = memory_landscape / peak if peak > 0 else np.zeros(N, dtype=np.float64)

        return {
            "memory_density": memory_density,
            "memory_gradient": memory_gradient,
            "memory_decay_rate": memory_decay_rate,
            "memory_half_life": memory_half_life,
            "memory_interference": memory_interference,
            "memory_landscape": memory_landscape,
            "memory_regime": memory_regime,
            "attraction_events": attraction_events,
            "repulsion_events": repulsion_events,
        }

    def get_state_contribution(self) -> NDArray:
        return self._state_contribution

    def _compute_memory_density(
        self, returns: NDArray, volume: NDArray, N: int
    ) -> NDArray:
        events = np.zeros(N, dtype=np.float64)
        ret_std = float(np.nanstd(returns))
        if ret_std > 1e-12:
            extreme_ret = np.abs(returns) > 2.0 * ret_std
            events[extreme_ret] += np.abs(returns[extreme_ret]) / ret_std
        vol_mean = float(np.nanmean(volume))
        vol_std = float(np.nanstd(volume))
        if vol_std > 1e-12:
            volume_spike = volume > vol_mean + 2.0 * vol_std
            events[volume_spike] += (volume[volume_spike] - vol_mean) / vol_std
        half = float(self.memory_decay_half_life)
        kernel_len = min(N, int(5.0 * half))
        kernel = np.exp(-np.arange(kernel_len, dtype=np.float64) / half)
        memory = np.convolve(events, kernel, mode="full")[:N]
        return memory

    def _compute_memory_decay(
        self, memory_density: NDArray
    ) -> tuple[float, float]:
        N = len(memory_density)
        if N < 5:
            return 0.0, 0.0
        max_lag = min(N // 2, 100)
        if max_lag < 2:
            return 0.0, 0.0
        mem = memory_density - np.nanmean(memory_density)
        mem_std = float(np.nanstd(mem))
        if mem_std < 1e-12:
            return 0.0, 0.0
        autocorr = np.zeros(max_lag, dtype=np.float64)
        for lag in range(1, max_lag + 1):
            c = float(np.nanmean(mem[lag:] * mem[:-lag]))
            autocorr[lag - 1] = c / (mem_std * mem_std + 1e-12)
        autocorr = np.clip(autocorr, 1e-12, 1.0)
        lags = np.arange(1, max_lag + 1, dtype=np.float64)
        log_ac = -np.log(autocorr)
        A = np.column_stack([lags, np.ones_like(lags)])
        try:
            tau, _ = np.linalg.lstsq(A, log_ac, rcond=None)[0]
        except np.linalg.LinAlgError:
            tau = 1.0
        tau = float(tau)
        tau = max(tau, 1e-12)
        decay_rate = 1.0 / tau
        half_life = tau * np.log(2.0)
        return decay_rate, half_life

    def _compute_memory_interference(
        self, returns: NDArray, N: int
    ) -> NDArray:
        context_window = min(20, max(3, N // 4))
        outcome_window = min(5, max(1, N // 10))
        ret_std = float(np.nanstd(returns))
        
        return _numba_memory_interference(returns, context_window, outcome_window, ret_std)

    def _empty_result(self) -> dict[str, Any]:
        self._state_contribution = np.zeros(1, dtype=np.float64)
        return {
            "memory_density": np.zeros(1, dtype=np.float64),
            "memory_gradient": np.zeros(1, dtype=np.float64),
            "memory_decay_rate": 0.0,
            "memory_half_life": 0.0,
            "memory_interference": np.zeros(1, dtype=np.float64),
            "memory_landscape": np.zeros(1, dtype=np.float64),
            "memory_regime": np.zeros(1, dtype=np.int32),
            "attraction_events": [],
            "repulsion_events": [],
        }
