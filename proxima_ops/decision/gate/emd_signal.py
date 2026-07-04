from __future__ import annotations

import math
from typing import Any


class ExecutionMicrostructureDrift:
    def __init__(self, ema_alpha: float = 0.3) -> None:
        self._fill_latencies: dict[str, list[float]] = {}
        self._slippages: dict[str, list[float]] = {}
        self._latency_ema: dict[str, float] = {}
        self._slippage_ema: dict[str, float] = {}
        self._ema_alpha = ema_alpha

    def record_fill(self, symbol: str, latency: float, expected_slippage: float, actual_slippage: float) -> None:
        if symbol not in self._fill_latencies:
            self._fill_latencies[symbol] = []
            self._slippages[symbol] = []
            self._latency_ema[symbol] = latency
            self._slippage_ema[symbol] = abs(actual_slippage - expected_slippage)
        else:
            self._latency_ema[symbol] = self._ema_alpha * latency + (1 - self._ema_alpha) * self._latency_ema[symbol]
            slippage_dev = abs(actual_slippage - expected_slippage)
            ema_dev = self._ema_alpha * slippage_dev + (1 - self._ema_alpha) * self._slippage_ema[symbol]
            self._slippage_ema[symbol] = ema_dev
        self._fill_latencies[symbol].append(latency)
        slippage_dev = abs(actual_slippage - expected_slippage)
        self._slippages[symbol].append(slippage_dev)
        max_history = 100
        if len(self._fill_latencies[symbol]) > max_history:
            self._fill_latencies[symbol] = self._fill_latencies[symbol][-max_history:]
            self._slippages[symbol] = self._slippages[symbol][-max_history:]

    def _winsorize(self, values: list[float], limits: tuple[float, float] = (0.05, 0.95)) -> list[float]:
        if len(values) < 5:
            return values[:]
        sorted_v = sorted(values)
        n = len(sorted_v)
        lo_idx = int(n * limits[0])
        hi_idx = int(n * limits[1]) - 1
        lo_val = sorted_v[lo_idx]
        hi_val = sorted_v[hi_idx]
        return [max(lo_val, min(v, hi_val)) for v in values]

    def compute_variance(self, values: list[float]) -> float:
        if len(values) < 3:
            return 0.0
        wins = self._winsorize(values)
        mean_v = sum(wins) / len(wins)
        return sum((v - mean_v) ** 2 for v in wins) / len(wins)

    def get_emd(self, symbol: str, dampen: bool = False) -> dict[str, float]:
        latencies = self._fill_latencies.get(symbol, [])
        slippages = self._slippages.get(symbol, [])
        latency_var = self.compute_variance(latencies)
        slippage_var = self.compute_variance(slippages)
        emd_score = math.sqrt(latency_var + slippage_var)
        emd_normalized = min(0.5, emd_score * 5.0)
        smoothed = self._latency_ema.get(symbol, 0.0) + self._slippage_ema.get(symbol, 0.0)
        emd_blended = 0.6 * emd_normalized + 0.4 * min(0.5, smoothed * 3.0)
        if dampen:
            emd_blended *= 0.5
        return {
            "emd_score": round(emd_blended, 4),
            "latency_variance": round(latency_var, 6),
            "slippage_deviation": round(slippage_var, 6),
        }
