from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import numba
from numpy.typing import NDArray

from research.temporal_reality.causality_analysis import AdaptiveTimeCausality


@dataclass
class SurvivalResult:
    generator: str
    cross_asset_consistency: float
    bootstrap_stability: float
    noise_stability: float
    regime_split_consistency: float
    survival_probability: float
    validated: bool

    def __post_init__(self) -> None:
        self.validated = self.survival_probability > 0.5


@numba.jit(nopython=True, cache=True)
def _bootstrap_indices(n: int, n_samples: int) -> NDArray[np.int32]:
    result = np.empty((n_samples, int(0.8 * n)), dtype=np.int32)
    for i in range(n_samples):
        for j in range(result.shape[1]):
            result[i, j] = np.random.randint(0, n)
    return result


@numba.jit(nopython=True, cache=True)
def _compute_stability_from_bootstrap(
    x: NDArray[np.float64], y: NDArray[np.float64], indices: NDArray[np.int32], max_lag: int
) -> float:
    n_samples = indices.shape[0]
    strengths = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        idx = indices[i]
        xb = x[idx]
        yb = y[idx]
        corr = _cross_correlate_nb(xb, yb, max_lag)
        peak = 0.0
        for j in range(len(corr)):
            if abs(corr[j]) > abs(peak):
                peak = corr[j]
        strengths[i] = peak
    return float(np.mean(strengths))


@numba.jit(nopython=True, cache=True)
def _cross_correlate_nb(x: NDArray[np.float64], y: NDArray[np.float64], max_lag: int) -> NDArray[np.float64]:
    n = len(x)
    result = np.empty(2 * max_lag + 1, dtype=np.float64)
    x_mean = np.mean(x)
    x_std = np.std(x)
    y_mean = np.mean(y)
    y_std = np.std(y)
    for k in range(-max_lag, max_lag + 1):
        result[k + max_lag] = _pearson_nb(x, y, k, x_mean, x_std, y_mean, y_std)
    return result


@numba.jit(nopython=True, cache=True)
def _pearson_nb(
    x: NDArray[np.float64], y: NDArray[np.float64], lag: int,
    x_mean: float, x_std: float, y_mean: float, y_std: float,
) -> float:
    n = len(x)
    if lag >= 0:
        sx, ex = lag, n
        sy, ey = 0, n - lag
    else:
        sx, ex = 0, n + lag
        sy, ey = -lag, n
    length = ex - sx
    if length < 2:
        return 0.0
    xs = x[sx:ex]
    ys = y[sy:ey]
    xsm = np.mean(xs)
    xss = np.std(xs)
    ysm = np.mean(ys)
    yss = np.std(ys)
    if xss == 0.0 or yss == 0.0:
        return 0.0
    s = 0.0
    for i in range(length):
        s += (xs[i] - xsm) / xss * (ys[i] - ysm) / yss
    r = s / length
    return max(-1.0, min(1.0, r))


@numba.jit(nopython=True, cache=True)
def _rolling_std(arr: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    n = len(arr)
    result = np.zeros(n, dtype=np.float64)
    for i in range(window, n):
        s = 0.0
        c = 0
        for j in range(i - window, i):
            s += arr[j]
            c += 1
        mean = s / c
        var = 0.0
        for j in range(i - window, i):
            var += (arr[j] - mean) ** 2
        result[i] = np.sqrt(var / c)
    return result


def _compute_causal_strength(source: np.ndarray, target: np.ndarray, max_lag: int) -> float:
    corr = AdaptiveTimeCausality._cross_correlate(source, target, max_lag)
    peak = float(corr[np.argmax(np.abs(corr))])
    return peak


class GeneratorSurvivalValidator:
    def __init__(self, max_lag: int = 50, n_bootstrap: int = 100) -> None:
        self.max_lag = max_lag
        self.n_bootstrap = n_bootstrap

    def validate_single(
        self, candidates: list[dict[str, Any]], data_single_asset: dict[str, np.ndarray]
    ) -> dict[str, SurvivalResult]:
        results: dict[str, SurvivalResult] = {}
        asset_keys = list(data_single_asset.keys())
        signals = {k: np.asarray(v, dtype=np.float64) for k, v in data_single_asset.items()}

        for c in candidates:
            source_name = str(c.get("source", ""))
            target_name = str(c.get("target", ""))
            generator_key = f"{source_name}\u2192{target_name}"

            source_sig = signals.get(source_name)
            target_sig = signals.get(target_name)
            if source_sig is None or target_sig is None:
                continue

            n = len(source_sig)
            bootstrap_stability = self._bootstrap_test(source_sig, target_sig)
            noise_stability = self._noise_test(source_sig, target_sig)
            regime_consistency = self._regime_test(source_sig, target_sig, signals.get("returns"))
            cross_asset = 0.0

            prob = self.composite_survival(cross_asset, bootstrap_stability, noise_stability, regime_consistency)
            results[generator_key] = SurvivalResult(
                generator=generator_key,
                cross_asset_consistency=cross_asset,
                bootstrap_stability=bootstrap_stability,
                noise_stability=noise_stability,
                regime_split_consistency=regime_consistency,
                survival_probability=prob,
                validated=prob > 0.5,
            )
        return results

    def validate_cross_asset(
        self, candidates_per_asset: dict[str, list[dict[str, Any]]]
    ) -> dict[str, float]:
        generator_strengths: dict[str, list[float]] = {}

        for asset_name, cand_list in candidates_per_asset.items():
            for c in cand_list:
                source = str(c.get("source", ""))
                target = str(c.get("target", ""))
                key = f"{source}\u2192{target}"
                strength = float(c.get("causal_strength", 0.0))
                generator_strengths.setdefault(key, []).append(strength)

        consistency: dict[str, float] = {}
        for key, strengths in generator_strengths.items():
            consistency[key] = float(np.std(strengths)) if len(strengths) > 1 else 0.0
        return consistency

    def composite_survival(
        self,
        cross_asset: float,
        bootstrap: float,
        noise: float,
        regime: float,
    ) -> float:
        raw = 0.0
        raw += max(0.0, 1.0 - cross_asset) * 0.3
        raw += max(0.0, min(1.0, abs(bootstrap))) * 0.3
        raw += max(0.0, min(1.0, noise)) * 0.2
        raw += max(0.0, min(1.0, regime)) * 0.2
        return min(1.0, raw)

    def _bootstrap_test(self, source: np.ndarray, target: np.ndarray) -> float:
        n = len(source)
        indices = _bootstrap_indices(n, self.n_bootstrap)
        return _compute_stability_from_bootstrap(source, target, indices, self.max_lag)

    def _noise_test(self, source: np.ndarray, target: np.ndarray) -> float:
        sigma_s = float(np.nanstd(source))
        sigma_t = float(np.nanstd(target))
        noise_s = np.random.normal(0.0, 0.05 * sigma_s, len(source)).astype(np.float64)
        noise_t = np.random.normal(0.0, 0.05 * sigma_t, len(target)).astype(np.float64)
        noisy_source = source + noise_s
        noisy_target = target + noise_t
        clean_strength = abs(_compute_causal_strength(source, target, self.max_lag))
        noisy_strength = abs(_compute_causal_strength(noisy_source, noisy_target, self.max_lag))
        if clean_strength < 1e-12:
            return 0.0
        return float(min(1.0, noisy_strength / clean_strength))

    def _regime_test(
        self, source: np.ndarray, target: np.ndarray, returns: Optional[np.ndarray]
    ) -> float:
        if returns is None or len(returns) < 40:
            return 0.5
        returns_f64 = np.asarray(returns, dtype=np.float64)
        vol = _rolling_std(returns_f64, 20)
        median_vol = float(np.nanmedian(vol[20:]))
        n = len(source)
        low_mask = vol < median_vol
        high_mask = vol >= median_vol
        low_strength = 0.0
        high_strength = 0.0
        low_count = int(np.sum(low_mask))
        high_count = int(np.sum(high_mask))
        if low_count > self.max_lag * 2:
            low_strength = abs(_compute_causal_strength(source[low_mask], target[low_mask], self.max_lag))
        if high_count > self.max_lag * 2:
            high_strength = abs(_compute_causal_strength(source[high_mask], target[high_mask], self.max_lag))
        denom = max(low_strength, high_strength, 1e-12)
        return float(1.0 - abs(low_strength - high_strength) / denom)
