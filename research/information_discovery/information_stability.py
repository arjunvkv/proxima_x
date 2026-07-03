from __future__ import annotations

from typing import Optional

import numpy as np
import numba
from numpy.typing import NDArray

from research.information_discovery.mi_estimator import (
    MIEstimator,
    _fast_percentile,
    _fast_digitize,
    _fast_entropy_digitized,
    _fast_joint_entropy_digitized,
)


@numba.jit(nopython=True, cache=True)
def _numba_rolling_mutual_info(
    feature: NDArray[np.float64],
    target: NDArray[np.float64],
    window: int,
    step: int,
    n_bins: int,
) -> NDArray[np.float64]:
    n = min(len(feature), len(target))
    n_steps = (n - window) // step + 1
    if n_steps <= 0:
        n_steps = 1

    mi_values = np.zeros(n_steps, dtype=np.float64)
    q = np.linspace(0.0, 1.0, n_bins + 1)

    for idx in range(n_steps):
        start = idx * step
        end = start + window
        if end > n:
            end = n

        f_slice = feature[start:end]
        t_slice = target[start:end]

        n_valid = 0
        for i in range(len(f_slice)):
            if not np.isnan(f_slice[i]) and not np.isnan(t_slice[i]):
                n_valid += 1

        if n_valid < 2:
            mi_values[idx] = 0.0
            continue

        f_clean = np.zeros(n_valid)
        t_clean = np.zeros(n_valid)
        k = 0
        for i in range(len(f_slice)):
            if not np.isnan(f_slice[i]) and not np.isnan(t_slice[i]):
                f_clean[k] = f_slice[i]
                t_clean[k] = t_slice[i]
                k += 1

        f_bins = _fast_percentile(f_clean, q)
        t_bins = _fast_percentile(t_clean, q)

        ux = np.unique(f_bins)
        uy = np.unique(t_bins)
        if len(ux) < 2 or len(uy) < 2:
            mi_values[idx] = 0.0
            continue

        dig_x = _fast_digitize(f_clean, f_bins)
        dig_y = _fast_digitize(t_clean, t_bins)

        hx = _fast_entropy_digitized(dig_x, n_bins)
        hy = _fast_entropy_digitized(dig_y, n_bins)
        hxy = _fast_joint_entropy_digitized(dig_x, dig_y, n_bins)

        mi_values[idx] = max(0.0, hx + hy - hxy)

    return mi_values


class InformationStability:
    def __init__(self, mi_estimator: Optional[MIEstimator] = None, window: int = 250, step: int = 50):
        self.mi = mi_estimator or MIEstimator(n_bins=20)
        self.window = window
        self.step = step

    def rolling_mutual_info(self, feature: NDArray, target: NDArray) -> tuple[NDArray, NDArray]:
        n = min(len(feature), len(target))
        positions = list(range(0, n - self.window, self.step))
        if not positions:
            positions = [0]
        
        mi_values = _numba_rolling_mutual_info(
            feature.astype(np.float64),
            target.astype(np.float64),
            self.window,
            self.step,
            self.mi.n_bins,
        )
        
        return np.array(positions, dtype=np.int32), mi_values

    def stability_score(self, mi_values: NDArray) -> float:
        if len(mi_values) < 2:
            return 0.0
        mean_mi = float(np.mean(mi_values))
        std_mi = float(np.std(mi_values))
        if mean_mi < 1e-10:
            return 0.0
        return mean_mi / (mean_mi + std_mi)

    def stability_classification(self, score: float) -> str:
        if score > 0.8:
            return "very_stable"
        if score > 0.6:
            return "stable"
        if score > 0.4:
            return "moderate"
        if score > 0.2:
            return "unstable"
        return "very_unstable"

    def compute_stability_for_feature(self, feature: NDArray, target: NDArray) -> dict:
        positions, mi_values = self.rolling_mutual_info(feature, target)
        score = self.stability_score(mi_values)
        return {
            "positions": positions,
            "mi_values": mi_values,
            "stability_score": score,
            "classification": self.stability_classification(score),
            "mean_mi": float(np.mean(mi_values)) if len(mi_values) > 0 else 0.0,
            "std_mi": float(np.std(mi_values)) if len(mi_values) > 0 else 0.0,
            "min_mi": float(np.min(mi_values)) if len(mi_values) > 0 else 0.0,
            "max_mi": float(np.max(mi_values)) if len(mi_values) > 0 else 0.0,
        }

    def cross_year_stability(self, feature: NDArray, target: NDArray, yearly_masks: dict[str, NDArray]) -> dict[str, float]:
        result: dict[str, float] = {}
        for year, mask in yearly_masks.items():
            f_slice = feature[mask]
            t_slice = target[mask]
            if len(f_slice) < 10:
                continue
            result[str(year)] = self.mi.mutual_info(f_slice, t_slice)
        return result

    def cross_regime_stability(self, feature: NDArray, target: NDArray, regime_masks: dict[str, NDArray]) -> dict[str, float]:
        result: dict[str, float] = {}
        for regime, mask in regime_masks.items():
            if mask.sum() < 10:
                continue
            f_slice = feature[mask]
            t_slice = target[mask]
            result[str(regime)] = self.mi.mutual_info(f_slice, t_slice)
        return result

    def compute_all_stability(self, features: dict[str, NDArray], targets: dict[str, NDArray]) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for fname, farr in features.items():
            result[fname] = {}
            for tname, tarr in targets.items():
                common = min(len(farr), len(tarr))
                stable = self.compute_stability_for_feature(farr[:common], tarr[:common])
                result[fname][tname] = stable["stability_score"]
        return result
