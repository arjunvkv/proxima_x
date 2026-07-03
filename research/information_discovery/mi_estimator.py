from __future__ import annotations

from typing import Optional

import numpy as np
import numba
from numpy.typing import NDArray
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score


@numba.jit(nopython=True, cache=True)
def _fast_percentile(x: NDArray[np.float64], q: NDArray[np.float64]) -> NDArray[np.float64]:
    x_sorted = np.sort(x)
    n = len(x_sorted)
    res = np.zeros(len(q))
    for i in range(len(q)):
        idx = q[i] * (n - 1)
        idx_low = int(np.floor(idx))
        idx_high = int(np.ceil(idx))
        if idx_low == idx_high:
            res[i] = x_sorted[idx_low]
        else:
            weight = idx - idx_low
            res[i] = x_sorted[idx_low] * (1.0 - weight) + x_sorted[idx_high] * weight
    return res


@numba.jit(nopython=True, cache=True)
def _fast_digitize(x: NDArray[np.float64], bins: NDArray[np.float64]) -> NDArray[np.int32]:
    n = len(x)
    nb = len(bins)
    res = np.zeros(n, dtype=np.int32)
    for i in range(n):
        val = x[i]
        if val <= bins[0]:
            res[i] = 0
        elif val >= bins[nb - 1]:
            res[i] = nb - 2
        else:
            low = 0
            high = nb - 2
            best = 0
            while low <= high:
                mid = (low + high) // 2
                if bins[mid] <= val:
                    best = mid
                    low = mid + 1
                else:
                    high = mid - 1
            res[i] = best
    return res


@numba.jit(nopython=True, cache=True)
def _fast_entropy_digitized(dig: NDArray[np.int32], n_bins: int) -> float:
    counts = np.zeros(n_bins, dtype=np.int32)
    for i in range(len(dig)):
        counts[dig[i]] += 1
    total = len(dig)
    entropy = 0.0
    for j in range(n_bins):
        if counts[j] > 0:
            p = counts[j] / total
            entropy -= p * np.log(p)
    return entropy


@numba.jit(nopython=True, cache=True)
def _fast_joint_entropy_digitized(dig_x: NDArray[np.int32], dig_y: NDArray[np.int32], n_bins: int) -> float:
    counts = np.zeros((n_bins, n_bins), dtype=np.int32)
    for i in range(len(dig_x)):
        counts[dig_x[i], dig_y[i]] += 1
    total = len(dig_x)
    entropy = 0.0
    for j in range(n_bins):
        for k in range(n_bins):
            if counts[j, k] > 0:
                p = counts[j, k] / total
                entropy -= p * np.log(p)
    return entropy


@numba.jit(nopython=True, cache=True)
def _fast_triple_entropy_digitized(dig_x: NDArray[np.int32], dig_y: NDArray[np.int32], dig_z: NDArray[np.int32], n_bins: int) -> float:
    counts = np.zeros((n_bins, n_bins, n_bins), dtype=np.int32)
    for i in range(len(dig_x)):
        counts[dig_x[i], dig_y[i], dig_z[i]] += 1
    total = len(dig_x)
    entropy = 0.0
    for j in range(n_bins):
        for k in range(n_bins):
            for l in range(n_bins):
                if counts[j, k, l] > 0:
                    p = counts[j, k, l] / total
                    entropy -= p * np.log(p)
    return entropy


@numba.jit(nopython=True, cache=True)
def _fast_mutual_info(x: NDArray[np.float64], y: NDArray[np.float64], n_bins: int) -> float:
    valid = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[valid]
    y_clean = y[valid]
    if len(x_clean) < 2:
        return 0.0

    q = np.linspace(0.0, 1.0, n_bins + 1)
    x_bins = _fast_percentile(x_clean, q)
    y_bins = _fast_percentile(y_clean, q)

    ux = np.unique(x_bins)
    uy = np.unique(y_bins)
    if len(ux) < 2 or len(uy) < 2:
        return 0.0

    dig_x = _fast_digitize(x_clean, x_bins)
    dig_y = _fast_digitize(y_clean, y_bins)

    hx = _fast_entropy_digitized(dig_x, n_bins)
    hy = _fast_entropy_digitized(dig_y, n_bins)
    hxy = _fast_joint_entropy_digitized(dig_x, dig_y, n_bins)

    return max(0.0, hx + hy - hxy)


@numba.jit(nopython=True, cache=True)
def _fast_conditional_mutual_info(x: NDArray[np.float64], y: NDArray[np.float64], z: NDArray[np.float64], n_bins: int) -> float:
    valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x_clean = x[valid]
    y_clean = y[valid]
    z_clean = z[valid]
    if len(x_clean) < 2:
        return 0.0

    q = np.linspace(0.0, 1.0, n_bins + 1)
    x_bins = _fast_percentile(x_clean, q)
    y_bins = _fast_percentile(y_clean, q)
    z_bins = _fast_percentile(z_clean, q)

    ux = np.unique(x_bins)
    uy = np.unique(y_bins)
    uz = np.unique(z_bins)
    if len(ux) < 2 or len(uy) < 2 or len(uz) < 2:
        return 0.0

    dig_x = _fast_digitize(x_clean, x_bins)
    dig_y = _fast_digitize(y_clean, y_bins)
    dig_z = _fast_digitize(z_clean, z_bins)

    h_xz = _fast_joint_entropy_digitized(dig_x, dig_z, n_bins)
    h_yz = _fast_joint_entropy_digitized(dig_y, dig_z, n_bins)
    h_z = _fast_entropy_digitized(dig_z, n_bins)
    h_xyz = _fast_triple_entropy_digitized(dig_x, dig_y, dig_z, n_bins)

    return max(0.0, h_xz + h_yz - h_z - h_xyz)


class MIEstimator:
    def __init__(self, n_bins: int = 20, n_neighbors: int = 5):
        self.n_bins = n_bins
        self.n_neighbors = n_neighbors

    def entropy(self, x: NDArray) -> float:
        x_clean = x[~np.isnan(x)].astype(np.float64)
        if len(x_clean) < 2:
            return 0.0
        q = np.linspace(0.0, 1.0, self.n_bins + 1)
        bins = _fast_percentile(x_clean, q)
        if len(np.unique(bins)) < 2:
            return 0.0
        dig = _fast_digitize(x_clean, bins)
        return _fast_entropy_digitized(dig, self.n_bins)

    def joint_entropy(self, x: NDArray, y: NDArray) -> float:
        valid = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[valid].astype(np.float64)
        y_clean = y[valid].astype(np.float64)
        if len(x_clean) < 2:
            return 0.0
        q = np.linspace(0.0, 1.0, self.n_bins + 1)
        x_bins = _fast_percentile(x_clean, q)
        y_bins = _fast_percentile(y_clean, q)
        if len(np.unique(x_bins)) < 2 or len(np.unique(y_bins)) < 2:
            return 0.0
        dig_x = _fast_digitize(x_clean, x_bins)
        dig_y = _fast_digitize(y_clean, y_bins)
        return _fast_joint_entropy_digitized(dig_x, dig_y, self.n_bins)

    def mutual_info(self, x: NDArray, y: NDArray, method: str = "histogram") -> float:
        if method == "histogram":
            return _fast_mutual_info(x.astype(np.float64), y.astype(np.float64), self.n_bins)
        
        valid = ~(np.isnan(x) | np.isnan(y))
        x_clean, y_clean = x[valid], y[valid]
        if len(x_clean) < 2:
            return 0.0
        if method == "sklearn":
            y_disc, _ = self._discretize(y_clean, self.n_bins)
            return max(0.0, float(mutual_info_regression(x_clean.reshape(-1, 1), y_disc, random_state=42)[0]))
        if method == "knn":
            return max(0.0, float(mutual_info_regression(x_clean.reshape(-1, 1), y_clean, n_neighbors=self.n_neighbors, random_state=42)[0]))
        return 0.0

    def conditional_mutual_info(self, x: NDArray, y: NDArray, z: NDArray) -> float:
        return _fast_conditional_mutual_info(x.astype(np.float64), y.astype(np.float64), z.astype(np.float64), self.n_bins)

    def _triple_entropy(self, x: NDArray, y: NDArray, z: NDArray, x_bins: NDArray, y_bins: NDArray, z_bins: NDArray) -> float:
        valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        x_clean = x[valid].astype(np.float64)
        y_clean = y[valid].astype(np.float64)
        z_clean = z[valid].astype(np.float64)
        if len(x_clean) < 2:
            return 0.0
        dig_x = _fast_digitize(x_clean, x_bins)
        dig_y = _fast_digitize(y_clean, y_bins)
        dig_z = _fast_digitize(z_clean, z_bins)
        return _fast_triple_entropy_digitized(dig_x, dig_y, dig_z, self.n_bins)

    def _discretize(self, x: NDArray, bins: int) -> tuple[NDArray, NDArray]:
        x_clean = x.astype(np.float64)
        q = np.linspace(0.0, 1.0, bins + 1)
        edges = _fast_percentile(x_clean, q)
        edges = np.unique(edges)
        if len(edges) < 2:
            return np.zeros(len(x), dtype=np.int32), edges
        labels = np.digitize(x_clean, edges[:-1], right=False)
        labels = np.clip(labels, 0, len(edges) - 2)
        return labels.astype(np.int32), edges

    def transfer_entropy(self, source: NDArray, target: NDArray, lag: int = 1) -> float:
        s, t = source[:-lag], target[lag:]
        t_lagged = target[:-lag]
        return self.conditional_mutual_info(t, s, t_lagged)

    def compute_all_mi(self, features: dict[str, NDArray], targets: dict[str, NDArray]) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for fname, farr in features.items():
            result[fname] = {}
            for tname, tarr in targets.items():
                common = min(len(farr), len(tarr))
                result[fname][tname] = self.mutual_info(farr[:common], tarr[:common])
        return result

    def compute_all_cmi(self, features: dict[str, NDArray], targets: dict[str, NDArray], condition_on: dict[str, NDArray]) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for fname, farr in features.items():
            result[fname] = {}
            for tname, tarr in targets.items():
                cond_arr = condition_on.get(tname, condition_on.get("volatility"))
                if cond_arr is None:
                    result[fname][tname] = 0.0
                else:
                    common = min(len(farr), len(tarr), len(cond_arr))
                    result[fname][tname] = self.conditional_mutual_info(farr[:common], tarr[:common], cond_arr[:common])
        return result

    def information_gain_ratio(self, x: NDArray, y: NDArray) -> float:
        mi = self.mutual_info(x, y)
        hx = self.entropy(x)
        if hx < 1e-10:
            return 0.0
        return mi / hx

    def compute_feature_information_ranking(self, features: dict[str, NDArray], targets: dict[str, NDArray]) -> list[tuple[str, str, float]]:
        rankings: list[tuple[str, str, float]] = []
        for fname, farr in features.items():
            for tname, tarr in targets.items():
                common = min(len(farr), len(tarr))
                mi = self.mutual_info(farr[:common], tarr[:common])
                rankings.append((fname, tname, mi))
        rankings.sort(key=lambda x: x[2], reverse=True)
        return rankings
