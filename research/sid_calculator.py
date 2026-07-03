from __future__ import annotations

from typing import Any

import numpy as np
import numba
from numpy.typing import NDArray


@numba.jit(nopython=True, cache=True)
def _entropy(values: NDArray[np.float64], n_bins: int) -> float:
    if len(values) == 0:
        return 0.0
    lo = np.min(values)
    hi = np.max(values)
    if hi - lo < 1e-15:
        return 0.0
    bin_edges = np.linspace(lo, hi, n_bins + 1)
    counts = np.zeros(n_bins, dtype=np.int64)
    n = len(values)
    for i in range(n):
        v = values[i]
        if v < bin_edges[0]:
            continue
        if v >= bin_edges[-1]:
            counts[-1] += 1
        else:
            for b in range(n_bins):
                if bin_edges[b] <= v < bin_edges[b + 1]:
                    counts[b] += 1
                    break
    total = np.sum(counts)
    if total < 1:
        return 0.0
    h = 0.0
    for b in range(n_bins):
        if counts[b] > 0:
            p = counts[b] / total
            h -= p * np.log(p)
    return h


@numba.jit(nopython=True, cache=True)
def _conditional_entropy_per_state(
    states: NDArray[np.int32],
    forward_values: NDArray[np.float64],
    unique_states: NDArray[np.int32],
    n_bins: int,
) -> NDArray[np.float64]:
    n_states = len(unique_states)
    result = np.zeros(n_states, dtype=np.float64)
    n = len(states)
    for s_idx in range(n_states):
        s = unique_states[s_idx]
        mask = np.zeros(n, dtype=np.bool_)
        count = 0
        for i in range(n):
            if states[i] == s:
                mask[i] = True
                count += 1
        if count < 2:
            result[s_idx] = 0.0
            continue
        subset = np.zeros(count, dtype=np.float64)
        j = 0
        for i in range(n):
            if mask[i]:
                subset[j] = forward_values[i]
                j += 1
        result[s_idx] = _entropy(subset, n_bins)
    return result


@numba.jit(nopython=True, cache=True)
def _marginal_entropy(forward_values: NDArray[np.float64], n_bins: int) -> float:
    return _entropy(forward_values, n_bins)


@numba.jit(nopython=True, cache=True)
def _compute_sid_single(
    states: NDArray[np.int32],
    forward_values: NDArray[np.float64],
    unique_states: NDArray[np.int32],
    n_bins: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    marginal_h = _marginal_entropy(forward_values, n_bins)
    cond_h = _conditional_entropy_per_state(states, forward_values, unique_states, n_bins)
    sid_values = np.zeros(len(unique_states), dtype=np.float64)
    for i in range(len(unique_states)):
        sid_values[i] = marginal_h - cond_h[i]
    return sid_values, cond_h


@numba.jit(nopython=True, cache=True)
def _sid_regime_change(
    states: NDArray[np.int32],
    future_regimes: NDArray[np.int32],
    unique_states: NDArray[np.int32],
    n_bins: int,
) -> float:
    n = min(len(states), len(future_regimes))
    s = states[:n]
    f = future_regimes[:n]
    unique_f = np.zeros(0, dtype=np.int32)
    uf_set = set()
    for i in range(n):
        uf_set.add(f[i])
    unique_f = np.array(list(uf_set), dtype=np.int32)
    marginal_h = _marginal_entropy(f.astype(np.float64), n_bins)
    cond_h = _conditional_entropy_per_state(s, f.astype(np.float64), unique_states, n_bins)
    avg_cond = 0.0
    total_count = 0
    for s_idx in range(len(unique_states)):
        sid = s == unique_states[s_idx]
        cnt = 0
        for i in range(n):
            if sid[i]:
                cnt += 1
        if cnt > 0:
            avg_cond += cond_h[s_idx] * cnt
            total_count += cnt
    if total_count > 0:
        avg_cond /= total_count
    return marginal_h - avg_cond


@numba.jit(nopython=True, cache=True)
def _rank_states_numba(
    unique_states: NDArray[np.int32],
    sid_matrix: NDArray[np.float64],
) -> NDArray[np.int64]:
    n_states = len(unique_states)
    n_horizons = sid_matrix.shape[1]
    avg_sid = np.zeros(n_states, dtype=np.float64)
    for i in range(n_states):
        total = 0.0
        for j in range(n_horizons):
            total += sid_matrix[i, j]
        avg_sid[i] = total / n_horizons
    order = np.argsort(-avg_sid)
    return order


class SIDCalculator:
    def __init__(self, n_bins: int = 20):
        self.n_bins = n_bins

    def compute_sid(
        self,
        states: NDArray[np.int32],
        forward_returns: NDArray[np.float64],
        forward_horizons: list[int] | None = None,
    ) -> dict:
        if forward_horizons is None:
            forward_horizons = [1, 5, 20, 50, 100]
        unique_states = np.unique(states)
        results: dict[str, dict] = {}
        for col_idx, h in enumerate(forward_horizons):
            h_key = str(h)
            fwd = forward_returns[:, col_idx] if forward_returns.ndim == 2 else forward_returns
            valid = ~np.isnan(fwd)
            s_valid = states[valid]
            f_valid = fwd[valid].astype(np.float64)
            if len(np.unique(s_valid)) < 2:
                results[h_key] = {
                    "sid_per_state": {int(s): 0.0 for s in unique_states},
                    "avg_sid": 0.0,
                    "max_sid_state": int(unique_states[0]),
                    "min_sid_state": int(unique_states[0]),
                }
                continue
            uq = np.unique(s_valid)
            sid_vals, cond_h = _compute_sid_single(s_valid, f_valid, uq, self.n_bins)
            sid_per_state = {int(uq[i]): float(sid_vals[i]) for i in range(len(uq))}
            avg_sid = float(np.mean(sid_vals))
            max_idx = int(uq[np.argmax(sid_vals)])
            min_idx = int(uq[np.argmin(sid_vals)])
            results[h_key] = {
                "sid_per_state": sid_per_state,
                "avg_sid": avg_sid,
                "max_sid_state": max_idx,
                "min_sid_state": min_idx,
            }
        return results

    def compute_sid_by_volatility(
        self,
        states: NDArray[np.int32],
        forward_vol: NDArray[np.float64],
        forward_horizons: list[int] | None = None,
    ) -> dict:
        if forward_horizons is None:
            forward_horizons = [1, 5, 20, 50, 100]
        return self.compute_sid(states, forward_vol, forward_horizons)

    def compute_sid_entropy_reduction(
        self,
        states: NDArray[np.int32],
        forward_entropy: NDArray[np.float64],
        forward_horizons: list[int],
    ) -> dict:
        return self.compute_sid(states, forward_entropy, forward_horizons)

    def compute_sid_regime_change(
        self,
        states: NDArray[np.int32],
        future_regimes: NDArray[np.int32],
    ) -> float:
        unique_states = np.unique(states)
        return float(_sid_regime_change(states, future_regimes, unique_states, self.n_bins))

    def compute_all_sid(self, states: NDArray[np.int32], data: dict) -> dict:
        results: dict[str, Any] = {}
        horizons_detected: set[int] = set()
        for key in data:
            if key.startswith("forward_returns_"):
                parts = key.split("_")
                h = int(parts[-1])
                horizons_detected.add(h)
        if not horizons_detected:
            horizons_detected = {1, 5, 20, 50, 100}
        horizons = sorted(horizons_detected)
        return_keys = [f"forward_returns_{h}" for h in horizons]
        vol_keys = [f"forward_vol_{h}" for h in horizons]
        entropy_keys = [f"forward_entropy_{h}" for h in horizons]
        fwd_returns_2d = None
        fwd_vol_2d = None
        fwd_entropy_2d = None
        for i, h in enumerate(horizons):
            rk = f"forward_returns_{h}"
            vk = f"forward_vol_{h}"
            ek = f"forward_entropy_{h}"
            if rk in data:
                arr = data[rk]
                if fwd_returns_2d is None:
                    fwd_returns_2d = np.zeros((len(arr), len(horizons)), dtype=np.float64)
                fwd_returns_2d[:, i] = arr
            if vk in data:
                arr = data[vk]
                if fwd_vol_2d is None:
                    fwd_vol_2d = np.zeros((len(arr), len(horizons)), dtype=np.float64)
                fwd_vol_2d[:, i] = arr
            if ek in data:
                arr = data[ek]
                if fwd_entropy_2d is None:
                    fwd_entropy_2d = np.zeros((len(arr), len(horizons)), dtype=np.float64)
                fwd_entropy_2d[:, i] = arr
        if fwd_returns_2d is not None:
            sid_ret = self.compute_sid(states, fwd_returns_2d, horizons)
            for k, v in sid_ret.items():
                results[f"sid_return_{k}"] = v
        if fwd_vol_2d is not None:
            sid_vol = self.compute_sid_by_volatility(states, fwd_vol_2d, horizons)
            for k, v in sid_vol.items():
                results[f"sid_vol_{k}"] = v
        if fwd_entropy_2d is not None:
            sid_ent = self.compute_sid_entropy_reduction(states, fwd_entropy_2d, horizons)
            for k, v in sid_ent.items():
                results[f"sid_entropy_{k}"] = v
        if "future_regimes" in data:
            results["sid_regime_change"] = self.compute_sid_regime_change(states, data["future_regimes"])
        results["horizons"] = horizons
        results["n_states"] = int(len(np.unique(states)))
        return results

    def rank_states_by_sid(self, sid_results: dict) -> list[tuple[int, float]]:
        sid_map: dict[int, list[float]] = {}
        for key, val in sid_results.items():
            if isinstance(val, dict) and "sid_per_state" in val:
                for state_id, sid_val in val["sid_per_state"].items():
                    if state_id not in sid_map:
                        sid_map[state_id] = []
                    sid_map[state_id].append(float(sid_val))
        averages: list[tuple[int, float]] = []
        for state_id, vals in sid_map.items():
            avg = float(np.mean(vals)) if vals else 0.0
            averages.append((state_id, avg))
        averages.sort(key=lambda x: -x[1])
        return averages

    def analyze_sid_distribution(self, sid_per_state: dict[int, float]) -> dict:
        vals = np.array(list(sid_per_state.values()), dtype=np.float64)
        if len(vals) == 0:
            return {"mean": 0.0, "median": 0.0, "std": 0.0, "max": 0.0, "min": 0.0}
        return {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals)),
            "max": float(np.max(vals)),
            "min": float(np.min(vals)),
        }
