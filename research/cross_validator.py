from __future__ import annotations

from typing import Callable

import numpy as np
import numba
from numpy.typing import NDArray
from scipy.stats import f_oneway


@numba.jit(nopython=True, cache=True)
def _state_id_diversity(states: NDArray[np.int32]) -> NDArray[np.float32]:
    n = len(states)
    if n < 2:
        return np.zeros(n, dtype=np.float32)
    result = np.zeros(n, dtype=np.float32)
    running: list[int] = [int(states[0])]
    for i in range(1, n):
        s = int(states[i])
        found = False
        for rs in running:
            if rs == s:
                found = True
                break
        if not found:
            running.append(s)
        result[i] = len(running) / max(i + 1, 1)
    return result


@numba.jit(nopython=True, cache=True)
def _sid_score(states: NDArray[np.int32], window: int = 50) -> float:
    n = len(states)
    if n < 2 * window:
        return 0.0
    halves = n // 2
    first_half = states[:halves]
    second_half = states[halves:2 * halves]
    unique_first = np.zeros(np.max(states) + 1, dtype=np.int32)
    unique_second = np.zeros(np.max(states) + 1, dtype=np.int32)
    for s in first_half:
        unique_first[s] += 1
    for s in second_half:
        unique_second[s] += 1
    total_first = len(first_half)
    total_second = len(second_half)
    divergence = 0.0
    for i in range(len(unique_first)):
        p1 = unique_first[i] / total_first if total_first > 0 else 0.0
        p2 = unique_second[i] / total_second if total_second > 0 else 0.0
        if p1 > 0 and p2 > 0:
            divergence += p1 * np.log(p1 / p2 + 1e-15) + p2 * np.log(p2 / p1 + 1e-15)
    return float(divergence)


@numba.jit(nopython=True, cache=True)
def _persistence_from_states(states: NDArray[np.int32]) -> float:
    n = len(states)
    if n < 2:
        return 0.0
    changes = 0
    for i in range(1, n):
        if states[i] != states[i - 1]:
            changes += 1
    return 1.0 - changes / (n - 1)


@numba.jit(nopython=True, cache=True)
def _n_states_unique(states: NDArray[np.int32]) -> int:
    if len(states) == 0:
        return 0
    max_s = np.max(states)
    seen = np.zeros(max_s + 1, dtype=np.bool_)
    count = 0
    for i in range(len(states)):
        if not seen[states[i]]:
            seen[states[i]] = True
            count += 1
    return count


class CrossValidator:

    def __init__(self) -> None:
        pass

    def validate_across_assets(self, state_discovery_fn: Callable, assets_data: dict[str, dict]) -> dict:
        result: dict[str, dict] = {}
        for asset_name, data in assets_data.items():
            states = state_discovery_fn(data)
            sid_scores: dict[str, float] = {}
            window = 50
            n_windows = max(1, len(states) // window)
            for w in range(n_windows - 1):
                seg1 = states[w * window:(w + 1) * window]
                seg2 = states[(w + 1) * window:(w + 2) * window]
                if len(seg1) > 0 and len(seg2) > 0:
                    sid_scores[f"win_{w}"] = float(_sid_score(np.concatenate((seg1, seg2)), window))
            avg_sid = float(np.mean(list(sid_scores.values()))) if sid_scores else 0.0
            result[asset_name] = {
                "states": states,
                "sid_scores": sid_scores,
                "avg_sid": avg_sid,
            }
        return result

    def compute_cross_asset_consistency(self, asset_results: dict) -> dict:
        n_states_list: list[float] = []
        sid_list: list[float] = []
        persistence_list: list[float] = []
        for asset_name, res in asset_results.items():
            states = res["states"]
            n_states_list.append(float(_n_states_unique(states)))
            sid_list.append(res["avg_sid"])
            persistence_list.append(float(_persistence_from_states(states)))
        return {
            "n_states_consistency": float(np.std(n_states_list)) if len(n_states_list) > 1 else 0.0,
            "sid_consistency": float(np.std(sid_list)) if len(sid_list) > 1 else 0.0,
            "persistence_consistency": float(np.std(persistence_list)) if len(persistence_list) > 1 else 0.0,
            "n_states_per_asset": n_states_list,
            "sid_per_asset": sid_list,
            "persistence_per_asset": persistence_list,
        }

    def validate_across_regimes(self, state_discovery_fn: Callable, regime_data: dict[str, dict]) -> dict:
        return self.validate_across_assets(state_discovery_fn, regime_data)

    def compute_regime_invariance(self, regime_results: dict) -> dict:
        n_states_list: list[float] = []
        sid_list: list[float] = []
        persistence_list: list[float] = []
        for regime_name, res in regime_results.items():
            states = res["states"]
            n_states_list.append(float(_n_states_unique(states)))
            sid_list.append(res["avg_sid"])
            persistence_list.append(float(_persistence_from_states(states)))
        return {
            "n_states_invariance": float(np.std(n_states_list)) if len(n_states_list) > 1 else 0.0,
            "sid_invariance": float(np.std(sid_list)) if len(sid_list) > 1 else 0.0,
            "persistence_invariance": float(np.std(persistence_list)) if len(persistence_list) > 1 else 0.0,
            "n_states_per_regime": n_states_list,
            "sid_per_regime": sid_list,
            "persistence_per_regime": persistence_list,
        }

    def compute_stability_score(self, cross_asset_results: dict, cross_regime_results: dict) -> float:
        asset_consistency = self.compute_cross_asset_consistency(cross_asset_results)
        regime_invariance = self.compute_regime_invariance(cross_regime_results)
        n_states_std = max(asset_consistency["n_states_consistency"], regime_invariance["n_states_invariance"])
        sid_std = max(asset_consistency["sid_consistency"], regime_invariance["sid_invariance"])
        persistence_std = max(asset_consistency["persistence_consistency"], regime_invariance["persistence_invariance"])
        score_n = 1.0 / (1.0 + n_states_std)
        score_sid = 1.0 / (1.0 + sid_std)
        score_p = 1.0 / (1.0 + persistence_std)
        return float(np.mean([score_n, score_sid, score_p]))

    def validate_all(self, state_discovery_fn: Callable, assets_data: dict, regime_data: dict) -> dict:
        asset_results = self.validate_across_assets(state_discovery_fn, assets_data)
        regime_results = self.validate_across_regimes(state_discovery_fn, regime_data)
        return {
            "cross_asset": asset_results,
            "cross_regime": regime_results,
            "asset_consistency": self.compute_cross_asset_consistency(asset_results),
            "regime_invariance": self.compute_regime_invariance(regime_results),
            "stability_score": self.compute_stability_score(asset_results, regime_results),
        }
