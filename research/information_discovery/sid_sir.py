from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from research.information_discovery.mi_estimator import MIEstimator


class SIDCalculator:
    def __init__(self, mi_estimator: Optional[MIEstimator] = None):
        self.mi = mi_estimator or MIEstimator(n_bins=20)

    def compute_sid(self, states: NDArray[np.int32], forward_values: NDArray[np.float64]) -> dict:
        unique = np.unique(states[states >= 0])
        if len(unique) < 1:
            return {"sid_per_state": {}, "avg_sid": 0.0, "n_states": 0}
        valid = states >= 0
        sid_per_state: dict[int, float] = {}
        h_y = self.mi.entropy(forward_values)
        for s in unique:
            mask = states == s
            if mask.sum() < 2:
                sid_per_state[int(s)] = -float("inf")
                continue
            h_y_given_s = self.mi.entropy(forward_values[mask])
            sid_per_state[int(s)] = h_y - h_y_given_s
        avg_sid = float(np.mean([v for v in sid_per_state.values() if v > -float("inf")])) if sid_per_state else 0.0
        return {"sid_per_state": sid_per_state, "avg_sid": avg_sid, "n_states": len(unique)}

    def compute_sid_horizons(self, states: NDArray[np.int32], forward_metrics: dict[str, NDArray]) -> dict[str, dict]:
        results: dict[str, dict] = {}
        for key, fwd in forward_metrics.items():
            common = min(len(states), len(fwd))
            results[key] = self.compute_sid(states[:common], fwd[:common])
        return results

    def compute_sid_regime(self, states: NDArray[np.int32], future_regimes: NDArray[np.int32]) -> float:
        valid = (states >= 0) & (future_regimes >= 0)
        states, regimes = states[valid], future_regimes[valid]
        if len(np.unique(regimes)) < 2 or len(np.unique(states)) < 1:
            return 0.0
        h_r = 0.0
        r_unique, r_counts = np.unique(regimes, return_counts=True)
        r_probs = r_counts / r_counts.sum()
        for prob in r_probs:
            if prob > 0:
                h_r -= prob * np.log(prob)
        h_r_given_s = 0.0
        for s in np.unique(states):
            s_mask = states == s
            if s_mask.sum() < 2:
                continue
            sub_regimes = regimes[s_mask]
            _, sub_counts = np.unique(sub_regimes, return_counts=True)
            sub_probs = sub_counts / sub_counts.sum()
            h_s = 0.0
            for prob in sub_probs:
                if prob > 0:
                    h_s -= prob * np.log(prob)
            h_r_given_s += (s_mask.sum() / len(states)) * h_s
        return max(0.0, h_r - h_r_given_s)


class SIRCalculator:
    def __init__(self, sid_calculator: Optional[SIDCalculator] = None):
        self.sid = sid_calculator or SIDCalculator()

    def state_complexity(self, states: NDArray[np.int32], compressed_dim: int) -> float:
        unique = np.unique(states[states >= 0])
        n_states = len(unique)
        if n_states < 2:
            return 1.0
        trans_count = 0
        for i in range(1, len(states)):
            if states[i] != states[i - 1] and states[i] >= 0 and states[i - 1] >= 0:
                trans_count += 1
        transition_rate = trans_count / len(states)
        state_probs = np.array([(states == s).sum() for s in unique], dtype=np.float64)
        state_probs = state_probs / state_probs.sum()
        entropy = -float(np.sum(state_probs * np.log(state_probs + 1e-10)))
        return float(compressed_dim) + transition_rate * 100 + entropy

    def compute_sir(self, states: NDArray[np.int32], forward_values: NDArray[np.float64], compressed_dim: int) -> float:
        sid_result = self.sid.compute_sid(states, forward_values)
        avg_sid = sid_result.get("avg_sid", 0.0)
        if avg_sid <= 0:
            return avg_sid
        complexity = self.state_complexity(states, compressed_dim)
        if complexity < 1e-10:
            return 0.0
        return avg_sid / complexity

    def compute_sir_all(self, states: NDArray[np.int32], forward_metrics: dict[str, NDArray], compressed_dim: int) -> dict[str, float]:
        sir_results: dict[str, float] = {}
        for key, fwd in forward_metrics.items():
            common = min(len(states), len(fwd))
            sir_results[key] = self.compute_sir(states[:common], fwd[:common], compressed_dim)
        return sir_results
