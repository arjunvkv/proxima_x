from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from research.information_discovery.mi_estimator import MIEstimator


class TransitionIntelligence:
    def __init__(self, mi_estimator: Optional[MIEstimator] = None):
        self.mi = mi_estimator or MIEstimator(n_bins=20)

    def compute_transition_matrix(self, states: NDArray[np.int32]) -> tuple[NDArray, NDArray]:
        valid = states >= 0
        s = states[valid]
        unique = np.unique(s)
        n = len(unique)
        if n < 2:
            return np.zeros((1, 1), dtype=np.float64), unique
        idx_map = {old: i for i, old in enumerate(sorted(unique))}
        mapped = np.array([idx_map[x] for x in s], dtype=np.int32)
        matrix = np.zeros((n, n), dtype=np.float64)
        for i in range(1, len(mapped)):
            matrix[mapped[i - 1], mapped[i]] += 1.0
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        matrix = matrix / row_sums
        return matrix, unique

    def transition_information_gain(self, transition_matrix: NDArray, state_probs: NDArray) -> float:
        n = len(transition_matrix)
        if n < 2:
            return 0.0
        h_future_given_current = 0.0
        for i in range(n):
            row = transition_matrix[i]
            row = row[row > 0]
            if len(row) > 0:
                h_future_given_current += state_probs[i] * (-float(np.sum(row * np.log(row))))
        h_future = -float(np.sum(state_probs * np.log(state_probs + 1e-10)))
        return max(0.0, h_future - h_future_given_current)

    def transition_entropy(self, transition_matrix: NDArray) -> NDArray:
        n = len(transition_matrix)
        entropies = np.zeros(n, dtype=np.float64)
        for i in range(n):
            row = transition_matrix[i]
            row = row[row > 0]
            if len(row) > 0:
                entropies[i] = -float(np.sum(row * np.log(row)))
        return entropies

    def transition_stability(self, transition_matrix: NDArray) -> float:
        n = len(transition_matrix)
        if n < 2:
            return 1.0
        diag_mean = float(np.mean(np.diag(transition_matrix)))
        return diag_mean

    def transition_survival_rate(self, states: NDArray[np.int32], lag: int = 1) -> float:
        valid = states >= 0
        s = states[valid]
        if len(s) < lag + 2:
            return 0.0
        same = np.sum(s[:-lag] == s[lag:])
        return same / (len(s) - lag)

    def find_informative_transitions(self, states: NDArray[np.int32], forward_target: NDArray) -> list[dict]:
        valid = states >= 0
        s, fwd = states[valid], forward_target[: len(states)][valid]
        if len(s) < 3:
            return []
        results: list[dict] = []
        for i in range(1, len(s)):
            from_state, to_state = int(s[i - 1]), int(s[i])
            if from_state < 0 or to_state < 0:
                continue
            transition_key = f"{from_state}->{to_state}"
            transition_fwd = fwd[i] if i < len(fwd) else 0.0
            results.append({
                "from": from_state,
                "to": to_state,
                "key": transition_key,
                "forward_value": float(transition_fwd),
                "index": i,
            })
        return results

    def compute_transition_mi(self, states: NDArray[np.int32], target: NDArray) -> float:
        valid = states >= 0
        s = states[valid]
        t = target[:len(states)][valid]
        if len(s) < 3:
            return 0.0
        transition_keys: list[str] = []
        aligned_targets: list[float] = []
        for i in range(1, len(s)):
            transition_keys.append(f"{s[i-1]}->{s[i]}")
            aligned_targets.append(t[i] if i < len(t) else 0.0)
        if len(np.unique(transition_keys)) < 2:
            return 0.0
        tk_int = np.array([hash(k) % 10000 for k in transition_keys], dtype=np.float64)
        ta = np.array(aligned_targets, dtype=np.float64)
        return self.mi.mutual_info(tk_int, ta)

    def compute_all_transition_metrics(self, states: NDArray[np.int32], forward_target: Optional[NDArray] = None) -> dict:
        matrix, unique = self.compute_transition_matrix(states)
        n = len(unique)
        state_counts = np.array([(states == s).sum() for s in unique], dtype=np.float64)
        state_probs = state_counts / state_counts.sum() if state_counts.sum() > 0 else np.ones(n) / n
        tig = self.transition_information_gain(matrix, state_probs) if n >= 2 else 0.0
        tent = self.transition_entropy(matrix) if n >= 2 else np.zeros(1)
        tstab = self.transition_stability(matrix) if n >= 2 else 1.0
        tsurv = self.transition_survival_rate(states)
        result: dict = {
            "transition_matrix": matrix,
            "unique_states": unique,
            "n_states": n,
            "transition_information_gain": tig,
            "transition_entropy": tent.tolist() if isinstance(tent, np.ndarray) else tent,
            "transition_stability": tstab,
            "transition_survival_rate": tsurv,
            "state_probs": state_probs.tolist(),
        }
        if forward_target is not None:
            result["transition_mi"] = self.compute_transition_mi(states, forward_target)
        return result
