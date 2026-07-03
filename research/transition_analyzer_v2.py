from __future__ import annotations

import numpy as np
import numba
from numpy.typing import NDArray
from sklearn.metrics import mutual_info_score


@numba.jit(nopython=True, cache=True)
def _build_transition_matrix_inner(states: NDArray[np.int32], n_states: int) -> NDArray[np.float32]:
    mat = np.zeros((n_states, n_states), dtype=np.float32)
    for i in range(1, len(states)):
        frm = states[i - 1]
        to = states[i]
        if 0 <= frm < n_states and 0 <= to < n_states:
            mat[frm, to] += 1.0
    for i in range(n_states):
        row_sum = np.sum(mat[i])
        if row_sum > 0:
            mat[i] = mat[i] / row_sum
    return mat


@numba.jit(nopython=True, cache=True)
def _transition_velocity_inner(states: NDArray[np.int32], window: int) -> NDArray[np.float32]:
    n = len(states)
    result = np.zeros(n, dtype=np.float32)
    changes = np.zeros(n, dtype=np.int32)
    for i in range(1, n):
        changes[i] = 1 if states[i] != states[i - 1] else 0
    for i in range(window - 1, n):
        total = 0.0
        for j in range(i - window + 1, i + 1):
            total += changes[j]
        result[i] = total / window
    return result


@numba.jit(nopython=True, cache=True)
def _transition_persistence_inner(states: NDArray[np.int32], window: int) -> NDArray[np.float32]:
    velocity = _transition_velocity_inner(states, window)
    result = np.zeros(len(velocity), dtype=np.float32)
    for i in range(len(velocity)):
        result[i] = 1.0 - velocity[i]
    return result


@numba.jit(nopython=True, cache=True)
def _row_entropy(row: NDArray[np.float32]) -> float:
    ent = 0.0
    for i in range(len(row)):
        if row[i] > 0:
            ent -= row[i] * np.log2(row[i] + 1e-15)
    return ent


@numba.jit(nopython=True, cache=True)
def _transition_entropy_inner(mat: NDArray[np.float32]) -> NDArray[np.float32]:
    n = mat.shape[0]
    result = np.zeros(n, dtype=np.float32)
    for i in range(n):
        result[i] = _row_entropy(mat[i])
    return result


@numba.jit(nopython=True, cache=True)
def _find_cycles_inner(mat: NDArray[np.float32], max_len: int) -> list[list[int]]:
    n = mat.shape[0]
    cycles_list: list[list[int]] = []
    seen_cycle_strs: list[str] = []
    for start in range(n):
        path = np.zeros(max_len + 1, dtype=np.int32)
        path[0] = start
        depth = 1
        stack_pos = np.zeros(max_len + 1, dtype=np.int32)
        stack_pos[0] = 0
        while depth > 0:
            node = path[depth - 1]
            pos = stack_pos[depth - 1]
            found = False
            for nxt in range(pos, n):
                if mat[node, nxt] > 0:
                    stack_pos[depth - 1] = nxt + 1
                    cycle_start = -1
                    for pi in range(depth):
                        if path[pi] == nxt:
                            cycle_start = pi
                            break
                    if cycle_start >= 0:
                        cycle_len = depth - cycle_start
                        if cycle_len >= 2:
                            cycle_buf = np.zeros(cycle_len, dtype=np.int32)
                            for ci in range(cycle_len):
                                cycle_buf[ci] = path[cycle_start + ci]
                            cycle_str = ""
                            for ci in range(cycle_len):
                                if ci > 0:
                                    cycle_str += ","
                                cycle_str += str(int(cycle_buf[ci]))
                            is_dup = False
                            for si in range(len(seen_cycle_strs)):
                                if seen_cycle_strs[si] == cycle_str:
                                    is_dup = True
                                    break
                            if not is_dup:
                                seen_cycle_strs.append(cycle_str)
                                cycle_list: list[int] = []
                                for ci in range(cycle_len):
                                    cycle_list.append(int(cycle_buf[ci]))
                                cycles_list.append(cycle_list)
                    else:
                        if depth < max_len + 1:
                            path[depth] = nxt
                            stack_pos[depth] = 0
                            depth += 1
                        found = True
                        break
            if not found:
                depth -= 1
    return cycles_list


@numba.jit(nopython=True, cache=True)
def _compute_markovity_inner(states: NDArray[np.int32], max_lag: int) -> NDArray[np.float32]:
    n = len(states)
    mi_values = np.zeros(max_lag, dtype=np.float32)
    for lag in range(1, max_lag + 1):
        n_pairs = n - lag - 1
        if n_pairs <= 0:
            continue
        x = states[lag:n - 1]
        y = states[lag + 1:n]
        joint = np.zeros((np.max(states) + 1, np.max(states) + 1), dtype=np.float64)
        for k in range(n_pairs):
            joint[x[k], y[k]] += 1.0
        joint /= n_pairs
        px = np.sum(joint, axis=1)
        py = np.sum(joint, axis=0)
        mi = 0.0
        for i in range(joint.shape[0]):
            for j in range(joint.shape[1]):
                if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += joint[i, j] * np.log(joint[i, j] / (px[i] * py[j]) + 1e-15)
        mi_values[lag - 1] = mi
    return mi_values


class TransitionGraphAnalyzer:

    def __init__(self, max_lag: int = 10) -> None:
        self.max_lag = max_lag

    def build_transition_matrix(self, states: NDArray[np.int32], n_states: int) -> NDArray[np.float32]:
        return _build_transition_matrix_inner(states, n_states)

    def compute_transition_velocity(self, states: NDArray[np.int32], window: int = 20) -> NDArray[np.float32]:
        return _transition_velocity_inner(states, window)

    def compute_transition_persistence(self, states: NDArray[np.int32], window: int = 20) -> NDArray[np.float32]:
        return _transition_persistence_inner(states, window)

    def compute_transition_entropy(self, transition_matrix: NDArray[np.float32]) -> NDArray[np.float32]:
        return _transition_entropy_inner(transition_matrix)

    def compute_transition_stability(self, transition_matrix: NDArray[np.float32]) -> float:
        n = min(transition_matrix.shape[0], transition_matrix.shape[1])
        total = 0.0
        for i in range(n):
            total += transition_matrix[i, i]
        return total / n

    def find_transition_cycles(self, transition_matrix: NDArray[np.float32], max_cycle_length: int = 5) -> list[list[int]]:
        cycles = _find_cycles_inner(transition_matrix, max_cycle_length)
        deduped: list[list[int]] = []
        seen: set[str] = set()
        for cycle in cycles:
            key = "-".join(str(s) for s in cycle)
            if key not in seen:
                seen.add(key)
                deduped.append(cycle)
        return deduped

    def compute_markovity(self, states: NDArray[np.int32], max_lag: int = 10) -> NDArray[np.float32]:
        return _compute_markovity_inner(states, max_lag)

    def compute_all(self, states: NDArray[np.int32], n_states: int) -> dict:
        tm = self.build_transition_matrix(states, n_states)
        return {
            "transition_matrix": tm,
            "transition_velocity": self.compute_transition_velocity(states),
            "transition_persistence": self.compute_transition_persistence(states),
            "transition_entropy": self.compute_transition_entropy(tm),
            "transition_stability": self.compute_transition_stability(tm),
            "transition_cycles": self.find_transition_cycles(tm),
            "markovity": self.compute_markovity(states, self.max_lag),
        }
