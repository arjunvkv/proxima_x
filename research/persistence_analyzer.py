from __future__ import annotations

import numpy as np
import numba
from numpy.typing import NDArray


@numba.jit(nopython=True, cache=True)
def _find_runs(state_series: NDArray[np.int32]) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]]:
    n = len(state_series)
    if n == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    max_runs = n
    values = np.zeros(max_runs, dtype=np.int32)
    starts = np.zeros(max_runs, dtype=np.int32)
    lengths = np.zeros(max_runs, dtype=np.int32)
    run_count = 0
    values[0] = state_series[0]
    starts[0] = 0
    lengths[0] = 1
    for i in range(1, n):
        if state_series[i] == state_series[i - 1]:
            lengths[run_count] += 1
        else:
            run_count += 1
            values[run_count] = state_series[i]
            starts[run_count] = i
            lengths[run_count] = 1
    run_count += 1
    return values[:run_count], starts[:run_count], lengths[:run_count]


@numba.jit(nopython=True, cache=True)
def _collect_durations(values: NDArray[np.int32], lengths: NDArray[np.int32]) -> tuple[NDArray[np.int32], list[NDArray[np.int32]]]:
    unique_vals = np.unique(values)
    durations_list: list[NDArray[np.int32]] = []
    for v in unique_vals:
        count = 0
        for j in range(len(values)):
            if values[j] == v:
                count += 1
        d = np.zeros(count, dtype=np.int32)
        idx = 0
        for j in range(len(values)):
            if values[j] == v:
                d[idx] = lengths[j]
                idx += 1
        durations_list.append(d)
    return unique_vals, durations_list


@numba.jit(nopython=True, cache=True)
def _empirical_survival(durations: NDArray[np.int32]) -> NDArray[np.float64]:
    if len(durations) == 0:
        return np.zeros(1, dtype=np.float64)
    max_len = int(np.max(durations))
    survival = np.zeros(max_len + 1, dtype=np.float64)
    for t in range(1, max_len + 1):
        count = 0
        for d in durations:
            if d >= t:
                count += 1
        survival[t] = count / len(durations)
    return survival


@numba.jit(nopython=True, cache=True)
def _half_life_from_survival(survival: NDArray[np.float64]) -> float:
    for t in range(1, len(survival)):
        if survival[t] < 0.5:
            return float(t)
    return float(len(survival) - 1)


@numba.jit(nopython=True, cache=True)
def _transition_prob_by_lag(state_series: NDArray[np.int32], max_lag: int) -> NDArray[np.float32]:
    n = len(state_series)
    counts_lag = np.zeros(max_lag + 1, dtype=np.int32)
    change_counts = np.zeros(max_lag + 1, dtype=np.int32)
    for i in range(1, n):
        current_lag = 1
        for lag in range(1, min(i + 1, max_lag + 1)):
            if state_series[i - lag] == state_series[i - 1]:
                if lag == 1:
                    current_lag = 1
                else:
                    current_lag = lag
            else:
                break
        if current_lag >= 1 and current_lag <= max_lag:
            counts_lag[current_lag] += 1
            if state_series[i] != state_series[i - 1]:
                change_counts[current_lag] += 1
    probs = np.zeros(max_lag, dtype=np.float32)
    for lag in range(1, max_lag + 1):
        if counts_lag[lag] > 0:
            probs[lag - 1] = change_counts[lag] / counts_lag[lag]
    return probs


class PersistenceAnalyzer:

    def __init__(self, min_duration: int = 2) -> None:
        self.min_duration = min_duration

    def compute_state_durations(self, state_series: NDArray[np.int32]) -> dict[int, NDArray[np.int32]]:
        values, _, lengths = _find_runs(state_series)
        unique_vals, durations_list = _collect_durations(values, lengths)
        result: dict[int, NDArray[np.int32]] = {}
        for i in range(len(unique_vals)):
            v = int(unique_vals[i])
            d = durations_list[i]
            result[v] = d
        return result

    def compute_average_duration(self, state_series: NDArray[np.int32]) -> dict[int, float]:
        durations = self.compute_state_durations(state_series)
        result: dict[int, float] = {}
        for state_id, durs in durations.items():
            if len(durs) > 0:
                result[state_id] = float(np.mean(durs.astype(np.float64)))
            else:
                result[state_id] = 0.0
        return result

    def compute_state_half_life(self, state_series: NDArray[np.int32]) -> dict[int, float]:
        durations = self.compute_state_durations(state_series)
        result: dict[int, float] = {}
        for state_id, durs in durations.items():
            survival = _empirical_survival(durs)
            result[state_id] = _half_life_from_survival(survival)
        return result

    def compute_persistence_score(self, state_series: NDArray[np.int32]) -> dict[int, float]:
        avg_durations = self.compute_average_duration(state_series)
        result: dict[int, float] = {}
        for state_id, avg in avg_durations.items():
            result[state_id] = avg / (avg + 1.0)
        return result

    def compute_transition_probability_given_duration(self, state_series: NDArray[np.int32], max_lag: int = 20) -> NDArray[np.float32]:
        return _transition_prob_by_lag(state_series, max_lag)

    def classify_states(self, state_series: NDArray[np.int32]) -> dict[str, list[int]]:
        avg_durations = self.compute_average_duration(state_series)
        result: dict[str, list[int]] = {
            "transient": [],
            "normal": [],
            "persistent": [],
            "stable": [],
        }
        for state_id, avg in avg_durations.items():
            if avg < 3:
                result["transient"].append(state_id)
            elif avg < 10:
                result["normal"].append(state_id)
            elif avg < 50:
                result["persistent"].append(state_id)
            else:
                result["stable"].append(state_id)
        return result

    def compute_all(self, state_series: NDArray[np.int32]) -> dict:
        return {
            "state_durations": self.compute_state_durations(state_series),
            "average_durations": self.compute_average_duration(state_series),
            "half_lives": self.compute_state_half_life(state_series),
            "persistence_scores": self.compute_persistence_score(state_series),
            "transition_prob_given_duration": self.compute_transition_probability_given_duration(state_series),
            "state_classifications": self.classify_states(state_series),
        }
