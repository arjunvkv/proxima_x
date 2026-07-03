from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from research.mechanism_discovery.base import BaseMechanism, MechanismScore


class StateLifecycle(BaseMechanism):
    MAX_ABSENCE: int = 10
    HALF_LIFE_BINS: int = 50

    def __init__(self, name: str = "state_lifecycle", category: str = "state_dynamics", min_duration: int = 5) -> None:
        super().__init__(name, category)
        self.min_duration = min_duration
        self._state_contribution: NDArray = np.array([], dtype=np.float64)

    def compute(self, data: dict[str, NDArray], states: Optional[NDArray] = None) -> dict[str, Any]:
        states_arr = np.asarray(data.get("states", states if states is not None else []), dtype=np.int32)
        price = np.asarray(data.get("price", []), dtype=np.float64)
        returns = np.asarray(data.get("returns", []), dtype=np.float64)

        N = len(states_arr)
        if N == 0:
            return self._empty_result()

        price = price if len(price) == N else np.zeros(N, dtype=np.float64)
        returns = returns if len(returns) == N else np.zeros(N, dtype=np.float64)

        unique_states = np.unique(states_arr[states_arr >= 0])
        state_ids = unique_states.astype(int).tolist() if len(unique_states) > 0 else []

        birth_events, birth_probability = self._detect_births(states_arr, N)
        death_events, death_probability = self._detect_deaths(states_arr, state_ids, N)
        mutation_matrix, mutation_probability, mutation_rate = self._detect_mutations(states_arr, state_ids, N)
        survival_curves, survival_probability, state_half_lives, state_mean_lifetimes = self._compute_survival(
            states_arr, state_ids, N
        )
        lifecycle_phases, regime_shift_detected, n_state_births, n_state_deaths = self._compute_lifecycle_model(
            states_arr, state_ids, birth_events, death_events, returns, N
        )
        net_state_growth = n_state_births - n_state_deaths

        self._state_contribution = lifecycle_phases.astype(np.float64) / max(1.0, float(np.max(lifecycle_phases)) if np.max(lifecycle_phases) > 0 else 1.0)

        return {
            "birth_probability": birth_probability,
            "death_probability": death_probability,
            "mutation_probability": mutation_probability,
            "survival_probability": survival_probability,
            "birth_events": birth_events,
            "death_events": death_events,
            "mutation_matrix": mutation_matrix,
            "mutation_rate": mutation_rate,
            "survival_curves": survival_curves,
            "state_half_lives": state_half_lives,
            "state_mean_lifetimes": state_mean_lifetimes,
            "lifecycle_phases": lifecycle_phases,
            "regime_shift_detected": regime_shift_detected,
            "n_state_births": n_state_births,
            "n_state_deaths": n_state_deaths,
            "net_state_growth": net_state_growth,
        }

    def get_state_contribution(self) -> NDArray:
        return self._state_contribution

    def _detect_births(self, states: NDArray, N: int) -> tuple[list[dict[str, Any]], float]:
        birth_events: list[dict[str, Any]] = []
        births = 0
        if N == 0:
            return birth_events, 0.0

        seen: dict[int, int] = {}
        last_seen: dict[int, int] = {}
        for t in range(N):
            s = int(states[t])
            if s < 0:
                continue
            if s not in seen:
                seen[s] = t
                birth_events.append({"state_id": s, "birth_time": t, "birth_condition": "first_appearance"})
                births += 1
            else:
                last_t = last_seen.get(s, seen[s])
                if t - last_t > self.MAX_ABSENCE:
                    birth_events.append({"state_id": s, "birth_time": t, "birth_condition": "reemergence"})
                    births += 1
            last_seen[s] = t

        birth_probability = births / float(N) if N > 0 else 0.0
        return birth_events, birth_probability

    def _detect_deaths(self, states: NDArray, state_ids: list[int], N: int) -> tuple[list[dict[str, Any]], float]:
        death_events: list[dict[str, Any]] = []
        deaths = 0
        if N == 0 or not state_ids:
            return death_events, 0.0

        for sid in state_ids:
            occurrences = np.where(states == sid)[0]
            if len(occurrences) == 0:
                continue
            birth_time = int(occurrences[0])
            last_time = int(occurrences[-1])
            duration_lived = last_time - birth_time
            gap = N - 1 - last_time
            if gap > max(self.min_duration, self.MAX_ABSENCE):
                death_events.append({
                    "state_id": sid,
                    "death_time": last_time,
                    "duration_lived": duration_lived,
                })
                deaths += 1

        death_probability = deaths / float(len(state_ids)) if state_ids else 0.0
        return death_events, death_probability

    def _detect_mutations(self, states: NDArray, state_ids: list[int], N: int) -> tuple[NDArray, float, float]:
        n_states = len(state_ids)
        if n_states == 0 or N < 2:
            return np.zeros((0, 0), dtype=np.float64), 0.0, 0.0

        sid_to_idx = {sid: i for i, sid in enumerate(state_ids)}
        half = N // 2
        first_half = np.zeros((n_states, n_states), dtype=np.float64)
        second_half = np.zeros((n_states, n_states), dtype=np.float64)

        half1_end = half
        half2_start = half

        for t in range(1, half1_end):
            prev = int(states[t - 1])
            curr = int(states[t])
            if prev < 0 or curr < 0:
                continue
            if prev in sid_to_idx and curr in sid_to_idx:
                first_half[sid_to_idx[prev], sid_to_idx[curr]] += 1.0

        for t in range(half2_start + 1, N):
            prev = int(states[t - 1])
            curr = int(states[t])
            if prev < 0 or curr < 0:
                continue
            if prev in sid_to_idx and curr in sid_to_idx:
                second_half[sid_to_idx[prev], sid_to_idx[curr]] += 1.0

        row_sums1 = np.sum(first_half, axis=1, keepdims=True)
        row_sums1 = np.where(row_sums1 > 0, row_sums1, 1.0)
        first_half_norm = first_half / row_sums1

        row_sums2 = np.sum(second_half, axis=1, keepdims=True)
        row_sums2 = np.where(row_sums2 > 0, row_sums2, 1.0)
        second_half_norm = second_half / row_sums2

        diff = first_half_norm - second_half_norm
        mutation_rate = float(np.linalg.norm(diff, ord="fro"))

        total_transitions = int(np.sum(first_half)) + int(np.sum(second_half))
        mutations = int(np.sum(diff != 0.0))
        mutation_probability = mutations / float(max(1, n_states * n_states))

        full_matrix = (first_half + second_half) / 2.0
        row_sums_full = np.sum(full_matrix, axis=1, keepdims=True)
        row_sums_full = np.where(row_sums_full > 0, row_sums_full, 1.0)
        full_matrix_norm = full_matrix / row_sums_full

        return full_matrix_norm, mutation_probability, mutation_rate

    def _compute_survival(
        self, states: NDArray, state_ids: list[int], N: int
    ) -> tuple[dict[int, NDArray], float, dict[int, float], dict[int, float]]:
        survival_curves: dict[int, NDArray] = {}
        state_half_lives: dict[int, float] = {}
        state_mean_lifetimes: dict[int, float] = {}

        if N == 0 or not state_ids:
            return survival_curves, 0.0, state_half_lives, state_mean_lifetimes

        for sid in state_ids:
            occurrences = np.where(states == sid)[0]
            if len(occurrences) < 2:
                survival_curves[int(sid)] = np.ones(self.HALF_LIFE_BINS, dtype=np.float64)
                state_half_lives[int(sid)] = float(self.HALF_LIFE_BINS)
                state_mean_lifetimes[int(sid)] = 0.0
                continue

            durations = np.diff(occurrences)
            if len(durations) == 0:
                survival_curves[int(sid)] = np.ones(self.HALF_LIFE_BINS, dtype=np.float64)
                state_half_lives[int(sid)] = float(self.HALF_LIFE_BINS)
                state_mean_lifetimes[int(sid)] = 0.0
                continue

            max_dur = int(np.max(durations)) + 1
            bins = min(self.HALF_LIFE_BINS, max_dur)
            if bins < 2:
                bins = 2
            hist, edges = np.histogram(durations, bins=bins, range=(0, float(max_dur)), density=True)
            cdf = np.cumsum(hist * np.diff(edges))
            survival = 1.0 - cdf
            survival = np.clip(survival, 0.0, 1.0)
            if len(survival) < self.HALF_LIFE_BINS:
                survival = np.pad(survival, (0, self.HALF_LIFE_BINS - len(survival)), mode="edge")
            survival = survival[:self.HALF_LIFE_BINS]

            survival_curves[int(sid)] = survival
            state_mean_lifetimes[int(sid)] = float(np.mean(durations))

            half_life_idx = int(np.searchsorted(survival, 0.5, side="right"))
            state_half_lives[int(sid)] = float(half_life_idx) if half_life_idx < len(survival) else float(len(survival))

        all_survival = np.mean(list(survival_curves.values()), axis=0) if survival_curves else np.ones(self.HALF_LIFE_BINS, dtype=np.float64)
        survival_probability = float(np.mean(all_survival))

        return survival_curves, survival_probability, state_half_lives, state_mean_lifetimes

    def _compute_lifecycle_model(
        self,
        states: NDArray,
        state_ids: list[int],
        birth_events: list[dict[str, Any]],
        death_events: list[dict[str, Any]],
        returns: NDArray,
        N: int,
    ) -> tuple[NDArray, bool, int, int]:
        lifecycle_phases = np.full(N, 1, dtype=np.int32)
        n_state_births = len(birth_events)
        n_state_deaths = len(death_events)

        if N == 0:
            return lifecycle_phases, False, 0, 0

        death_times = {e["state_id"]: e["death_time"] for e in death_events}
        birth_times = {e["state_id"]: e["birth_time"] for e in birth_events}

        state_ages: dict[int, int] = {}
        state_max_ages: dict[int, int] = {}

        for sid in state_ids:
            occ = np.where(states == sid)[0]
            if len(occ) == 0:
                continue
            state_max_ages[sid] = int(occ[-1]) - int(occ[0]) + 1

        for t in range(N):
            s = int(states[t])
            if s < 0:
                continue
            if s not in state_ages:
                state_ages[s] = 0
                if s in death_times:
                    lifecycle_phases[t] = 0
                continue

            state_ages[s] += 1
            max_age = state_max_ages.get(s, 1)
            age_ratio = state_ages[s] / float(max(1, max_age))

            if s in death_times and t >= death_times[s]:
                lifecycle_phases[t] = 3
            elif age_ratio > 0.75:
                lifecycle_phases[t] = 2
            elif age_ratio > 0.25:
                lifecycle_phases[t] = 1
            else:
                lifecycle_phases[t] = 0

        regime_shift_detected = False
        if len(returns) == N:
            ret_std = float(np.nanstd(returns))
            if ret_std > 1e-12:
                for e in death_events:
                    dt = e["death_time"]
                    if 0 <= dt < N:
                        local_ret = np.abs(returns[max(0, dt - 2):min(N, dt + 3)])
                        if len(local_ret) > 0 and float(np.mean(local_ret)) > 2.0 * ret_std:
                            regime_shift_detected = True
                            break

        return lifecycle_phases, regime_shift_detected, n_state_births, n_state_deaths

    def _empty_result(self) -> dict[str, Any]:
        self._state_contribution = np.zeros(1, dtype=np.float64)
        return {
            "birth_probability": 0.0,
            "death_probability": 0.0,
            "mutation_probability": 0.0,
            "survival_probability": 0.0,
            "birth_events": [],
            "death_events": [],
            "mutation_matrix": np.zeros((0, 0), dtype=np.float64),
            "mutation_rate": 0.0,
            "survival_curves": {},
            "state_half_lives": {},
            "state_mean_lifetimes": {},
            "lifecycle_phases": np.array([1], dtype=np.int32),
            "regime_shift_detected": False,
            "n_state_births": 0,
            "n_state_deaths": 0,
            "net_state_growth": 0,
        }
