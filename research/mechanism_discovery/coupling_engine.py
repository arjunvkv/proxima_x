from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from research.information_discovery.mi_estimator import MIEstimator
from research.mechanism_discovery.base import BaseMechanism


class CouplingEngine:
    def __init__(self, mi_estimator: Optional[MIEstimator] = None):
        self.mi = mi_estimator or MIEstimator(n_bins=20)

    def compute_combined_contribution(self, mechanisms: dict[str, BaseMechanism], data: dict, states: NDArray) -> NDArray:
        contributions: list[NDArray] = []
        for name, mech in mechanisms.items():
            mech.reset()
            mech.compute(data, states)
            c = mech.get_state_contribution()
            if len(c) > 0:
                contributions.append(c)
        if not contributions:
            return np.zeros(len(states), dtype=np.float64)
        min_len = min(len(c) for c in contributions)
        aligned = np.column_stack([c[:min_len] for c in contributions])
        return np.mean(aligned, axis=1)

    def compute_combined_without(self, mechanisms: dict[str, BaseMechanism], exclude_name: str, data: dict, states: NDArray) -> NDArray:
        subset = {n: m for n, m in mechanisms.items() if n != exclude_name}
        return self.compute_combined_contribution(subset, data, states)

    def compute_coupling(self, mech1: BaseMechanism, mech2: BaseMechanism, data: dict, states: NDArray) -> dict[str, float]:
        mech1.reset(); mech1.compute(data, states); c1 = mech1.get_state_contribution()
        mech2.reset(); mech2.compute(data, states); c2 = mech2.get_state_contribution()
        if len(c1) < 2 or len(c2) < 2:
            return {"coupling_corr": 0.0, "coupling_mi": 0.0}
        min_len = min(len(c1), len(c2))
        corr = float(np.corrcoef(c1[:min_len], c2[:min_len])[0, 1])
        mi_val = self.mi.mutual_info(c1[:min_len], c2[:min_len])
        return {"coupling_corr": max(0.0, corr) if not np.isnan(corr) else 0.0, "coupling_mi": mi_val}

    def compute_all_couplings(self, mechanisms: dict[str, BaseMechanism], data: dict, states: NDArray) -> dict[tuple[str, str], dict[str, float]]:
        couplings: dict[tuple[str, str], dict[str, float]] = {}
        names = list(mechanisms.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                coupling = self.compute_coupling(mechanisms[names[i]], mechanisms[names[j]], data, states)
                couplings[(names[i], names[j])] = coupling
        return couplings

    def compute_coupling_information(self, mechanisms: dict[str, BaseMechanism], data: dict, states: NDArray, target: NDArray) -> dict[str, Any]:
        combined = self.compute_combined_contribution(mechanisms, data, states)
        if len(combined) < 2:
            return {"coupling_information_gain": 0.0, "coupling_sid": 0.0, "coupling_sir": 0.0}
        min_len = min(len(combined), len(target))
        ig = self.mi.mutual_info(combined[:min_len], target[:min_len])
        n_bins = min(10, len(np.unique(combined[:min_len])))
        if n_bins < 2:
            return {"coupling_information_gain": ig, "coupling_sid": 0.0, "coupling_sir": 0.0}
        _, edges = self.mi._discretize(combined[:min_len], n_bins)
        labels = np.digitize(combined[:min_len], edges[:-1]).astype(np.int32)
        h_y = self.mi.entropy(target[:min_len])
        sid = 0.0
        for s in np.unique(labels):
            mask = labels == s
            if mask.sum() < 2:
                continue
            h_y_given_s = self.mi.entropy(target[:min_len][mask])
            sid += (mask.sum() / len(labels)) * (h_y - h_y_given_s)
        complexity = len(mechanisms) + len(np.unique(labels))
        sir = sid / complexity if complexity > 0 else 0.0
        return {"coupling_information_gain": ig, "coupling_sid": max(0.0, sid), "coupling_sir": max(0.0, sir)}

    def detect_emergence(self, mechanisms: dict[str, BaseMechanism], data: dict, states: NDArray, target: NDArray) -> dict[str, Any]:
        combined = self.compute_combined_contribution(mechanisms, data, states)
        if len(combined) < 2:
            return {"emergent_information_gain": 0.0, "emergence_detected": False}
        min_len = min(len(combined), len(target))
        combined_mi = self.mi.mutual_info(combined[:min_len], target[:min_len])
        individual_mis: list[float] = []
        for name, mech in mechanisms.items():
            mech.reset(); mech.compute(data, states)
            c = mech.get_state_contribution()
            if len(c) >= 2:
                cl = min(len(c), len(target))
                individual_mis.append(self.mi.mutual_info(c[:cl], target[:cl]))
        max_individual = max(individual_mis) if individual_mis else 0.0
        emergent = max(0.0, combined_mi - max_individual)
        return {"combined_mi": combined_mi, "max_individual_mi": max_individual, "emergent_information_gain": emergent, "emergence_detected": emergent > 0.01}

    def compute_hierarchy(self, mechanisms: dict[str, BaseMechanism], data: dict, states: NDArray, transitions: NDArray, outcomes: NDArray) -> dict[str, Any]:
        hierarchy: dict[str, Any] = {"mechanism_to_state": {}, "state_to_transition": {}, "transition_to_outcome": {}}
        for name, mech in mechanisms.items():
            mech.reset(); mech.compute(data, states)
            c = mech.get_state_contribution()
            if len(c) < 2:
                continue
            min_len_s = min(len(c), len(states))
            mi_s = self.mi.mutual_info(c[:min_len_s], states[:min_len_s].astype(np.float64))
            hierarchy["mechanism_to_state"][name] = mi_s
        unique_states = np.unique(states[states >= 0])
        if len(unique_states) >= 2 and len(transitions) >= 2:
            min_len_t = min(len(states), len(transitions))
            mi_t = self.mi.mutual_info(states[:min_len_t].astype(np.float64), transitions[:min_len_t].astype(np.float64))
            hierarchy["state_to_transition"]["overall"] = mi_t
        if len(outcomes) >= 2 and len(transitions) >= 2:
            min_len_o = min(len(transitions), len(outcomes))
            mi_o = self.mi.mutual_info(transitions[:min_len_o].astype(np.float64), outcomes[:min_len_o].astype(np.float64))
            hierarchy["transition_to_outcome"]["overall"] = mi_o
        return hierarchy
