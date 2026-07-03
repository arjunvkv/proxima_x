from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from research.information_discovery.mi_estimator import MIEstimator
from research.information_discovery.feature_scorer import FeatureScorer


class InformationCompression:
    def __init__(self, mi_estimator: Optional[MIEstimator] = None):
        self.mi = mi_estimator or MIEstimator(n_bins=20)

    def select_by_mutual_info(self, features: dict[str, NDArray], target: NDArray, top_k: int = 20) -> list[str]:
        mi_values: list[tuple[str, float]] = []
        for fname, farr in features.items():
            common = min(len(farr), len(target))
            mi = self.mi.mutual_info(farr[:common], target[:common])
            mi_values.append((fname, mi))
        mi_values.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in mi_values[:top_k]]

    def select_by_threshold(self, features: dict[str, NDArray], target: NDArray, threshold: float = 0.01) -> list[str]:
        selected: list[str] = []
        for fname, farr in features.items():
            common = min(len(farr), len(target))
            mi = self.mi.mutual_info(farr[:common], target[:common])
            if mi > threshold:
                selected.append(fname)
        return selected

    def eliminate_redundant(self, features: dict[str, NDArray], target: NDArray, mi_threshold: float = 0.005, redundancy_threshold: float = 0.7) -> list[str]:
        candidates = self.select_by_threshold(features, target, mi_threshold)
        if not candidates:
            return []
        selected: list[str] = [candidates[0]]
        for candidate in candidates[1:]:
            redundant = False
            for existing in selected:
                common = min(len(features[candidate]), len(features[existing]))
                mi_between = self.mi.mutual_info(features[candidate][:common], features[existing][:common])
                h_candidate = self.mi.entropy(features[candidate][:common])
                if h_candidate > 1e-10 and mi_between / h_candidate > redundancy_threshold:
                    redundant = True
                    break
            if not redundant:
                selected.append(candidate)
        return selected

    def information_bottleneck(self, features: dict[str, NDArray], target: NDArray, compression_ratio: float = 0.5) -> list[str]:
        scorer = FeatureScorer(mi_estimator=self.mi)
        scores = scorer.score_all_features(features, {"target": target})
        total_mi = sum(s.information_gain for s in scores)
        if total_mi < 1e-10:
            return []
        selected: list[str] = []
        accumulated = 0.0
        sorted_scores = sorted(scores, key=lambda x: x.information_gain, reverse=True)
        target_mi = total_mi * compression_ratio
        for s in sorted_scores:
            if accumulated >= target_mi:
                break
            selected.append(s.name)
            accumulated += s.information_gain
        return selected

    def multi_target_selection(self, features: dict[str, NDArray], targets: dict[str, NDArray], top_k_per_target: int = 10) -> list[str]:
        selected_set: set[str] = set()
        for tname, tarr in targets.items():
            selected = self.select_by_mutual_info(features, tarr, top_k_per_target)
            selected_set.update(selected)
        return list(selected_set)

    def rank_by_information_density(self, features: dict[str, NDArray], target: NDArray) -> list[tuple[str, float]]:
        density: list[tuple[str, float]] = []
        for fname, farr in features.items():
            common = min(len(farr), len(target))
            mi = self.mi.mutual_info(farr[:common], target[:common])
            density.append((fname, mi))
        density.sort(key=lambda x: x[1], reverse=True)
        return density
