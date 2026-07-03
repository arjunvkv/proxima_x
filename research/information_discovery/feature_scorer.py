from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from research.information_discovery.mi_estimator import MIEstimator


@dataclass
class FeatureScore:
    name: str
    information_gain: float = 0.0
    conditional_gain: float = 0.0
    stability: float = 0.0
    robustness: float = 0.0
    cross_asset_score: float = 0.0
    cross_regime_score: float = 0.0
    entropy: float = 0.0
    information_gain_ratio: float = 0.0
    mi_by_target: dict[str, float] = field(default_factory=dict)
    cmi_by_target: dict[str, float] = field(default_factory=dict)

    @property
    def composite_score(self) -> float:
        return (
            self.information_gain * 0.30
            + self.conditional_gain * 0.20
            + self.stability * 0.15
            + self.robustness * 0.10
            + self.cross_asset_score * 0.15
            + self.cross_regime_score * 0.10
        )

    @property
    def survives(self) -> bool:
        return self.information_gain > 0 and self.stability > 0


class FeatureScorer:
    def __init__(self, mi_estimator: Optional[MIEstimator] = None):
        self.mi = mi_estimator or MIEstimator(n_bins=20)

    def score_feature(self, name: str, feature: NDArray, targets: dict[str, NDArray], conditions: Optional[dict[str, NDArray]] = None) -> FeatureScore:
        mi_by_target: dict[str, float] = {}
        cmi_by_target: dict[str, float] = {}
        for tname, tarr in targets.items():
            common = min(len(feature), len(tarr))
            mi_by_target[tname] = self.mi.mutual_info(feature[:common], tarr[:common])
        if conditions:
            for tname, tarr in targets.items():
                cond_arr = conditions.get(tname, conditions.get("volatility"))
                if cond_arr is not None:
                    common = min(len(feature), len(tarr), len(cond_arr))
                    cmi_by_target[tname] = self.mi.conditional_mutual_info(feature[:common], tarr[:common], cond_arr[:common])
        avg_mi = float(np.mean(list(mi_by_target.values()))) if mi_by_target else 0.0
        avg_cmi = float(np.mean(list(cmi_by_target.values()))) if cmi_by_target else 0.0
        entropy = self.mi.entropy(feature)
        igr = self.mi.information_gain_ratio(feature, list(targets.values())[0]) if targets else 0.0
        return FeatureScore(
            name=name,
            information_gain=avg_mi,
            conditional_gain=avg_cmi,
            entropy=entropy,
            information_gain_ratio=igr,
            mi_by_target=mi_by_target,
            cmi_by_target=cmi_by_target,
        )

    def score_all_features(self, features: dict[str, NDArray], targets: dict[str, NDArray], conditions: Optional[dict[str, NDArray]] = None) -> list[FeatureScore]:
        scores: list[FeatureScore] = []
        for fname, farr in features.items():
            scores.append(self.score_feature(fname, farr, targets, conditions))
        return scores


class FeatureSurvivalEngine:
    def __init__(self, mi_threshold: float = 0.0, cmi_threshold: float = 0.0, stability_threshold: float = 0.0):
        self.mi_threshold = mi_threshold
        self.cmi_threshold = cmi_threshold
        self.stability_threshold = stability_threshold

    def filter_survivors(self, scores: list[FeatureScore]) -> list[FeatureScore]:
        survivors: list[FeatureScore] = []
        for s in scores:
            if s.information_gain <= self.mi_threshold:
                continue
            survivors.append(s)
        survivors.sort(key=lambda x: x.composite_score, reverse=True)
        return survivors

    def rank_survivors(self, survivors: list[FeatureScore]) -> list[FeatureScore]:
        return sorted(survivors, key=lambda x: x.composite_score, reverse=True)

    def get_surviving_names(self, survivors: list[FeatureScore]) -> list[str]:
        return [s.name for s in survivors]

    def apply_cross_asset_filter(self, survivors: list[FeatureScore], cross_asset_mi: dict[str, float]) -> list[FeatureScore]:
        filtered: list[FeatureScore] = []
        for s in survivors:
            cami = cross_asset_mi.get(s.name, 0.0)
            s.cross_asset_score = cami
            if cami <= 0:
                continue
            filtered.append(s)
        return filtered

    def apply_cross_regime_filter(self, survivors: list[FeatureScore], cross_regime_mi: dict[str, float]) -> list[FeatureScore]:
        filtered: list[FeatureScore] = []
        for s in survivors:
            crmi = cross_regime_mi.get(s.name, 0.0)
            s.cross_regime_score = crmi
            if crmi <= 0:
                continue
            filtered.append(s)
        return filtered
