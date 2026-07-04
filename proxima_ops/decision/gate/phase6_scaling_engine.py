from __future__ import annotations

from typing import Any


STABILITY_TIERS = [
    {"label": "critical", "max_score": 0.30, "multiplier": 0.01},
    {"label": "low", "max_score": 0.50, "multiplier": 0.02},
    {"label": "medium", "max_score": 0.70, "multiplier": 0.03},
    {"label": "elevated", "max_score": 0.85, "multiplier": 0.05},
    {"label": "high", "max_score": 1.00, "multiplier": 1.00},
]


class Phase6ScalingEngine:
    def __init__(self, window_size: int = 20) -> None:
        self._alignment_history: list[float] = []
        self._rc_history: list[float] = []
        self._emd_history: list[float] = []
        self._window_size = window_size
        self._scaling_log: list[dict] = []

    def update(self, alignment: float, rc_veto_rate: float, emd_score: float) -> None:
        self._alignment_history.append(alignment)
        self._rc_history.append(rc_veto_rate)
        self._emd_history.append(emd_score)
        if len(self._alignment_history) > self._window_size:
            self._alignment_history.pop(0)
            self._rc_history.pop(0)
            self._emd_history.pop(0)

    def compute_stability_score(self) -> float:
        if len(self._alignment_history) < 5:
            return 0.0
        mean_align = sum(self._alignment_history) / len(self._alignment_history)
        mean_rc = sum(self._rc_history) / len(self._rc_history)
        mean_emd = sum(self._emd_history) / len(self._emd_history)
        score = (mean_align * 0.5) + ((1.0 - mean_rc) * 0.3) + ((1.0 - mean_emd) * 0.2)
        return max(0.0, min(1.0, score))

    def get_multiplier(self) -> float:
        score = self.compute_stability_score()
        for tier in STABILITY_TIERS:
            if score <= tier["max_score"]:
                return tier["multiplier"]
        return 0.01

    def evaluate(self, alignment: float, rc_veto_rate: float, emd_score: float) -> dict:
        self.update(alignment, rc_veto_rate, emd_score)
        score = self.compute_stability_score()
        multiplier = self.get_multiplier()
        for tier in STABILITY_TIERS:
            if score <= tier["max_score"]:
                label = tier["label"]
                break
        else:
            label = "critical"

        result = {
            "stability_score": round(score, 4),
            "stability_tier": label,
            "position_size_multiplier": multiplier,
            "alignment_mean": round(sum(self._alignment_history) / max(1, len(self._alignment_history)), 4),
            "rc_veto_mean": round(sum(self._rc_history) / max(1, len(self._rc_history)), 4),
            "emd_mean": round(sum(self._emd_history) / max(1, len(self._emd_history)), 4),
            "history_size": len(self._alignment_history),
        }
        self._scaling_log.append(result)
        return result

    def get_log(self) -> list[dict]:
        return list(self._scaling_log)
