from __future__ import annotations

import time
from typing import Any


class AdaptiveWeightEngine:
    def __init__(self) -> None:
        self._error_history: list[float] = []
        self._weight_history: list[dict[str, Any]] = []
        self._max_error_history = 100

    def compute_weights(self, context: dict) -> dict:
        regime = context.get("regime", "neutral")
        regime_stability = context.get("regime_stability", 0.5)
        fsv_entropy = context.get("fsv_entropy", 0.5)
        technical_volatility = context.get("technical_volatility", 0.5)
        recent_prediction_error = context.get("recent_prediction_error", 0.0)
        exposure_concentration = context.get("exposure_concentration", 0.5)

        if regime == "transition":
            weights = {
                "technical_weight": 0.20,
                "fundamental_weight": 0.40,
                "macro_weight": 0.30,
                "exposure_weight": 0.10,
            }
        elif regime == "risk_on" and regime_stability > 0.7:
            weights = {
                "technical_weight": 0.45,
                "fundamental_weight": 0.15,
                "macro_weight": 0.30,
                "exposure_weight": 0.10,
            }
        elif regime == "risk_off":
            weights = {
                "technical_weight": 0.15,
                "fundamental_weight": 0.30,
                "macro_weight": 0.20,
                "exposure_weight": 0.35,
            }
        else:
            weights = {
                "technical_weight": 0.30,
                "fundamental_weight": 0.30,
                "macro_weight": 0.20,
                "exposure_weight": 0.20,
            }

        if recent_prediction_error > 0.3:
            shift = min(recent_prediction_error * 0.3, 0.15)
            reduction = min(shift, weights["technical_weight"] - 0.05)
            weights["technical_weight"] -= reduction
            weights["fundamental_weight"] += reduction

        if fsv_entropy > 0.7:
            shift = min((fsv_entropy - 0.7) * 0.4, 0.15)
            reduction = min(shift, weights["exposure_weight"] - 0.05)
            weights["exposure_weight"] -= reduction
            weights["macro_weight"] += reduction

        weights = self.normalize_weights(weights)

        confidence = self._compute_confidence(weights, regime_stability, recent_prediction_error)

        result: dict[str, Any] = {
            "technical_weight": weights["technical_weight"],
            "fundamental_weight": weights["fundamental_weight"],
            "macro_weight": weights["macro_weight"],
            "exposure_weight": weights["exposure_weight"],
            "regime": regime,
            "confidence": confidence,
        }

        self._weight_history.append({"timestamp": time.time(), **result})

        return result

    def update_weight_memory(self, error_signal: float) -> None:
        self._error_history.append(error_signal)
        if len(self._error_history) > self._max_error_history:
            self._error_history.pop(0)

    def normalize_weights(self, weights: dict) -> dict:
        floor = 0.05
        for key in ("technical_weight", "fundamental_weight", "macro_weight", "exposure_weight"):
            if key in weights:
                weights[key] = max(weights[key], floor)

        total = sum(weights.get(k, 0.0) for k in ("technical_weight", "fundamental_weight", "macro_weight", "exposure_weight"))

        if total > 0:
            for key in ("technical_weight", "fundamental_weight", "macro_weight", "exposure_weight"):
                if key in weights:
                    weights[key] = weights[key] / total

        return weights

    def get_weight_history(self) -> list[dict]:
        return list(self._weight_history)

    def get_default_weights(self) -> dict:
        return {
            "technical_weight": 0.35,
            "fundamental_weight": 0.30,
            "macro_weight": 0.20,
            "exposure_weight": 0.15,
        }

    def _compute_confidence(self, weights: dict, regime_stability: float, recent_prediction_error: float) -> float:
        base = regime_stability * 0.6 + (1.0 - recent_prediction_error) * 0.4
        return max(0.0, min(1.0, base))
