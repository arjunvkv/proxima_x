from __future__ import annotations

import time
from statistics import mean
from typing import Any, Dict, List, Tuple
import numpy as np


class RealityGate:
    DEFAULT_LAYERS = ("tpi", "observer", "regime", "calibration")
    WEIGHT_EXPONENTS = {"tpi": 0.40, "observer": 0.30, "regime": 0.20, "calibration": 0.10}

    def __init__(self) -> None:
        self.trust_weights: Dict[str, float] = {
            layer: 1.0 for layer in self.DEFAULT_LAYERS
        }
        self.trust_decay_rate: float = 0.95
        self.min_trust: float = 0.1
        self.calibration_veto_threshold: float = 0.15
        self._score_history: List[float] = []  # all-candidate scores (for adaptive threshold)
        self._score_window: int = 200
        self._entry_percentile: float = 0.70  # P70 for entry
        self._maintain_percentile: float = 0.55  # P55 for maintain (hysteresis)
        self._last_update: float = time.time()
        self._regime_trust: Dict[str, Dict[str, float]] = {}

    def update_trust(self, honesty_scores) -> None:
        now = time.time()
        elapsed = max(now - self._last_update, 0.0)
        decay_factor = self.trust_decay_rate ** max(elapsed, 1.0)

        for layer in self.trust_weights:
            decayed = self.trust_weights[layer] * decay_factor
            self.trust_weights[layer] = self._clamp(decayed)

        for score in honesty_scores:
            layer_name = getattr(score, "layer_name", None)
            raw_score = getattr(score, "score", None)
            if layer_name is None or raw_score is None:
                continue
            normalized = self._clamp(raw_score / 100.0)
            self.trust_weights[layer_name] = normalized

            regime = getattr(score, "regime", None)
            if regime:
                if regime not in self._regime_trust:
                    self._regime_trust[regime] = {}
                self._regime_trust[regime][layer_name] = normalized

        self._last_update = now

    def record_candidate_score(self, score: float) -> None:
        """Record a pre-gate candidate score for unbiased threshold calibration.
        Must be called upstream (before RealityGate evaluates) for ALL candidates."""
        self._score_history.append(score)
        if len(self._score_history) > self._score_window:
            self._score_history.pop(0)

    def compute_execution_score(self, perception_state: Dict[str, Any]) -> float:
        tpi = abs(perception_state.get("tpi", 0.0))
        observer_confidence = perception_state.get("observer_confidence", 0.0)
        regime = perception_state.get("regime", "default")

        calibration_weight = self.regime_conditioned_weight(regime, "calibration")
        if calibration_weight < self.calibration_veto_threshold:
            return 0.0

        tpi_weight = self.regime_conditioned_weight(regime, "tpi")
        observer_weight = self.regime_conditioned_weight(regime, "observer")
        regime_weight = self.regime_conditioned_weight(regime, "regime")

        components = {
            "tpi": tpi * tpi_weight,
            "observer": observer_confidence * observer_weight,
            "regime": regime_weight,
        }

        log_sum = sum(
            self.WEIGHT_EXPONENTS[k] * max(components.get(k, 0.0), 1e-10)
            for k in components
            if k in self.WEIGHT_EXPONENTS
        )
        norm = sum(self.WEIGHT_EXPONENTS.get(k, 0.0) for k in components)
        exec_score = log_sum / max(norm, 1e-10)

        return float(exec_score)

    def _adaptive_threshold(self, percentile: float) -> float:
        if len(self._score_history) < 30:
            return 0.60  # conservative boot threshold until enough data
        arr = np.array(self._score_history)
        return float(np.percentile(arr, percentile * 100))

    def should_execute(self, perception_state: Dict[str, Any], is_already_in: bool = False) -> Tuple[bool, float]:
        exec_score = self.compute_execution_score(perception_state)
        if is_already_in:
            thresh = self._adaptive_threshold(self._maintain_percentile)
        else:
            thresh = self._adaptive_threshold(self._entry_percentile)
        return exec_score >= thresh, exec_score, thresh

    def decompose(self, perception_state: Dict[str, Any]) -> Dict[str, float]:
        tpi = abs(perception_state.get("tpi", 0.0))
        observer_confidence = perception_state.get("observer_confidence", 0.0)
        regime = perception_state.get("regime", "default")
        return {
            "calibration_veto": self.regime_conditioned_weight(regime, "calibration"),
            "tpi_contribution": tpi * self.regime_conditioned_weight(regime, "tpi"),
            "observer_contribution": observer_confidence * self.regime_conditioned_weight(regime, "observer"),
            "regime_contribution": self.regime_conditioned_weight(regime, "regime"),
        }

    def regime_conditioned_weight(self, regime: str, layer: str) -> float:
        if regime in self._regime_trust and layer in self._regime_trust[regime]:
            return self._regime_trust[regime][layer]
        return self.trust_weights.get(layer, 1.0)

    def adapt_threshold(self, honesty_scores) -> None:
        if not honesty_scores:
            return
        scores = [getattr(s, "score", 0.0) / 100.0 for s in honesty_scores]
        avg_honesty = mean(scores)
        threshold = 0.5 * (1 - avg_honesty) + 0.05
        self.execution_threshold = self._clamp(threshold, lower=0.05, upper=0.50)

    def _clamp(self, value: float, lower: float = None, upper: float = 1.0) -> float:
        lower = self.min_trust if lower is None else lower
        return max(lower, min(value, upper))
