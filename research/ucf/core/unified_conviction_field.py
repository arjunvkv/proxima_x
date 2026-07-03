from __future__ import annotations

import math
import time
from collections import Counter
from typing import Any

from .adaptive_weight_engine import AdaptiveWeightEngine
from .bidirectional_fusion import BidirectionalFusionLayer


class UnifiedConvictionField:
    def __init__(self) -> None:
        self.weight_engine = AdaptiveWeightEngine()
        self.fusion_layer = BidirectionalFusionLayer()
        self.history: list[dict[str, Any]] = []

    def compute(
        self,
        symbols: list[str],
        technical_convictions: dict[str, dict[str, Any]],
        fundamental_convictions: dict[str, dict[str, Any]],
        exposure_convictions: dict[str, dict[str, Any]],
        regime_context: dict[str, Any],
    ) -> dict[str, Any]:
        weights = self.weight_engine.compute_weights(regime_context)

        states: list[dict[str, Any]] = []
        for symbol in symbols:
            tech = technical_convictions.get(symbol, {"conviction": 0.0, "direction": 0, "stability": 0.5})
            fund = fundamental_convictions.get(symbol, {"conviction": 0.0, "direction": 0, "stability": 0.5})
            expo = exposure_convictions.get(symbol, {"conviction": 0.0, "direction": 0, "stability": 0.5})
            state: dict[str, Any] = {
                "symbol": symbol,
                "technical": tech,
                "fundamental": fund,
                "exposure": expo,
                "weights": {
                    "technical_weight": weights["technical_weight"],
                    "fundamental_weight": weights["fundamental_weight"],
                    "exposure_weight": weights["exposure_weight"],
                },
            }
            states.append(state)

        fused_results = self.fusion_layer.fuse_batch(states)

        regime = regime_context.get("regime", "neutral")
        timestamp = time.time()

        field: dict[str, Any] = {}
        direction_counts: Counter[int] = Counter()
        total_agreement = 0.0

        for symbol in symbols:
            fused = fused_results.get(symbol, {})
            tech = technical_convictions.get(symbol, {"conviction": 0.0, "direction": 0, "stability": 0.5})
            fund = fundamental_convictions.get(symbol, {"conviction": 0.0, "direction": 0, "stability": 0.5})
            expo = exposure_convictions.get(symbol, {"conviction": 0.0, "direction": 0, "stability": 0.5})

            conviction_score = fused.get("conviction_score", 0.0)
            direction = fused.get("direction", 0)
            stability = fused.get("stability", 0.5)
            component_breakdown = fused.get(
                "component_breakdown",
                {
                    "technical_contribution": 0.0,
                    "fundamental_contribution": 0.0,
                    "exposure_contribution": 0.0,
                },
            )
            agreement = fused.get("agreement", 0.0)

            directions = [tech.get("direction", 0), fund.get("direction", 0), expo.get("direction", 0)]
            entropy = self._compute_entropy(directions)

            regime_adapted = regime != "neutral"

            field[symbol] = {
                "conviction_score": max(0.0, min(1.0, conviction_score)),
                "direction": direction,
                "stability": max(0.0, min(1.0, stability)),
                "entropy": max(0.0, min(1.0, entropy)),
                "component_breakdown": component_breakdown,
                "agreement": max(-1.0, min(1.0, agreement)),
                "regime_adapted": regime_adapted,
            }

            direction_counts[direction] += 1
            total_agreement += agreement

        field_coherence = total_agreement / len(symbols) if symbols else 0.0
        dominant_direction = direction_counts.most_common(1)[0][0] if direction_counts else 0

        result: dict[str, Any] = {
            "field": field,
            "weights": {
                "technical_weight": weights["technical_weight"],
                "fundamental_weight": weights["fundamental_weight"],
                "macro_weight": weights["macro_weight"],
                "exposure_weight": weights["exposure_weight"],
                "confidence": weights["confidence"],
            },
            "regime": regime,
            "field_coherence": field_coherence,
            "dominant_direction": dominant_direction,
            "timestamp": timestamp,
        }

        self.history.append(result)
        if len(self.history) > 100:
            self.history.pop(0)

        return result

    def get_field_snapshot(self) -> dict[str, Any]:
        if not self.history:
            return {}
        return dict(self.history[-1])

    def get_coherence_timeline(self, limit: int = 50) -> list[float]:
        return [entry["field_coherence"] for entry in self.history[-limit:]]

    def get_weight_evolution(self, limit: int = 50) -> list[dict[str, Any]]:
        return [dict(entry["weights"]) for entry in self.history[-limit:]]

    def reset(self) -> None:
        self.history.clear()

    def _compute_entropy(self, directions: list[int]) -> float:
        total = len(directions)
        if total == 0:
            return 0.0
        counts: Counter[int] = Counter(directions)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log(p)
        max_entropy = math.log(3)
        return entropy / max_entropy if max_entropy > 0 else 0.0
