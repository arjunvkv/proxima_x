from __future__ import annotations

from typing import Dict, Optional
import math
import numpy as np


class ObserverDecayEngine:
    __slots__ = ("_birth_state", "_half_life_ticks", "_lambda",
                 "_execute_threshold", "_tick_count", "_decayed",
                 "_rejected_signals")

    def __init__(self, half_life_ticks: int = 300,
                 execute_threshold: float = 0.55) -> None:
        self._birth_state: Dict = {}
        self._half_life_ticks = half_life_ticks
        self._lambda = math.log(2) / max(half_life_ticks, 1)
        self._execute_threshold = execute_threshold
        self._tick_count = 0
        self._decayed: Dict[str, bool] = {}
        # P0.13: Track rejected signals to measure selection bias
        self._rejected_signals: list[dict] = []

    def birth(self, signal_id: str, tpi_confidence: float,
              persistence_streak: int, normalized_entropy: float,
              regime: str) -> None:
        self._birth_state[signal_id] = {
            "tpi_confidence": tpi_confidence,
            "persistence_streak": persistence_streak,
            "normalized_entropy": normalized_entropy,
            "regime": regime,
            "birth_tick": self._tick_count,
        }
        self._decayed[signal_id] = False

    def compute(self, signal_id: str, current_persistence_streak: int,
                current_entropy: float, current_regime: str) -> Dict:
        birth = self._birth_state.get(signal_id)
        if birth is None:
            return {"confidence": 0.0, "state": "SUPPRESS",
                    "decayed": True, "reason": "NO_BIRTH_RECORD"}

        delta_ticks = self._tick_count - birth["birth_tick"]

        temporal = math.exp(-self._lambda * delta_ticks)

        bp = max(birth["persistence_streak"], 1)
        s_p = min(1.0, current_persistence_streak / bp)

        s_e = 1.0 - abs(current_entropy - birth["normalized_entropy"])

        s_r = 1.0 if current_regime == birth["regime"] else 0.6

        s = 0.5 * s_p + 0.3 * s_e + 0.2 * s_r

        base = float(birth["tpi_confidence"])
        confidence = base * temporal * max(0.0, min(1.0, s))

        if confidence >= self._execute_threshold:
            state = "EXECUTE"
        elif confidence >= 0.55:
            state = "HESITATE"
        elif confidence >= 0.35:
            state = "REVIEW"
        else:
            state = "SUPPRESS"

        if state != "EXECUTE" and self._decayed.get(signal_id, False) is False:
            self._decayed[signal_id] = True

        return {
            "confidence": float(confidence),
            "state": state,
            "temporal": float(temporal),
            "survival": float(s),
            "survival_components": {
                "persistence": float(s_p),
                "entropy": float(s_e),
                "regime": float(s_r),
            },
            "delta_ticks": delta_ticks,
            "decayed": self._decayed.get(signal_id, False),
        }

    def tick(self, n: int = 1) -> None:
        self._tick_count += n

    def get_active_signals(self) -> list:
        return [sid for sid, d in self._decayed.items() if not d]

    def summary(self) -> Dict:
        total = len(self._birth_state)
        active = len(self.get_active_signals())
        return {
            "total_signals": total,
            "active": active,
            "decayed": total - active,
            "tick_count": self._tick_count,
        }

    def reject(self, signal_id: str, reason: str,
               tpi_confidence: float = 0.0, persistence_streak: int = 0,
               normalized_entropy: float = 0.0, regime: str = "") -> None:
        """P0.13: Track a rejected signal to measure selection bias.
        Logs characteristics so we can compare accepted vs rejected populations.
        """
        self._rejected_signals.append({
            "signal_id": signal_id,
            "reason": reason,
            "tpi_confidence": tpi_confidence,
            "persistence_streak": persistence_streak,
            "normalized_entropy": normalized_entropy,
            "regime": regime,
            "reject_tick": self._tick_count,
        })
        if len(self._rejected_signals) > 500:
            self._rejected_signals = self._rejected_signals[-250:]

    def selection_bias(self) -> dict:
        """P0.13: Compare accepted vs rejected populations for bias detection."""
        total_births = len(self._birth_state)
        total_rejected = len(self._rejected_signals)
        if total_births + total_rejected == 0:
            return {"accept_rate": 0.0}
        birth_confidences = [b.get("tpi_confidence", 0) for b in self._birth_state.values()]
        reject_confidences = [r.get("tpi_confidence", 0) for r in self._rejected_signals]
        return {
            "accept_rate": total_births / max(total_births + total_rejected, 1),
            "total_births": total_births,
            "total_rejected": total_rejected,
            "mean_accepted_confidence": float(np.mean(birth_confidences)) if birth_confidences else 0.0,
            "mean_rejected_confidence": float(np.mean(reject_confidences)) if reject_confidences else 0.0,
            "bias_gap": float(np.mean(birth_confidences) - np.mean(reject_confidences))
                       if birth_confidences and reject_confidences else 0.0,
        }

    def clear(self) -> None:
        self._birth_state.clear()
        self._decayed.clear()
        self._rejected_signals.clear()
        self._tick_count = 0
