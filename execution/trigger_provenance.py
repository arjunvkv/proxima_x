from __future__ import annotations

from typing import Dict, List, Optional


class TriggerProvenance:
    def __init__(self) -> None:
        self._log: List[Dict] = []

    def record(self, trade_id: str, symbol: str,
               trigger_layer: str, observer_state: str,
               calibration_state: str, reality_score: float,
               rejection_reason: str = "") -> Dict:
        entry = {
            "trade_id": trade_id,
            "symbol": symbol,
            "trigger_layer": trigger_layer,
            "observer_state": observer_state,
            "calibration_state": calibration_state,
            "reality_score": reality_score,
            "rejection_reason": rejection_reason,
        }
        self._log.append(entry)
        return entry

    def get_log(self) -> List[Dict]:
        return list(self._log)

    def summary(self) -> Dict:
        if not self._log:
            return {"total": 0}
        layers: Dict[str, int] = {}
        rejections: Dict[str, int] = {}
        for e in self._log:
            layers[e["trigger_layer"]] = layers.get(e["trigger_layer"], 0) + 1
            if e["rejection_reason"]:
                rejections[e["rejection_reason"]] = rejections.get(e["rejection_reason"], 0) + 1
        observer_count = sum(1 for e in self._log if e["observer_state"] == "EXECUTE")
        return {
            "total": len(self._log),
            "observer_approved": observer_count,
            "causal_fidelity_ratio": observer_count / len(self._log) if self._log else 0.0,
            "trigger_layer_distribution": layers,
            "rejection_reasons": rejections,
        }

    def clear(self) -> None:
        self._log.clear()
