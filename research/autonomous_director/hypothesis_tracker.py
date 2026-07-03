import json
import os
from datetime import datetime
from typing import Optional


class HypothesisTracker:
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(data_dir, exist_ok=True)
        self._path = os.path.join(data_dir, "hypotheses.json")
        self._hypotheses = {
            "energy_storage_alpha": {"confidence": 0.50, "evidence_for": 0, "evidence_against": 0, "history": []},
            "residual_alpha": {"confidence": 0.50, "evidence_for": 0, "evidence_against": 0, "history": []},
            "frequency_controller": {"confidence": 0.50, "evidence_for": 0, "evidence_against": 0, "history": []},
            "at_overlay": {"confidence": 0.50, "evidence_for": 0, "evidence_against": 0, "history": []},
            "persistence_forecast": {"confidence": 0.50, "evidence_for": 0, "evidence_against": 0, "history": []}}
        self._load()

    def _load(self):
        if os.path.exists(self._path):
            with open(self._path) as f:
                data = json.load(f)
                for k, v in data.items():
                    if k in self._hypotheses:
                        self._hypotheses[k] = v

    def _save(self):
        with open(self._path, "w") as f:
            json.dump(self._hypotheses, f, indent=2)

    def record_evidence(self, hypothesis: str, supports: bool, weight: float = 0.1):
        if hypothesis not in self._hypotheses:
            return
        h = self._hypotheses[hypothesis]
        if supports:
            h["evidence_for"] += 1
            h["confidence"] = min(1.0, h["confidence"] + weight)
        else:
            h["evidence_against"] += 1
            h["confidence"] = max(0.0, h["confidence"] - weight)
        h["history"].append({
            "timestamp": datetime.now().isoformat(),
            "supports": supports, "weight": weight,
            "new_confidence": round(h["confidence"], 3)})
        if len(h["history"]) > 1000:
            h["history"] = h["history"][-500:]
        self._save()

    def confidence(self, hypothesis: str) -> float:
        return self._hypotheses.get(hypothesis, {}).get("confidence", 0.5)

    def all_confidences(self) -> dict:
        return {k: round(v["confidence"], 3) for k, v in self._hypotheses.items()}

    def summary(self) -> dict:
        return dict(self._hypotheses)
