"""
Causal Fusion Layer — Wave 12.

Binds field output to external reality constraints:
1. Causal anchoring (prediction vs realized drift binding)
2. Exogenous perturbation injection (controlled reality shock forcing)
"""
from layer7.causal_anchor import CausalAnchor
from layer7.perturbation_injector import PerturbationInjector
import random


class CausalFusionLayer:
    def __init__(self):
        self.anchor = CausalAnchor(window=32)
        self.injector = PerturbationInjector(intensity=0.08)

    def process(self, field_output: dict, realized: float) -> dict:
        signal = field_output["signal"]
        anchor = self.anchor.update(signal, realized)
        signal = anchor["anchored_signal"]
        signal, shocked = self.injector.inject(signal, field_output["instability"])
        return {
            "signal": signal,
            "alignment": anchor["alignment"],
            "drift": anchor["drift"],
            "error": anchor["error"],
            "shock": shocked,
        }

    def compute_ucf_metrics(self, ucf_field) -> dict[str, float]:
        if ucf_field is None:
            return {"alignment_drift": 0.0, "confidence_tension": 0.0}
        scores = [e.get("ucf_score", 0.0) for e in ucf_field.ranked_symbols]
        directions = [e.get("direction", 0) for e in ucf_field.ranked_symbols]
        mean_score = sum(scores) / len(scores) if scores else 0.0
        score_var = sum((s - mean_score) ** 2 for s in scores) / len(scores) if scores else 0.0
        alignment_drift = min(1.0, score_var * 2.0)
        unique_dirs = len(set(d for d in directions if d != 0))
        confidence_tension = unique_dirs / max(1, len(directions)) * ucf_field.field_coherence
        return {"alignment_drift": round(alignment_drift, 4), "confidence_tension": round(confidence_tension, 4)}
