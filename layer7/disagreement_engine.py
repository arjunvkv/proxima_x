"""
Disagreement Engine — Wave 8.

Aggregates multiple internal interpretations into a structured disagreement field.
"""
from layer7.internal_agents import InternalAgent
import math


class DisagreementEngine:
    def __init__(self):
        self.agents = [
            InternalAgent("trend_lens", bias=0.15, memory_decay=0.05),
            InternalAgent("mean_reversion_lens", bias=-0.1, memory_decay=0.08),
            InternalAgent("noise_lens", bias=0.0, memory_decay=0.2),
            InternalAgent("momentum_amplifier", bias=0.25, memory_decay=0.03),
            InternalAgent("fragility_detector", bias=-0.2, memory_decay=0.1),
        ]

    def evaluate(self, signal: float) -> dict:
        outputs = [a.process(signal) for a in self.agents]
        signals = [o.signal for o in outputs]
        confidences = [o.confidence for o in outputs]
        mean_signal = sum(signals) / len(signals)
        variance = sum((s - mean_signal) ** 2 for s in signals) / len(signals)
        disagreement = math.sqrt(variance)
        consensus_strength = 1.0 / (1.0 + disagreement)
        weighted = sum(s * c for s, c in zip(signals, confidences))
        norm = sum(confidences) + 1e-9
        consensus_signal = weighted / norm
        return {
            "consensus_signal": consensus_signal,
            "disagreement": disagreement,
            "consensus_strength": consensus_strength,
            "agent_outputs": outputs,
        }
