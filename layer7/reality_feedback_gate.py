"""
Reality Feedback Gate — Wave 7 enforcement layer.

Prevents decision loops from becoming self-reinforcing hallucination systems.
"""
from layer7.observer_collapse import ObserverCollapseModel


class RealityFeedbackGate:
    def __init__(self):
        self.occl = ObserverCollapseModel()

    def process(self, signal: float, decision: float, outcome: float) -> dict:
        state = self.occl.update(signal, decision, outcome)

        if state.collapse_risk > 0.72:
            signal *= 0.5

        return {
            "signal": signal,
            "observer_bias": state.observer_bias,
            "entropy": state.feedback_entropy,
            "collapse_risk": state.collapse_risk,
            "corrected_confidence": state.corrected_confidence,
        }
