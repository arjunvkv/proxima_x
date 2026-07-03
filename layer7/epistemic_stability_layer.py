"""
Epistemic Stability Layer — Waves 8+9 enforcement layer.

Wave 8: Converts multi-agent disagreement into actionable stability metrics.
Wave 9: Adds recursive belief contagion — agents influence each other mid-interpretation.
"""
from layer7.disagreement_engine import DisagreementEngine
from layer7.recurrent_belief_network import RecurrentBeliefNetwork


class EpistemicStabilityLayer:
    def __init__(self):
        self.engine = DisagreementEngine()
        self.rb_network = RecurrentBeliefNetwork(iterations=3)

    def process(self, signal: float) -> dict:
        base = self.engine.evaluate(signal)
        contagion = self.rb_network.run(base["agent_outputs"])
        disagreement = base["disagreement"]

        # Contagion-modified consensus signal
        signal = (base["consensus_signal"] * 0.5) + (contagion["signal"] * 0.5)

        instability = (disagreement > 0.65) or (contagion["stability"] < 0.4)
        if instability:
            signal *= 0.7

        return {
            "signal": signal,
            "disagreement": disagreement,
            "contagion_stability": contagion["stability"],
            "unstable": instability,
            "consensus_strength": base["consensus_strength"],
        }
