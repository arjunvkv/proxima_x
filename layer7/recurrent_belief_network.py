"""
Recurrent Belief Network — Wave 9.

Iteratively applies contagion until interpretive equilibrium stabilizes.
"""
from layer7.belief_contagion import BeliefContagionModel


class RecurrentBeliefNetwork:
    def __init__(self, iterations: int = 3):
        self.iterations = iterations
        self.model = BeliefContagionModel()

    def run(self, agent_outputs) -> dict:
        state_signal = None
        stability = 0.0
        current = agent_outputs

        for _ in range(self.iterations):
            result = self.model.propagate(current)
            state_signal = result.signal
            stability = result.stability
            # feedback loop: agents re-seed from contagion output
            for a in current:
                a.signal = (a.signal * 0.7) + (state_signal * 0.3)

        return {"signal": state_signal, "stability": stability, "iterations": self.iterations}
