"""
Non-Markovian Belief Field — Wave 11.

Field evolution depends on persistent historical deformation.
Same input produces different output depending on history shape.
"""
from layer7.belief_field import BeliefField
from layer7.field_memory import FieldMemory


class NonMarkovianBeliefField:
    def __init__(self):
        self.field = BeliefField(resolution=7, diffusion=0.28)
        self.memory = FieldMemory(decay=0.94)

    def step(self, signal: float) -> dict:
        self.field.inject(signal)
        state = None
        for _ in range(2):
            state = self.field.step()
        memory_state = self.memory.update(state.field_energy)
        curvature_bias = memory_state["curvature_pressure"] * 0.5
        raw_signal = state.global_signal + curvature_bias
        instability = state.instability + abs(memory_state["residual_energy"]) * 0.05
        return {
            "signal": raw_signal,
            "instability": min(1.0, instability),
            "coherence": 1.0 - min(1.0, instability),
            "memory_residual": memory_state["residual_energy"],
            "memory_drag": memory_state["memory_drag"],
            "adaptive_diffusion": self.field.diffusion + (0.15 * memory_state["memory_drag"]),
        }
