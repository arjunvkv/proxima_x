"""
Non-Markovian memory substrate for belief field evolution — Wave 11.

Memory is not storage — it is deformation.
Past signals exist as lingering curvature in energy space.
The field is not only what it is now — it is what it has never fully stopped being.
"""
import math


class FieldMemory:
    def __init__(self, decay: float = 0.92):
        self.decay = decay
        self.residual = 0.0
        self.prev_energy = 0.0

    def update(self, current_energy: float):
        self.residual = (self.residual * self.decay) + current_energy
        curvature = current_energy - self.prev_energy
        self.prev_energy = current_energy
        memory_drag = math.tanh(self.residual) * 0.3
        return {
            "residual_energy": self.residual,
            "curvature_pressure": curvature,
            "memory_drag": memory_drag,
        }
