"""
Continuous Belief Field Dynamics — Wave 10.

No agents. No disagreement engine. No contagion between entities.
Only a continuous belief field evolving over time with local perturbations
acting as "interpretation waves" propagating through structure.
"""
import math


class FieldState:
    def __init__(self, global_signal: float, field_energy: float, instability: float):
        self.global_signal = global_signal
        self.field_energy = field_energy
        self.instability = instability


class BeliefField:
    def __init__(self, resolution: int = 5, diffusion: float = 0.35):
        self.resolution = resolution
        self.diffusion = diffusion
        self.field = [0.0 for _ in range(resolution)]

    def inject(self, signal: float):
        center = self.resolution // 2
        self.field[center] += signal

    def step(self):
        new_field = [0.0 for _ in range(self.resolution)]
        for i in range(self.resolution):
            left = self.field[i - 1] if i > 0 else self.field[i]
            right = self.field[i + 1] if i < self.resolution - 1 else self.field[i]
            center = self.field[i]
            spread = (left + right - 2 * center) * self.diffusion
            reaction = math.tanh(center)
            new_field[i] = center + spread + reaction * 0.1
        self.field = new_field
        return self.compute_state()

    def compute_state(self):
        global_signal = sum(self.field) / len(self.field)
        energy = sum(x * x for x in self.field) / len(self.field)
        mean = global_signal
        variance = sum((x - mean) ** 2 for x in self.field) / len(self.field)
        instability = math.tanh(variance)
        return FieldState(
            global_signal=global_signal,
            field_energy=energy,
            instability=instability,
        )
