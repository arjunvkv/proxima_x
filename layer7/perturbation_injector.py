"""
External shock injection layer — Wave 12.

Prevents closed-system convergence collapse by injecting
controlled stochastic perturbations when overconfidence rises.
"""
import random


class PerturbationInjector:
    def __init__(self, intensity: float = 0.08):
        self.intensity = intensity

    def inject(self, signal: float, instability: float) -> tuple:
        shock_prob = 0.05 + 0.25 * (1.0 - instability)
        if random.random() < shock_prob:
            shock = random.uniform(-self.intensity, self.intensity)
            return signal + shock, True
        return signal, False
