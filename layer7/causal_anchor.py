"""
Causal Anchor — Wave 12 external reality tether.

Maintains bounded prediction-vs-reality coupling to prevent
closed-loop belief formation from becoming self-validating.
"""
from collections import deque


class CausalAnchor:
    def __init__(self, window: int = 32):
        self.window = window
        self.predictions = deque(maxlen=window)
        self.reality = deque(maxlen=window)

    def update(self, predicted: float, realized: float) -> dict:
        self.predictions.append(predicted)
        self.reality.append(realized)
        n = len(self.predictions)
        if n < 2:
            return {"drift": 0.0, "alignment": 1.0, "error": 0.0, "anchored_signal": predicted}
        error = sum(abs(p - r) for p, r in zip(self.predictions, self.reality)) / n
        drift = sum(p - r for p, r in zip(self.predictions, self.reality)) / n
        alignment = 1.0 / (1.0 + error)
        anchored_signal = predicted - 0.5 * drift
        return {"drift": drift, "alignment": alignment, "error": error, "anchored_signal": anchored_signal}
