"""
Observation Bias — Wave 5 P0.13: ObserverDecay correction.

Corrects for selection bias in observed signals:
- Executed signals increase selection_bias (we observe what we filter)
- Corrected signal = raw / (1 + bias)
- Bias decays via EMA when signals are not executed
"""
from collections import defaultdict


class ObserverDecay:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.selection_bias = defaultdict(float)

    def update_bias(self, symbol: str, was_executed: bool) -> None:
        if was_executed:
            self.selection_bias[symbol] = (
                (1 - self.alpha) * self.selection_bias[symbol] + self.alpha * 1.0
            )
        else:
            self.selection_bias[symbol] = (
                (1 - self.alpha) * self.selection_bias[symbol]
            )

    def corrected_signal(self, symbol: str, raw_signal: float) -> float:
        bias = self.selection_bias.get(symbol, 0.0)
        return raw_signal / (1.0 + bias)
