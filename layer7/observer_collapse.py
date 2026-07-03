"""
Observer Collapse Correction Layer (OCCL) — Wave 7.

Models how past decisions corrupt future interpretation:
- Observer bias: self-confirmation pressure from decision-outcome alignment
- Feedback entropy: divergence between signal and outcome learning
- Collapse risk: overconfidence in self-reinforced regime
- Corrected confidence: anti-self-confirmation damping
"""
from collections import deque
import math
from typing import List


class OCCLState:
    def __init__(self, observer_bias: float, feedback_entropy: float,
                 collapse_risk: float, corrected_confidence: float):
        self.observer_bias = observer_bias
        self.feedback_entropy = feedback_entropy
        self.collapse_risk = collapse_risk
        self.corrected_confidence = corrected_confidence


class ObserverCollapseModel:
    """
    Tracks how past decisions reshape interpretation of new signals.
    Runs strictly POST-decision — outcome must be realized before calling update().
    """

    def __init__(self, window: int = 64):
        self.window = window
        self.decision_history: deque = deque(maxlen=window)
        self.outcome_history: deque = deque(maxlen=window)
        self.signal_history: deque = deque(maxlen=window)

    def update(self, signal: float, decision: float, outcome: float) -> OCCLState:
        """
        signal: raw system signal [-1..1]
        decision: executed action strength [-1..1]
        outcome: realized feedback [-1..1]
        """
        self.signal_history.append(signal)
        self.decision_history.append(decision)
        self.outcome_history.append(outcome)

        n = len(self.signal_history)
        if n < 4:
            return OCCLState(0.0, 0.0, 0.0, signal)

        # Observer bias: self-confirmation pressure
        alignment = sum(d * o for d, o in zip(self.decision_history, self.outcome_history)) / n
        mean_signal = sum(self.signal_history) / n
        signal_var = sum((x - mean_signal) ** 2 for x in self.signal_history) / n + 1e-9
        observer_bias = alignment / (math.sqrt(signal_var) + 1e-9)
        observer_bias = max(-5.0, min(5.0, observer_bias))  # clamp to prevent blowup

        # Feedback entropy: divergence between signal and outcome
        divergence = sum(abs(s - o) for s, o in zip(self.signal_history, self.outcome_history)) / n
        entropy = math.tanh(divergence)

        # Collapse risk: overconfidence in self-reinforced regime
        confidence_loop = abs(alignment) * (1.0 - entropy)
        collapse_risk = math.tanh(confidence_loop * observer_bias)

        # Corrected confidence: anti-self-confirmation damping
        stability = 1.0 / (1.0 + signal_var)
        distortion = observer_bias * stability * (1.0 - entropy)
        corrected_confidence = signal * math.exp(-0.7 * abs(distortion))

        return OCCLState(
            observer_bias=observer_bias,
            feedback_entropy=entropy,
            collapse_risk=collapse_risk,
            corrected_confidence=corrected_confidence,
        )
