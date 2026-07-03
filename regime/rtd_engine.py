from typing import Dict, Any


class RegimeTransitionDetector:
    """
    RTD — Regime Transition Detector

    Adds stability layer on top of raw regime inference.
    """

    def __init__(self,
                 enter_threshold: float = 0.65,
                 exit_threshold: float = 0.55,
                 min_persistence: int = 3):

        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.min_persistence = min_persistence

        self._last_regime = None
        self._counter = 0

    def detect(self, eval_data: Dict[str, Dict[str, Any]]) -> str:
        entropies = [
            float(v.get("entropy", 0.5)) for v in eval_data.values()
        ]

        if not entropies:
            return "TRANSITION"

        avg_entropy = sum(entropies) / len(entropies)

        if avg_entropy > self.enter_threshold:
            raw = "CHAOTIC"
        elif avg_entropy < self.exit_threshold:
            raw = "STRUCTURED"
        else:
            raw = "TRANSITION"

        if raw == self._last_regime:
            self._counter += 1
        else:
            self._counter = 1
            self._last_regime = raw

        if raw == "TRANSITION":
            return "TRANSITION"

        if self._counter < self.min_persistence:
            return "TRANSITION"

        if raw == "STRUCTURED":
            return "STABLE_STRUCTURED"
        elif raw == "CHAOTIC":
            return "STABLE_CHAOTIC"

        return "TRANSITION"
