from typing import Dict, List


class LongHorizonConvergenceTracker:
    """
    LCT — Long-Horizon Convergence Tracker

    Measures whether system is improving over time.
    """

    def __init__(self, window: int = 50):
        self.window = window
        self.history: List[float] = []

    def record(self, doa_results: Dict[str, float]):
        if not doa_results:
            self.history.append(0.0)
            return

        avg_score = sum(doa_results.values()) / len(doa_results)
        self.history.append(avg_score)

        if len(self.history) > self.window:
            self.history.pop(0)

    def convergence_score(self) -> float:
        if len(self.history) < 10:
            return 0.0

        n = len(self.history)
        x = list(range(n))
        y = self.history

        x_mean = sum(x) / n
        y_mean = sum(y) / n

        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        slope = numerator / denominator

        return max(-1.0, min(1.0, slope * 10))
