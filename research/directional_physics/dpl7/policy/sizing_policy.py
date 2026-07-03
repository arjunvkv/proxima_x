class SizingPolicy:
    def __init__(self, thresholds=None, multipliers=None):
        self.thresholds = thresholds or [0.2, 0.4, 0.7]
        self.multipliers = multipliers or [0.0, 0.5, 1.0, 1.5]

    def compute(self, signal: dict) -> float:
        c = signal["confidence"]
        for i, thresh in enumerate(self.thresholds):
            if c < thresh:
                return self.multipliers[i]
        return self.multipliers[-1]

    def compute_batch(self, signals: list) -> list:
        return [self.compute(s) for s in signals]
