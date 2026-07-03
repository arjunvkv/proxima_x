from typing import Dict, Any


class FeatureWeightOptimizer:
    """
    FWO — Feature Weight Optimizer

    Updates feature importance based on CAL attribution.
    """

    def __init__(self,
                 lr: float = 0.05):
        self.lr = lr
        self.weights = {
            "ecdf": 0.40,
            "entropy": 0.35,
            "spread": 0.15,
            "signal": 0.10,
        }

    def update(self,
               cal_report: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        if not cal_report:
            return self.weights

        agg = {"ecdf": 0.0, "entropy": 0.0, "spread": 0.0, "signal": 0.0}
        count = 0

        for sym, contrib in cal_report.items():
            agg["ecdf"] += contrib.get("ecdf_contrib", 0.0)
            agg["entropy"] += contrib.get("entropy_contrib", 0.0)
            agg["spread"] += contrib.get("spread_contrib", 0.0)
            agg["signal"] += contrib.get("signal_contrib", 0.0)
            count += 1

        if count == 0:
            return self.weights

        for k in agg:
            agg[k] /= count

        for feature in self.weights:
            self.weights[feature] += self.lr * agg[feature]

        for feature in self.weights:
            self.weights[feature] = max(0.05, min(0.80, self.weights[feature]))

        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

        return self.weights
