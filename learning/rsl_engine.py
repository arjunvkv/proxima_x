from typing import Dict, Any


class RegimeSegmentedLearning:
    """
    RSL — Regime-Segmented Learning Layer

    Maintains separate feature weights per market regime.
    """

    def __init__(self, base_weights: Dict[str, float]):
        self.regimes = {
            "STRUCTURED": base_weights.copy(),
            "TRANSITION": base_weights.copy(),
            "CHAOTIC": base_weights.copy(),
        }
        self.lr = 0.03

    def update(self,
               regime: str,
               cal_report: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        if regime not in self.regimes:
            return self.regimes["TRANSITION"]

        weights = self.regimes[regime]

        agg = {"ecdf": 0.0, "entropy": 0.0, "spread": 0.0, "signal": 0.0}
        count = 0

        for sym, contrib in cal_report.items():
            agg["ecdf"] += contrib.get("ecdf_contrib", 0.0)
            agg["entropy"] += contrib.get("entropy_contrib", 0.0)
            agg["spread"] += contrib.get("spread_contrib", 0.0)
            agg["signal"] += contrib.get("signal_contrib", 0.0)
            count += 1

        if count > 0:
            for k in agg:
                agg[k] /= count

        for feature in weights:
            weights[feature] += self.lr * agg[feature]

        for feature in weights:
            weights[feature] = max(0.05, min(0.80, weights[feature]))

        total = sum(weights.values())
        self.regimes[regime] = {k: v / total for k, v in weights.items()}

        return self.regimes[regime]

    def get_weights(self, regime: str) -> Dict[str, float]:
        return self.regimes.get(regime, self.regimes["TRANSITION"])
