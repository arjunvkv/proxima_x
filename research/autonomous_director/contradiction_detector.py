import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger("proxima_ops.ard.contradiction")


class ContradictionDetector:
    def __init__(self):
        self._contradictions: list[dict] = []

    def check(self, expected: float, observed: float, metric: str,
              tolerance: float = 0.25) -> Optional[dict]:
        if expected == 0:
            return None
        deviation = abs(observed - expected) / abs(expected)
        if deviation > tolerance:
            rec = {
                "timestamp": datetime.now().isoformat(),
                "metric": metric, "expected": expected,
                "observed": observed, "deviation_pct": round(deviation, 3),
                "direction": "ABOVE" if observed > expected else "BELOW"}
            self._contradictions.append(rec)
            logger.warning(f"Contradiction: {metric} expected={expected} observed={observed}")
            return rec
        return None

    def full_scan(self, expected_sharpe: float, observed_sharpe: float,
                   expected_pp: float, observed_pp: float,
                   expected_freq: float, observed_freq: float,
                   expected_es: float, observed_es: float) -> list[dict]:
        results = []
        c = self.check(expected_sharpe, observed_sharpe, "sharpe")
        if c:
            results.append(c)
        c = self.check(expected_pp, observed_pp, "profit_probability")
        if c:
            results.append(c)
        c = self.check(expected_freq, observed_freq, "frequency")
        if c:
            results.append(c)
        c = self.check(expected_es, observed_es, "es_rank_mean")
        if c:
            results.append(c)
        return results

    def recent(self, n: int = 10) -> list[dict]:
        return self._contradictions[-n:]

    def count(self) -> int:
        return len(self._contradictions)
