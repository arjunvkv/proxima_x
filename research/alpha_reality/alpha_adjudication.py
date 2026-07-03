"""RQ10: Final classification — would this survive live trading?"""

from __future__ import annotations

from typing import Any

import numpy as np

from research.alpha_reality.arl_validator import ARLValidator, ARLResult, HORIZONS, _future_returns


CLASSIFICATIONS = [
    "FALSE_ALPHA",
    "FRAGILE_ALPHA",
    "CONDITIONAL_ALPHA",
    "ROBUST_ALPHA",
    "INSTITUTIONAL_GRADE_ALPHA",
]


class AlphaAdjudication:
    def __init__(self, validator: ARLValidator, asset: str = "EURJPY",
                 prior_results: dict[str, ARLResult] | None = None):
        self.validator = validator
        self.asset = asset
        self.prior = prior_results or {}

    def _score(self, rq_name: str) -> float:
        r = self.prior.get(rq_name)
        if r is None:
            return 0.0
        if r.status == "PASSED":
            return 1.0
        if r.status in ("COMPLETE",):
            return 0.8
        return 0.0

    def run(self) -> ARLResult:
        data = self.validator.load_asset_data(self.asset)
        price = data["price"]
        signals = self.validator.compute_signals(data)
        fr_all = _future_returns(price, np.array(HORIZONS, dtype=np.int32))

        alpha = self.validator.alpha_signal(signals)
        h20_eval = self.validator.eval_alpha(alpha, fr_all, 2)

        evidence = {
            "trend_independence": self._score("RQ1: Trend Independence"),
            "volatility_independence": self._score("RQ2: Volatility Independence"),
            "randomization_test": self._score("RQ3: Randomization Test"),
            "execution_reality": self._score("RQ4: Execution Reality"),
            "cross_asset_transfer": self._score("RQ5: Cross-Asset Transfer"),
            "cross_time_transfer": self._score("RQ6: Cross-Time Transfer"),
            "threshold_stability": self._score("RQ7: Threshold Stability"),
            "capacity_analysis": self._score("RQ9: Capacity Analysis"),
        }

        passed = sum(1 for v in evidence.values() if v >= 0.8)
        total = len(evidence)
        avg_score = np.mean(list(evidence.values()))

        net_mean = h20_eval.get("mean", 0)
        net_sharpe = h20_eval.get("sharpe", 0)
        net_pp = h20_eval.get("pp", 0.5)
        net_n = h20_eval.get("n", 0)

        # Classification logic
        if passed >= 7 and avg_score > 0.85 and net_sharpe > 0.5:
            cls = "INSTITUTIONAL_GRADE_ALPHA"
            confidence = avg_score
        elif passed >= 5 and avg_score > 0.6 and net_sharpe > 0.3 and net_pp > 0.55:
            cls = "ROBUST_ALPHA"
            confidence = avg_score
        elif passed >= 3 and net_pp > 0.53 and net_mean > 0:
            cls = "CONDITIONAL_ALPHA"
            confidence = avg_score
        elif net_pp > 0.51 and net_mean > 0:
            cls = "FRAGILE_ALPHA"
            confidence = max(0.1, avg_score)
        else:
            cls = "FALSE_ALPHA"
            confidence = 1.0 - avg_score

        would_survive = cls in ("ROBUST_ALPHA", "INSTITUTIONAL_GRADE_ALPHA")

        print(f"  Alpha Adjudication:")
        print(f"    Passed: {passed}/{total} tests")
        print(f"    Overall score: {avg_score:.2f}")
        print(f"    H20 net mean: {net_mean:.6f}, sharpe: {net_sharpe:.3f}, pp: {net_pp:.3f}")
        print(f"    Classification: {cls}")
        print(f"    Would survive live trading: {'YES' if would_survive else 'NO'}")
        print()
        for rq, score in evidence.items():
            icon = "PASS" if score >= 0.8 else "FAIL"
            print(f"    {icon} {rq:30s}: {score:.2f}")

        return ARLResult("alpha_adjudication", cls, metrics={
            "classification": cls,
            "confidence": confidence,
            "would_survive_live_trading": would_survive,
            "evidence": evidence,
            "passed": passed,
            "total": total,
            "avg_score": avg_score,
            "h20_performance": h20_eval,
        })
