from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class GauntletResult:
    passed: bool
    deflated_sharpe: Optional[float] = None
    deflated_sharpe_pvalue: Optional[float] = None
    prob_backtest_overfit: Optional[float] = None
    cpcv_score: Optional[float] = None
    walk_forward_score: Optional[float] = None
    sign_test_pvalue: Optional[float] = None
    regime_consistency: Optional[float] = None
    cost_stress_test: Optional[Dict[str, Any]] = None
    details: Dict[str, Any] = field(default_factory=dict)


class OverfitGauntlet:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = {
            "min_dsr": 0.5,
            "max_pbo": 0.3,
            "min_cpcv": 0.0,
            "min_wf": 0.0,
        }
        if config:
            self.config.update(config)

        self._purgedcv_available = False
        self._init_purgedcv()

    def _init_purgedcv(self) -> None:
        try:
            import purgedcv  # type: ignore[import-untyped]
            self._purgedcv = purgedcv
            self._purgedcv_available = True
        except ImportError:
            self._purgedcv_available = False

    def run(
        self,
        returns: List[float],
        strategy_label: str,
        benchmark_returns: Optional[List[float]] = None,
        market_regimes: Optional[List[str]] = None,
        cost_scenarios: Optional[List[Dict[str, Any]]] = None,
    ) -> GauntletResult:
        details: Dict[str, Any] = {
            "strategy": strategy_label,
            "n_observations": len(returns),
        }

        dsr = self._calc_deflated_sharpe(returns)
        details["deflated_sharpe"] = dsr

        pbo = self._calc_pbo(returns)
        details["prob_backtest_overfit"] = pbo

        cpcv = self._calc_cpcv(returns)
        details["cpcv_score"] = cpcv

        sign_p = self._calc_sign_permutation(returns)
        details["sign_test_pvalue"] = sign_p

        regime_consistency = None
        if market_regimes is not None and len(market_regimes) == len(returns):
            regime_consistency = self._calc_regime_consistency(returns, market_regimes)
            details["regime_consistency"] = regime_consistency

        cost_stress = None
        if cost_scenarios is not None:
            cost_stress = self._calc_cost_stress(returns, cost_scenarios)
            details["cost_stress_test"] = cost_stress
        else:
            default_scenarios = [{"name": "1x", "multiplier": 1.0}]
            cost_stress = self._calc_cost_stress(returns, default_scenarios)
            details["cost_stress_test"] = cost_stress

        passed = self._evaluate_passed(details)

        return GauntletResult(
            passed=passed,
            deflated_sharpe=dsr.get("value") if dsr else None,
            deflated_sharpe_pvalue=dsr.get("pvalue") if dsr else None,
            prob_backtest_overfit=pbo.get("value") if pbo else None,
            cpcv_score=cpcv.get("value") if cpcv else None,
            sign_test_pvalue=sign_p,
            regime_consistency=regime_consistency,
            cost_stress_test=cost_stress,
            details=details,
        )

    def _calc_deflated_sharpe(self, returns: List[float]) -> Optional[Dict[str, Any]]:
        if not self._purgedcv_available:
            return {"note": "purgedcv not installed", "value": None, "pvalue": None}
        try:
            result = self._purgedcv.deflated_sharpe_ratio(returns)
            return {"value": float(result.get("dsr", 0)), "pvalue": float(result.get("pvalue", 1))}
        except Exception as exc:
            return {"error": str(exc), "value": None, "pvalue": None}

    def _calc_pbo(self, returns: List[float]) -> Optional[Dict[str, Any]]:
        if not self._purgedcv_available:
            return {"note": "purgedcv not installed", "value": None}
        try:
            result = self._purgedcv.probabilistic_backtest_overfitting(returns)
            return {"value": float(result.get("pbo", 1))}
        except Exception as exc:
            return {"error": str(exc), "value": None}

    def _calc_cpcv(self, returns: List[float]) -> Optional[Dict[str, Any]]:
        if not self._purgedcv_available:
            return {"note": "purgedcv not installed", "value": None}
        try:
            result = self._purgedcv.combinatorial_purged_cross_validation(returns)
            return {"value": float(result.get("score", 0))}
        except Exception as exc:
            return {"error": str(exc), "value": None}

    def _calc_sign_permutation(self, returns: List[float], n_permutations: int = 1000) -> float:
        arr = np.array(returns)
        observed_sharpe = np.mean(arr) / (np.std(arr) + 1e-10) * math.sqrt(len(arr))

        count_extreme = 0
        for _ in range(n_permutations):
            signs = np.random.choice([1.0, -1.0], size=len(arr))
            permuted = arr * signs
            perm_sharpe = np.mean(permuted) / (np.std(permuted) + 1e-10) * math.sqrt(len(permuted))
            if perm_sharpe >= observed_sharpe:
                count_extreme += 1

        return (count_extreme + 1) / (n_permutations + 1)

    def _calc_regime_consistency(self, returns: List[float], regimes: List[str]) -> float:
        regime_map: Dict[str, List[float]] = {}
        for r, ret in zip(regimes, returns):
            if r not in regime_map:
                regime_map[r] = []
            regime_map[r].append(ret)

        sharpes: List[float] = []
        for regime_rets in regime_map.values():
            arr = np.array(regime_rets)
            if len(arr) < 2:
                continue
            s = np.mean(arr) / (np.std(arr) + 1e-10) * math.sqrt(len(arr))
            sharpes.append(s)

        if len(sharpes) < 2:
            return 1.0

        consistency = 1.0 / (1.0 + float(np.std(sharpes)))
        return consistency

    def _calc_cost_stress(
        self, returns: List[float], scenarios: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        gross_pnl = sum(returns)
        turnover = sum(abs(r) for r in returns)

        results: Dict[str, Any] = {}
        for scenario in scenarios:
            name = scenario.get("name", "unknown")
            multiplier = float(scenario.get("multiplier", 1.0))
            cost = turnover * multiplier * 0.001
            net_pnl = gross_pnl - cost
            results[name] = {
                "gross_pnl": gross_pnl,
                "cost": cost,
                "net_pnl": net_pnl,
                "cost_multiplier": multiplier,
            }

        return results

    def _evaluate_passed(self, details: Dict[str, Any]) -> bool:
        dsr_value = None
        if details.get("deflated_sharpe") and details["deflated_sharpe"].get("value") is not None:
            dsr_value = details["deflated_sharpe"]["value"]

        pbo_value = None
        if details.get("prob_backtest_overfit") and details["prob_backtest_overfit"].get("value") is not None:
            pbo_value = details["prob_backtest_overfit"]["value"]

        if dsr_value is not None and dsr_value < self.config["min_dsr"]:
            return False
        if pbo_value is not None and pbo_value > self.config["max_pbo"]:
            return False

        return True
