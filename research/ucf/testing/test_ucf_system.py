from __future__ import annotations

import math
import random
import statistics
import time
from typing import Any

from ..core.unified_conviction_field import UnifiedConvictionField
from ..core.adaptive_weight_engine import AdaptiveWeightEngine
from ..core.bidirectional_fusion import BidirectionalFusionLayer
from ..integration.regime_adaptive_modulator import RegimeAdaptiveModulator


class UCFIntegrationTests:
    def __init__(self) -> None:
        self.test_results: dict[str, bool] = {}
        self.failures: list[str] = []

    def _has_nan(self, obj: Any) -> bool:
        if isinstance(obj, float):
            return math.isnan(obj)
        if isinstance(obj, dict):
            return any(self._has_nan(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return any(self._has_nan(v) for v in obj)
        return False

    def _make_conviction(self, conviction: float, direction: int, stability: float = 0.5) -> dict[str, Any]:
        return {"conviction": conviction, "direction": direction, "stability": stability}

    def _make_regime_context(self, regime: str = "neutral", stability: float = 0.5, fsv_entropy: float = 0.5, tech_vol: float = 0.5, pred_err: float = 0.0, expo_conc: float = 0.5) -> dict[str, Any]:
        return {
            "regime": regime,
            "regime_stability": stability,
            "fsv_entropy": fsv_entropy,
            "technical_volatility": tech_vol,
            "recent_prediction_error": pred_err,
            "exposure_concentration": expo_conc,
        }

    def test_end_to_end_pipeline_integrity(self) -> bool:
        ucf = UnifiedConvictionField()
        symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        tech = {s: self._make_conviction(0.7, 1, 0.8) for s in symbols}
        fund = {s: self._make_conviction(0.5, 1, 0.6) for s in symbols}
        expo = {s: self._make_conviction(0.3, 0, 0.5) for s in symbols}
        regime = self._make_regime_context("neutral")
        try:
            result = ucf.compute(symbols, tech, fund, expo, regime)
        except Exception:
            return False
        field = result.get("field", {})
        for s in symbols:
            entry = field.get(s, {})
            if "conviction_score" not in entry:
                return False
            if "direction" not in entry:
                return False
            if "component_breakdown" not in entry:
                return False
            cs = entry["conviction_score"]
            if not (0.0 <= cs <= 1.0):
                return False
            d = entry["direction"]
            if d not in (-1, 0, 1):
                return False
        fc = result.get("field_coherence", -1.0)
        if not (0.0 <= fc <= 1.0):
            return False
        return True

    def test_regime_stability_transition(self) -> bool:
        ucf = UnifiedConvictionField()
        symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        tech = {s: self._make_conviction(0.6, 1, 0.7) for s in symbols}
        fund = {s: self._make_conviction(0.6, 1, 0.7) for s in symbols}
        expo = {s: self._make_conviction(0.6, 1, 0.7) for s in symbols}
        for regime in ("risk_on", "risk_off", "neutral"):
            ctx = self._make_regime_context(regime, stability=0.8)
            try:
                result = ucf.compute(symbols, tech, fund, expo, ctx)
            except Exception:
                return False
            fc = result.get("field_coherence", 0.0)
            if fc <= 0.3:
                return False
            if self._has_nan(result):
                return False
        return True

    def test_no_dominance_constraint(self) -> bool:
        ucf = UnifiedConvictionField()
        symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        tech = {s: self._make_conviction(1.0, 1, 1.0) for s in symbols}
        fund = {s: self._make_conviction(0.0, 0, 0.5) for s in symbols}
        expo = {s: self._make_conviction(0.0, 0, 0.5) for s in symbols}
        seen_weights: set[tuple[tuple[str, float], ...]] = set()
        for regime in ("risk_on", "risk_off", "neutral", "transition"):
            stability = 0.8 if regime in ("risk_on", "risk_off") else 0.5
            ctx = self._make_regime_context(regime, stability=stability)
            try:
                result = ucf.compute(symbols, tech, fund, expo, ctx)
            except Exception:
                return False
            field = result.get("field", {})
            for s in symbols:
                entry = field.get(s, {})
                cb = entry.get("component_breakdown", {})
                for k in ("technical_contribution", "fundamental_contribution", "exposure_contribution"):
                    if cb.get(k, 0.0) > 0.65:
                        return False
            weights = result.get("weights", {})
            seen_weights.add(tuple(sorted((k, v) for k, v in weights.items() if k != "regime" and k != "confidence")))
        if len(seen_weights) < 2:
            return False
        return True

    def test_cross_system_consistency(self) -> bool:
        ucf = UnifiedConvictionField()
        symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        tech_agree = {s: self._make_conviction(0.9, 1, 0.9) for s in symbols}
        fund_agree = {s: self._make_conviction(0.9, 1, 0.9) for s in symbols}
        expo_agree = {s: self._make_conviction(0.9, 1, 0.9) for s in symbols}
        ctx = self._make_regime_context("neutral")
        try:
            result_agree = ucf.compute(symbols, tech_agree, fund_agree, expo_agree, ctx)
        except Exception:
            return False
        for s in symbols:
            cs = result_agree.get("field", {}).get(s, {}).get("conviction_score", 0.0)
            if cs <= 0.6:
                return False
        tech_dis = {s: self._make_conviction(0.55, 1, 0.9) for s in symbols}
        fund_dis = {s: self._make_conviction(0.55, -1, 0.9) for s in symbols}
        expo_dis = {s: self._make_conviction(0.55, 0, 0.9) for s in symbols}
        try:
            result_dis = ucf.compute(symbols, tech_dis, fund_dis, expo_dis, ctx)
        except Exception:
            return False
        for s in symbols:
            cs = result_dis.get("field", {}).get(s, {}).get("conviction_score", 1.0)
            if cs >= 0.5:
                return False
            cb = result_dis.get("field", {}).get(s, {}).get("component_breakdown", {})
            for k in ("technical_contribution", "fundamental_contribution", "exposure_contribution"):
                if cb.get(k, 0.0) == 0.0:
                    return False
        return True

    def test_ranking_stability(self) -> bool:
        ucf = UnifiedConvictionField()
        symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
        convictions = [0.9, 0.7, 0.5, 0.3, 0.1]
        tech = {}
        fund = {}
        expo = {}
        for s, c in zip(symbols, convictions):
            tech[s] = self._make_conviction(c, 1, 0.5)
            fund[s] = self._make_conviction(c, 1, 0.5)
            expo[s] = self._make_conviction(c, 1, 0.5)
        ctx = self._make_regime_context("neutral")
        try:
            result = ucf.compute(symbols, tech, fund, expo, ctx)
        except Exception:
            return False
        field = result.get("field", {})
        ranking = sorted(symbols, key=lambda s: field.get(s, {}).get("conviction_score", 0.0), reverse=True)
        random.seed(42)
        tech2 = {}
        fund2 = {}
        expo2 = {}
        for s, c in zip(symbols, convictions):
            delta = random.uniform(-0.01, 0.01)
            nc = max(0.0, min(1.0, c + delta))
            tech2[s] = self._make_conviction(nc, 1, 0.5)
            fund2[s] = self._make_conviction(nc, 1, 0.5)
            expo2[s] = self._make_conviction(nc, 1, 0.5)
        try:
            result2 = ucf.compute(symbols, tech2, fund2, expo2, ctx)
        except Exception:
            return False
        field2 = result2.get("field", {})
        ranking2 = sorted(symbols, key=lambda s: field2.get(s, {}).get("conviction_score", 0.0), reverse=True)
        return ranking == ranking2

    def test_adaptive_weights_no_extreme(self) -> bool:
        engine = AdaptiveWeightEngine()
        for regime in ("risk_on", "risk_off", "neutral", "transition"):
            stability = 0.8 if regime == "risk_on" else 0.5
            ctx: dict[str, Any] = {
                "regime": regime,
                "regime_stability": stability,
                "fsv_entropy": 0.5,
                "technical_volatility": 0.5,
                "recent_prediction_error": 0.0,
                "exposure_concentration": 0.5,
            }
            try:
                weights = engine.compute_weights(ctx)
            except Exception:
                return False
            w_keys = ("technical_weight", "fundamental_weight", "macro_weight", "exposure_weight")
            for k in w_keys:
                v = weights.get(k, 0.0)
                if v < 0.05 or v > 0.60:
                    return False
            total = sum(weights.get(k, 0.0) for k in w_keys)
            if not (0.999 <= total <= 1.001):
                return False
        return True

    def test_bidirectional_fusion_agreement_bonus(self) -> bool:
        fusion = BidirectionalFusionLayer()
        tech_a = {"conviction": 0.7, "direction": 1, "stability": 0.8}
        fund_a = {"conviction": 0.7, "direction": 1, "stability": 0.8}
        expo_a = {"conviction": 0.7, "direction": 1, "stability": 0.8}
        weights = {"technical": 0.34, "fundamental": 0.33, "exposure": 0.33}
        result_a = fusion.fuse_states(tech_a, fund_a, expo_a, "neutral", weights)
        w_avg = 0.34 * 0.7 + 0.33 * 0.7 + 0.33 * 0.7
        if result_a["fused_conviction"] <= w_avg:
            return False
        tech_d = {"conviction": 0.7, "direction": 1, "stability": 0.8}
        fund_d = {"conviction": 0.7, "direction": -1, "stability": 0.8}
        expo_d = {"conviction": 0.7, "direction": 0, "stability": 0.8}
        result_d = fusion.fuse_states(tech_d, fund_d, expo_d, "neutral", weights)
        w_avg_d = 0.34 * 0.7 + 0.33 * 0.7 + 0.33 * 0.7
        if result_d["fused_conviction"] >= w_avg_d:
            return False
        return True

    def test_modulator_no_fixed_caps(self) -> bool:
        modulator = RegimeAdaptiveModulator()
        base = 0.5
        risk_on = modulator.modulate(base, "risk_on", 0.5, 0.8)
        if risk_on <= base:
            return False
        risk_off = modulator.modulate(base, "risk_off", 0.5, -0.8)
        if risk_off >= base:
            return False
        transition = modulator.modulate(base, "transition", 0.7, 0.5)
        neutral = modulator.modulate(base, "neutral", 0.5, 0.0)
        if abs(transition - base) <= abs(neutral - base):
            return False
        extreme_on = modulator.modulate(0.5, "risk_on", 0.5, 1.0)
        if extreme_on <= 0.5 * 1.15:
            return False
        return True

    def test_nan_propagation_prevention(self) -> bool:
        ucf = UnifiedConvictionField()
        symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        tech = {s: {"conviction": float("nan"), "direction": 1, "stability": 0.5} for s in symbols}
        fund = {s: self._make_conviction(0.5, 1, 0.5) for s in symbols}
        expo = {s: self._make_conviction(0.5, 1, 0.5) for s in symbols}
        ctx = self._make_regime_context("neutral")
        try:
            result = ucf.compute(symbols, tech, fund, expo, ctx)
        except Exception:
            return False
        if self._has_nan(result):
            return False
        return True

    def test_full_system_regression(self) -> bool:
        ucf = UnifiedConvictionField()
        symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "USDCHF"]
        random.seed(42)
        tech: dict[str, dict[str, Any]] = {}
        fund: dict[str, dict[str, Any]] = {}
        expo: dict[str, dict[str, Any]] = {}
        for s in symbols:
            tech[s] = {"conviction": random.uniform(0.0, 1.0), "direction": random.choice([-1, 0, 1]), "stability": random.uniform(0.0, 1.0)}
            fund[s] = {"conviction": random.uniform(0.0, 1.0), "direction": random.choice([-1, 0, 1]), "stability": random.uniform(0.0, 1.0)}
            expo[s] = {"conviction": random.uniform(0.0, 1.0), "direction": random.choice([-1, 0, 1]), "stability": random.uniform(0.0, 1.0)}
        ctx = self._make_regime_context(random.choice(["risk_on", "risk_off", "neutral", "transition"]))
        try:
            result = ucf.compute(symbols, tech, fund, expo, ctx)
        except Exception:
            return False
        field = result.get("field", {})
        if not field:
            return False
        for s in symbols:
            entry = field.get(s, {})
            if "conviction_score" not in entry:
                return False
            if self._has_nan(entry):
                return False
        return True

    def run_all(self) -> dict[str, Any]:
        tests: list[tuple[str, Any]] = [
            ("test_end_to_end_pipeline_integrity", self.test_end_to_end_pipeline_integrity),
            ("test_regime_stability_transition", self.test_regime_stability_transition),
            ("test_no_dominance_constraint", self.test_no_dominance_constraint),
            ("test_cross_system_consistency", self.test_cross_system_consistency),
            ("test_ranking_stability", self.test_ranking_stability),
            ("test_adaptive_weights_no_extreme", self.test_adaptive_weights_no_extreme),
            ("test_bidirectional_fusion_agreement_bonus", self.test_bidirectional_fusion_agreement_bonus),
            ("test_modulator_no_fixed_caps", self.test_modulator_no_fixed_caps),
            ("test_nan_propagation_prevention", self.test_nan_propagation_prevention),
            ("test_full_system_regression", self.test_full_system_regression),
        ]
        results: dict[str, bool] = {}
        failures: list[str] = []
        for name, test_fn in tests:
            try:
                passed = test_fn()
                results[name] = passed
                if not passed:
                    failures.append(name)
            except Exception:
                results[name] = False
                failures.append(name)
        passed_count = sum(1 for v in results.values() if v)
        failed_count = len(results) - passed_count
        total_count = len(results)
        stability = passed_count / total_count if total_count > 0 else 0.0
        return {
            "passed": passed_count,
            "failed": failed_count,
            "total": total_count,
            "results": results,
            "failures": failures,
            "stability_score": stability,
            "timestamp": time.time(),
        }

    def stress_suite(self) -> dict[str, Any]:
        tests: list[tuple[str, Any]] = [
            ("test_regime_stability_transition", self.test_regime_stability_transition),
            ("test_no_dominance_constraint", self.test_no_dominance_constraint),
            ("test_ranking_stability", self.test_ranking_stability),
            ("test_modulator_no_fixed_caps", self.test_modulator_no_fixed_caps),
            ("test_nan_propagation_prevention", self.test_nan_propagation_prevention),
        ]
        results: dict[str, bool] = {}
        failures: list[str] = []
        for name, test_fn in tests:
            try:
                passed = test_fn()
                results[name] = passed
                if not passed:
                    failures.append(name)
            except Exception:
                results[name] = False
                failures.append(name)
        passed_count = sum(1 for v in results.values() if v)
        failed_count = len(results) - passed_count
        total_count = len(results)
        stability = passed_count / total_count if total_count > 0 else 0.0
        return {
            "passed": passed_count,
            "failed": failed_count,
            "total": total_count,
            "results": results,
            "failures": failures,
            "stability_score": stability,
            "timestamp": time.time(),
        }
