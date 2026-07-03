import time
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
import numpy as np

from research.trading_relevance.outcome_distribution import OutcomeDistributionAnalyzer, OutcomeDistributionReport
from research.trading_relevance.trade_survivability import TradeSurvivabilityAnalyzer, TradeSurvivabilityReport
from research.trading_relevance.risk_profile import RiskProfileAnalyzer, RiskProfileReport
from research.trading_relevance.mechanism_interaction import MechanismInteractionAnalyzer, MechanismInteractionReport
from research.trading_relevance.cross_asset import CrossAssetRelevanceAnalyzer, CrossAssetRelevanceReport
from research.trading_relevance.economic_value import EconomicValueAnalyzer, EconomicValueReport


VERDICTS = [
    "ACADEMIC_ONLY",
    "WEAK_OPERATIONAL_VALUE",
    "MODERATE_OPERATIONAL_VALUE",
    "STRONG_OPERATIONAL_VALUE",
    "RISK_FILTER",
    "REGIME_FILTER",
    "POSITION_SIZING_VARIABLE",
    "MARKET_EVOLUTION_VARIABLE",
    "MULTI_PURPOSE_VARIABLE",
]


@dataclass
class TradingValidationReport:
    asset: str
    outcome_distribution: OutcomeDistributionReport
    trade_survivability: TradeSurvivabilityReport
    risk_profile: RiskProfileReport
    mechanism_interaction: MechanismInteractionReport
    cross_asset: Optional[CrossAssetRelevanceReport]
    economic_value: EconomicValueReport
    timing: dict
    final_verdict: str


class TradingRelevanceValidator:
    """
    Orchestrates all Reality Phase 5 analyses and produces a final verdict
    about the trading relevance of adaptive_time_coordinate.
    
    The verdict is based on whether adaptive_time changes outcome distributions,
    risk profiles, trade survivability, and uncertainty in ways that
    improve decision quality.
    """
    
    def __init__(self, asset: str = "EURJPY"):
        self.asset = asset
        self.outcome_analyzer = OutcomeDistributionAnalyzer()
        self.survivability_analyzer = TradeSurvivabilityAnalyzer()
        self.risk_analyzer = RiskProfileAnalyzer()
        self.mechanism_analyzer = MechanismInteractionAnalyzer()
        self.economic_analyzer = EconomicValueAnalyzer()
        self.cross_asset_analyzer = CrossAssetRelevanceAnalyzer()
    
    def validate(self, data: dict) -> TradingValidationReport:
        """
        Run all trading relevance analyses.
        
        data dict must contain:
        - adaptive_time: NDArray
        - returns: NDArray
        - states: NDArray
        - price: NDArray
        - energy_storage: NDArray
        - memory_density: NDArray
        - memory_gradient: NDArray
        """
        wall = time.time()
        timing = {}
        
        # 1. Outcome Distribution (RQ1, RQ4, RQ7, RQ8)
        t0 = time.time()
        outcome_report = self.outcome_analyzer.compute(
            data["adaptive_time"], data["returns"], data["states"], data["price"])
        timing["outcome_distribution"] = time.time() - t0
        
        # 2. Trade Survivability (RQ2)
        t0 = time.time()
        surv_report = self.survivability_analyzer.compute(
            data["adaptive_time"], data["returns"])
        timing["trade_survivability"] = time.time() - t0
        
        # 3. Risk Profile (RQ3)
        t0 = time.time()
        risk_report = self.risk_analyzer.compute(
            data["adaptive_time"], data["returns"], data["states"])
        timing["risk_profile"] = time.time() - t0
        
        # 4. Mechanism Interaction (RQ5, RQ6)
        t0 = time.time()
        mech_report = self.mechanism_analyzer.compute(
            data["adaptive_time"], data["returns"],
            data.get("energy_storage"), data.get("memory_density"),
            data.get("memory_gradient"), data.get("states"))
        timing["mechanism_interaction"] = time.time() - t0
        
        # 5. Economic Value (RQ10)
        t0 = time.time()
        econ_report = self.economic_analyzer.compute(
            data["adaptive_time"], data["returns"])
        timing["economic_value"] = time.time() - t0
        
        timing["total"] = time.time() - wall
        
        final_verdict = self._compute_verdict(
            outcome_report, surv_report, risk_report, mech_report, econ_report)
        
        return TradingValidationReport(
            asset=self.asset,
            outcome_distribution=outcome_report,
            trade_survivability=surv_report,
            risk_profile=risk_report,
            mechanism_interaction=mech_report,
            cross_asset=None,
            economic_value=econ_report,
            timing=timing,
            final_verdict=final_verdict,
        )
    
    def validate_multi_asset(self, primary_asset: str,
                              all_data: dict) -> TradingValidationReport:
        """Run validation including cross-asset analysis."""
        report = self.validate(all_data[primary_asset])
        
        # Cross-asset relevance (RQ9)
        cross_data = {}
        for asset, data in all_data.items():
            cross_data[asset] = {
                "adaptive_time": data["adaptive_time"],
                "returns": data["returns"],
                "states": data["states"],
            }
        cross_report = self.cross_asset_analyzer.compute(cross_data)
        report.cross_asset = cross_report
        
        # Recompute verdict with cross-asset info
        report.final_verdict = self._compute_verdict(
            report.outcome_distribution, report.trade_survivability,
            report.risk_profile, report.mechanism_interaction,
            report.economic_value, cross_report)
        
        return report
    
    def _compute_verdict(self, outcome: OutcomeDistributionReport,
                          survivability: TradeSurvivabilityReport,
                          risk: RiskProfileReport,
                          mechanism: MechanismInteractionReport,
                          economic: EconomicValueReport,
                          cross: CrossAssetRelevanceReport = None) -> str:
        """
        Compute final verdict based on all evidence.
        
        Scoring logic:
        - Outcome distribution materially changes: +2 to STRONG_OPERATIONAL
        - Outcome separation > 0.3: +1 to STRONG_OPERATIONAL
        - Survivability changes across buckets: +1 to MODERATE_OPERATIONAL
        - Risk monotonic with adaptive_time: +2 to RISK_FILTER
        - Adaptive_time improves mechanism IG: +1 to REGIME_FILTER
        - Uncertainty reduction > 0.1: +1 to POSITION_SIZING
        - Cross-asset consistency: +1 to MULTI_PURPOSE
        
        Returns the highest scoring verdict category.
        """
        scores = {v: 0.0 for v in VERDICTS}
        
        # Outcome distribution material change
        if hasattr(outcome, 'verdict') and "change_materially" in outcome.verdict:
            scores["STRONG_OPERATIONAL_VALUE"] += 2.0
            scores["MULTI_PURPOSE_VARIABLE"] += 1.0
        
        # Outcome separation
        sep = getattr(outcome, 'outcome_separation_avg', 0.0)
        if sep > 0.3:
            scores["STRONG_OPERATIONAL_VALUE"] += 1.0
        elif sep > 0.15:
            scores["MODERATE_OPERATIONAL_VALUE"] += 1.0
        else:
            scores["ACADEMIC_ONLY"] += 0.5
        
        # Survivability effect
        surv_verdict = survivability.verdict if hasattr(survivability, 'verdict') else ""
        if "affects_survivability" in surv_verdict:
            scores["MODERATE_OPERATIONAL_VALUE"] += 1.0
        
        # Risk monotonic
        risk_verdict = risk.verdict if hasattr(risk, 'verdict') else ""
        if "monotonic" in risk_verdict:
            scores["RISK_FILTER"] += 2.0
        elif "alters_risk" in risk_verdict:
            scores["RISK_FILTER"] += 1.0
        
        # Mechanism interaction
        improvement = getattr(mechanism, 'adaptive_time_improvement', 0.0)
        if improvement > 0.05:
            scores["REGIME_FILTER"] += 1.5
            scores["MULTI_PURPOSE_VARIABLE"] += 1.0
        
        # Regime filter
        ig_diff = getattr(mechanism, 'ig_difference', 0.0)
        if abs(ig_diff) > 0.05:
            scores["REGIME_FILTER"] += 1.0
        
        # Economic value - uncertainty reduction
        uncert_red = getattr(economic, 'uncertainty_reduction', 0.0)
        if uncert_red > 0.15:
            scores["POSITION_SIZING_VARIABLE"] += 1.5
            scores["STRONG_OPERATIONAL_VALUE"] += 1.0
        elif uncert_red > 0.05:
            scores["POSITION_SIZING_VARIABLE"] += 1.0
            scores["WEAK_OPERATIONAL_VALUE"] += 1.0
        
        # Cross-asset consistency
        if cross and hasattr(cross, 'verdict'):
            if cross.verdict == "operational_usefulness_transfers":
                scores["MULTI_PURPOSE_VARIABLE"] += 2.0
                scores["STRONG_OPERATIONAL_VALUE"] += 1.0
        
        # Market evolution variable
        if scores["REGIME_FILTER"] > 1.0 and scores["RISK_FILTER"] > 1.0:
            scores["MARKET_EVOLUTION_VARIABLE"] += 1.5
        
        return max(scores, key=scores.get)
