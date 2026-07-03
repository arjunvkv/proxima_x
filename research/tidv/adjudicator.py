VERDICTS = [
    "ACADEMIC_ARTIFACT",
    "WEAK_CONTEXT_VARIABLE",
    "REGIME_FILTER",
    "RISK_FILTER",
    "POSITION_SIZING_VARIABLE",
    "HOLDING_PERIOD_VARIABLE",
    "DECISION_QUALITY_VARIABLE",
    "MARKET_EVOLUTION_VARIABLE",
    "MULTI_PURPOSE_DECISION_LAYER",
]

INTEGRATION_MAP = {
    "ACADEMIC_ARTIFACT": "REJECT",
    "WEAK_CONTEXT_VARIABLE": "RESEARCH_FURTHER",
    "REGIME_FILTER": "INTEGRATE_AS_FILTER",
    "RISK_FILTER": "INTEGRATE_AS_FILTER",
    "POSITION_SIZING_VARIABLE": "INTEGRATE_AS_CONTEXT",
    "HOLDING_PERIOD_VARIABLE": "INTEGRATE_AS_CONTEXT",
    "DECISION_QUALITY_VARIABLE": "INTEGRATE_AS_CONTEXT",
    "MARKET_EVOLUTION_VARIABLE": "INTEGRATE_AS_CORE_LAYER",
    "MULTI_PURPOSE_DECISION_LAYER": "INTEGRATE_AS_CORE_LAYER",
}

class TIDVVerdict:
    def __init__(self, classification: str, integration_recommendation: str,
                 scores: dict, evidence: dict):
        self.classification = classification
        self.integration_recommendation = integration_recommendation
        self.scores = scores
        self.evidence = evidence

    def __repr__(self):
        return (f"TIDVVerdict({self.classification}, "
                f"integration={self.integration_recommendation})")

class TIDVAdjudicator:
    def adjudicate(self, results: dict) -> TIDVVerdict:
        scores = {v: 0.0 for v in VERDICTS}
        evidence = {}

        exp_a = results.get("experiment_a", {})
        exp_b = results.get("experiment_b", {})
        exp_c = results.get("experiment_c", {})
        exp_d = results.get("experiment_d", {})
        exp_e = results.get("experiment_e", {})
        exp_f = results.get("experiment_f", {})
        exp_g = results.get("experiment_g")
        exp_h = results.get("experiment_h", {})

        # Experiment A: Regime Filter
        mech_improv = exp_a.get("adaptive_time_improvement", 0.0)
        ig_diff = exp_a.get("ig_difference", 0.0)
        a_verdict = exp_a.get("verdict", "")
        if mech_improv > 0.05:
            scores["REGIME_FILTER"] += 2.0
            scores["MULTI_PURPOSE_DECISION_LAYER"] += 1.0
            evidence["regime_filter_improvement"] = mech_improv
        if abs(ig_diff) > 0.05:
            scores["REGIME_FILTER"] += 1.5
            scores["MARKET_EVOLUTION_VARIABLE"] += 0.5
            evidence["regime_filter_ig_diff"] = ig_diff

        # Experiment B: Decision Quality
        uncert_red = exp_b.get("uncertainty_reduction", 0.0)
        info_gain = exp_b.get("information_gain", 0.0)
        if uncert_red > 0.05 and exp_b.get("distribution_separation", 0.0) > 0.1:
            scores["DECISION_QUALITY_VARIABLE"] += 2.5
            scores["MULTI_PURPOSE_DECISION_LAYER"] += 1.5
            evidence["decision_quality_uncert_red"] = uncert_red
        elif info_gain > 0.02:
            scores["DECISION_QUALITY_VARIABLE"] += 1.0
            scores["WEAK_CONTEXT_VARIABLE"] += 1.0
            evidence["decision_quality_info_gain"] = info_gain
        else:
            evidence["no_decision_quality_gain"] = True

        # Experiment C: Risk Conditioning
        c_verdict = exp_c.get("verdict", "")
        c_buckets = exp_c.get("buckets", {})
        risk_scores = [b["risk_score"] for b in c_buckets.values() if b.get("count", 0) >= 2]
        if "monotonic" in c_verdict:
            scores["RISK_FILTER"] += 2.5
            scores["MARKET_EVOLUTION_VARIABLE"] += 1.0
            evidence["risk_monotonic"] = True
        elif "alters_risk" in c_verdict:
            scores["RISK_FILTER"] += 1.5
            evidence["risk_alters"] = True
        if risk_scores and (max(risk_scores) - min(risk_scores)) > 0.1:
            scores["RISK_FILTER"] += 0.5
            evidence["risk_spread"] = max(risk_scores) - min(risk_scores)

        # Experiment D: Position Sizing
        d_sharpe_improv = exp_d.get("avg_sharpe_improvement", 0.0)
        if d_sharpe_improv > 0.1:
            scores["POSITION_SIZING_VARIABLE"] += 2.0
            scores["MULTI_PURPOSE_DECISION_LAYER"] += 1.0
            evidence["position_sizing_sharpe_improv"] = d_sharpe_improv
        elif d_sharpe_improv > 0.0:
            scores["POSITION_SIZING_VARIABLE"] += 0.5
            evidence["position_sizing_sharpe_improv"] = d_sharpe_improv
        else:
            evidence["no_position_sizing_gain"] = d_sharpe_improv

        # Experiment E: Holding Period
        e_buckets = exp_e.get("buckets", {})
        times_profit = [b["time_to_profit"] for b in e_buckets.values() if b.get("count", 0) >= 2]
        times_state = [b["time_to_state_change"] for b in e_buckets.values() if b.get("count", 0) >= 2]
        if times_profit and (max(times_profit) - min(times_profit)) > 50:
            scores["HOLDING_PERIOD_VARIABLE"] += 1.5
            scores["MULTI_PURPOSE_DECISION_LAYER"] += 0.5
            evidence["holding_period_profit_spread"] = max(times_profit) - min(times_profit)
        if times_state and (max(times_state) - min(times_state)) > 50:
            scores["HOLDING_PERIOD_VARIABLE"] += 1.0
            evidence["holding_period_state_spread"] = max(times_state) - min(times_state)

        # Experiment F: Trade Survivability
        f_verdict = exp_f.get("verdict", "")
        if "affects_survivability" in f_verdict:
            scores["WEAK_CONTEXT_VARIABLE"] += 1.0
            scores["RISK_FILTER"] += 0.5
            evidence["survivability_effect"] = True
        else:
            evidence["no_survivability_effect"] = True

        # Experiment G: Cross-Asset
        if exp_g:
            g_verdict = exp_g.get("verdict", "")
            if g_verdict == "operational_usefulness_transfers":
                scores["MULTI_PURPOSE_DECISION_LAYER"] += 2.0
                scores["MARKET_EVOLUTION_VARIABLE"] += 1.5
                evidence["cross_asset_transfer"] = True
            elif g_verdict == "mixed":
                scores["WEAK_CONTEXT_VARIABLE"] += 0.5
                evidence["cross_asset_mixed"] = True
        else:
            evidence["cross_asset_not_tested"] = True

        # Experiment H: Economic Value
        h_sep = exp_h.get("outcome_separation_avg", 0.0)
        h_ig = exp_h.get("information_gain_avg", 0.0)
        h_ur = exp_h.get("uncertainty_reduction", 0.0)
        if h_ur > 0.05 and h_sep > 0.1:
            scores["DECISION_QUALITY_VARIABLE"] += 1.5
            scores["MULTI_PURPOSE_DECISION_LAYER"] += 1.0
            evidence["economic_value_positive"] = True
        elif h_sep > 0.01 or h_ig > 0.02:
            scores["WEAK_CONTEXT_VARIABLE"] += 0.5
            evidence["economic_value_marginal"] = True
        else:
            evidence["no_economic_value"] = True

        # Fallback: if nothing scored, it's an academic artifact
        total_score = sum(scores.values())
        if total_score < 0.5:
            scores["ACADEMIC_ARTIFACT"] = 1.0
            evidence["no_practical_value"] = True

        classification = max(scores, key=lambda k: (scores[k], VERDICTS.index(k)))
        integration = INTEGRATION_MAP.get(classification, "RESEARCH_FURTHER")

        return TIDVVerdict(classification, integration, scores, evidence)
