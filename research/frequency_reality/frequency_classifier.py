class FrequencyClassifier:
    def __init__(self, cost_analysis):
        self._analysis = cost_analysis

    # RQ10
    def classify(self) -> dict:
        leakage = self._analysis.leakage_rate()
        adr = self._analysis.alpha_destruction_ratio(horizon_tag="h20")
        oc = self._analysis.opportunity_cost(horizon_tag="h20")
        blocked_mean = oc["blocked"]["mean_return"]
        executed_mean = oc["executed"]["mean_return"]

        if executed_mean > 0 and blocked_mean <= executed_mean:
            classification = "ALPHA_PROTECTOR"
            confidence = min(1.0, (executed_mean - blocked_mean) / max(abs(executed_mean), 0.0001))
        elif adr < 0.15:
            classification = "NEUTRAL_CONTROLLER"
            confidence = 1.0 - adr
        elif blocked_mean > executed_mean and adr > 0.25:
            classification = "ALPHA_DESTROYER"
            confidence = min(1.0, adr)
        else:
            classification = "NEUTRAL_CONTROLLER"
            confidence = 0.5

        return {
            "classification": classification,
            "confidence": round(confidence, 3),
            "adr": round(adr, 3),
            "leakage_rate": leakage["leakage_rate"],
            "blocked_mean_return": round(blocked_mean, 6),
            "executed_mean_return": round(executed_mean, 6),
            "blocked_profitable": leakage["blocked_profitable"],
            "blocked_total": leakage["blocked_total"]}

    def dashboard_section(self) -> str:
        c = self.classify()
        le = self._analysis.leakage_rate()
        return (
            f"  INVALID SPREAD AUDIT\n"
            f"  Blocked Signals:          {le['blocked_total']}\n"
            f"  Profitable Blocked:       {le['blocked_profitable']}\n"
            f"  Leakage Rate:             {le['leakage_rate']}%\n"
            f"  ADR:                      {c['adr']}\n"
            f"  Classification:           {c['classification']} (conf={c['confidence']})")
