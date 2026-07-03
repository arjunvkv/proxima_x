class RQ10Adjudication:
    """RQ10: Final adjudication — which mechanism is the root cause?"""

    def __init__(self, results: dict):
        self.results = results

    def run(self) -> dict:
        r = self.results

        # Extract key findings from each RQ
        rq8 = r.get("rq8_threshold_drift_order", {})
        rq1 = r.get("rq1_persistence_drivers", {})
        rq2 = r.get("rq2_survival_curves", {})
        rq3 = r.get("rq3_threshold_mapping", {})
        rq4 = r.get("rq4_residual_lifespan", {})
        rq7 = r.get("rq7_walk_forward", {})
        rq9 = r.get("rq9_regime_classifier", {})

        # Evidence A: Threshold Drift causes failure
        causal_direction = rq8.get("causal_direction", "mutual_or_unknown")
        threshold_leads = causal_direction == "threshold_leads_persistence"
        peak_lag = rq8.get("peak_lag", 0)

        evidence_a = {
            "causal_direction": causal_direction,
            "threshold_leads_persistence": threshold_leads,
            "peak_lag": peak_lag,
            "score": 1.0 if threshold_leads else 0.0,
            "summary": (
                "Threshold drift causes persistence collapse"
                if threshold_leads else
                "Threshold drift does NOT cause persistence collapse"
            ),
        }

        # Evidence B: Persistence Collapse causes failure
        persistence_leads = causal_direction == "persistence_leads_threshold"
        break_ordering = rq8.get("break_ordering", "")
        persistence_breaks_first = break_ordering == "persistence_breaks_first"
        forecastable = rq7.get("forecastable", False)
        r2 = rq7.get("mean_r2", 0.0)

        evidence_b = {
            "causal_direction": causal_direction,
            "persistence_leads_threshold": persistence_leads,
            "persistence_breaks_first": persistence_breaks_first,
            "forecastable": forecastable,
            "r2": r2,
            "score": (1.0 if persistence_leads else 0.0) +
                     (0.5 if persistence_breaks_first else 0.0) +
                     (0.3 if forecastable else 0.0),
            "summary": (
                "Persistence collapse causes threshold drift"
                if persistence_leads else
                "Persistence collapse does NOT cause threshold drift"
            ),
        }

        # Evidence C: Residual Energy Collapse causes failure
        best_decay = rq4.get("best_decay_model", "unknown")
        decay_rate = 0.0
        if best_decay == "exponential":
            decay_rate = rq4.get("aggregate_fits", {}).get("exponential", {}).get("rate_mean", 0.0)
        elif best_decay == "power_law":
            decay_rate = rq4.get("aggregate_fits", {}).get("power_law", {}).get("exponent_mean", 0.0)
        elif best_decay == "linear":
            decay_rate = abs(rq4.get("aggregate_fits", {}).get("linear", {}).get("slope_mean", 0.0))

        re_is_top_driver = rq1.get("top_entry_layer") == "residual_energy"
        re_r2 = rq1.get("top_entry_score", 0.0)

        evidence_c = {
            "best_decay_model": best_decay,
            "decay_rate": decay_rate,
            "re_is_top_driver": re_is_top_driver,
            "re_driver_score": re_r2,
            "score": (0.5 if best_decay != "linear" else 0.0) +
                     (0.5 if re_is_top_driver else 0.0),
            "summary": (
                f"Residual energy decays via {best_decay} model"
                if best_decay != "unknown" else
                "Residual energy decay model could not be determined"
            ),
        }

        # Classification
        scores = {
            "threshold_drift": evidence_a["score"],
            "persistence_collapse": evidence_b["score"],
            "residual_energy_collapse": evidence_c["score"],
        }
        max_score = max(scores.values()) if scores else 0.0

        if max_score == 0:
            classification = "UNRESOLVED"
            confidence = 0.0
        else:
            winner = max(scores, key=scores.get)
            total = sum(scores.values())
            confidence = max_score / total if total > 0 else 0.0

            if winner == "persistence_collapse" and confidence > 0.5:
                classification = "PERSISTENCE_ROOT_CAUSE"
            elif winner == "threshold_drift" and confidence > 0.5:
                classification = "THRESHOLD_ROOT_CAUSE"
            elif confidence > 0.4:
                classification = "MIXED_CAUSALITY"
            else:
                classification = "UNRESOLVED"

        return {
            "classification": classification,
            "confidence": float(confidence),
            "evidence": {
                "threshold_drift_causes_failure": evidence_a,
                "persistence_collapse_causes_failure": evidence_b,
                "residual_energy_collapse_causes_failure": evidence_c,
            },
            "scores": scores,
            "rq8_causal_direction": causal_direction,
            "rq1_top_driver": rq1.get("top_entry_layer", "unknown"),
            "rq2_half_life_trajectory": rq2.get("half_life_trajectory", {}),
            "rq2_structural_breakpoints": rq2.get("structural_breakpoints", []),
            "rq3_elasticity": rq3.get("mean_elasticity", 0.0),
            "rq4_best_decay": best_decay,
            "rq7_r2": r2,
            "rq9_f1": rq9.get("persistence_classifier", {}).get("f1_macro", 0.0),
        }
