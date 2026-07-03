from typing import Dict


class SystemValidationReduction:
    def compute_fitness(self,
                        doa_results: Dict[str, float],
                        drift_scores: Dict[str, float],
                        lct_score: float,
                        stability_score: float,
                        allocation_entropy: float = 0.5) -> float:
        if not doa_results:
            return 0.0
        performance = sum(doa_results.values()) / len(doa_results)
        drift_penalty = sum(drift_scores.values()) / len(drift_scores) if drift_scores else 0.0
        stability_bonus = stability_score
        trend_bonus = lct_score
        entropy_penalty = allocation_entropy
        fitness = (
            0.5 * performance +
            0.2 * stability_bonus +
            0.2 * trend_bonus -
            0.3 * drift_penalty -
            0.1 * entropy_penalty
        )
        return max(-1.0, min(1.0, fitness))

    def redundancy_map(self) -> Dict[str, str]:
        return {
            "RTD": "regime stability (KEEP)",
            "CDM": "signal consistency (KEEP but MERGE with MSO signals)",
            "MSO": "oscillation control (KEEP but subset of SSOL)",
            "SSOL": "overarching controller (KEEP PRIMARY)",
            "DRL": "redundant with SSOL drift control (MERGE)",
            "AFL": "redundant with FWO (MERGE INTO FWO)",
            "FWO": "feature weighting (KEEP)",
            "RSL": "regime learning (KEEP)",
            "CAL/TCA": "merge into CWF only (REMOVE SEPARATION)",
            "CWF": "KEEP (unified attribution engine)",
        }
