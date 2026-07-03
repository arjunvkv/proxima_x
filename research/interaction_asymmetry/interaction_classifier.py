from __future__ import annotations

import numpy as np

from research.interaction_asymmetry.interaction_validator import InteractionValidator, IAEResult


CLASSIFICATIONS = ["NO_EDGE", "WEAK_EDGE", "CONDITIONAL_EDGE", "ROBUST_EDGE", "SUPERIOR_TO_ES"]

ALPHA_LABELS = [
    "DivergenceAlpha", "SynchronizationAlpha", "FrictionAlpha",
    "LeadershipAlpha", "ContradictionAlpha", "TensionAlpha",
    "TransitionPressureAlpha",
]


class InteractionClassifier:
    def __init__(self, validator: InteractionValidator, prior_results: dict[str, IAEResult] | None = None):
        self.validator = validator
        self.prior_results = prior_results or {}

    def _get_pp(self, key: str, subkey: str = "divergence_alpha", default: float = 0.5) -> float:
        rq = self.prior_results.get(key)
        if rq is None or not hasattr(rq, "metrics"):
            return default
        m = rq.metrics.get(subkey, {})
        if isinstance(m, dict):
            return float(m.get("pp", default))
        if isinstance(m, list) and len(m) > 0:
            best = max(x.get("pp", default) for x in m if isinstance(x, dict))
            return best if best != default else default
        return default

    def _get_sharpe(self, key: str, subkey: str = "divergence_alpha", default: float = 0.0) -> float:
        rq = self.prior_results.get(key)
        if rq is None or not hasattr(rq, "metrics"):
            return default
        m = rq.metrics.get(subkey, {})
        if isinstance(m, dict):
            return float(m.get("sharpe", default))
        if isinstance(m, list) and len(m) > 0:
            best = max(x.get("sharpe", default) for x in m if isinstance(x, dict))
            return best if best != default else default
        return default

    def _classify(self, pp: float, sharpe: float, es_pp: float, es_sharpe: float) -> str:
        if pp > es_pp and sharpe > es_sharpe * 1.05:
            return "SUPERIOR_TO_ES"
        if pp > 0.65 and sharpe > 0.4:
            return "ROBUST_EDGE"
        if pp > 0.58 and sharpe > 0.2:
            return "CONDITIONAL_EDGE"
        if pp > 0.53:
            return "WEAK_EDGE"
        return "NO_EDGE"

    def run(self) -> IAEResult:
        es_pp = self._get_pp("RQ1: Divergence Alpha", "benchmark_es_alpha", 0.74)
        es_sharpe = self._get_sharpe("RQ1: Divergence Alpha", "benchmark_es_alpha", 0.69)

        results_pp: dict[str, float] = {}
        results_sharpe: dict[str, float] = {}

        rq_map = {
            "DivergenceAlpha": ("RQ1: Divergence Alpha", "best_divergence"),
            "SynchronizationAlpha": ("RQ2: Synchronization States", "strongest_state_metrics"),
            "FrictionAlpha": ("RQ3: Temporal Friction", "friction_alpha"),
            "LeadershipAlpha": ("RQ4: Leadership Rotation", "leader_change_alpha"),
            "ContradictionAlpha": ("RQ5: Hidden Contradictions", "agreement_vs_contradiction"),
            "TensionAlpha": ("RQ6: Tension Surface", "tension_alpha"),
            "TransitionPressureAlpha": ("RQ7: Transition Pressure", "pressure_alpha"),
        }

        for label, (rq_key, metric_key) in rq_map.items():
            results_pp[label] = self._get_pp(rq_key, metric_key, 0.5)
            results_sharpe[label] = self._get_sharpe(rq_key, metric_key, 0.0)

        classifications: dict[str, str] = {}
        for label in ALPHA_LABELS:
            classifications[label] = self._classify(
                results_pp.get(label, 0.5), results_sharpe.get(label, 0.0),
                es_pp, es_sharpe,
            )

        n_superior = sum(1 for c in classifications.values() if c == "SUPERIOR_TO_ES")
        n_robust = sum(1 for c in classifications.values() if c == "ROBUST_EDGE")

        print(f"\n{'='*72}")
        print("RQ10: Interaction Alpha Adjudication")
        print(f"{'='*72}")
        print(f"  ES Benchmark: PP={es_pp:.3f}, Sharpe={es_sharpe:.3f}")
        print(f"\n  {'Alpha':25s} {'PP':>8s} {'Sharpe':>8s} {'Classification':>20s}")
        print(f"  {'-'*61}")
        for label in ALPHA_LABELS:
            pp = results_pp.get(label, 0.5)
            sh = results_sharpe.get(label, 0.0)
            cls = classifications.get(label, "NO_EDGE")
            print(f"  {label:25s} {pp:>8.3f} {sh:>8.3f} {cls:>20s}")

        if n_superior > 0:
            verdict = "ORIGINAL_HYPOTHESIS_SURVIVES"
            detail = f"{n_superior} interaction alpha(s) exceed ES. The edge is in the relationships."
        else:
            verdict = "ORIGINAL_HYPOTHESIS_REJECTED"
            detail = "No interaction alpha exceeds ES. The edge is in the variable itself."

        print(f"\n  Verdict: {verdict}")
        print(f"  Detail: {detail}")
        print(f"  Total interaction alphas: {len(ALPHA_LABELS)}")
        print(f"  Superior to ES: {n_superior}")
        print(f"  Robust edge: {n_robust}")

        return IAEResult(
            rq_name="RQ10: Interaction Adjudication",
            status=verdict,
            metrics={
                "es_benchmark_pp": es_pp,
                "es_benchmark_sharpe": es_sharpe,
                "interaction_alphas": {label: {"pp": results_pp.get(label, 0.5),
                                                "sharpe": results_sharpe.get(label, 0.0),
                                                "classification": classifications.get(label, "NO_EDGE")}
                                       for label in ALPHA_LABELS},
                "verdict": verdict,
                "detail": detail,
                "n_superior_to_es": n_superior,
                "n_robust_edge": n_robust,
            },
        )
