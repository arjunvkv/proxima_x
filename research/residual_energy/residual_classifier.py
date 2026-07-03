from __future__ import annotations

import numpy as np

from research.residual_energy.residual_validator import ResidualEnergyValidator, REPResult, RESIDUAL_TYPES


CLASSIFICATIONS = ["ARTIFACT", "VOLATILITY_RESIDUAL", "PROXIMA_FACTOR", "NOVEL_MARKET_SIGNAL"]


class ResidualClassifier:
    def __init__(self, validator: ResidualEnergyValidator, prior_results: dict[str, REPResult] | None = None):
        self.validator = validator
        self.prior_results = prior_results or {}

    def _get_pp(self, key: str, *path, default: float = 0.5) -> float:
        rq = self.prior_results.get(key)
        if rq is None or not hasattr(rq, "metrics"):
            return default
        m = rq.metrics
        for p in path:
            if isinstance(m, dict) and p in m:
                m = m[p]
            else:
                return default
        if isinstance(m, dict):
            return float(m.get("pp", default))
        return float(m) if isinstance(m, (int, float)) else default

    def _best_res_pp(self) -> float:
        return self._get_pp("REP-1: Residual Constructor", "residual_results", "xgboost", "alpha_h20", default=0.5)

    def _es_pp(self) -> float:
        return self._get_pp("REP-1: Residual Constructor", "es_alpha", default=0.738)

    def run(self) -> REPResult:
        es_pp = self._es_pp()
        res_pp = self._best_res_pp()

        beats_es = res_pp > es_pp
        decom = self.prior_results.get("REP-3: Residual Decomposition", REPResult("", "", {}))
        most_correlated = decom.metrics.get("most_correlated_layer", "unknown") if hasattr(decom, "metrics") else "unknown"
        max_corr = decom.metrics.get("max_correlation", 0.0) if hasattr(decom, "metrics") else 0.0

        cross_asset = self.prior_results.get("REP-5+6: Residual Reality", REPResult("", "", {}))
        n_assets_beats = cross_asset.metrics.get("n_assets_residual_beats_es", 0) if hasattr(cross_asset, "metrics") else 0

        ortho = self.prior_results.get("REP-4: Orthogonality Test", REPResult("", "", {}))
        residual_adds_more = ortho.metrics.get("residual_adds_more_info", False) if hasattr(ortho, "metrics") else False

        wf = self.prior_results.get("REP-7: Walk-Forward", REPResult("", "", {}))
        res_survives_better_wf = wf.metrics.get("residual_survives_better", False) if hasattr(wf, "metrics") else False

        dep = self.prior_results.get("REP-8+9: Deployment", REPResult("", "", {}))
        res_rank = dep.metrics.get("residual_rank", 99) if hasattr(dep, "metrics") else 99
        plateau = dep.metrics.get("plateau_size", 0.0) if hasattr(dep, "metrics") else 0.0

        evidence = {
            "beats_es": beats_es,
            "res_pp": res_pp,
            "es_pp": es_pp,
            "most_correlated_layer": most_correlated,
            "max_correlation": max_corr,
            "n_assets_residual_beats_es": n_assets_beats,
            "residual_adds_more_info": residual_adds_more,
            "residual_survives_better_walk_forward": res_survives_better_wf,
            "residual_benchmark_rank": res_rank,
            "plateau_size": plateau,
        }

        if not beats_es:
            classification = "ARTIFACT"
            detail = "Residual alpha collapses. It is a statistical artifact of the vol regression."
        elif max_corr > 0.8 and most_correlated in ("realized_vol", "atr", "parkinson_vol"):
            classification = "VOLATILITY_RESIDUAL"
            detail = f"Residual dominated by {most_correlated} (r={max_corr:.3f}). Still a volatility phenomenon."
        elif beats_es and n_assets_beats >= 3 and residual_adds_more and plateau > 0.3:
            classification = "NOVEL_MARKET_SIGNAL"
            detail = "Residual passes all tests. It is a genuinely novel, orthogonal market signal."
        else:
            classification = "PROXIMA_FACTOR"
            detail = f"Residual (PP={res_pp:.3f}) beats ES on {n_assets_beats}/5 assets, near-orthogonal to all layers (max r={max_corr:.3f}), plateau={plateau:.0%}. Orthogonality definitional: residual is the non-volatility component of ES. Deepest surviving Proxima alpha."

        print(f"\n{'='*72}")
        print("REP-10: Final Adjudication")
        print(f"{'='*72}")
        print(f"  ES PP:          {es_pp:.3f}")
        print(f"  Residual PP:    {res_pp:.3f}")
        print(f"  Beats ES:       {beats_es}")
        print(f"  Most Correlated: {most_correlated} (r={max_corr:.3f})")
        print(f"  Assets Beats ES: {n_assets_beats}/5")
        print(f"  Adds More Info: {residual_adds_more}")
        print(f"  WF Survives Better: {res_survives_better_wf}")
        print(f"  Benchmark Rank: {res_rank}")
        print(f"  Plateau:        {plateau:.1%}")
        print(f"\n  Classification: {classification}")
        print(f"  Detail: {detail}")

        return REPResult(
            rq_name="REP-10: Final Adjudication",
            status=classification,
            metrics={
                "classification": classification,
                "detail": detail,
                "evidence": evidence,
                "es_pp": es_pp,
                "residual_pp": res_pp,
                "beats_es": beats_es,
            },
        )
