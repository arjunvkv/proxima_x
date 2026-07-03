from __future__ import annotations

from typing import Any

import numpy as np

from research.memory_physics.memory_validator import MemoryValidator, MPRResult


class GeneratorTournament:
    """RQ9: Round-robin generator competition.

    Candidates: memory_conflict, memory_density, energy_storage, compression, adaptive_time.
    Each pair is evaluated for causal strength in both directions.

    Ranking by: causal_explanatory_power, survival_score, replacement_score.
    """

    def __init__(self, validator: MemoryValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset
        self._max_lag = 200

    def run(self) -> MPRResult:
        data = self.validator.load_asset_data(self.asset)
        signals = self.validator.compute_signals(data)

        n = len(signals.get("adaptive_time", np.zeros(1)))
        returns = np.asarray(signals["returns"], dtype=np.float64)

        compression = np.zeros(n, dtype=np.float64)
        for i in range(20, n):
            compression[i] = np.std(returns[i - 20:i])

        signals["compression"] = compression

        candidates = ["memory_conflict", "memory_density", "energy_storage", "compression", "adaptive_time"]
        downstream_targets = ["state_mutation_rate", "regime_change_probability"]

        from research.memory_physics.memory_validator import _find_peak_lag

        # Round 1: Head-to-head causal strength between all pairs
        pair_results = {}
        for src in candidates:
            for tgt in candidates:
                if src == tgt:
                    continue
                sig_src = np.asarray(signals.get(src, np.zeros(n)), dtype=np.float64)
                sig_tgt = np.asarray(signals.get(tgt, np.zeros(n)), dtype=np.float64)
                common = min(len(sig_src), len(sig_tgt))
                if common < self._max_lag * 2 + 1:
                    continue
                lag, corr = _find_peak_lag(sig_src[:common], sig_tgt[:common], self._max_lag)
                flow = self.validator.information_flow(src, tgt, signals)
                pair_results[f"{src}->{tgt}"] = {"lag": lag, "corr": corr, "flow": flow}

        # Score each candidate: how well it LEADS others (lag < 0)
        lead_scores = {}
        for c in candidates:
            leads = 0
            total_corr = 0.0
            for key, val in pair_results.items():
                if key.startswith(f"{c}->") and val["lag"] < 0:
                    leads += 1
                    total_corr += abs(val["corr"])
            lead_scores[c] = {"leads": leads, "total_corr": total_corr, "avg_corr": total_corr / max(leads, 1)}

        # Score each candidate: downstream explanatory power
        downstream_scores = {}
        for c in candidates:
            total_exp = 0.0
            for tgt in downstream_targets:
                flow = self.validator.information_flow(c, tgt, signals)
                total_exp += flow
            downstream_scores[c] = total_exp

        # Replacement score: can candidate X predict what Y predicts?
        replacement_scores = {}
        for src in candidates:
            for tgt in candidates:
                if src == tgt:
                    continue
                key = f"{src}_replaces_{tgt}"
                r_score = 0.0
                for dt in downstream_targets:
                    flow_src = self.validator.information_flow(src, dt, signals)
                    flow_tgt = self.validator.information_flow(tgt, dt, signals)
                    if flow_tgt > 0:
                        r_score += flow_src / flow_tgt
                replacement_scores[f"{src}->{tgt}"] = r_score / max(len(downstream_targets), 1)

        # Composite ranking
        rankings = {}
        for c in candidates:
            ls = lead_scores.get(c, {})
            ds = downstream_scores.get(c, 0)
            composite = ls.get("avg_corr", 0) * 0.3 + ls.get("leads", 0) / max(len(candidates) - 1, 1) * 0.3 + min(ds * 10, 1.0) * 0.4
            rankings[c] = {
                "lead_count": ls.get("leads", 0),
                "avg_lead_corr": ls.get("avg_corr", 0),
                "downstream_explanatory_power": ds,
                "composite_score": composite,
            }

        sorted_ranking = sorted(rankings.items(), key=lambda x: x[1]["composite_score"], reverse=True)
        rank_order = [name for name, _ in sorted_ranking]

        metrics = {
            "candidates": candidates,
            "pairwise_results": pair_results,
            "lead_scores": lead_scores,
            "downstream_scores": downstream_scores,
            "replacement_scores": replacement_scores,
            "composite_rankings": rankings,
            "rank_order": rank_order,
        }

        print(f"  Generator Tournament Results:")
        for i, (name, score) in enumerate(sorted_ranking):
            leads = score["lead_count"]
            ds = score["downstream_explanatory_power"]
            print(f"    #{i + 1} {name}: composite={score['composite_score']:.4f}, leads={leads}, downstream={ds:.6f}")

        return MPRResult("generator_tournament", "PASSED", metrics=metrics)
