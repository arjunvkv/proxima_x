from __future__ import annotations

from typing import Any

import numpy as np

from research.interaction_asymmetry.interaction_validator import InteractionValidator, IAEResult
from research.adaptive_alpha_engine.aae_validator import HORIZONS, TARGET_ASSETS, TIME_WINDOWS, _zscore, _future_returns


class CrossAssetValidator:
    def __init__(self, validator: InteractionValidator):
        self.validator = validator

    def run(self) -> IAEResult:
        per_asset: dict[str, Any] = {}
        divergence_signals: dict[str, np.ndarray] = {}
        variable_data: dict[str, dict[str, np.ndarray]] = {}

        for asset in TARGET_ASSETS:
            self.validator.load(asset)
            md_z = self.validator.md_z()
            es_z = self.validator.es_z()
            at_z = self.validator.at_z()

            md_es_div = self.validator.divergence(md_z, es_z, "difference")
            sync = self.validator.classify_synchronization(md_z, es_z, at_z)
            leader = self.validator.detect_leader(md_z, es_z, at_z)
            contradiction = self.validator.detect_contradictions(md_z, es_z, at_z)
            tension = self.validator.tension_index(md_z, es_z, at_z, window=20)

            leader_change = np.zeros(len(leader), dtype=np.float64)
            if len(leader) > 1:
                leader_change[1:] = (leader[1:] != leader[:-1]).astype(np.float64)

            contradiction_active = (contradiction > 0).astype(np.float64)

            tension_z = np.zeros_like(tension)
            if len(tension) > 1 and np.nanstd(tension) > 1e-12:
                tension_z = _zscore(np.nan_to_num(tension.copy(), nan=0.0))

            div_alpha = self.validator.eval_alpha(np.abs(md_es_div), 2)
            sync_alpha = self.validator.eval_alpha((sync == 0).astype(np.float64), 2)
            leader_alpha = self.validator.eval_alpha(leader_change, 2)
            contradiction_alpha = self.validator.eval_alpha(contradiction_active, 2)
            tension_alpha = self.validator.eval_alpha(tension_z, 2)
            es_alpha = self.validator.benchmark_es_alpha()

            per_asset[asset] = {
                "divergence_alpha": div_alpha,
                "sync_alpha": sync_alpha,
                "leader_alpha": leader_alpha,
                "contradiction_alpha": contradiction_alpha,
                "tension_alpha": tension_alpha,
                "es_alpha": es_alpha,
            }

            divergence_signals[asset] = md_es_div
            variable_data[asset] = {"md_z": md_z, "es_z": es_z, "at_z": at_z}

        asset_list = list(TARGET_ASSETS)

        corr_vals = []
        for i in range(len(asset_list)):
            for j in range(i + 1, len(asset_list)):
                a1, a2 = asset_list[i], asset_list[j]
                s1 = divergence_signals[a1]
                s2 = divergence_signals[a2]
                min_len = min(len(s1), len(s2))
                if min_len > 5:
                    c = np.corrcoef(s1[:min_len], s2[:min_len])[0, 1]
                    if not np.isnan(c):
                        corr_vals.append(c)

        interaction_similarity = float(np.mean(corr_vals)) if corr_vals else 0.0

        var_corr_vals = []
        for i in range(len(asset_list)):
            for j in range(i + 1, len(asset_list)):
                a1, a2 = asset_list[i], asset_list[j]
                for var_name in ["md_z", "es_z", "at_z"]:
                    v1 = variable_data[a1][var_name]
                    v2 = variable_data[a2][var_name]
                    min_len = min(len(v1), len(v2))
                    if min_len > 5:
                        c = np.corrcoef(v1[:min_len], v2[:min_len])[0, 1]
                        if not np.isnan(c):
                            var_corr_vals.append(c)

        variable_similarity = float(np.mean(var_corr_vals)) if var_corr_vals else 0.0

        interactions_more_universal = interaction_similarity > variable_similarity

        n_assets_divergence_beats_es = 0
        for asset in asset_list:
            d_pp = per_asset[asset]["divergence_alpha"].get("pp", 0.5)
            e_pp = per_asset[asset]["es_alpha"].get("pp", 0.5)
            if d_pp > e_pp:
                n_assets_divergence_beats_es += 1

        interaction_type_positive_counts: dict[str, int] = {}
        for itype in ["divergence_alpha", "sync_alpha", "leader_alpha", "contradiction_alpha", "tension_alpha"]:
            count = sum(1 for asset in asset_list if per_asset[asset][itype].get("pp", 0.5) > 0.5)
            interaction_type_positive_counts[itype.replace("_alpha", "")] = count

        print("=== CrossAssetValidator (RQ8) ===")
        header = f"{'Asset':>10s} | {'Div_PP':>8s} | {'Sync_PP':>8s} | {'Lead_PP':>8s} | {'Contr_PP':>8s} | {'Tens_PP':>8s} | {'ES_PP':>8s}"
        print(header)
        print("-" * len(header))
        for asset in asset_list:
            d = per_asset[asset]
            print(f"{asset:>10s} | {d['divergence_alpha'].get('pp', 0.5):>8.4f} | {d['sync_alpha'].get('pp', 0.5):>8.4f} | {d['leader_alpha'].get('pp', 0.5):>8.4f} | {d['contradiction_alpha'].get('pp', 0.5):>8.4f} | {d['tension_alpha'].get('pp', 0.5):>8.4f} | {d['es_alpha'].get('pp', 0.5):>8.4f}")
        print(f"\nInteraction similarity (avg cross-asset divergence corr): {interaction_similarity:.4f}")
        print(f"Variable similarity (avg cross-asset variable corr): {variable_similarity:.4f}")
        print(f"Interactions more universal? {interactions_more_universal}")
        print(f"Assets where divergence beats ES: {n_assets_divergence_beats_es} / {len(asset_list)}")
        print(f"Interaction type positive counts: {interaction_type_positive_counts}")

        metrics: dict[str, Any] = {
            "per_asset": per_asset,
            "interaction_similarity": interaction_similarity,
            "variable_similarity": variable_similarity,
            "interactions_more_universal": interactions_more_universal,
            "n_assets_divergence_beats_es": n_assets_divergence_beats_es,
            "interaction_type_positive_counts": interaction_type_positive_counts,
        }

        return IAEResult(rq_name="RQ8_Cross_Asset_Universality", status="COMPLETE", metrics=metrics)
