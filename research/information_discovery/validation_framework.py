from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from research.information_discovery.mi_estimator import MIEstimator
from research.information_discovery.feature_scorer import FeatureScore


class ValidationFramework:
    def __init__(self, mi_estimator: Optional[MIEstimator] = None):
        self.mi = mi_estimator or MIEstimator(n_bins=20)

    def cross_asset_validation(self, feature_scores_per_asset: dict[str, list[FeatureScore]]) -> dict:
        feature_stds: dict[str, list[float]] = {}
        for asset_id, scores in feature_scores_per_asset.items():
            for fs in scores:
                if fs.name not in feature_stds:
                    feature_stds[fs.name] = []
                feature_stds[fs.name].append(fs.information_gain)

        per_feature: dict[str, dict] = {}
        all_stds: list[float] = []
        for fname, gains in feature_stds.items():
            std_val = float(np.std(gains)) if len(gains) > 1 else 0.0
            mean_val = float(np.mean(gains))
            per_feature[fname] = {
                "cross_asset_mean": mean_val,
                "cross_asset_std": std_val,
                "cross_asset_stability": 1.0 / (1.0 + std_val) if std_val > 0 else 1.0,
            }
            all_stds.append(std_val)

        mean_std = float(np.mean(all_stds)) if all_stds else 0.0
        surviving: list[str] = [
            fname for fname, d in per_feature.items()
            if d["cross_asset_std"] <= mean_std
        ]

        return {
            "per_feature": per_feature,
            "overall_cross_asset_score": float(np.mean([
                d["cross_asset_stability"] for d in per_feature.values()
            ])) if per_feature else 0.0,
            "surviving_features": surviving,
            "mean_std_threshold": mean_std,
        }

    def cross_regime_validation(self, feature_mi_per_regime: dict[str, dict[str, float]]) -> dict:
        regime_features: dict[str, list[float]] = {}
        for regime_label, features in feature_mi_per_regime.items():
            for fname, mi_score in features.items():
                if fname not in regime_features:
                    regime_features[fname] = []
                regime_features[fname].append(mi_score)

        per_feature: dict[str, dict] = {}
        for fname, scores in regime_features.items():
            std_val = float(np.std(scores)) if len(scores) > 1 else 0.0
            mean_val = float(np.mean(scores))
            per_feature[fname] = {
                "regime_mean": mean_val,
                "regime_std": std_val,
                "regime_invariance": 1.0 / (1.0 + std_val),
            }

        threshold = float(np.mean([d["regime_std"] for d in per_feature.values()])) if per_feature else 0.0
        surviving: list[str] = [
            fname for fname, d in per_feature.items()
            if d["regime_std"] <= threshold
        ]

        return {
            "per_feature": per_feature,
            "overall_regime_score": float(np.mean([
                d["regime_invariance"] for d in per_feature.values()
            ])) if per_feature else 0.0,
            "surviving_features": surviving,
            "threshold": threshold,
        }

    def out_of_sample_validation(self, features_in_sample: dict, features_oos: dict, target_is: NDArray, target_oos: NDArray) -> dict:
        per_feature: dict[str, dict] = {}
        all_features = set(features_in_sample.keys()) & set(features_oos.keys())
        for fname in all_features:
            mi_is = self.mi.mutual_info(features_in_sample[fname], target_is)
            mi_oos = self.mi.mutual_info(features_oos[fname], target_oos)
            oos_drop = (mi_is - mi_oos) / mi_is if mi_is > 1e-10 else 1.0
            oos_drop = max(0.0, min(1.0, oos_drop))
            per_feature[fname] = {
                "mi_in_sample": mi_is,
                "mi_out_of_sample": mi_oos,
                "oos_drop": oos_drop,
                "survives": oos_drop <= 0.5,
            }

        surviving = [fname for fname, d in per_feature.items() if d["survives"]]
        return {
            "per_feature": per_feature,
            "overall_oos_score": float(np.mean([
                1.0 - d["oos_drop"] for d in per_feature.values()
            ])) if per_feature else 0.0,
            "surviving_features": surviving,
        }

    def bootstrap_validation(self, feature: NDArray, target: NDArray, n_bootstrap: int = 100) -> dict:
        valid = ~(np.isnan(feature) | np.isnan(target))
        f, t = feature[valid], target[valid]
        n = len(f)
        if n < 2:
            return {"mean": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "p_value": 1.0}

        boot_mis: list[float] = []
        rng = np.random.default_rng()
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            mi_val = self.mi.mutual_info(f[idx], t[idx])
            boot_mis.append(mi_val)

        boot_arr = np.array(boot_mis)
        mean = float(np.mean(boot_arr))
        std = float(np.std(boot_arr))
        ci_lower = float(np.percentile(boot_arr, 2.5))
        ci_upper = float(np.percentile(boot_arr, 97.5))
        p_value = float(np.mean(boot_arr <= 0.0))

        return {
            "mean": mean,
            "std": std,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": p_value,
        }

    def compute_feature_reliability(self, in_sample_mi: float, oos_mi: float, bootstrap_std: float) -> float:
        stability_ratio = min(oos_mi / in_sample_mi, 1.0) if in_sample_mi > 1e-10 else 0.0
        variance_penalty = 1.0 / (1.0 + bootstrap_std)
        return stability_ratio * variance_penalty

    def validate_all(self, feature_scores_per_asset: dict, feature_mi_per_regime: dict, features_is: dict, features_oos: dict, target_is: NDArray, target_oos: NDArray) -> dict:
        cross_asset = self.cross_asset_validation(feature_scores_per_asset)
        cross_regime = self.cross_regime_validation(feature_mi_per_regime)
        oos = self.out_of_sample_validation(features_is, features_oos, target_is, target_oos)

        all_feature_names: set[str] = set()
        for d in [cross_asset["per_feature"], cross_regime["per_feature"], oos["per_feature"]]:
            all_feature_names.update(d.keys())

        bootstrap_results: dict[str, dict] = {}
        for fname in all_feature_names:
            if fname in features_is and fname in features_oos:
                combined = np.concatenate([features_is[fname], features_oos[fname]])
                combined_target = np.concatenate([target_is, target_oos])
                bootstrap_results[fname] = self.bootstrap_validation(combined, combined_target)

        per_feature: dict[str, dict] = {}
        for fname in all_feature_names:
            ca = cross_asset["per_feature"].get(fname, {})
            cr = cross_regime["per_feature"].get(fname, {})
            oo = oos["per_feature"].get(fname, {})
            bt = bootstrap_results.get(fname, {})

            ca_survive = fname in cross_asset["surviving_features"]
            cr_survive = fname in cross_regime["surviving_features"]
            oos_survive = oo.get("survives", False)
            bt_survive = bt.get("p_value", 1.0) < 0.05

            n_survived = sum([ca_survive, cr_survive, oos_survive, bt_survive])
            reliability = self.compute_feature_reliability(
                oo.get("mi_in_sample", 0.0),
                oo.get("mi_out_of_sample", 0.0),
                bt.get("std", 1.0),
            )

            per_feature[fname] = {
                "cross_asset_stability": ca.get("cross_asset_stability", 0.0),
                "regime_invariance": cr.get("regime_invariance", 0.0),
                "oos_drop": oo.get("oos_drop", 1.0),
                "bootstrap_p_value": bt.get("p_value", 1.0),
                "bootstrap_ci": (bt.get("ci_lower", 0.0), bt.get("ci_upper", 0.0)),
                "survives_cross_asset": ca_survive,
                "survives_cross_regime": cr_survive,
                "survives_oos": oos_survive,
                "survives_bootstrap": bt_survive,
                "n_survival_tests_passed": n_survived,
                "reliability": reliability,
                "survives": n_survived >= 3,
            }

        return {
            "per_feature": per_feature,
            "cross_asset": cross_asset,
            "cross_regime": cross_regime,
            "out_of_sample": oos,
            "bootstrap": bootstrap_results,
            "surviving_features": [
                fname for fname, d in per_feature.items() if d["survives"]
            ],
        }

    def generate_validation_report(self, validation_results: dict) -> str:
        lines: list[str] = []
        lines.append("# Validation Report")
        lines.append("")

        survivors = validation_results.get("surviving_features", [])
        lines.append(f"**Surviving Features:** {len(survivors)}")
        lines.append("")

        per_feature = validation_results.get("per_feature", {})
        if per_feature:
            lines.append("## Per-Feature Summary")
            header = (
                f"{'Feature':<25} {'CA Stab':<10} {'Regime Inv':<12} "
                f"{'OOS Drop':<10} {'BS P-Val':<10} {'Reliab':<8} {'Pass'}"
            )
            lines.append(header)
            lines.append("-" * len(header))
            for fname in sorted(per_feature.keys()):
                d = per_feature[fname]
                lines.append(
                    f"{fname:<25} {d['cross_asset_stability']:<10.4f} "
                    f"{d['regime_invariance']:<12.4f} {d['oos_drop']:<10.4f} "
                    f"{d['bootstrap_p_value']:<10.4f} {d['reliability']:<8.4f} "
                    f"{'✓' if d['survives'] else '✗'}"
                )
            lines.append("")

        ca = validation_results.get("cross_asset", {})
        if ca:
            lines.append("## Cross-Asset Validation")
            lines.append(f"- Overall Score: {ca.get('overall_cross_asset_score', 0.0):.4f}")
            lines.append(f"- Surviving: {len(ca.get('surviving_features', []))} features")
            lines.append("")

        cr = validation_results.get("cross_regime", {})
        if cr:
            lines.append("## Cross-Regime Validation")
            lines.append(f"- Overall Score: {cr.get('overall_regime_score', 0.0):.4f}")
            lines.append(f"- Surviving: {len(cr.get('surviving_features', []))} features")
            lines.append("")

        oos = validation_results.get("out_of_sample", {})
        if oos:
            lines.append("## Out-of-Sample Validation")
            lines.append(f"- Overall OOS Score: {oos.get('overall_oos_score', 0.0):.4f}")
            lines.append(f"- Surviving: {len(oos.get('surviving_features', []))} features")
            lines.append("")

        bt = validation_results.get("bootstrap", {})
        if bt:
            lines.append("## Bootstrap Validation")
            for fname, d in sorted(bt.items()):
                lines.append(
                    f"- {fname}: mean={d.get('mean', 0.0):.4f}, "
                    f"std={d.get('std', 0.0):.4f}, "
                    f"CI=[{d.get('ci_lower', 0.0):.4f}, {d.get('ci_upper', 0.0):.4f}], "
                    f"p={d.get('p_value', 1.0):.4f}"
                )
            lines.append("")

        lines.append("## Summary")
        total = len(per_feature)
        passed = len(survivors)
        lines.append(f"- {passed}/{total} features survived all validation checks")
        return "\n".join(lines)
