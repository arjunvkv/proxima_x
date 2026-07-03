from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.stats import pearsonr

from research.information_discovery.mi_estimator import MIEstimator
from research.mechanism_discovery.base import BaseMechanism


class AttackSuite:
    def __init__(self, mi_estimator: Optional[MIEstimator] = None):
        self.mi = mi_estimator or MIEstimator(n_bins=20)
        self.results: dict[str, dict[str, Any]] = {}

    def cross_asset_transfer(self, mechanism: BaseMechanism, train_data: dict, test_assets: dict[str, dict], states: NDArray) -> dict:
        train_result = mechanism.compute(train_data, states)
        train_contrib = mechanism.get_state_contribution()
        scores: dict[str, float] = {}
        for asset_id, asset_data in test_assets.items():
            mechanism.reset()
            _ = mechanism.compute(asset_data)
            test_contrib = mechanism.get_state_contribution()
            if len(train_contrib) < 2 or len(test_contrib) < 2:
                scores[asset_id] = 0.0
                continue
            min_len = min(len(train_contrib), len(test_contrib))
            corr = float(np.corrcoef(train_contrib[:min_len], test_contrib[:min_len])[0, 1])
            scores[asset_id] = max(0.0, corr) if not np.isnan(corr) else 0.0
        asset_transfer_score = float(np.mean(list(scores.values()))) if scores else 0.0
        return {"per_asset": scores, "asset_transfer_score": asset_transfer_score, "passed": asset_transfer_score > 0.3}

    def cross_time_transfer(self, mechanism: BaseMechanism, train_data: dict, test_data: dict, states: NDArray) -> dict:
        mechanism.reset()
        train_result = mechanism.compute(train_data, states)
        train_contrib = mechanism.get_state_contribution()
        mechanism.reset()
        test_result = mechanism.compute(test_data)
        test_contrib = mechanism.get_state_contribution()
        if len(train_contrib) < 2 or len(test_contrib) < 2:
            return {"time_transfer_score": 0.0, "passed": False}
        min_len = min(len(train_contrib), len(test_contrib))
        corr = float(np.corrcoef(train_contrib[:min_len], test_contrib[:min_len])[0, 1])
        time_transfer_score = max(0.0, corr) if not np.isnan(corr) else 0.0
        return {"time_transfer_score": time_transfer_score, "passed": time_transfer_score > 0.3}

    def regime_transfer(self, mechanism: BaseMechanism, data: dict, regime_masks: dict[str, NDArray], states: NDArray) -> dict:
        scores: dict[str, float] = {}
        baseline_result = mechanism.compute(data, states)
        baseline_contrib = mechanism.get_state_contribution()
        for regime, mask in regime_masks.items():
            if mask.sum() < 10:
                continue
            regime_data = {k: v[mask] for k, v in data.items() if isinstance(v, np.ndarray) and len(v) == len(mask)}
            mechanism.reset()
            _ = mechanism.compute(regime_data)
            regime_contrib = mechanism.get_state_contribution()
            if len(baseline_contrib) < 2 or len(regime_contrib) < 2:
                scores[str(regime)] = 0.0
                continue
            min_len = min(len(baseline_contrib), len(regime_contrib))
            corr = float(np.corrcoef(baseline_contrib[:min_len], regime_contrib[:min_len])[0, 1])
            scores[str(regime)] = max(0.0, corr) if not np.isnan(corr) else 0.0
        reg_transfer_score = float(np.mean(list(scores.values()))) if scores else 0.0
        return {"per_regime": scores, "regime_transfer_score": reg_transfer_score, "passed": reg_transfer_score > 0.3}

    def noise_injection(self, mechanism: BaseMechanism, data: dict, states: NDArray, noise_levels: list[float] | None = None) -> dict:
        if noise_levels is None:
            noise_levels = [0.05, 0.10, 0.20, 0.30, 0.40]
        baseline_result = mechanism.compute(data, states)
        baseline_contrib = mechanism.get_state_contribution()
        if len(baseline_contrib) < 2:
            return {"per_noise_level": {}, "noise_survival_score": 0.0, "passed": False}
        scores: dict[float, float] = {}
        for nl in noise_levels:
            noisy_data: dict[str, NDArray] = {}
            for k, v in data.items():
                if isinstance(v, np.ndarray) and v.dtype.kind == "f":
                    noise = np.random.randn(*v.shape) * float(np.std(v)) * nl
                    noisy_data[k] = v + noise
                else:
                    noisy_data[k] = v
            mechanism.reset()
            _ = mechanism.compute(noisy_data, states)
            noisy_contrib = mechanism.get_state_contribution()
            if len(noisy_contrib) < 2:
                scores[nl] = 0.0
                continue
            min_len = min(len(baseline_contrib), len(noisy_contrib))
            corr = float(np.corrcoef(baseline_contrib[:min_len], noisy_contrib[:min_len])[0, 1])
            scores[nl] = max(0.0, corr) if not np.isnan(corr) else 0.0
        noise_survival = float(np.mean(list(scores.values()))) if scores else 0.0
        return {"per_noise_level": scores, "noise_survival_score": noise_survival, "passed": noise_survival > 0.5}

    def adversarial_perturbation(self, mechanism: BaseMechanism, data: dict, states: NDArray) -> dict:
        baseline_result = mechanism.compute(data, states)
        baseline_contrib = mechanism.get_state_contribution()
        if len(baseline_contrib) < 2:
            return {"adversarial_score": 0.0, "passed": False}
        scores: dict[str, float] = {}
        n = len(states)
        for pname, perturbed_states in [
            ("feature_removal", np.full_like(states, -1)),
            ("state_scramble", np.random.permutation(states)),
            ("sequence_scramble", states[np.random.permutation(n)]),
        ]:
            mechanism.reset()
            _ = mechanism.compute(data, perturbed_states)
            p_contrib = mechanism.get_state_contribution()
            if len(p_contrib) < 2:
                scores[pname] = 0.0
                continue
            min_len = min(len(baseline_contrib), len(p_contrib))
            corr = float(np.corrcoef(baseline_contrib[:min_len], p_contrib[:min_len])[0, 1])
            scores[pname] = max(0.0, corr) if not np.isnan(corr) else 0.0
        adv_score = float(np.mean(list(scores.values()))) if scores else 0.0
        return {"per_perturbation": scores, "adversarial_score": adv_score, "passed": adv_score > 0.3}

    def complexity_collapse(self, mechanism: BaseMechanism, data: dict, states: NDArray, removal_rates: list[float] | None = None) -> dict:
        if removal_rates is None:
            removal_rates = [0.5, 0.75, 0.9, 0.95]
        baseline_result = mechanism.compute(data, states)
        baseline_contrib = mechanism.get_state_contribution()
        if len(baseline_contrib) < 2:
            return {"per_rate": {}, "complexity_survival": 0.0, "passed": False}
        scores: dict[float, float] = {}
        data_keys = [k for k in data if isinstance(data[k], np.ndarray) and data[k].dtype.kind == "f" and len(data[k]) == len(states)]
        if not data_keys:
            return {"per_rate": {}, "complexity_survival": 0.0, "passed": False}
        for rate in removal_rates:
            n_remove = max(1, int(len(data_keys) * rate))
            removed = set(np.random.choice(data_keys, n_remove, replace=False))
            collapsed = {k: v for k, v in data.items() if k not in removed}
            mechanism.reset()
            _ = mechanism.compute(collapsed, states)
            c_contrib = mechanism.get_state_contribution()
            if len(c_contrib) < 2:
                scores[rate] = 0.0
                continue
            min_len = min(len(baseline_contrib), len(c_contrib))
            corr = float(np.corrcoef(baseline_contrib[:min_len], c_contrib[:min_len])[0, 1])
            scores[rate] = max(0.0, corr) if not np.isnan(corr) else 0.0
        complexity_survival = float(np.mean(list(scores.values()))) if scores else 0.0
        return {"per_rate": scores, "complexity_survival": complexity_survival, "passed": complexity_survival > 0.3}

    def bootstrap_stability(self, mechanism: BaseMechanism, data: dict, states: NDArray, n_iterations: list[int] | None = None) -> dict:
        if n_iterations is None:
            n_iterations = [100, 500, 1000]
        baseline_result = mechanism.compute(data, states)
        baseline_contrib = mechanism.get_state_contribution()
        if len(baseline_contrib) < 2:
            return {"per_n": {}, "bootstrap_stability": 0.0, "passed": False}
        n = len(states)
        scores: dict[int, float] = {}
        for n_iter in n_iterations:
            corrs: list[float] = []
            for _ in range(min(n_iter, 50)):
                idx = np.random.choice(n, n, replace=True)
                boot_states = states[idx]
                boot_data = {k: v[idx] if isinstance(v, np.ndarray) and len(v) == n else v for k, v in data.items()}
                mechanism.reset()
                _ = mechanism.compute(boot_data, boot_states)
                boot_contrib = mechanism.get_state_contribution()
                if len(boot_contrib) < 2:
                    continue
                min_len = min(len(baseline_contrib), len(boot_contrib))
                corr = float(np.corrcoef(baseline_contrib[:min_len], boot_contrib[:min_len])[0, 1])
                corrs.append(max(0.0, corr) if not np.isnan(corr) else 0.0)
            scores[n_iter] = float(np.mean(corrs)) if corrs else 0.0
        boot_stability = float(np.mean(list(scores.values()))) if scores else 0.0
        return {"per_n": scores, "bootstrap_stability": boot_stability, "passed": boot_stability > 0.5}

    def randomization_test(self, mechanism: BaseMechanism, data: dict, states: NDArray) -> dict:
        baseline_result = mechanism.compute(data, states)
        baseline_contrib = mechanism.get_state_contribution()
        if len(baseline_contrib) < 2:
            return {"randomization_resistance": 0.0, "passed": False}
        scores: dict[str, float] = {}
        for rname, shuffled in [
            ("shuffled_returns", np.random.permutation(data.get("returns", states))),
            ("shuffled_states", np.random.permutation(states)),
        ]:
            shuffle_data = dict(data)
            if rname == "shuffled_returns":
                shuffle_data["returns"] = shuffled
            mechanism.reset()
            _ = mechanism.compute(shuffle_data, states if rname == "shuffled_returns" else shuffled)
            s_contrib = mechanism.get_state_contribution()
            if len(s_contrib) < 2:
                scores[rname] = 0.0
                continue
            min_len = min(len(baseline_contrib), len(s_contrib))
            mi_val = self.mi.mutual_info(baseline_contrib[:min_len], s_contrib[:min_len])
            scores[rname] = mi_val
        rand_resistance = 1.0 - min(1.0, float(np.mean(list(scores.values()))) / (max(list(scores.values())) + 1e-10))
        if np.isnan(rand_resistance):
            rand_resistance = 0.0
        return {"per_shuffle": scores, "randomization_resistance": rand_resistance, "passed": rand_resistance > 0.5}

    def delayed_information_test(self, mechanism: BaseMechanism, data: dict, states: NDArray, delays: list[int] | None = None) -> dict:
        if delays is None:
            delays = [1, 5, 10, 20]
        baseline_result = mechanism.compute(data, states)
        baseline_contrib = mechanism.get_state_contribution()
        if len(baseline_contrib) < 2:
            return {"per_delay": {}, "temporal_sensitivity": 0.0, "passed": False}
        scores: dict[int, float] = {}
        for d in delays:
            if d >= len(baseline_contrib):
                scores[d] = 0.0
                continue
            delayed = baseline_contrib[:-d]
            original = baseline_contrib[d:]
            if len(original) < 2 or len(delayed) < 2:
                scores[d] = 0.0
                continue
            mi_val = self.mi.mutual_info(original, delayed)
            scores[d] = mi_val
        temp_sensitivity = float(np.mean(list(scores.values()))) if scores else 0.0
        return {"per_delay": scores, "temporal_sensitivity": temp_sensitivity, "passed": temp_sensitivity > 0.01}

    def mechanism_isolation(self, mechanism: BaseMechanism, all_mechanisms: dict[str, BaseMechanism], data: dict, states: NDArray, target: NDArray) -> dict:
        from research.mechanism_discovery.coupling_engine import CouplingEngine
        ce = CouplingEngine(mi_estimator=self.mi)
        combined_all = ce.compute_combined_contribution(all_mechanisms, data, states)
        combined_without = ce.compute_combined_without(all_mechanisms, mechanism.name, data, states)
        if len(combined_all) < 2 or len(combined_without) < 2:
            return {"unique_information_contribution": 0.0, "passed": False}
        min_len = min(len(combined_all), len(target))
        mi_all = self.mi.mutual_info(combined_all[:min_len], target[:min_len])
        mi_without = self.mi.mutual_info(combined_without[:min_len], target[:min_len])
        unique_contrib = max(0.0, mi_all - mi_without)
        return {"mi_all": mi_all, "mi_without": mi_without, "unique_information_contribution": unique_contrib, "passed": unique_contrib > 0.01}

    def run_all_attacks(self, mechanism: BaseMechanism, data: dict, states: NDArray, target: NDArray,
                        all_mechanisms: dict[str, BaseMechanism], test_assets: dict[str, dict] | None = None,
                        test_data: dict | None = None, regime_masks: dict[str, NDArray] | None = None) -> dict[str, Any]:
        results: dict[str, Any] = {}
        results["noise_injection"] = self.noise_injection(mechanism, data, states)
        results["adversarial_perturbation"] = self.adversarial_perturbation(mechanism, data, states)
        results["complexity_collapse"] = self.complexity_collapse(mechanism, data, states)
        results["bootstrap_stability"] = self.bootstrap_stability(mechanism, data, states)
        results["randomization_test"] = self.randomization_test(mechanism, data, states)
        results["delayed_information"] = self.delayed_information_test(mechanism, data, states)
        results["mechanism_isolation"] = self.mechanism_isolation(mechanism, all_mechanisms, data, states, target)
        if test_assets:
            results["cross_asset_transfer"] = self.cross_asset_transfer(mechanism, data, test_assets, states)
        if test_data:
            results["cross_time_transfer"] = self.cross_time_transfer(mechanism, data, test_data, states)
        if regime_masks:
            results["regime_transfer"] = self.regime_transfer(mechanism, data, regime_masks, states)
        self.results[mechanism.name] = results
        return results
