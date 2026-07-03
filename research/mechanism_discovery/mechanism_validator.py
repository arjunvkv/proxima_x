from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from research.mechanism_discovery.base import BaseMechanism, MechanismScore
from research.information_discovery.mi_estimator import MIEstimator
from research.information_discovery.sid_sir import SIDCalculator, SIRCalculator
from research.information_discovery.validation_framework import ValidationFramework


class MechanismValidator:
    def __init__(
        self,
        mi_estimator: Optional[MIEstimator] = None,
        sid_calc: Optional[SIDCalculator] = None,
        sir_calc: Optional[SIRCalculator] = None,
    ) -> None:
        self.mi = mi_estimator or MIEstimator(n_bins=20)
        self.sid_calc = sid_calc or SIDCalculator(mi_estimator=self.mi)
        self.sir_calc = sir_calc or SIRCalculator(sid_calculator=self.sid_calc)

    def information_test(
        self,
        mechanism_result: dict[str, Any],
        target: NDArray,
        states: NDArray,
    ) -> dict[str, Any]:
        state_contrib = mechanism_result.get("state_contribution", None)
        if state_contrib is None:
            return {"information_gain": 0.0, "target_mi": 0.0, "state_mi": 0.0, "passed": False}

        min_len = min(len(state_contrib), len(target))
        if min_len < 2:
            return {"information_gain": 0.0, "target_mi": 0.0, "state_mi": 0.0, "passed": False}

        target_mi = self.mi.mutual_info(state_contrib[:min_len], target[:min_len])
        min_states = min(len(state_contrib), len(states))
        state_mi = self.mi.mutual_info(state_contrib[:min_states], states[:min_states]) if min_states >= 2 else 0.0
        information_gain = target_mi + state_mi
        passed = information_gain > 1e-8

        return {
            "information_gain": information_gain,
            "target_mi": target_mi,
            "state_mi": state_mi,
            "passed": bool(passed),
        }

    def sid_test(
        self,
        mechanism_result: dict[str, Any],
        states: NDArray,
        forward_returns: NDArray,
    ) -> dict[str, Any]:
        state_contrib = mechanism_result.get("state_contribution", None)
        if state_contrib is None:
            return {"sid": 0.0, "sid_per_state": {}, "passed": False}

        min_len = min(len(state_contrib), len(states), len(forward_returns))
        if min_len < 2:
            return {"sid": 0.0, "sid_per_state": {}, "passed": False}

        state_contrib_trim = state_contrib[:min_len]
        states_trim = states[:min_len]
        forward_trim = forward_returns[:min_len]

        state_labels: NDArray[np.int32]
        if state_contrib_trim.dtype.kind in ("i", "u"):
            state_labels = state_contrib_trim.astype(np.int32)
        else:
            n_bins = min(20, len(np.unique(state_contrib_trim)))
            if n_bins < 2:
                return {"sid": 0.0, "sid_per_state": {}, "passed": False}
            _, bin_edges = self.mi._discretize(state_contrib_trim, n_bins)
            state_labels = np.digitize(state_contrib_trim, bin_edges[:-1]).astype(np.int32)

        sid_result = self.sid_calc.compute_sid(state_labels, forward_trim)
        avg_sid = sid_result.get("avg_sid", 0.0)
        passed = bool(avg_sid > 1e-8)

        return {
            "sid": avg_sid,
            "sid_per_state": sid_result.get("sid_per_state", {}),
            "passed": passed,
        }

    def sir_test(
        self,
        mechanism_result: dict[str, Any],
        states: NDArray,
        forward_returns: NDArray,
        compressed_dim: int,
    ) -> dict[str, Any]:
        state_contrib = mechanism_result.get("state_contribution", None)
        if state_contrib is None:
            return {"sir": 0.0, "passed": False}

        min_len = min(len(state_contrib), len(states), len(forward_returns))
        if min_len < 2:
            return {"sir": 0.0, "passed": False}

        state_contrib_trim = state_contrib[:min_len]
        forward_trim = forward_returns[:min_len]

        state_labels: NDArray[np.int32]
        if state_contrib_trim.dtype.kind in ("i", "u"):
            state_labels = state_contrib_trim.astype(np.int32)
        else:
            n_bins = min(20, len(np.unique(state_contrib_trim)))
            if n_bins < 2:
                return {"sir": 0.0, "passed": False}
            _, bin_edges = self.mi._discretize(state_contrib_trim, n_bins)
            state_labels = np.digitize(state_contrib_trim, bin_edges[:-1]).astype(np.int32)

        sir = self.sir_calc.compute_sir(state_labels, forward_trim, compressed_dim)
        passed = bool(sir > 1e-8)

        return {"sir": float(sir), "passed": passed}

    def persistence_test(
        self,
        mechanism_result: dict[str, Any],
        n_splits: int = 5,
    ) -> dict[str, Any]:
        state_contrib = mechanism_result.get("state_contribution", None)
        if state_contrib is None or len(state_contrib) < n_splits:
            return {"persistence_score": 0.0, "per_window_metrics": [], "passed": False}

        n = len(state_contrib)
        window_size = n // n_splits
        per_window_metrics: list[float] = []

        for i in range(n_splits):
            start = i * window_size
            end = start + window_size if i < n_splits - 1 else n
            window_data = state_contrib[start:end]
            if len(window_data) < 2:
                metric = 0.0
            else:
                metric = float(np.mean(np.abs(window_data)))
            per_window_metrics.append(metric)

        metrics_arr = np.array(per_window_metrics, dtype=np.float64)
        mean_val = float(np.mean(metrics_arr))
        std_val = float(np.std(metrics_arr))
        persistence_score = 1.0 / (1.0 + std_val) if mean_val > 1e-10 else 0.0
        passed = bool(persistence_score > 0.5)

        return {
            "persistence_score": persistence_score,
            "per_window_metrics": per_window_metrics,
            "passed": passed,
        }

    def cross_asset_test(
        self,
        mechanism: BaseMechanism,
        assets_data: dict[str, dict[str, NDArray]],
        states_dict: dict[str, NDArray],
    ) -> dict[str, Any]:
        per_asset_results: dict[str, dict[str, Any]] = {}
        contributions: list[NDArray] = []

        for asset_id, data in assets_data.items():
            states_asset = states_dict.get(asset_id, None)
            result = mechanism.compute(data, states_asset)
            contrib = result.get("state_contribution", None)
            if contrib is not None and len(contrib) > 0:
                contributions.append(contrib)
            per_asset_results[asset_id] = {
                "mean_contribution": float(np.mean(np.abs(contrib))) if contrib is not None and len(contrib) > 0 else 0.0,
                "std_contribution": float(np.std(contrib)) if contrib is not None and len(contrib) > 1 else 0.0,
            }

        if len(contributions) < 2:
            return {
                "per_asset_results": per_asset_results,
                "consistency": 0.0,
                "cross_asset_score": 0.0,
                "passed": False,
            }

        min_len = min(len(c) for c in contributions)
        aligned = np.stack([c[:min_len] for c in contributions], axis=0)
        corr_matrix = np.corrcoef(aligned)
        triu_idx = np.triu_indices_from(corr_matrix, k=1)
        correlations = corr_matrix[triu_idx]
        consistency = float(np.nanmean(correlations)) if len(correlations) > 0 else 0.0
        consistency = max(-1.0, min(1.0, consistency)) if not np.isnan(consistency) else 0.0
        cross_asset_score = max(0.0, consistency)
        passed = bool(cross_asset_score > 0.3)

        return {
            "per_asset_results": per_asset_results,
            "consistency": consistency,
            "cross_asset_score": cross_asset_score,
            "passed": passed,
        }

    def cross_regime_test(
        self,
        mechanism: BaseMechanism,
        data: dict[str, NDArray],
        regime_masks: dict[str, NDArray],
        states: NDArray,
    ) -> dict[str, Any]:
        per_regime_results: dict[str, dict[str, Any]] = {}
        regime_metrics: list[float] = []

        for regime_label, mask in regime_masks.items():
            mask_bool = mask.astype(bool)
            if mask_bool.sum() < 10:
                per_regime_results[regime_label] = {"mean_contribution": 0.0, "std_contribution": 0.0, "n_samples": int(mask_bool.sum())}
                continue

            regime_data: dict[str, NDArray] = {}
            for key, arr in data.items():
                regime_data[key] = arr[mask_bool]
            regime_states = states[mask_bool] if len(states) == len(mask_bool) else None

            result = mechanism.compute(regime_data, regime_states)
            contrib = result.get("state_contribution", None)
            if contrib is not None and len(contrib) > 0:
                mean_c = float(np.mean(np.abs(contrib)))
                std_c = float(np.std(contrib))
                regime_metrics.append(mean_c)
            else:
                mean_c = 0.0
                std_c = 0.0

            per_regime_results[regime_label] = {
                "mean_contribution": mean_c,
                "std_contribution": std_c,
                "n_samples": int(mask_bool.sum()),
            }

        if len(regime_metrics) < 2:
            return {
                "per_regime_results": per_regime_results,
                "invariance": 0.0,
                "score": 0.0,
                "passed": False,
            }

        metrics_arr = np.array(regime_metrics, dtype=np.float64)
        std_across = float(np.std(metrics_arr))
        mean_across = float(np.mean(metrics_arr))
        invariance = 1.0 / (1.0 + std_across) if mean_across > 1e-10 else 0.0
        score = float(np.mean([1.0 / (1.0 + abs(m - mean_across)) for m in regime_metrics]))
        passed = bool(invariance > 0.5)

        return {
            "per_regime_results": per_regime_results,
            "invariance": invariance,
            "score": score,
            "passed": passed,
        }

    def oos_test(
        self,
        mechanism: BaseMechanism,
        data_is: dict[str, NDArray],
        data_oos: dict[str, NDArray],
        states_is: NDArray,
        states_oos: NDArray,
    ) -> dict[str, Any]:
        is_result = mechanism.compute(data_is, states_is)
        oos_result = mechanism.compute(data_oos, states_oos)

        is_output = is_result.get("state_contribution", np.array([], dtype=np.float64))
        oos_output = oos_result.get("state_contribution", np.array([], dtype=np.float64))

        if len(is_output) < 2 or len(oos_output) < 2:
            return {"is_output": is_output, "oos_output": oos_output, "correlation": 0.0, "score": 0.0, "passed": False}

        is_mean = float(np.mean(np.abs(is_output)))
        oos_mean = float(np.mean(np.abs(oos_output)))

        min_len = min(len(is_output), len(oos_output))
        if min_len < 2:
            return {"is_output": is_output, "oos_output": oos_output, "correlation": 0.0, "score": 0.0, "passed": False}

        corr_val = float(np.corrcoef(is_output[:min_len], oos_output[:min_len])[0, 1])
        if np.isnan(corr_val):
            corr_val = 0.0
        corr_val = max(-1.0, min(1.0, corr_val))

        is_std = float(np.std(is_output)) if len(is_output) > 1 else 0.0
        oos_std = float(np.std(oos_output)) if len(oos_output) > 1 else 0.0
        std_ratio = min(is_std / (oos_std + 1e-10), 1.0 / (is_std / (oos_std + 1e-10) + 1e-10)) if is_std > 1e-10 and oos_std > 1e-10 else 0.0

        score = max(0.0, corr_val) * (0.5 + 0.5 * std_ratio)
        passed = bool(corr_val > 0.3 and score > 0.3)

        return {
            "is_output": is_output,
            "oos_output": oos_output,
            "correlation": corr_val,
            "is_mean": is_mean,
            "oos_mean": oos_mean,
            "score": score,
            "passed": passed,
        }

    def validate_all(
        self,
        mechanism: BaseMechanism,
        data: dict[str, NDArray],
        states: NDArray,
        forward_returns: NDArray,
        compressed_dim: int = 5,
    ) -> dict[str, Any]:
        mechanism_result = mechanism.compute(data, states)

        info_result = self.information_test(mechanism_result, forward_returns, states)
        sid_result = self.sid_test(mechanism_result, states, forward_returns)
        sir_result = self.sir_test(mechanism_result, states, forward_returns, compressed_dim)
        persistence_result = self.persistence_test(mechanism_result)

        results: dict[str, Any] = {
            "mechanism_name": mechanism.name,
            "mechanism_category": mechanism.category,
            "information_test": info_result,
            "sid_test": sid_result,
            "sir_test": sir_result,
            "persistence_test": persistence_result,
        }

        all_passed = all(
            r.get("passed", False)
            for r in [info_result, sid_result, sir_result, persistence_result]
        )

        results["overall"] = {
            "all_passed": all_passed,
            "n_passed": int(sum(
                r.get("passed", False)
                for r in [info_result, sid_result, sir_result, persistence_result]
            )),
            "n_total": 4,
        }

        return results

    def validate_mechanism_score(self, score: MechanismScore) -> dict[str, bool]:
        info_pass = score.information_gain > 1e-8
        sid_pass = score.sid > 1e-8
        sir_pass = score.sir > 1e-8
        persistence_pass = score.persistence > 0.5
        cross_asset_pass = score.cross_asset_score > 0.3
        cross_regime_pass = score.cross_regime_score > 0.3
        oos_pass = score.oos_score > 0.3
        all_pass = all([info_pass, sid_pass, sir_pass, persistence_pass, cross_asset_pass, cross_regime_pass, oos_pass])

        return {
            "information_gain": info_pass,
            "sid": sid_pass,
            "sir": sir_pass,
            "persistence": persistence_pass,
            "cross_asset": cross_asset_pass,
            "cross_regime": cross_regime_pass,
            "oos": oos_pass,
            "all": all_pass,
        }
