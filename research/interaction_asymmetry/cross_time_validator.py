from __future__ import annotations

from typing import Any

import numpy as np

from research.interaction_asymmetry.interaction_validator import InteractionValidator, IAEResult
from research.adaptive_alpha_engine.aae_validator import HORIZONS, TARGET_ASSETS, TIME_WINDOWS, _zscore, _future_returns


class CrossTimeValidator:
    def __init__(self, validator: InteractionValidator):
        self.validator = validator

    def run(self) -> IAEResult:
        per_window: dict[str, Any] = {}
        divergence_signals: dict[str, np.ndarray] = {}

        for start, end, label in TIME_WINDOWS:
            data = self.validator.aae.load_data_window("EURJPY", start, end)
            sig = self.validator.aae.compute_signals(data)
            price = sig["price"]
            md = np.asarray(sig["memory_density"], dtype=np.float64)
            es = np.asarray(sig["energy_storage"], dtype=np.float64)
            at = np.asarray(sig["adaptive_time"], dtype=np.float64)

            if len(price) < 30:
                print(f"Window {label}: insufficient data ({len(price)} bars), skipping")
                continue

            md_z = _zscore(md.copy())
            es_z = _zscore(es.copy())
            at_z = _zscore(at.copy())

            divergence_signal = self.validator.divergence(md_z, es_z, "difference")
            sync = self.validator.classify_synchronization(md_z, es_z, at_z)
            leader = self.validator.detect_leader(md_z, es_z, at_z)
            tension = self.validator.tension_index(md_z, es_z, at_z, window=20)

            horizons_arr = np.array(HORIZONS, dtype=np.int32)
            fut_ret = _future_returns(price, horizons_arr)

            es_alpha = self.validator.aae.eval_alpha(es_z, fut_ret, 2)

            interaction_score = np.abs(divergence_signal)
            interaction_alpha = self.validator.aae.eval_alpha(interaction_score, fut_ret, 2)

            per_window[label] = {
                "es_alpha": es_alpha,
                "interaction_alpha": interaction_alpha,
                "n_bars": len(price),
            }

            divergence_signals[label] = divergence_signal

        n_windows = len(per_window)
        variable_survival_rate = 0.0
        interaction_survival_rate = 0.0

        if n_windows > 0:
            es_ok = sum(1 for w in per_window.values() if w["es_alpha"].get("pp", 0.5) > 0.55)
            int_ok = sum(1 for w in per_window.values() if w["interaction_alpha"].get("pp", 0.5) > 0.55)
            variable_survival_rate = es_ok / n_windows
            interaction_survival_rate = int_ok / n_windows

        interaction_survives_better = interaction_survival_rate > variable_survival_rate

        labels = [w[2] for w in TIME_WINDOWS]
        corr_vals = []
        for i in range(len(labels) - 1):
            l1, l2 = labels[i], labels[i + 1]
            if l1 in divergence_signals and l2 in divergence_signals:
                s1 = divergence_signals[l1]
                s2 = divergence_signals[l2]
                min_len = min(len(s1), len(s2))
                if min_len > 5:
                    c = np.corrcoef(s1[:min_len], s2[:min_len])[0, 1]
                    if not np.isnan(c):
                        corr_vals.append(c)

        interaction_consistency = float(np.mean(corr_vals)) if corr_vals else 0.0

        print("=== CrossTimeValidator (RQ9) ===")
        header = f"{'Window':>12s} | {'ES_PP':>8s} | {'Int_PP':>8s} | {'N_Bars':>8s}"
        print(header)
        print("-" * len(header))
        for label, w in per_window.items():
            print(f"{label:>12s} | {w['es_alpha'].get('pp', 0.5):>8.4f} | {w['interaction_alpha'].get('pp', 0.5):>8.4f} | {w['n_bars']:>8d}")
        print(f"\nVariable survival rate (ES pp > 0.55): {variable_survival_rate:.4f}")
        print(f"Interaction survival rate (Int pp > 0.55): {interaction_survival_rate:.4f}")
        print(f"Interaction survives better? {interaction_survives_better}")
        print(f"Interaction consistency (adjacent window divergence corr): {interaction_consistency:.4f}")

        metrics: dict[str, Any] = {
            "per_window": per_window,
            "variable_survival_rate": variable_survival_rate,
            "interaction_survival_rate": interaction_survival_rate,
            "interaction_survives_better": interaction_survives_better,
            "interaction_consistency": interaction_consistency,
        }

        return IAEResult(rq_name="RQ9_Cross_Time_Survival", status="COMPLETE", metrics=metrics)
