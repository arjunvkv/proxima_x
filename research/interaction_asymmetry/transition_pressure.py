from __future__ import annotations

from typing import Any

import numpy as np

from research.interaction_asymmetry.interaction_validator import InteractionValidator, IAEResult
from research.adaptive_alpha_engine.aae_validator import HORIZONS, TARGET_ASSETS, TIME_WINDOWS, _zscore, _future_returns


class TransitionPressure:
    def __init__(self, validator: InteractionValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset

    def run(self) -> IAEResult:
        self.validator.load(self.asset)
        md_z = self.validator.md_z()
        es_z = self.validator.es_z()
        at_z = self.validator.at_z()

        state_mutation_rate = self.validator.signals["state_mutation_rate"]
        states = self.validator.signals["states"]

        pressure = self.validator.interaction_pressure(md_z, es_z, at_z, window=20)

        md_es_div = np.abs(self.validator.divergence(md_z, es_z, "difference"))
        md_at_div = np.abs(self.validator.divergence(md_z, at_z, "difference"))
        es_at_div = np.abs(self.validator.divergence(es_z, at_z, "difference"))
        avg_pairwise_div = (md_es_div + md_at_div + es_at_div) / 3.0

        tension = self.validator.tension_index(md_z, es_z, at_z, window=20)

        mutation_events = [i for i in range(1, len(state_mutation_rate)) if state_mutation_rate[i] > 0 or states[i] != states[i - 1]]

        pre_event_pressures = []
        pre_event_divergences = []
        pre_event_tensions = []

        for ev in mutation_events:
            start = max(0, ev - 20)
            end = ev
            if end - start >= 5:
                pre_event_pressures.append(np.nanmean(pressure[start:end]))
                pre_event_divergences.append(np.nanmean(avg_pairwise_div[start:end]))
                pre_event_tensions.append(np.nanmean(tension[start:end]))

        baseline_pressures = []
        n_valid = max(0, len(pressure) - 20)
        if n_valid > 0:
            n_samples = min(500, n_valid)
            rng = np.random.default_rng(42)
            for _ in range(n_samples):
                idx = rng.integers(0, n_valid)
                baseline_pressures.append(np.nanmean(pressure[idx:idx + 20]))

        pre_event_pressure_mean = float(np.nanmean(pre_event_pressures)) if pre_event_pressures else 0.0
        pre_event_divergence_mean = float(np.nanmean(pre_event_divergences)) if pre_event_divergences else 0.0
        pre_event_tension_mean = float(np.nanmean(pre_event_tensions)) if pre_event_tensions else 0.0
        baseline_pressure_mean = float(np.nanmean(baseline_pressures)) if baseline_pressures else 0.0
        pressure_ratio = pre_event_pressure_mean / max(baseline_pressure_mean, 1e-12)

        pressure_threshold = np.nanpercentile(pressure, 90) if len(pressure) > 10 else 0.0
        pressure_signal = np.where(pressure > pressure_threshold, 1.0, 0.0)
        pressure_alpha = self.validator.eval_alpha(pressure_signal, 2)

        div_threshold = np.nanpercentile(avg_pairwise_div, 90) if len(avg_pairwise_div) > 10 else 0.0
        div_signal = np.where(avg_pairwise_div > div_threshold, 1.0, 0.0)
        divergence_pressure_alpha = self.validator.eval_alpha(div_signal, 2)

        print(f"=== TransitionPressure (RQ7) - {self.asset} ===")
        print(f"Transition events detected: {len(mutation_events)}")
        print(f"Valid pre-event windows: {len(pre_event_pressures)}")
        print(f"Pre-event pressure mean: {pre_event_pressure_mean:.6f}")
        print(f"Pre-event divergence mean: {pre_event_divergence_mean:.6f}")
        print(f"Pre-event tension mean: {pre_event_tension_mean:.6f}")
        print(f"Baseline pressure mean: {baseline_pressure_mean:.6f}")
        print(f"Pressure ratio (pre-event / baseline): {pressure_ratio:.4f}")
        print(f"Pressure alpha (pp): {pressure_alpha.get('pp', 0.5):.4f}")
        print(f"Pressure alpha (sharpe): {pressure_alpha.get('sharpe', 0.0):.4f}")
        print(f"Divergence pressure alpha (pp): {divergence_pressure_alpha.get('pp', 0.5):.4f}")
        print(f"Divergence pressure alpha (sharpe): {divergence_pressure_alpha.get('sharpe', 0.0):.4f}")

        metrics: dict[str, Any] = {
            "pre_event_pressure_mean": pre_event_pressure_mean,
            "baseline_pressure_mean": baseline_pressure_mean,
            "pressure_ratio": pressure_ratio,
            "pressure_alpha": pressure_alpha,
            "divergence_pressure_alpha": divergence_pressure_alpha,
            "n_transition_events": len(mutation_events),
        }

        return IAEResult(rq_name="RQ7_Transition_Pressure", status="COMPLETE", metrics=metrics)
