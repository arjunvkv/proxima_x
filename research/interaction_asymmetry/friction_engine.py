from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from research.interaction_asymmetry.interaction_validator import InteractionValidator, IAEResult
from research.adaptive_alpha_engine.aae_validator import HORIZONS, _zscore


class FrictionEngine:
    def __init__(self, validator: InteractionValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset

    def run(self) -> IAEResult:
        self.validator.load(self.asset)
        md_z = self.validator.md_z()
        es_z = self.validator.es_z()
        at_z = self.validator.at_z()

        v_md = self.validator.velocity(md_z)
        v_es = self.validator.velocity(es_z)
        v_at = self.validator.velocity(at_z)

        a_md = self.validator.acceleration(md_z)
        a_es = self.validator.acceleration(es_z)
        a_at = self.validator.acceleration(at_z)

        friction = np.abs(v_md - v_es) + np.abs(v_md - v_at) + np.abs(v_es - v_at)
        friction = np.nan_to_num(friction, nan=0.0, posinf=0.0, neginf=0.0)
        friction_z = self.validator.z(friction)

        accel_friction = np.abs(a_md - a_es) + np.abs(a_md - a_at) + np.abs(a_es - a_at)
        accel_friction = np.nan_to_num(accel_friction, nan=0.0, posinf=0.0, neginf=0.0)
        accel_friction_z = self.validator.z(accel_friction)

        benchmark_es = self.validator.benchmark_es_alpha()

        def safe_eval(signal: NDArray[np.float64], horizon_idx: int = 2) -> dict:
            sig = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
            if len(sig) < 10:
                return {"mean": 0.0, "pp": 0.5, "sharpe": 0.0, "std": 0.0, "n": 0}
            return self.validator.eval_alpha(sig, horizon_idx)

        friction_alpha = safe_eval(friction_z, 2)
        accel_friction_alpha = safe_eval(accel_friction_z, 2)

        velocity_alphas: dict[str, dict] = {}
        for vname, vsig in [("memory_density", v_md), ("energy_storage", v_es), ("adaptive_time", v_at)]:
            vsig_clean = np.nan_to_num(vsig, nan=0.0, posinf=0.0, neginf=0.0)
            velocity_alphas[vname] = safe_eval(self.validator.z(vsig_clean), 2)

        print(f"{'Metric':>30s} | {'Mean':>10s} | {'PP':>8s} | {'Sharpe':>8s} | {'Std':>10s} | {'N':>6s}")
        print("-" * 80)
        print(f"{'Benchmark ES (H20)':>30s} | {benchmark_es.get('mean', 0.0):>10.6f} | {benchmark_es.get('pp', 0.5):>8.4f} | {benchmark_es.get('sharpe', 0.0):>8.4f} | {benchmark_es.get('std', 0.0):>10.6f} | {benchmark_es.get('n', 0):>6d}")

        for rname, rdict in [("Friction (velocity-based)", friction_alpha), ("Friction (acceleration-based)", accel_friction_alpha)]:
            print(f"{rname:>30s} | {rdict.get('mean', 0.0):>10.6f} | {rdict.get('pp', 0.5):>8.4f} | {rdict.get('sharpe', 0.0):>8.4f} | {rdict.get('std', 0.0):>10.6f} | {rdict.get('n', 0):>6d}")

        for vname, valpha in velocity_alphas.items():
            print(f"{f'Velocity {vname}':>30s} | {valpha.get('mean', 0.0):>10.6f} | {valpha.get('pp', 0.5):>8.4f} | {valpha.get('sharpe', 0.0):>8.4f} | {valpha.get('std', 0.0):>10.6f} | {valpha.get('n', 0):>6d}")

        b_pp = benchmark_es.get("pp", 0.5)
        b_sharpe = benchmark_es.get("sharpe", 0.0)
        beats_es_friction = friction_alpha["pp"] > b_pp or friction_alpha["sharpe"] > b_sharpe * 1.1
        beats_es_accel = accel_friction_alpha["pp"] > b_pp or accel_friction_alpha["sharpe"] > b_sharpe * 1.1

        print(f"\nFriction beats ES: {beats_es_friction}")
        print(f"Acceleration friction beats ES: {beats_es_accel}")

        candidates = {
            "friction_sharpe": friction_alpha.get("sharpe", 0.0),
            "accel_friction_sharpe": accel_friction_alpha.get("sharpe", 0.0),
        }
        best_metric = max(candidates, key=lambda k: abs(candidates[k])) if any(candidates.values()) else "none"

        metrics: dict[str, Any] = {
            "benchmark_es_alpha": benchmark_es,
            "friction_alpha": friction_alpha,
            "acceleration_friction_alpha": accel_friction_alpha,
            "velocity_alphas": velocity_alphas,
            "beats_es": beats_es_friction or beats_es_accel,
            "best_friction_metric": best_metric,
        }

        return IAEResult(rq_name="RQ3_Temporal_Friction", status="COMPLETE", metrics=metrics)
