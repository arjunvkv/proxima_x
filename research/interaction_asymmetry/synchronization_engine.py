from __future__ import annotations

from typing import Any

import numpy as np

from research.interaction_asymmetry.interaction_validator import InteractionValidator, IAEResult, SYNC_STATES
from research.adaptive_alpha_engine.aae_validator import HORIZONS, _zscore


class SynchronizationEngine:
    def __init__(self, validator: InteractionValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset

    def run(self) -> IAEResult:
        self.validator.load(self.asset)
        md_z = self.validator.md_z()
        es_z = self.validator.es_z()
        at_z = self.validator.at_z()
        fut_ret = self.validator.fut_ret

        states = self.validator.classify_synchronization(md_z, es_z, at_z)
        n_total = len(states)

        state_distribution: dict[str, int] = {}
        forward_metrics: dict[str, dict[str, float]] = {}

        for sid, sname in enumerate(SYNC_STATES):
            mask = states == sid
            count = int(np.sum(mask))
            state_distribution[sname] = count
            pct = count / max(n_total, 1) * 100.0

            if count < 5:
                forward_metrics[sname] = {"mean": 0.0, "std": 0.0, "sharpe": 0.0, "pp": 0.5, "n": 0}
                print(f"{sname:>30s} | count={count:>8d} | {pct:>6.2f}% | (insufficient data)")
                continue

            h20_ret = fut_ret[mask, 2]
            h20_ret = h20_ret[~np.isnan(h20_ret)]
            if len(h20_ret) < 5:
                forward_metrics[sname] = {"mean": 0.0, "std": 0.0, "sharpe": 0.0, "pp": 0.5, "n": 0}
                print(f"{sname:>30s} | count={count:>8d} | {pct:>6.2f}% | (insufficient returns)")
                continue

            m = float(np.mean(h20_ret))
            s = float(np.std(h20_ret))
            pp = float(np.mean(h20_ret > 0))
            sh = m / max(s, 1e-12)

            forward_metrics[sname] = {"mean": m, "std": s, "sharpe": sh, "pp": pp, "n": len(h20_ret)}
            print(f"{sname:>30s} | count={count:>8d} | {pct:>6.2f}% | mean={m:>10.6f} | std={s:>10.6f} | sharpe={sh:>8.4f} | pp={pp:>8.4f} | n={len(h20_ret):>6d}")

        strongest_state = ""
        weakest_state = ""
        best_abs_sh = -1.0
        worst_abs_sh = 1e10
        for sname in SYNC_STATES:
            sh = forward_metrics[sname].get("sharpe", 0.0)
            ash = abs(sh)
            if forward_metrics[sname].get("n", 0) >= 5:
                if ash > best_abs_sh:
                    best_abs_sh = ash
                    strongest_state = sname
                if ash < worst_abs_sh:
                    worst_abs_sh = ash
                    weakest_state = sname

        print(f"\nStrongest state: {strongest_state} (abs sharpe={best_abs_sh:.4f})")
        print(f"Weakest state: {weakest_state} (abs sharpe={worst_abs_sh:.4f})")

        n_states = len(SYNC_STATES)
        transition_matrix = [[0.0] * n_states for _ in range(n_states)]
        for i in range(n_total - 1):
            cur = int(states[i])
            nxt = int(states[i + 1])
            transition_matrix[cur][nxt] += 1.0

        for i in range(n_states):
            row_total = sum(transition_matrix[i])
            if row_total > 0:
                for j in range(n_states):
                    transition_matrix[i][j] /= row_total

        print("\nState Transition Matrix:")
        print(" " * 28 + "From/To", end="")
        for sname in SYNC_STATES:
            print(f"{sname:>20s}", end="")
        print()
        for i, sname in enumerate(SYNC_STATES):
            row_str = f"{sname:>30s}"
            for j in range(n_states):
                row_str += f"{transition_matrix[i][j]:>20.4f}"
            print(row_str)

        metrics: dict[str, Any] = {
            "state_distribution": state_distribution,
            "forward_metrics": forward_metrics,
            "strongest_state": strongest_state,
            "weakest_state": weakest_state,
            "transition_matrix": transition_matrix,
        }

        return IAEResult(rq_name="RQ2_Synchronization_States", status="COMPLETE", metrics=metrics)
