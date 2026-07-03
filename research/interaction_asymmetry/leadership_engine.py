from __future__ import annotations

from typing import Any

import numpy as np

from research.interaction_asymmetry.interaction_validator import InteractionValidator, IAEResult
from research.adaptive_alpha_engine.aae_validator import HORIZONS, TARGET_ASSETS, TIME_WINDOWS, _zscore, _future_returns


class LeadershipEngine:
    def __init__(self, validator: InteractionValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset

    def run(self) -> IAEResult:
        self.validator.load(self.asset)
        md_z = self.validator.md_z()
        es_z = self.validator.es_z()
        at_z = self.validator.at_z()

        leader = self.validator.detect_leader(md_z, es_z, at_z)
        n = len(leader)

        durations = []
        current_leader = int(leader[0])
        current_len = 1
        for i in range(1, n):
            if int(leader[i]) == current_leader:
                current_len += 1
            else:
                durations.append((current_leader, current_len))
                current_leader = int(leader[i])
                current_len = 1
        durations.append((current_leader, current_len))

        leader_runs: dict[int, list[int]] = {0: [], 1: [], 2: []}
        for ldr, dur in durations:
            leader_runs[ldr].append(dur)

        leader_distribution = {str(k): len(v) for k, v in leader_runs.items()}
        avg_duration_per_leader = {str(k): float(np.mean(v)) if v else 0.0 for k, v in leader_runs.items()}

        transition_matrix = np.zeros((3, 3), dtype=np.int64)
        for i in range(1, n):
            prev = int(leader[i - 1])
            curr = int(leader[i])
            if prev != curr:
                transition_matrix[prev, curr] += 1
        n_transitions = int(np.sum(transition_matrix))

        state_mutation = self.validator.signals["state_mutation_rate"]
        mutation_bars = np.where(state_mutation > 0)[0]
        if len(mutation_bars) > 0:
            pre_rot = 0
            for mb in mutation_bars:
                start = max(0, int(mb) - 5)
                wins = leader[start:int(mb)]
                if len(wins) >= 2 and np.any(wins[1:] != wins[:-1]):
                    pre_rot += 1
            pre_mutation_rate = pre_rot / len(mutation_bars)
        else:
            pre_mutation_rate = 0.0

        leader_change = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if leader[i] != leader[i - 1]:
                leader_change[i] = 1.0

        if np.sum(leader_change) >= 5:
            lc_alpha = self.validator.eval_alpha(leader_change, 2)
        else:
            lc_alpha = {"mean": 0.0, "pp": 0.5, "std": 0.0, "sharpe": 0.0, "n": 0}

        leader_forward_metrics: dict[str, Any] = {}
        for k in range(3):
            mask = (leader == k).astype(np.float64)
            if np.sum(mask) >= 5:
                leader_forward_metrics[f"leader_{k}"] = self.validator.eval_alpha(mask, 2)
            else:
                leader_forward_metrics[f"leader_{k}"] = {"mean": 0.0, "pp": 0.5, "std": 0.0, "sharpe": 0.0, "n": 0}

        total_bars = sum(len(v) for v in leader_runs.values())
        if total_bars > 0:
            probs = np.array([len(leader_runs[k]) / total_bars for k in range(3)], dtype=np.float64)
            probs = probs[probs > 0]
            if len(probs) > 0:
                entropy = float(-np.sum(probs * np.log(probs)))
                max_ent = np.log(3)
                concentration = 1.0 - (entropy / max_ent) if max_ent > 0 else 0.0
            else:
                concentration = 0.0
        else:
            concentration = 0.0

        print("=== Leadership Engine (RQ4) ===")
        print(f"\nLeader Distribution: {leader_distribution}")
        print(f"Avg Duration Per Leader: {avg_duration_per_leader}")
        print("Transition Matrix (raw counts):")
        for i in range(3):
            print(f"  From {i}: {transition_matrix[i].tolist()}")
        print(f"Total Transitions: {n_transitions}")
        print(f"Pre-Mutation Leadership Rotation Rate: {pre_mutation_rate:.4f}")
        print(f"Leader Change Alpha: mean={lc_alpha.get('mean', 0.0):.6f} pp={lc_alpha.get('pp', 0.5):.4f} sharpe={lc_alpha.get('sharpe', 0.0):.4f}")
        print(f"Leadership Concentration: {concentration:.4f}")
        for k in range(3):
            m = leader_forward_metrics[f"leader_{k}"]
            print(f"Leader {k} Forward Metrics: mean={m.get('mean', 0.0):.6f} pp={m.get('pp', 0.5):.4f} sharpe={m.get('sharpe', 0.0):.4f} n={m.get('n', 0)}")

        metrics: dict[str, Any] = {
            "leader_distribution": leader_distribution,
            "avg_duration_per_leader": avg_duration_per_leader,
            "transition_matrix": transition_matrix.tolist(),
            "pre_mutation_rotation_rate": pre_mutation_rate,
            "leader_change_alpha": lc_alpha,
            "leader_forward_metrics": leader_forward_metrics,
            "leadership_concentration": concentration,
        }

        return IAEResult(rq_name="RQ4_Leadership_Rotation", status="COMPLETE", metrics=metrics)
