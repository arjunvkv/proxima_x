"""
SRFM — State Rehydration Failure Model

Detects partial subsystem desynchronization: some subsystems alive,
others in latent sleep state.

Reads wave12 cycle log (JSONL) and computes per-subsystem activation
metrics, sync index, entropy, mismatch, and dominant sleep subsystem.
"""

from __future__ import annotations

import json
import math
import os
from collections import OrderedDict
from typing import Any


class StateRehydrationFailure:
    """Analyze subsystem activation to detect desync / sleep drift."""

    SUBSYSTEMS = ("signal", "decision", "execution", "governance")

    def __init__(self, log_path: str = "state/wave12_cycle_log.jsonl") -> None:
        self.log_path = os.path.abspath(log_path)

    # ------------------------------------------------------------------
    # Per-subsystem activation logic
    # ------------------------------------------------------------------

    @staticmethod
    def _signal_activation(record: dict[str, Any]) -> float:
        """S: total_signals > 0 → 1.0, else 0.0."""
        return 1.0 if record.get("total_signals", 0) > 0 else 0.0

    @staticmethod
    def _decision_activation(record: dict[str, Any]) -> float:
        """D: best_signal exists → 1.0, else 0.0.

        Infers best_signal presence from the pipeline_trace.execution
        message: if it contains "no best_signal", treat as absent.
        """
        try:
            exc: str = record.get("pipeline_trace", {}).get("execution", "")
        except AttributeError:
            return 0.0
        # "no best_signal" → absent; anything else implies it exists
        return 1.0 if "no best_signal" not in exc else 0.0

    @staticmethod
    def _execution_activation(record: dict[str, Any]) -> float:
        """E:
        - decision not in ("HOLD", "SKIP") → 1.0
        - segl_state == "ARMED" and signals present → 0.3 (semi-active)
        - else 0.0
        """
        decision = record.get("decision", "HOLD")
        if decision not in ("HOLD", "SKIP"):
            return 1.0

        segl = record.get("segl_state", "")
        total_sig = record.get("total_signals", 0)
        if segl == "ARMED" and total_sig > 0:
            return 0.3

        return 0.0

    @staticmethod
    def _governance_activation(record: dict[str, Any]) -> float:
        """G:
        - segl_state == "ARMED" and not circuit_breaker → 1.0
        - segl_state == "ARMED" and circuit_breaker → 0.5
        - segl_state == "OBSERVE"                      → 0.2
        - else 0.0
        """
        segl = record.get("segl_state", "")
        cb = record.get("circuit_breaker", False)

        if segl == "ARMED" and not cb:
            return 1.0
        if segl == "ARMED" and cb:
            return 0.5
        if segl == "OBSERVE":
            return 0.2

        return 0.0

    # ------------------------------------------------------------------
    # Aggregate helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _activation_vector(record: dict[str, Any]) -> list[float]:
        """Return [S, D, E, G] for a single cycle record."""
        return [
            StateRehydrationFailure._signal_activation(record),
            StateRehydrationFailure._decision_activation(record),
            StateRehydrationFailure._execution_activation(record),
            StateRehydrationFailure._governance_activation(record),
        ]

    @staticmethod
    def _shannon_entropy(values: list[float]) -> float:
        """Shannon entropy of a probability-like distribution.

        0 if all values are equal (uniform activation → no information).
        """
        total = sum(values)
        if total == 0:
            return 0.0
        probs = [v / total for v in values]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        return entropy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, n_recent_cycles: int = 500) -> dict[str, Any]:
        """Compute SRFM diagnostics over the most recent N cycles.

        Returns the dict structure specified in the SRFM interface.
        """
        result: dict[str, Any] = {
            "activation_vector_trajectory": OrderedDict(),
            "subsystem_sync_index": 0.0,
            "activation_entropy": 0.0,
            "mismatch_index": 0.0,
            "desync_cycles": 0,
            "dominant_sleep_subsystem": "unknown",
            "by_subsystem": {
                s: {"mean_activation": 0.0, "active_ratio": 0.0}
                for s in self.SUBSYSTEMS
            },
        }

        try:
            # ---- read log -------------------------------------------------
            if not os.path.isfile(self.log_path):
                return result

            records: list[dict[str, Any]] = []
            with open(self.log_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))

            if not records:
                return result

            # keep most recent N
            tail = records[-n_recent_cycles:]

            # ---- compute per-cycle vectors --------------------------------
            vectors: list[list[float]] = []
            for rec in tail:
                vec = self._activation_vector(rec)
                vectors.append(vec)
                cycle_label = f"cycle_{rec.get('cycle', 0)}"
                result["activation_vector_trajectory"][cycle_label] = vec

            if not vectors:
                return result

            n_cycles = len(vectors)

            # ---- by-subsystem stats ---------------------------------------
            subs_activations: dict[str, list[float]] = {s: [] for s in self.SUBSYSTEMS}
            for vec in vectors:
                for idx, subs in enumerate(self.SUBSYSTEMS):
                    subs_activations[subs].append(vec[idx])

            for subs in self.SUBSYSTEMS:
                acts = subs_activations[subs]
                result["by_subsystem"][subs]["mean_activation"] = (
                    sum(acts) / len(acts) if acts else 0.0
                )
                # active_ratio = fraction of cycles where activation > 0
                active_count = sum(1 for a in acts if a > 0)
                result["by_subsystem"][subs]["active_ratio"] = (
                    active_count / n_cycles if n_cycles else 0.0
                )

            # ---- subsystem sync index ------------------------------------
            # sync = 1.0 - std of the *mean* activation across subsystems
            means = [
                result["by_subsystem"][s]["mean_activation"]
                for s in self.SUBSYSTEMS
            ]
            mean_of_means = sum(means) / len(means) if means else 0.0
            variance = (
                sum((m - mean_of_means) ** 2 for m in means) / len(means)
                if means
                else 0.0
            )
            std = math.sqrt(variance)
            result["subsystem_sync_index"] = round(max(0.0, 1.0 - std), 4)

            # ---- activation entropy (over mean activations) ---------------
            result["activation_entropy"] = round(
                self._shannon_entropy(means), 4
            )

            # ---- mismatch index -------------------------------------------
            result["mismatch_index"] = round(max(means) - min(means), 4) if means else 0.0

            # ---- desync cycles --------------------------------------------
            desync_count = 0
            for vec in vectors:
                mx = max(vec)
                mn = min(vec)
                if (mx - mn) > 0.5:
                    desync_count += 1
            result["desync_cycles"] = desync_count

            # ---- dominant sleep subsystem ---------------------------------
            lowest_mean = min(means)
            for subs in self.SUBSYSTEMS:
                if (
                    abs(
                        result["by_subsystem"][subs]["mean_activation"]
                        - lowest_mean
                    )
                    < 1e-9
                ):
                    result["dominant_sleep_subsystem"] = subs
                    break

        except Exception:
            # On any error return the skeleton with safe defaults
            pass

        return result
