"""GDE — Governance Decontamination Engine.

Compute memory contamination from historical CB events, apply decay
correction, and reduce false-positive safety triggers.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from typing import Any


class GovernanceDecontamination:
    """Compute memory contamination from historical CB events.

    Parameters
    ----------
    log_path : str
        Path to the JSONL cycle log (default
        ``"state/wave12_cycle_log.jsonl"``).
    decay_rate : float
        Exponential decay rate applied to contamination over time
        (default 0.05).
    """

    def __init__(
        self, log_path: str = "state/wave12_cycle_log.jsonl", decay_rate: float = 0.05
    ) -> None:
        self.log_path = log_path
        self.decay_rate = decay_rate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, n_recent_cycles: int = 500) -> dict[str, Any]:
        """Run the full GDE analysis.

        Parameters
        ----------
        n_recent_cycles : int
            Number of most-recent cycles to analyse (default 500).

        Returns
        -------
        dict
            See class docstring for schema.
        """
        try:
            rows = self._load_log()
        except FileNotFoundError:
            return self._empty_result()
        except json.JSONDecodeError:
            return self._empty_result()

        if not rows:
            return self._empty_result()

        # Keep only the N most recent cycles
        rows = rows[-n_recent_cycles:]

        try:
            return self._compute(rows)
        except Exception:
            return self._empty_result()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_log(self) -> list[dict[str, Any]]:
        """Parse every JSON line from *log_path*."""
        rows: list[dict[str, Any]] = []
        with open(self.log_path, "r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped:
                    rows.append(json.loads(stripped))
        return rows

    # ------------------------------------------------------------------

    def _empty_result(self) -> dict[str, Any]:
        return {
            "contamination_trajectory": {},
            "decontaminated_state": {},
            "false_positive_cb_triggers": 0,
            "memory_kernel_decay_rate": self.decay_rate,
            "signal_acceptance_improvement": 0.0,
            "decay_correction_applied": 0.0,
            "by_subsystem": {
                "circuit_breaker": {"before": 0.0, "after": 0.0, "improvement": 0.0},
                "segl_state": {"before": 0.0, "after": 0.0, "improvement": 0.0},
            },
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _is_cb_trigger(entry: dict[str, Any]) -> bool:
        """Return True if the cycle entry records a CB trigger."""
        exec_str: str = (
            entry.get("pipeline_trace", {}).get("execution", "") or ""
        )
        return "CB" in exec_str or "circuit" in exec_str.lower()

    # ------------------------------------------------------------------

    @staticmethod
    def _has_tick_failure(entry: dict[str, Any]) -> bool:
        """Return True if the cycle entry contains a tick failure."""
        exec_str: str = (
            entry.get("pipeline_trace", {}).get("execution", "") or ""
        )
        return "tick" in exec_str.lower() and "fail" in exec_str.lower()

    # ------------------------------------------------------------------

    @staticmethod
    def _mof_score_bin(score: float) -> int:
        """Bin mof_score into discrete levels for similarity comparison.

        Bins::

            0: score < 0.2
            1: 0.2 <= score < 0.4
            2: 0.4 <= score < 0.6
            3: 0.6 <= score < 0.8
            4: 0.8 <= score <= 1.0
        """
        if score < 0.2:
            return 0
        if score < 0.4:
            return 1
        if score < 0.6:
            return 2
        if score < 0.8:
            return 3
        return 4

    # ------------------------------------------------------------------

    @staticmethod
    def _extract_features(
        entry: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract the feature vector used for contamination comparison."""
        return {
            "mof_state": entry.get("mof_state", "UNKNOWN"),
            "mof_score_bin": GovernanceDecontamination._mof_score_bin(
                entry.get("mof_score", 0.0) or 0.0
            ),
            "open_positions": entry.get("open_positions", 0) or 0,
            "total_signals": entry.get("total_signals", 0) or 0,
        }

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _compute(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Core analysis logic on pre-loaded, trimmed *rows*."""

        # ── 1. Identify CB blocks ─────────────────────────────────────
        # A CB block is a contiguous sequence of cycles where the
        # pipeline_trace.execution field contains a CB denial.
        cb_block_ids: dict[int, int] = {}  # row_index -> block_id
        cb_blocks: dict[int, list[int]] = defaultdict(list)  # block_id -> [row_index]
        block_id = 0
        in_block = False

        for idx, r in enumerate(rows):
            if self._is_cb_trigger(r):
                if not in_block:
                    block_id += 1
                    in_block = True
                cb_block_ids[idx] = block_id
                cb_blocks[block_id].append(idx)
            else:
                in_block = False

        total_cb_blocks = len(cb_blocks)

        # ── 2. Build CB trigger profile ───────────────────────────────
        # The profile is the set of feature values that most commonly
        # appear during CB blocks.
        cb_feature_values: dict[str, Counter] = {
            "mof_state": Counter(),
            "mof_score_bin": Counter(),
            "open_positions": Counter(),
            "total_signals": Counter(),
        }

        for r in rows:
            if self._is_cb_trigger(r):
                feat = self._extract_features(r)
                for k, v in feat.items():
                    cb_feature_values[k][v] += 1

        cb_profile: dict[str, Any] = {}
        for k, counter in cb_feature_values.items():
            if counter:
                cb_profile[k] = counter.most_common(1)[0][0]
            else:
                cb_profile[k] = None

        # If no CB blocks exist, return empty result with zeros
        if not cb_profile.get("mof_state"):
            return self._empty_result()

        # ── 3. Per-cycle contamination & decontamination ──────────────
        contamination_trajectory: dict[str, float] = {}
        decontaminated_state: dict[str, float] = {}
        decay_correction_total = 0.0

        # Pre-compute: for each row index, the most recent CB row index
        # at or before that position.
        last_cb_idx: int | None = None
        most_recent_cb_index: list[int | None] = []
        for idx in range(len(rows)):
            if self._is_cb_trigger(rows[idx]):
                last_cb_idx = idx
            most_recent_cb_index.append(last_cb_idx)

        # Track for false-positive detection
        false_positive_cb_count = 0
        cb_block_start_indices: set[int] = set()
        for block_indices in cb_blocks.values():
            if block_indices:
                cb_block_start_indices.add(block_indices[0])

        for idx, r in enumerate(rows):
            cycle = r.get("cycle", 0)
            feat = self._extract_features(r)

            # Contamination = fraction of features matching CB profile
            matches = 0
            total_features = len(cb_profile)
            for k, profile_val in cb_profile.items():
                if profile_val is not None and feat.get(k) == profile_val:
                    matches += 1
            contamination = matches / total_features if total_features > 0 else 0.0

            # Cycles since last CB trigger measured in row-distance
            last_cb = most_recent_cb_index[idx]
            if last_cb is not None:
                cycles_since = idx - last_cb
            else:
                # No CB trigger observed yet in this window — use a large
                # effective distance so decontamination → 0.
                cycles_since = len(rows)

            # Apply decay correction
            decontaminated = contamination * math.exp(
                -self.decay_rate * cycles_since
            )

            key = f"cycle_{cycle}"
            contamination_trajectory[key] = round(contamination, 4)
            decontaminated_state[key] = round(decontaminated, 4)
            decay_correction_total += contamination - decontaminated

            # ── 4. False-positive detection ───────────────────────────
            # Check only the start of each CB block (first cycle of block)
            if idx in cb_block_start_indices:
                is_false_positive = (
                    contamination > 0.5
                    and r.get("mof_state") == "INFORMATION_RICH"
                    and (r.get("open_positions", 0) or 0) == 0
                    and not self._has_tick_failure(r)
                )
                if is_false_positive:
                    false_positive_cb_count += 1

        # ── 5. Aggregate metrics ──────────────────────────────────────
        # Signal acceptance improvement = false positives / total CB blocks
        signal_acceptance_improvement = (
            false_positive_cb_count / total_cb_blocks
            if total_cb_blocks > 0
            else 0.0
        )

        # By-subsystem: circuit_breaker and segl_state
        # Compute average contamination (before) and decontaminated (after)
        # across all cycles
        all_contaminations = list(contamination_trajectory.values())
        all_decontaminated = list(decontaminated_state.values())

        avg_before = (
            sum(all_contaminations) / len(all_contaminations)
            if all_contaminations
            else 0.0
        )
        avg_after = (
            sum(all_decontaminated) / len(all_decontaminated)
            if all_decontaminated
            else 0.0
        )
        cb_improvement = avg_before - avg_after if avg_before > 0 else 0.0

        return {
            "contamination_trajectory": dict(
                sorted(
                    contamination_trajectory.items(),
                    key=lambda kv: int(kv[0].split("_")[1]),
                )
            ),
            "decontaminated_state": dict(
                sorted(
                    decontaminated_state.items(),
                    key=lambda kv: int(kv[0].split("_")[1]),
                )
            ),
            "false_positive_cb_triggers": false_positive_cb_count,
            "memory_kernel_decay_rate": self.decay_rate,
            "signal_acceptance_improvement": round(
                signal_acceptance_improvement, 4
            ),
            "decay_correction_applied": round(decay_correction_total, 4),
            "by_subsystem": {
                "circuit_breaker": {
                    "before": round(avg_before, 4),
                    "after": round(avg_after, 4),
                    "improvement": round(cb_improvement, 4),
                },
                "segl_state": {
                    "before": round(avg_before, 4),
                    "after": round(avg_after, 4),
                    "improvement": round(cb_improvement, 4),
                },
            },
        }
