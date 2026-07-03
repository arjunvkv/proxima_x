"""
GMCI — Governance Memory Contamination Index

Governance layer remembers past failures incorrectly, contaminating current decisions.

Reads from state/wave12_cycle_log.jsonl and computes memory contamination metrics
by tracking how past circuit-breaker triggers and blocked states influence current
governance decisions.
"""

import json
import math
import os


class GovMemoryContamination:
    """Quantify how much past governance failures contaminate current decisions."""

    def __init__(self, log_path: str = "state/wave12_cycle_log.jsonl", memory_decay: int = 100):
        self.log_path = os.path.abspath(log_path)
        self.memory_decay = memory_decay

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_cb_triggered(row: dict) -> bool:
        """Determine if the circuit-breaker was triggered for this cycle."""
        cb = str(row.get("cb_decision", "")).lower().strip()
        if cb and cb not in ("", "none", "allowed"):
            return True
        pipeline = row.get("pipeline_trace", {}) or {}
        execution = str(pipeline.get("execution", ""))
        if "denied cb" in execution.lower() or "circuit breaker" in execution.lower():
            return True
        denial = str(row.get("denial_reason", ""))
        if "cb" in denial.lower() or "circuit breaker" in denial.lower():
            return True
        return False

    @staticmethod
    def _is_confirm_gate_blocked(row: dict) -> bool:
        """Determine if confirm_gate was a blocker for this cycle."""
        confirm = row.get("confirm_cycles", 2)
        if isinstance(confirm, (int, float)) and confirm < 1:
            return True
        pipeline = row.get("pipeline_trace", {}) or {}
        confirm_raw = pipeline.get("confirm_gate", [])
        if isinstance(confirm_raw, str) and "fail" in confirm_raw.lower():
            return True
        denial = str(row.get("denial_reason", ""))
        if "confirm" in denial.lower():
            return True
        return False

    @staticmethod
    def _is_segl_blocked(row: dict) -> bool:
        """Determine if segl_state was a blocker for this cycle."""
        segl = str(row.get("segl_state", "")).upper()
        if segl in ("OBSERVE",):
            return True
        pipeline = row.get("pipeline_trace", {}) or {}
        governor = pipeline.get("governor_gate", [])
        if isinstance(governor, list):
            for g in governor:
                if "segl_state=OBSERVE" in str(g):
                    return True
        denial = str(row.get("denial_reason", ""))
        if "segl_state" in denial.lower() or "obs" in denial.lower():
            return True
        return False

    @staticmethod
    def _compute_similarity(current: dict, past: dict) -> float:
        """
        Compare current cycle state to a past cycle state.

        Conditions compared:
          - mof_state matches (exact string match)
          - spread level within 20% (or both zero / unavailable)
          - open_positions match (exact integer match)
          - total_signals within 30% relative difference

        Returns fraction of matching conditions (0.0 – 1.0).
        """
        conditions = 4
        matches = 0

        # 1) mof_state
        cur_mof = str(current.get("mof_state", "")).upper().strip()
        past_mof = str(past.get("mof_state", "")).upper().strip()
        if cur_mof and past_mof and cur_mof == past_mof:
            matches += 1
        elif not cur_mof and not past_mof:
            matches += 1  # both absent → treat as match

        # 2) spread level
        cur_spread = current.get("spread_level")
        past_spread = past.get("spread_level")
        if cur_spread is not None and past_spread is not None:
            try:
                cs = float(cur_spread)
                ps = float(past_spread)
                if cs == 0.0 and ps == 0.0:
                    matches += 1
                elif cs > 0 and ps > 0:
                    ratio = min(cs, ps) / max(cs, ps)
                    if ratio >= 0.8:
                        matches += 1
            except (ValueError, TypeError):
                pass
        else:
            # spread data unavailable — skip condition (reduces denominator)
            conditions -= 1

        # 3) open_positions
        cur_op = current.get("open_positions")
        past_op = past.get("open_positions")
        if cur_op is not None and past_op is not None:
            try:
                if int(cur_op) == int(past_op):
                    matches += 1
            except (ValueError, TypeError):
                pass
        else:
            conditions -= 1

        # 4) total_signals
        cur_ts = current.get("total_signals")
        past_ts = past.get("total_signals")
        if cur_ts is not None and past_ts is not None:
            try:
                cts = float(cur_ts)
                pts = float(past_ts)
                if cts == 0.0 and pts == 0.0:
                    matches += 1
                elif max(abs(cts), abs(pts)) > 0:
                    rel_diff = abs(cts - pts) / max(abs(cts), abs(pts))
                    if rel_diff <= 0.3:
                        matches += 1
            except (ValueError, TypeError):
                pass
        else:
            conditions -= 1

        if conditions == 0:
            return 0.0
        return matches / conditions

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, n_recent_cycles: int = 500) -> dict:
        """
        Compute the Governance Memory Contamination Index.

        Parameters
        ----------
        n_recent_cycles : int
            Number of most-recent cycles to analyse (default 500).

        Returns
        -------
        dict with keys:
          memory_contamination_trajectory : {cycle_N: float}
          mean_contamination              : float
          similarity_to_past_blocks       : {cycle_N: float}
          contamination_spikes            : int
          decay_kernel_half_life          : float
          memory_driven_block_rate        : float
          by_subsystem_contamination      : dict
        """
        try:
            # ---- load cycles -------------------------------------------------
            if not os.path.isfile(self.log_path):
                return {"error": f"Log file not found: {self.log_path}"}

            cycles: list[dict] = []
            with open(self.log_path, "r", encoding="utf-8") as fh:
                for raw_line in fh:
                    line = raw_line.strip()
                    if line:
                        try:
                            cycles.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

            if not cycles:
                return {"error": "No cycle data found"}

            # ---- take the N most-recent --------------------------------------
            tail = cycles[-n_recent_cycles:] if n_recent_cycles else cycles

            # ---- identify CB trigger cycles (past blocked states) ------------
            cb_cycles: list[int] = []
            cb_cycle_data: dict[int, dict] = {}
            for row in tail:
                c = row.get("cycle")
                if c is None:
                    continue
                if self._is_cb_triggered(row):
                    cb_cycles.append(c)
                    cb_cycle_data[c] = row

            # ---- compute memory kernel weights for all cycles ---------------
            # Each cycle gets a cumulative weight from all past CB triggers.
            # weight = sum over past CB triggers of exp(-distance / memory_decay)
            sorted_cycles = sorted(row.get("cycle") for row in tail if row.get("cycle") is not None)

            contamination: dict[str, float] = {}
            similarity: dict[str, float] = {}
            current_cycle_num = max(sorted_cycles) if sorted_cycles else 0

            # Most recent cycle as the "current" state for similarity comparisons
            current_row = None
            for row in tail:
                if row.get("cycle") == current_cycle_num:
                    current_row = row
                    break

            for c in sorted_cycles:
                if c not in cb_cycle_data:
                    continue  # skip non-CB cycles for contamination

                row = cb_cycle_data[c]

                # Memory kernel: weighted contribution from all *earlier* CB triggers
                kernel_weight = 0.0
                for past_c in cb_cycles:
                    if past_c >= c:
                        continue
                    distance = c - past_c
                    kernel_weight += math.exp(-distance / self.memory_decay)

                # Clamp kernel weight to [0, 1] for interpretability
                kernel_weight = min(1.0, kernel_weight)

                # Similarity to past blocked states
                sim_sum = 0.0
                sim_count = 0
                for past_c in cb_cycles:
                    if past_c >= c:
                        continue
                    sim_sum += self._compute_similarity(row, cb_cycle_data[past_c])
                    sim_count += 1
                avg_similarity = sim_sum / sim_count if sim_count > 0 else 0.0

                # Contamination = weighted similarity x memory_kernel_weight
                cont = avg_similarity * kernel_weight
                contamination[f"cycle_{c}"] = round(cont, 6)
                similarity[f"cycle_{c}"] = round(avg_similarity, 6)

            # ---- similarity of current state to ALL past blocked states ------
            sim_to_past: dict[str, float] = {}
            if current_row is not None:
                for past_c in cb_cycles:
                    if past_c >= current_cycle_num:
                        continue
                    sim_val = self._compute_similarity(current_row, cb_cycle_data[past_c])
                    sim_to_past[f"cycle_{past_c}"] = round(sim_val, 6)

            # ---- mean contamination ------------------------------------------
            cont_values = list(contamination.values())
            mean_cont = round(sum(cont_values) / len(cont_values), 6) if cont_values else 0.0

            # ---- contamination spikes (contamination > 0.7) ------------------
            spikes = sum(1 for v in cont_values if v > 0.7)

            # ---- decay kernel half-life --------------------------------------
            half_life = round(self.memory_decay * math.log(2), 6)

            # ---- memory-driven block rate ------------------------------------
            # Fraction of current CB blocks where contamination > 0.5
            cb_block_count = len(cb_cycles)
            contaminated_blocks = sum(
                1 for c in cb_cycles
                if contamination.get(f"cycle_{c}", 0) > 0.5
            )
            block_rate = round(
                contaminated_blocks / cb_block_count, 6
            ) if cb_block_count > 0 else 0.0

            # ---- by-subsystem memory load ------------------------------------
            # How much each subsystem's past state contributes to contamination.
            subsystem_memory: dict[str, list[float]] = {
                "circuit_breaker": [],
                "segl_state": [],
                "confirm_gate": [],
            }

            for c in cb_cycles:
                row = cb_cycle_data.get(c, {})
                # circuit_breaker memory load: kernel weight from past CB triggers
                cb_load = 0.0
                for past_c in cb_cycles:
                    if past_c >= c:
                        continue
                    distance = c - past_c
                    cb_load += math.exp(-distance / self.memory_decay)
                cb_load = min(1.0, cb_load)
                subsystem_memory["circuit_breaker"].append(cb_load)

                # segl_state memory load: how much past segl blocks contribute
                segl_load = 0.0
                segl_count = 0
                for past_c in cb_cycles:
                    if past_c >= c:
                        continue
                    past_row = cb_cycle_data.get(past_c, {})
                    if self._is_segl_blocked(past_row):
                        distance = c - past_c
                        segl_load += math.exp(-distance / self.memory_decay)
                        segl_count += 1
                segl_load = min(1.0, segl_load / max(1, segl_count))
                subsystem_memory["segl_state"].append(segl_load)

                # confirm_gate memory load: how much past confirm blocks contribute
                confirm_load = 0.0
                confirm_count = 0
                for past_c in cb_cycles:
                    if past_c >= c:
                        continue
                    past_row = cb_cycle_data.get(past_c, {})
                    if self._is_confirm_gate_blocked(past_row):
                        distance = c - past_c
                        confirm_load += math.exp(-distance / self.memory_decay)
                        confirm_count += 1
                confirm_load = min(1.0, confirm_load / max(1, confirm_count))
                subsystem_memory["confirm_gate"].append(confirm_load)

            by_subsystem = {}
            for sub, vals in subsystem_memory.items():
                by_subsystem[sub] = {
                    "memory_load": round(
                        sum(vals) / len(vals), 6
                    ) if vals else 0.0
                }

            return {
                "memory_contamination_trajectory": contamination,
                "mean_contamination": mean_cont,
                "similarity_to_past_blocks": similarity,
                "contamination_spikes": spikes,
                "decay_kernel_half_life": half_life,
                "memory_driven_block_rate": block_rate,
                "by_subsystem_contamination": by_subsystem,
            }

        except Exception as exc:
            return {
                "error": f"GMCI analysis failed: {exc}",
                "memory_contamination_trajectory": {},
                "mean_contamination": 0.0,
                "similarity_to_past_blocks": {},
                "contamination_spikes": 0,
                "decay_kernel_half_life": 0.0,
                "memory_driven_block_rate": 0.0,
                "by_subsystem_contamination": {
                    "circuit_breaker": {"memory_load": 0.0},
                    "segl_state": {"memory_load": 0.0},
                    "confirm_gate": {"memory_load": 0.0},
                },
            }


# ------------------------------------------------------------------
# Quick CLI demo
# ------------------------------------------------------------------
if __name__ == "__main__":
    gmci = GovMemoryContamination()
    result = gmci.analyze(n_recent_cycles=500)
    print(json.dumps(result, indent=2, default=str))
