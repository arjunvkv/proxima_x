"""
EPRG — Execution Pathway Reconstitution Graph

Model the S→C→G→E signal pipeline as a directed graph with three edges:
  - S_C  (Signal → Confirm)
  - C_G  (Confirm → Governor)
  - G_E  (Governor → Execute)

Identify broken edges (weight < 0.1), compute execution reachability
probability, minimum cut sets, and a fragmentation index.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("proxima_ops.diagnostics.eprg")

# ---------------------------------------------------------------------------
# Edge weight threshold for "broken"
# ---------------------------------------------------------------------------
_BROKEN_THRESHOLD = 0.1

# ---------------------------------------------------------------------------
# Edge names
# ---------------------------------------------------------------------------
_EDGES = ("S_C", "C_G", "G_E")


class ExecutionPathwayGraph:
    """Directed-graph model of the signal execution pipeline.

    Parameters
    ----------
    log_path : str
        Path to the JSONL cycle log (default
        ``"state/wave12_cycle_log.jsonl"``).
    """

    def __init__(self, log_path: str = "state/wave12_cycle_log.jsonl") -> None:
        self.log_path = log_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, n_recent_cycles: int = 500) -> dict[str, Any]:
        """Analyse the last *n_recent_cycles* and produce EPRG metrics.

        Returns
        -------
        dict
            Keys: ``adjacency_matrix``, ``execution_reachability_probability``,
            ``minimum_cut_sets``, ``broken_edges``,
            ``path_probability_trajectory``, ``fragmentation_index``.
        """
        cycles = self._load_cycles(n_recent_cycles)

        if not cycles:
            logger.warning("No cycles loaded — returning empty report.")
            return self._empty_report()

        # ---- per-cycle edge activity ---------------------------------
        s_c_active: list[bool] = []
        c_g_active: list[bool] = []
        g_e_active: list[bool] = []
        cycle_labels: list[str] = []

        for cyc in cycles:
            s_c_active.append(self._is_s_c_active(cyc))
            c_g_active.append(self._is_c_g_active(cyc))
            g_e_active.append(self._is_g_e_active(cyc))
            cycle_labels.append(str(cyc.get("cycle", "?")))

        n = len(cycles)

        # ---- adjacency matrix (edge weights) -------------------------
        adjacency_matrix: dict[str, float] = {
            "S_C": sum(s_c_active) / n,
            "C_G": sum(c_g_active) / n,
            "G_E": sum(g_e_active) / n,
        }

        # ---- execution reachability ----------------------------------
        reachability = (
            adjacency_matrix["S_C"]
            * adjacency_matrix["C_G"]
            * adjacency_matrix["G_E"]
        )

        # ---- minimum cut sets (edges below threshold) ----------------
        sorted_edges = sorted(
            _EDGES, key=lambda e: adjacency_matrix[e]
        )
        minimum_cut_sets = [e for e in sorted_edges if adjacency_matrix[e] < _BROKEN_THRESHOLD]

        # ---- broken edges --------------------------------------------
        broken_edges = [
            e for e in _EDGES if adjacency_matrix[e] < _BROKEN_THRESHOLD
        ]

        # ---- path probability trajectory (per-cycle) -----------------
        path_probability_trajectory: dict[str, float] = {}
        for i in range(n):
            p = (
                (1.0 if s_c_active[i] else 0.0)
                * (1.0 if c_g_active[i] else 0.0)
                * (1.0 if g_e_active[i] else 0.0)
            )
            path_probability_trajectory[f"cycle_{cycle_labels[i]}"] = p

        # ---- fragmentation index -------------------------------------
        edges_above = sum(
            1 for e in _EDGES if adjacency_matrix[e] > _BROKEN_THRESHOLD
        )
        fragmentation_index = 1.0 - (edges_above / len(_EDGES))

        return {
            "adjacency_matrix": adjacency_matrix,
            "execution_reachability_probability": reachability,
            "minimum_cut_sets": minimum_cut_sets,
            "broken_edges": broken_edges,
            "path_probability_trajectory": path_probability_trajectory,
            "fragmentation_index": round(fragmentation_index, 4),
        }

    # ------------------------------------------------------------------
    # Internal helpers — edge-activity predicates
    # ------------------------------------------------------------------

    @staticmethod
    def _is_s_c_active(cyc: dict[str, Any]) -> bool:
        """S→C edge: signal *exists* AND confirm transition *happened*.

        Active when ``total_signals > 0`` and ``confirm_cycles >= 1``.
        """
        try:
            return (
                cyc.get("total_signals", 0) > 0
                and cyc.get("confirm_cycles", 0) >= 1
            )
        except Exception:
            return False

    @staticmethod
    def _is_c_g_active(cyc: dict[str, Any]) -> bool:
        """C→G edge: confirm *passed*.

        Active when ``confirm_cycles >= 2`` (cross-cycle projection
        threshold reached).
        """
        try:
            return cyc.get("confirm_cycles", 0) >= 2
        except Exception:
            return False

    @staticmethod
    def _is_g_e_active(cyc: dict[str, Any]) -> bool:
        """G→E edge: governor + CB *passed*.

        Active when:
          - ``segl_state == "ARMED"``
          - The execution outcome is *not* a denial (does not start with
            ``"DENIED"``) and is not a no-signal outcome.

        This captures cycles where the governor allowed execution
        regardless of whether the MT5 broker call succeeded or failed.
        """
        try:
            if cyc.get("segl_state") != "ARMED":
                return False

            pt = cyc.get("pipeline_trace", {})
            if not isinstance(pt, dict):
                return False

            execution = pt.get("execution", "")
            if not isinstance(execution, str):
                return False

            # Governor / CB / VEL denial
            if execution.startswith("DENIED"):
                return False

            # No signal reached the execution gate
            if execution.startswith("NO_SIGNAL"):
                return False

            # Everything else (FAILED, PLACED, ...) means execution was
            # at least attempted — edge is active.
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_cycles(self, n_recent: int) -> list[dict[str, Any]]:
        """Read the last *n_recent* JSONL records from the log file."""
        records: list[dict[str, Any]] = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        records.append(json.loads(stripped))
                    except json.JSONDecodeError:
                        logger.warning("Skipping unparseable line: %r", stripped[:120])
        except FileNotFoundError:
            logger.error("Log file not found: %s", self.log_path)
            return []
        except Exception as exc:
            logger.error("Failed to read log file: %s", exc)
            return []

        if n_recent <= 0 or n_recent >= len(records):
            return records
        return records[-n_recent:]

    # ------------------------------------------------------------------
    # Empty-report fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_report() -> dict[str, Any]:
        return {
            "adjacency_matrix": {"S_C": 0.0, "C_G": 0.0, "G_E": 0.0},
            "execution_reachability_probability": 0.0,
            "minimum_cut_sets": [],
            "broken_edges": [],
            "path_probability_trajectory": {},
            "fragmentation_index": 1.0,
        }
