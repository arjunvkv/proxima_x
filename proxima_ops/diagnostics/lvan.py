"""
LVAN — Latent Veto Attribution Network

Model every HOLD decision as a superposition of latent veto forces.
Produces a vector of veto contributions per cycle from the wave cycle log.
"""

import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional


class LatentVetoAttribution:
    """Analyse HOLD decisions as a superposition of latent veto forces."""

    VETO_KEYS = [
        "volatility_rejection",
        "confirmation_decay",
        "execution_hesitation",
        "liquidity_ambiguity",
        "signal_disagreement",
    ]

    def __init__(self, log_path: str = "state/wave12_cycle_log.jsonl") -> None:
        self.log_path = os.path.abspath(log_path)

    # ------------------------------------------------------------------
    # Per-cycle veto estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_v1_volatility_rejection(entry: dict) -> float:
        """V1 — high when MoF is not information-rich."""
        mof_state = entry.get("mof_state", "")
        if mof_state != "INFORMATION_RICH":
            return 0.7
        return 0.2

    @staticmethod
    def _estimate_v2_confirmation_decay(entry: dict) -> float:
        """V2 — high when confirm_cycles < 2 (insufficient cross-projection confirmation)."""
        confirm_cycles = entry.get("confirm_cycles", 0)
        if isinstance(confirm_cycles, (int, float)) and confirm_cycles < 2:
            return 0.8
        return 0.2

    @staticmethod
    def _estimate_v3_execution_hesitation(entry: dict) -> float:
        """V3 — high when pipeline execution was DENIED or FAILED."""
        pipeline = entry.get("pipeline_trace", {}) or {}
        execution = (pipeline.get("execution") or "").strip()
        if any(kw in execution.upper() for kw in ("DENIED", "FAILED")):
            return 0.8
        return 0.2

    @staticmethod
    def _estimate_v4_liquidity_ambiguity(entry: dict) -> float:
        """V4 — high when denial_reason mentions missing tick data."""
        denial = entry.get("denial_reason") or ""
        if "No tick data" in denial or "no tick data" in denial.lower():
            return 0.8
        return 0.2

    @staticmethod
    def _estimate_v5_signal_disagreement(entry: dict) -> float:
        """V5 — high when generated / threshold-gate signals show mixed BUY (dir=1) and SELL (dir=-1)."""
        pipeline = entry.get("pipeline_trace", {}) or {}

        # Collect raw text from threshold_gate or generated (handles both str and list).
        raw_parts: list[str] = []
        for field in ("threshold_gate", "generated"):
            val = pipeline.get(field)
            if isinstance(val, str):
                raw_parts.append(val)
            elif isinstance(val, list):
                raw_parts.extend(str(item) for item in val)

        raw = " ".join(raw_parts)

        dirs = set(re.findall(r"dir=([-\d]+)", raw))
        # dir=1  → BUY,  dir=-1 → SELL
        if "1" in dirs and "-1" in dirs:
            return 0.7
        return 0.2

    # ------------------------------------------------------------------
    # Shapley attribution helpers  (count blocker sources)
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_shapley_source(entry: dict) -> Optional[str]:
        """Return the subsystem name that was the explicit blocker, or None."""
        denial = entry.get("denial_reason") or ""
        if not denial:
            return None

        d_lower = denial.lower()

        # confirm_gate
        if "confirm" in d_lower or "cross_confirm" in d_lower:
            return "confirm_gate"

        # governor_segl  (SEGL state blocker)
        if "segl_state" in d_lower or "obs" in d_lower:
            return "governor_segl"

        # governor_cb  (commitment-boundary / confidence-block)
        if "cb" in denial or "confidence_block" in d_lower or "commitment_boundary" in d_lower:
            return "governor_cb"

        # vel  (exposure smoothing / burst prevention)
        if "vel" in d_lower or "exposure_smoothing" in d_lower or "burst_prevention" in d_lower:
            return "vel"

        # execution  (MT5 / order failure)
        if "fail" in d_lower or "mt5" in d_lower or "none" in denial:
            return "execution"

        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, n_recent_cycles: int = 500) -> Dict[str, Any]:
        """
        Read the cycle log and produce a full LVAN diagnostic dictionary.

        Parameters
        ----------
        n_recent_cycles : int
            Number of most-recent cycles to analyse (default 500).

        Returns
        -------
        dict
            {
              "total_cycles": int,
              "hold_cycles": int,
              "veto_decomposition": { ... },
              "dominant_veto": str,
              "hold_reconstruction_accuracy": float,
              "per_cycle_top_veto": { "cycle_N": str },
              "shapley_breakdown": { ... }
            }
        """
        result: Dict[str, Any] = {
            "total_cycles": 0,
            "hold_cycles": 0,
            "veto_decomposition": {k: 0.0 for k in self.VETO_KEYS},
            "dominant_veto": "",
            "hold_reconstruction_accuracy": 0.0,
            "per_cycle_top_veto": {},
            "shapley_breakdown": {
                "confirm_gate": 0.0,
                "governor_segl": 0.0,
                "governor_cb": 0.0,
                "vel": 0.0,
                "execution": 0.0,
            },
        }

        try:
            if not os.path.isfile(self.log_path):
                result["error"] = f"Log file not found: {self.log_path}"
                return result

            # ---- load cycles -------------------------------------------------
            cycles: List[dict] = []
            with open(self.log_path, "r", encoding="utf-8") as fh:
                for raw_line in fh:
                    line = raw_line.strip()
                    if line:
                        try:
                            cycles.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue  # skip malformed lines

            result["total_cycles"] = len(cycles)

            # ---- take the N most-recent --------------------------------------
            tail = cycles[-n_recent_cycles:] if n_recent_cycles else cycles

            hold_entries = [c for c in tail if c.get("decision", "").upper() == "HOLD"]
            result["hold_cycles"] = len(hold_entries)

            if not hold_entries:
                # No holds in window — return zeroed decomposition
                return result

            # ---- per-cycle veto vectors --------------------------------------
            veto_vectors: List[Dict[str, float]] = []
            shapley_counts: Dict[str, int] = defaultdict(int)
            correct_reconstructions = 0

            for entry in hold_entries:
                v1 = self._estimate_v1_volatility_rejection(entry)
                v2 = self._estimate_v2_confirmation_decay(entry)
                v3 = self._estimate_v3_execution_hesitation(entry)
                v4 = self._estimate_v4_liquidity_ambiguity(entry)
                v5 = self._estimate_v5_signal_disagreement(entry)

                veto_vec = {
                    "volatility_rejection": v1,
                    "confirmation_decay": v2,
                    "execution_hesitation": v3,
                    "liquidity_ambiguity": v4,
                    "signal_disagreement": v5,
                }
                veto_vectors.append(veto_vec)

                # Dominant veto for this cycle
                top_veto = max(veto_vec, key=veto_vec.get)  # type: ignore[arg-type]
                cycle_key = f"cycle_{entry.get('cycle', '?')}"
                result["per_cycle_top_veto"][cycle_key] = top_veto

                # Reconstruction accuracy: max(veto) > 0.5
                if max(veto_vec.values()) > 0.5:
                    correct_reconstructions += 1

                # Shapley: count explicit blocker sources
                source = self._classify_shapley_source(entry)
                if source is not None:
                    shapley_counts[source] += 1

            # ---- veto decomposition (average across hold cycles) -------------
            n_hold = len(hold_entries)
            for key in self.VETO_KEYS:
                result["veto_decomposition"][key] = round(
                    sum(v[key] for v in veto_vectors) / n_hold, 4
                )

            # ---- dominant veto -----------------------------------------------
            result["dominant_veto"] = max(
                result["veto_decomposition"], key=result["veto_decomposition"].get  # type: ignore[arg-type]
            )

            # ---- reconstruction accuracy -------------------------------------
            result["hold_reconstruction_accuracy"] = round(
                correct_reconstructions / n_hold, 4
            )

            # ---- shapley breakdown (fraction of hold cycles) -----------------
            for source_key in result["shapley_breakdown"]:
                result["shapley_breakdown"][source_key] = round(
                    shapley_counts.get(source_key, 0) / n_hold, 4
                )

        except Exception as exc:
            result["error"] = f"LVAN analysis failed: {exc}"

        return result
