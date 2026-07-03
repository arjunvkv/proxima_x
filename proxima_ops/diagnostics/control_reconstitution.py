"""SYSTEM_CONTROL_RECONSTITUTION_PROFILE — Batch 7 Integration.

Combines all 5 control reconstitution modules into a unified profile.
Classifies each cycle into one of 5 states:
- HYSTERESIS_LOCKED_SYSTEM
- BROKEN_EXECUTION_TOPOLOGY
- ENERGY_TRAPPED_ATTRACTOR
- MEMORY_BIASED_SAFETY_SYSTEM
- OFF_MANIFOLD_STATE
"""

import json
import logging

logger = logging.getLogger("proxima_ops.diagnostics.control_reconstitution")

try:
    from .hnc import HysteresisNeutralization
    from .eprg import ExecutionPathwayGraph
    from .aeem import AttractorEscapeEnergy
    from .gde import GovernanceDecontamination
    from .emrs import ExecutionManifoldReinjection
    _HAS_MODULES = True
except ImportError as e:
    _HAS_MODULES = False
    logger.warning("Some control modules unavailable: %s", e)


class SystemControlReconstitutionProfile:
    """Unified profile combining all 5 Batch 7 modules."""

    def __init__(self, log_path: str = "state/wave12_cycle_log.jsonl"):
        self.log_path = log_path
        self._hnc = HysteresisNeutralization(log_path) if _HAS_MODULES else None
        self._eprg = ExecutionPathwayGraph(log_path) if _HAS_MODULES else None
        self._aeem = AttractorEscapeEnergy(log_path) if _HAS_MODULES else None
        self._gde = GovernanceDecontamination(log_path) if _HAS_MODULES else None
        self._emrs = ExecutionManifoldReinjection(log_path) if _HAS_MODULES else None

    def analyze(self, n_recent_cycles: int = 500) -> dict:
        try:
            hnc_r = self._hnc.analyze(n_recent_cycles) if self._hnc else {}
            eprg_r = self._eprg.analyze(n_recent_cycles) if self._eprg else {}
            aeem_r = self._aeem.analyze(n_recent_cycles) if self._aeem else {}
            gde_r = self._gde.analyze(n_recent_cycles) if self._gde else {}
            emrs_r = self._emrs.analyze(n_recent_cycles) if self._emrs else {}

            cycles = self._load_cycles(n_recent_cycles)
            profiles = {}
            classifications = {}

            for c in cycles:
                cyc = c.get("cycle", 0)
                profile = self._build_profile(c, hnc_r, eprg_r, aeem_r, gde_r, emrs_r)
                profiles[str(cyc)] = profile
                classifications[str(cyc)] = self._classify(profile)

            return {
                "total_cycles": len(cycles),
                "classification_distribution": self._distribution(classifications),
                "profiles": profiles,
                "classifications": classifications,
                "source_modules": {
                    "hnc": bool(hnc_r),
                    "eprg": bool(eprg_r),
                    "aeem": bool(aeem_r),
                    "gde": bool(gde_r),
                    "emrs": bool(emrs_r),
                },
            }
        except Exception as exc:
            logger.error("Control reconstitution failed: %s", exc, exc_info=True)
            return {"error": str(exc)}

    def _load_cycles(self, n: int) -> list:
        records = []
        try:
            with open(self.log_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning("Cannot load cycles: %s", e)
            return []
        return records[-n:] if n and n < len(records) else records

    def _build_profile(self, cycle: dict, hnc_r: dict, eprg_r: dict,
                       aeem_r: dict, gde_r: dict, emrs_r: dict) -> dict:
        return {
            "hysteresis_symmetry_score": hnc_r.get("symmetry_convergence_score", 0.5),
            "execution_path_reachability": eprg_r.get("execution_reachability_probability", 0.0),
            "escape_energy_required": aeem_r.get("escape_energy_required", 0.5),
            "governance_contamination_level": gde_r.get("memory_kernel_decay_rate", 0.5),
            "manifold_distance": emrs_r.get("current_manifold_distance", 0.5),
            "reentry_success_probability": emrs_r.get("reentry_probability", 0.0),
        }

    def _classify(self, p: dict) -> str:
        if p["hysteresis_symmetry_score"] < 0.3:
            return "HYSTERESIS_LOCKED_SYSTEM"
        if p["execution_path_reachability"] < 0.1:
            return "BROKEN_EXECUTION_TOPOLOGY"
        if p["escape_energy_required"] > 0.7:
            return "ENERGY_TRAPPED_ATTRACTOR"
        if p["governance_contamination_level"] > 0.6:
            return "MEMORY_BIASED_SAFETY_SYSTEM"
        if p["manifold_distance"] > 0.5:
            return "OFF_MANIFOLD_STATE"
        return "HYSTERESIS_LOCKED_SYSTEM"

    def _distribution(self, classifications: dict) -> dict:
        dist = {}
        for v in classifications.values():
            dist[v] = dist.get(v, 0) + 1
        total = max(len(classifications), 1)
        return {k: {"count": v, "pct": round(v / total * 100, 1)} for k, v in sorted(dist.items())}
