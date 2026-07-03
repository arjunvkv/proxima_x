"""Unified SYSTEM_REHYDRATION_PROFILE combining all Batch 6 diagnostic modules.

Classifies each HOLD cycle into one of 5 dominant mechanisms:
- DEEP_LATCHED_BLOCK
- PARTIAL_REHYDRATION_FAILURE
- MEMORY_LOCKED_SAFETY
- RECOVERY_GRADIENT_LOW
- NEAR_BASIN_READY
"""

import json
import logging
import statistics

logger = logging.getLogger("proxima_ops.diagnostics.integration")

try:
    from .cblpt import CBLatchPersistence
    from .srfm import StateRehydrationFailure
    from .rfg import RecoveryFieldGradient
    from .gmci import GovMemoryContamination
    from .erbm import ExecutionRestartBasin
    _HAS_MODULES = True
except ImportError as e:
    _HAS_MODULES = False
    logger.warning("Some diagnostic modules unavailable: %s", e)


class SystemRehydrationProfile:
    """Unified diagnostic combining all 5 Batch 6 modules."""

    def __init__(self, log_path: str = "state/wave12_cycle_log.jsonl"):
        self.log_path = log_path
        self._cblpt = CBLatchPersistence(log_path) if _HAS_MODULES else None
        self._srfm = StateRehydrationFailure(log_path) if _HAS_MODULES else None
        self._rfg = RecoveryFieldGradient(log_path) if _HAS_MODULES else None
        self._gmci = GovMemoryContamination(log_path) if _HAS_MODULES else None
        self._erbm = ExecutionRestartBasin(log_path) if _HAS_MODULES else None

    def analyze(self, n_recent_cycles: int = 500) -> dict:
        try:
            cblpt_r = self._cblpt.analyze(n_recent_cycles) if self._cblpt else {}
            srfm_r = self._srfm.analyze(n_recent_cycles) if self._srfm else {}
            rfg_r = self._rfg.analyze(n_recent_cycles) if self._rfg else {}
            gmci_r = self._gmci.analyze(n_recent_cycles) if self._gmci else {}
            erbm_r = self._erbm.analyze(n_recent_cycles) if self._erbm else {}

            cycles = self._load_cycles(n_recent_cycles)
            profiles = {}
            classifications = {}

            for c in cycles:
                cyc = c.get("cycle", 0)
                profile = self._build_profile(c, cblpt_r, srfm_r, rfg_r, gmci_r, erbm_r)
                profiles[str(cyc)] = profile
                classifications[str(cyc)] = self._classify_hold(profile)

            return {
                "total_cycles": len(cycles),
                "classification_distribution": self._distribution(classifications),
                "profiles": profiles,
                "classifications": classifications,
                "source_modules": {
                    "cblpt": bool(cblpt_r),
                    "srfm": bool(srfm_r),
                    "rfg": bool(rfg_r),
                    "gmci": bool(gmci_r),
                    "erbm": bool(erbm_r),
                },
            }
        except Exception as exc:
            logger.error("Integration analysis failed: %s", exc, exc_info=True)
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

    def _build_profile(self, cycle: dict, cblpt_r: dict, srfm_r: dict,
                       rfg_r: dict, gmci_r: dict, erbm_r: dict) -> dict:
        cyc = str(cycle.get("cycle", 0))
        cb_latch = self._extract_latch_strength(cycle, cblpt_r, cyc)
        sync = self._extract_sync(srfm_r, cyc)
        rfg = self._extract_rfg(rfg_r, cyc)
        mem = self._extract_mem(gmci_r, cyc)
        basin_dist = self._extract_basin(erbm_r, cyc)
        reentry = self._compute_reentry(cblpt_r)
        return {
            "cb_latch_strength": cb_latch,
            "subsystem_sync_index": sync,
            "recovery_gradient": rfg,
            "memory_contamination_score": mem,
            "basin_distance": basin_dist,
            "execution_reentry_probability": reentry,
        }

    def _extract_latch_strength(self, cycle: dict, cblpt_r: dict, cyc: str) -> float:
        denial = cycle.get("denial_reason", "")
        if "CircuitBreaker" in denial:
            return 1.0
        if cblpt_r:
            depth = cblpt_r.get("hysteresis_depth", 0.0)
            return min(depth * 1.2, 1.0)
        return 0.0

    def _extract_sync(self, srfm_r: dict, cyc: str) -> float:
        if srfm_r:
            return srfm_r.get("subsystem_sync_index", 0.5)
        return 0.5

    def _extract_rfg(self, rfg_r: dict, cyc: str) -> float:
        if rfg_r:
            traj = rfg_r.get("rfg_trajectory", {})
            return traj.get(cyc, rfg_r.get("mean_rfg", 0.5))
        return 0.5

    def _extract_mem(self, gmci_r: dict, cyc: str) -> float:
        if gmci_r:
            traj = gmci_r.get("memory_contamination_trajectory", {})
            return traj.get(cyc, gmci_r.get("mean_contamination", 0.5))
        return 0.5

    def _extract_basin(self, erbm_r: dict, cyc: str) -> float:
        if erbm_r:
            traj = erbm_r.get("basin_distance_trajectory", {})
            return traj.get(cyc, 0.5)
        return 0.5

    def _compute_reentry(self, cblpt_r: dict) -> float:
        if cblpt_r:
            return cblpt_r.get("reentry_probability", 0.0)
        return 0.0

    def _classify_hold(self, p: dict) -> str:
        if p["cb_latch_strength"] > 0.7:
            return "DEEP_LATCHED_BLOCK"
        if p["subsystem_sync_index"] < 0.4:
            return "PARTIAL_REHYDRATION_FAILURE"
        if p["memory_contamination_score"] > 0.6:
            return "MEMORY_LOCKED_SAFETY"
        if p["recovery_gradient"] < 0.3:
            return "RECOVERY_GRADIENT_LOW"
        if p["basin_distance"] < 0.3:
            return "NEAR_BASIN_READY"
        return "DEEP_LATCHED_BLOCK"

    def _distribution(self, classifications: dict) -> dict:
        dist = {}
        for v in classifications.values():
            dist[v] = dist.get(v, 0) + 1
        total = max(len(classifications), 1)
        return {k: {"count": v, "pct": round(v / total * 100, 1)} for k, v in sorted(dist.items())}
