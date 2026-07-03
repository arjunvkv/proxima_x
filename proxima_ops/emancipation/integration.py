"""EXECUTION_EMANCIPATION_PROFILE — Batch 8 Integration.

Combines all 5 emancipation modules into unified execution profile.
"""

import json
import logging

logger = logging.getLogger("proxima_ops.emancipation.integration")

try:
    from .dce import DecisionCollapseEngine
    from .tamk import TradeAuthorizationMinimalKernel
    from .ecp import ExecutionCompressionPipeline
    from .loef import LatentOpportunityExtractionField
    from .era import ExecutionRealityAnchor
    _HAS_MODULES = True
except ImportError as e:
    _HAS_MODULES = False
    logger.warning("Some emancipation modules unavailable: %s", e)


class ExecutionEmancipationProfile:
    """Unified profile combining all 5 Batch 8 emancipation modules."""

    def __init__(self):
        self._dce = DecisionCollapseEngine() if _HAS_MODULES else None
        self._tamk = TradeAuthorizationMinimalKernel() if _HAS_MODULES else None
        self._ecp = ExecutionCompressionPipeline() if _HAS_MODULES else None
        self._loef = LatentOpportunityExtractionField() if _HAS_MODULES else None
        self._era = ExecutionRealityAnchor() if _HAS_MODULES else None

    def evaluate(self, signals: list, confirm_counts: dict, readiness: dict,
                 governor_state: str, cb_triggered: bool, cb_latch_cycles: int,
                 sil_scores: dict, activation: dict, rsi_dict: dict,
                 erf: float, escape_energy: float, rfg: float,
                 gmci_score: float, eprg_reachability: float,
                 mt5_tick: dict, mt5_account: dict, open_positions: list,
                 best_signal: dict = None) -> dict:
        try:
            dce_r = self._dce.collapse(signals, confirm_counts, readiness,
                                        governor_state, cb_triggered,
                                        sil_scores, activation, rsi_dict) if self._dce else {}

            tamk_r = self._tamk.authorize(erf, escape_energy, rfg,
                                           cb_triggered, cb_latch_cycles,
                                           gmci_score, governor_state,
                                           mt5_tick is not None) if self._tamk else {}

            loef_r = self._loef.compute(signals, sil_scores, rsi_dict,
                                         activation, readiness, erf,
                                         escape_energy, gmci_score,
                                         eprg_reachability) if self._loef else {}

            era_r = self._era.validate(
                dce_r.get("action", "HOLD"),
                dce_r.get("symbol", ""),
                best_signal.get("confidence", 0.1) if best_signal else 0.1,
                mt5_tick.get("ask" if dce_r.get("action") == "BUY" else "bid", 0) if mt5_tick else 0,
                mt5_tick, mt5_account, open_positions, sil_scores
            ) if self._era and dce_r.get("action") != "HOLD" else self._hold_era()

            decision_entropy = dce_r.get("decision_entropy", 1.0)
            authorized = tamk_r.get("authorized", False)
            pipeline_depth = 2
            opp_density = loef_r.get("opportunity_density", 0.0)
            reality_score = era_r.get("reality_alignment_score", 0.0) if era_r else 0.0

            exec_prob = self._compute_exec_prob(authorized, dce_r, era_r, loef_r)

            return {
                "decision_entropy": decision_entropy,
                "authorization_status": "ALLOW" if authorized else "BLOCK",
                "pipeline_depth": pipeline_depth,
                "opportunity_density_peak": opp_density,
                "reality_alignment_score": reality_score,
                "execution_probability": exec_prob,
                "details": {
                    "dce": dce_r,
                    "tamk": tamk_r,
                    "loef": loef_r,
                    "era": era_r,
                },
            }
        except Exception as exc:
            logger.error("Emancipation evaluation failed: %s", exc, exc_info=True)
            return {"error": str(exc), "execution_probability": 0.0}

    def _hold_era(self) -> dict:
        return {
            "valid": False,
            "rejection_reason": "No action to validate (HOLD)",
            "reality_alignment_score": 0.0,
        }

    def _compute_exec_prob(self, authorized: bool, dce: dict, era: dict, loef: dict) -> float:
        if not authorized:
            return 0.0
        dce_conf = dce.get("confidence", 0.0)
        era_score = era.get("reality_alignment_score", 0.0) if era else 0.0
        loef_peak = loef.get("opportunity_density", 0.0)
        return round((dce_conf * 0.4 + era_score * 0.35 + loef_peak * 0.25), 4)
