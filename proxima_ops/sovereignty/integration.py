"""EXECUTION_SOVEREIGNTY_PROFILE — Batch 9 Integration.

Combines all 5 sovereignty modules into unified execution profile.
"""

import json
import logging

logger = logging.getLogger("proxima_ops.sovereignty.integration")

try:
    from .ses import SingleExecutionSovereign
    from .eacl import ExecutionArbitrationCollapse
    from .ecl import ExecutionCommitmentLock
    from .awns import AuthorityWeightNormalization
    from .efk import ExecutionFinalityKernel
    _HAS_MODULES = True
except ImportError as e:
    _HAS_MODULES = False
    logger.warning("Some sovereignty modules unavailable: %s", e)


class ExecutionSovereigntyProfile:
    """Unified profile combining all 5 Batch 9 sovereignty modules."""

    def __init__(self, mt5_connector=None):
        self._ses = SingleExecutionSovereign() if _HAS_MODULES else None
        self._eacl = ExecutionArbitrationCollapse() if _HAS_MODULES else None
        self._ecl = ExecutionCommitmentLock() if _HAS_MODULES else None
        self._awns = AuthorityWeightNormalization() if _HAS_MODULES else None
        self._efk = ExecutionFinalityKernel(mt5_connector) if _HAS_MODULES else None

    def execute_cycle(self, signal: dict, mt5_tick: dict, mt5_account: dict,
                      open_positions: list, sil_scores: dict, rsi_dict: dict,
                      activation: dict, readiness: dict, governor_state: str,
                      cb_triggered: bool, cb_latch_cycles: int,
                      confirm_counts: dict, signals: list,
                      dce_confidence: float = 0.0, erf: float = 0.0,
                      loef_density: float = 0.0, gmci_score: float = 0.0,
                      aeem_escape: float = 0.0, rfg: float = 0.0,
                      eprg_reachability: float = 0.0,
                      best_signal: dict = None) -> dict:
        try:
            all_signals = signals or ([signal] if signal else [])
            dce_r = self._ses if False else {}
            dce_action = "HOLD"
            dce_confidence_val = 0.0

            if self._ses and all_signals:
                dce_r = {"action": signal.get("direction", "HOLD") if signal else "HOLD",
                         "symbol": signal.get("symbol") if signal else None,
                         "confidence": signal.get("confidence", 0.0) if signal else 0.0}
                dce_action = dce_r.get("action", "HOLD")
                dce_confidence_val = dce_r.get("confidence", 0.0)

            tamk_r = self._ses.evaluate(
                {"action": dce_action, "symbol": dce_r.get("symbol"),
                 "confidence": dce_confidence_val, "action_value": dce_confidence_val},
                {"valid": True, "reality_alignment_score": 0.8,
                 "adjusted_price": mt5_tick.get("ask" if dce_action == "BUY" else "bid") if mt5_tick else None,
                 "adjusted_volume": 0.01},
                {"authorized": not cb_triggered, "override_active": cb_latch_cycles > 100},
                {"opportunity_density": loef_density, "top_k_symbols": []},
                signal, mt5_tick
            ) if self._ses else {}

            tamk_bool = not cb_triggered
            era_r = {"valid": True, "reality_alignment_score": 0.8,
                     "adjusted_price": mt5_tick.get("ask" if dce_action == "BUY" else "bid") if mt5_tick else None,
                     "adjusted_volume": 0.01}

            awns_r = self._awns.normalize(
                dce_confidence_val, erf, loef_density, gmci_score,
                aeem_escape, rfg, eprg_reachability, tamk_bool
            ) if self._awns else {}

            ecl_state = self._ecl.get_lock_state() if self._ecl else {"locked": False}

            commit_r = self._ecl.commit(
                tamk_r.get("order_params"), tamk_r
            ) if self._ecl and tamk_r.get("emit_order") else {}

            eacl_r = self._eacl.resolve(
                dce_r, tamk_r, era_r,
                {"opportunity_density": loef_density},
                {}, tamk_r
            ) if self._eacl else {}

            efk_r = self._efk.finalize(
                tamk_r, self._ecl.get_lock_state() if self._ecl else {"locked": False},
                awns_r, mt5_tick, signal
            ) if self._efk else {}

            if efk_r and efk_r.get("order_emitted"):
                self._ecl.release()

            afi = self._compute_afi(tamk_r)
            entropy = awns_r.get("authority_entropy", 1.0)
            convergence = eacl_r.get("arbitration_loops", 0)
            finality = 1.0 if efk_r.get("pipeline_terminated") else 0.0
            commit_ok = 1.0 if commit_r.get("locked") else 0.0

            return {
                "authority_fragmentation_index": afi,
                "arbitration_convergence_steps": convergence,
                "commit_lock_integrity": commit_ok,
                "normalized_authority_entropy": entropy,
                "execution_finality_rate": finality,
                "mt5_order_emitted": efk_r.get("order_emitted", False),
                "mt5_ticket": efk_r.get("ticket"),
                "order_params": tamk_r.get("order_params"),
                "ses_verdict": tamk_r.get("emit_order", False),
                "efk_result": efk_r,
            }
        except Exception as exc:
            logger.error("Sovereignty cycle failed: %s", exc, exc_info=True)
            return {"error": str(exc), "mt5_order_emitted": False}

    def _compute_afi(self, ses_r: dict) -> float:
        if not ses_r:
            return 1.0
        chain = ses_r.get("authority_chain", [])
        if not chain:
            return 1.0
        allow_count = sum(1 for c in chain if "ALLOW" in c)
        return 1.0 - (allow_count / len(chain))
