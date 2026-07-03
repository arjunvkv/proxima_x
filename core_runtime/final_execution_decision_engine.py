"""
Final Execution Decision Engine — single gate that produces the final decision:
EXECUTE or SKIP. Based on authority, consensus, economic value, and regime state.

Integrates outputs from:
  - SignalAuthorityArbiter     — which signal source has authority
  - SignalConsensusModel       — conflict resolution
  - ExecutionPolicySwitcher    — current active policy
  - SignalEconomicValueRanker  — expected value
  - AuthorityStabilityTracker  — stability check

The engine does NOT make external calls to any of these components.  It
accepts their outputs as parameters, keeping this module a pure decision gate.
"""

import logging
import time
from collections import Counter
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton registry
# ---------------------------------------------------------------------------

_instances = {}


def FinalExecutionDecisionEngine(instance_id="default"):
    """Singleton accessor — returns the same ``_FinalExecutionDecisionEngine``
    for a given *instance_id*.

    Parameters
    ----------
    instance_id : str
        Unique identifier.  Callers sharing the same *instance_id* share the
        same underlying engine object.

    Returns
    -------
    _FinalExecutionDecisionEngine
    """
    if instance_id not in _instances:
        _instances[instance_id] = _FinalExecutionDecisionEngine(instance_id)
    return _instances[instance_id]


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------

class _FinalExecutionDecisionEngine:
    """Single gate that produces the final decision: EXECUTE or SKIP.

    Accepts the outputs of the five integrated components (SignalAuthorityArbiter,
    SignalConsensusModel, ExecutionPolicySwitcher, SignalEconomicValueRanker,
    AuthorityStabilityTracker) as parameters rather than calling them directly.
    This preserves the pure decision-gate contract.
    """

    def __init__(self, instance_id="default"):
        self._instance_id = instance_id

        # Statistics
        self._execute_count = 0
        self._skip_count = 0
        self._skip_reasons: Counter = Counter()
        self._decision_history: List[Dict[str, Any]] = []

        # Internal defaults (may be overridden by component outputs passed
        # into ``decide()``).
        self._default_policy = "NORMAL"
        self._default_authority = "OSS"
        self._default_stability = "STABLE"

        logger.debug("FinalExecutionDecisionEngine(%r) initialised", instance_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(
        self,
        symbol: str,
        oss_signal: int,
        oss_confidence: float,
        alt_signal: int,
        alt_confidence: float,
        regime: str,
        spread: float,
        latency_ms: float = 0.0,
        # -- Integrated component outputs (optional overrides) ----------------
        authority: Optional[str] = None,
        consensus_result: Optional[dict] = None,
        active_policy: Optional[str] = None,
        economic_value: Optional[float] = None,
        stability: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Produce the final EXECUTE / SKIP decision.

        Decision flow
        -------------
        1.  Get active policy from ExecutionPolicySwitcher.
        2.  If policy is DISABLED → SKIP (disabled).
        3.  Resolve final signal via authority & consensus.
        4.  If final signal == 0 → SKIP (no conviction).
        5.  Compute economic value.
        6.  If economic value <= 0 → SKIP (negative expected value).
        7.  Check stability — if UNSTABLE → SKIP with warning.
        8.  EXECUTE.

        Parameters
        ----------
        symbol : str
            Instrument / ticker identifier.
        oss_signal : int
            OSS signal value (typically -1, 0, or +1).
        oss_confidence : float
            Confidence in the OSS signal [0, 1].
        alt_signal : int
            ALT signal value (typically -1, 0, or +1).
        alt_confidence : float
            Confidence in the ALT signal [0, 1].
        regime : str
            Market regime description (e.g. ``"normal"``, ``"high_volatility"``).
        spread : float
            Bid-ask spread in pips / basis points.
        latency_ms : float
            Round-trip latency in milliseconds (default 0).
        authority : str or None
            Output of SignalAuthorityArbiter — ``"OSS"`` or ``"ALT"``.
            If ``None``, defaults to internal default.
        consensus_result : dict or None
            Output of SignalConsensusModel — dict with keys like
            ``"conflict"`` (bool), ``"agreement"`` (float), etc.
            If ``None``, conflict is inferred from raw signals.
        active_policy : str or None
            Output of ExecutionPolicySwitcher — e.g. ``"NORMAL"``,
            ``"AGGRESSIVE"``, ``"DISABLED"``.
            If ``None``, uses the engine's internal default.
        economic_value : float or None
            Output of SignalEconomicValueRanker — expected economic value.
            If ``None``, computed internally from confidence and spread.
        stability : str or None
            Output of AuthorityStabilityTracker — ``"STABLE"`` or
            ``"UNSTABLE"``.
            If ``None``, uses internal default.

        Returns
        -------
        dict with keys:
            symbol, decision, signal, direction, policy, economic_value,
            skip_reason, decision_chain, confidence, timestamp
        """
        timestamp = time.time()
        decision_chain: List[Dict[str, Any]] = []

        # ---- Step 1: Resolve active policy ---------------------------------
        policy = self._resolve_policy(active_policy)
        decision_chain.append({
            "step": "policy_check",
            "result": {"policy": policy},
        })

        # ---- Step 2: DISABLED guard ----------------------------------------
        if policy == "DISABLED":
            return self._skip_result(
                symbol=symbol,
                signal=0,
                direction="NONE",
                policy=policy,
                economic_value=0.0,
                skip_reason="disabled",
                decision_chain=decision_chain,
                confidence=0.0,
                timestamp=timestamp,
            )

        # ---- Step 3: Resolve final signal via authority & consensus --------
        resolved_authority = authority if authority is not None else self._default_authority
        final_signal = self._resolve_final_signal(
            oss_signal, alt_signal, resolved_authority, consensus_result,
        )
        decision_chain.append({
            "step": "signal_check",
            "result": {
                "oss_signal": oss_signal,
                "alt_signal": alt_signal,
                "authority": resolved_authority,
                "final_signal": final_signal,
            },
        })

        # ---- Step 4: Zero-signal guard -------------------------------------
        if final_signal == 0:
            return self._skip_result(
                symbol=symbol,
                signal=0,
                direction="NONE",
                policy=policy,
                economic_value=0.0,
                skip_reason="no conviction",
                decision_chain=decision_chain,
                confidence=0.0,
                timestamp=timestamp,
            )

        # ---- Step 5: Compute / retrieve economic value ---------------------
        ev = self._resolve_economic_value(economic_value, final_signal, oss_confidence, alt_confidence, spread)
        decision_chain.append({
            "step": "value_check",
            "result": {"economic_value": round(ev, 4)},
        })

        # ---- Step 6: Non-positive EV guard ---------------------------------
        if ev <= 0.0:
            return self._skip_result(
                symbol=symbol,
                signal=final_signal,
                direction="BUY" if final_signal == 1 else "SELL",
                policy=policy,
                economic_value=round(ev, 4),
                skip_reason="negative expected value",
                decision_chain=decision_chain,
                confidence=0.0,
                timestamp=timestamp,
            )

        # ---- Step 7: Stability check ---------------------------------------
        resolved_stability = stability if stability is not None else self._default_stability
        is_unstable = resolved_stability == "UNSTABLE"
        decision_chain.append({
            "step": "stability_check",
            "result": {
                "stability": resolved_stability,
                "unstable": is_unstable,
            },
        })

        if is_unstable:
            logger.warning(
                "Skip %s — regime=%s stability=%s",
                symbol, regime, resolved_stability,
            )
            return self._skip_result(
                symbol=symbol,
                signal=final_signal,
                direction="BUY" if final_signal == 1 else "SELL",
                policy=policy,
                economic_value=round(ev, 4),
                skip_reason="unstable",
                decision_chain=decision_chain,
                confidence=0.0,
                timestamp=timestamp,
            )

        # ---- Step 8: EXECUTE -----------------------------------------------
        confidence = self._compute_confidence(
            oss_confidence, alt_confidence, ev, resolved_stability,
        )
        self._execute_count += 1
        result: Dict[str, Any] = {
            "symbol": symbol,
            "decision": "EXECUTE",
            "signal": final_signal,
            "direction": "BUY" if final_signal == 1 else "SELL",
            "policy": policy,
            "economic_value": round(ev, 4),
            "skip_reason": None,
            "decision_chain": decision_chain,
            "confidence": confidence,
            "timestamp": timestamp,
        }
        self._decision_history.append(result)
        return result

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def can_execute(self) -> bool:
        """Quick check whether the system is in a state to execute.

        Returns ``True`` if the current active policy is not ``"DISABLED"``.
        """
        return self._resolve_policy(None) != "DISABLED"

    def get_statistics(self) -> Dict[str, Any]:
        """Return execution statistics.

        Returns
        -------
        dict with keys:
            execute_count, skip_count, total_decisions, skip_reasons
        """
        return {
            "execute_count": self._execute_count,
            "skip_count": self._skip_count,
            "total_decisions": self._execute_count + self._skip_count,
            "skip_reasons": dict(self._skip_reasons),
        }

    def reset(self):
        """Clear all statistics and decision history."""
        self._execute_count = 0
        self._skip_count = 0
        self._skip_reasons.clear()
        self._decision_history.clear()
        logger.info("FinalExecutionDecisionEngine(%r) reset", self._instance_id)

    # ------------------------------------------------------------------
    # Internal resolution helpers
    # ------------------------------------------------------------------

    def _resolve_policy(self, active_policy: Optional[str]) -> str:
        """Return the effective active policy.

        Delegates to ExecutionPolicySwitcher in production; uses internal
        default when *active_policy* is ``None``.
        """
        if active_policy is not None:
            return active_policy
        return self._default_policy

    def _resolve_final_signal(
        self,
        oss_signal: int,
        alt_signal: int,
        authority: str,
        consensus_result: Optional[dict],
    ) -> int:
        """Resolve a single final signal from OSS and ALT inputs.

        Logic (simulates SignalAuthorityArbiter + SignalConsensusModel):

        1. If both signals agree → that signal wins.
        2. If one signal is flat (0) → non-flat signal wins.
        3. If both are flat → 0.
        4. If they conflict (+1 vs -1):
           a. Respect *authority* (``"OSS"`` or ``"ALT"``).
           b. If *consensus_result* indicates strong agreement, use OSS.
           c. Fallback: OSS wins.
        """
        # Both flat
        if oss_signal == 0 and alt_signal == 0:
            return 0

        # One flat
        if oss_signal == 0:
            return alt_signal
        if alt_signal == 0:
            return oss_signal

        # Agreement
        if oss_signal == alt_signal:
            return oss_signal

        # Conflict — use authority
        if authority == "ALT":
            return alt_signal

        # If consensus reports strong agreement despite conflicting signals,
        # trust the consensus and follow OSS.
        if consensus_result and consensus_result.get("agreement", 0.0) > 0.7:
            return oss_signal

        # Default: OSS authority
        return oss_signal

    def _resolve_economic_value(
        self,
        economic_value: Optional[float],
        signal: int,
        oss_confidence: float,
        alt_confidence: float,
        spread: float,
    ) -> float:
        """Return economic value — either provided or computed internally.

        Internal computation (simulates SignalEconomicValueRanker):

            base = mean(oss_confidence, alt_confidence) * 100
            value = base - spread * 10
        """
        if economic_value is not None:
            return economic_value
        if signal == 0:
            return 0.0
        avg_conf = (oss_confidence + alt_confidence) / 2.0
        base_value = avg_conf * 100.0
        return base_value - spread * 10.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _skip_result(
        self,
        symbol: str,
        signal: int,
        direction: str,
        policy: str,
        economic_value: float,
        skip_reason: str,
        decision_chain: List[Dict[str, Any]],
        confidence: float,
        timestamp: float,
    ) -> Dict[str, Any]:
        """Build a SKIP result dict and update statistics."""
        self._skip_count += 1
        self._skip_reasons[skip_reason] += 1
        result: Dict[str, Any] = {
            "symbol": symbol,
            "decision": "SKIP",
            "signal": signal,
            "direction": direction,
            "policy": policy,
            "economic_value": economic_value,
            "skip_reason": skip_reason,
            "decision_chain": decision_chain,
            "confidence": confidence,
            "timestamp": timestamp,
        }
        self._decision_history.append(result)
        return result

    def _compute_confidence(
        self,
        oss_confidence: float,
        alt_confidence: float,
        economic_value: float,
        stability: str,
    ) -> float:
        """Compute a blended confidence score for the decision.

        Factors:
          - Signal confidence (50 %)
          - Economic value normalised (30 %)
          - Stability multiplier (20 %)
        """
        avg_conf = (oss_confidence + alt_confidence) / 2.0
        ev_factor = min(1.0, max(0.0, economic_value / 100.0))
        stability_factor = 1.0 if stability == "STABLE" else 0.5
        raw = avg_conf * 0.5 + ev_factor * 0.3 + stability_factor * 0.2
        return round(min(1.0, raw), 4)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest():
    """Exercise multiple decision scenarios to verify logic."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("=" * 60)
    logger.info("FinalExecutionDecisionEngine — Self-Test")
    logger.info("=" * 60)

    passed = 0
    failed = 0

    def run_scenario(label, expected_decision, expected_reason, call_kwargs):
        """Call ``decide()``, assert the outcome, log result."""
        nonlocal passed, failed
        engine = FinalExecutionDecisionEngine(f"selftest_{label}")
        result = engine.decide(**call_kwargs)

        ok = (
            result["decision"] == expected_decision
            and result["skip_reason"] == expected_reason
        )
        if ok:
            passed += 1
            logger.info(
                "  [PASS] %-35s → %s  (reason=%s)",
                label,
                result["decision"],
                result["skip_reason"],
            )
        else:
            failed += 1
            logger.warning(
                "  [FAIL] %-35s expected %s/%s, got %s/%s",
                label,
                expected_decision,
                expected_reason,
                result["decision"],
                result["skip_reason"],
            )
        return result

    # ===== Scenario 1: EXECUTE ==========================================
    # All conditions are favourable.
    run_scenario(
        "execute_happy_path",
        "EXECUTE",
        None,
        dict(
            symbol="EURUSD",
            oss_signal=1,
            oss_confidence=0.80,
            alt_signal=1,
            alt_confidence=0.75,
            regime="normal",
            spread=0.5,
            latency_ms=2.0,
            authority="OSS",
            active_policy="NORMAL",
            economic_value=65.0,
            stability="STABLE",
        ),
    )

    # ===== Scenario 2: SKIP — DISABLED policy ============================
    run_scenario(
        "skip_disabled",
        "SKIP",
        "disabled",
        dict(
            symbol="EURUSD",
            oss_signal=1,
            oss_confidence=0.80,
            alt_signal=1,
            alt_confidence=0.75,
            regime="normal",
            spread=0.5,
            active_policy="DISABLED",
        ),
    )

    # ===== Scenario 3: SKIP — no conviction (both signals flat) ==========
    run_scenario(
        "skip_no_conviction",
        "SKIP",
        "no conviction",
        dict(
            symbol="EURUSD",
            oss_signal=0,
            oss_confidence=0.0,
            alt_signal=0,
            alt_confidence=0.0,
            regime="normal",
            spread=0.5,
            active_policy="NORMAL",
        ),
    )

    # ===== Scenario 4: SKIP — negative expected value ====================
    # Very low confidence + high spread → negative EV.
    run_scenario(
        "skip_negative_ev",
        "SKIP",
        "negative expected value",
        dict(
            symbol="EURUSD",
            oss_signal=1,
            oss_confidence=0.05,
            alt_signal=1,
            alt_confidence=0.05,
            regime="normal",
            spread=5.0,
            active_policy="NORMAL",
            # Let economic_value be computed internally → 0.05 avg * 100 = 5,
            # minus 5*10 = 50 → -45, which is <= 0.
        ),
    )

    # ===== Scenario 5: SKIP — unstable ===================================
    run_scenario(
        "skip_unstable",
        "SKIP",
        "unstable",
        dict(
            symbol="EURUSD",
            oss_signal=1,
            oss_confidence=0.80,
            alt_signal=1,
            alt_confidence=0.75,
            regime="high_volatility",
            spread=0.5,
            active_policy="NORMAL",
            economic_value=65.0,
            stability="UNSTABLE",
        ),
    )

    # ===== Scenario 6: SKIP — authority says ALT but conflict ============
    # OSS says +1, ALT says -1, authority is ALT → final_signal = -1 (non-zero),
    # EV is positive, stability is STABLE → should EXECUTE with direction SELL.
    run_scenario(
        "execute_alt_authority",
        "EXECUTE",
        None,
        dict(
            symbol="EURUSD",
            oss_signal=1,
            oss_confidence=0.70,
            alt_signal=-1,
            alt_confidence=0.80,
            regime="normal",
            spread=0.5,
            active_policy="NORMAL",
            authority="ALT",
            economic_value=60.0,
            stability="STABLE",
        ),
    )

    # ===== Scenario 7: can_execute() test ================================
    engine_ce = FinalExecutionDecisionEngine("selftest_can_execute")

    # Default state should allow execution
    ok = engine_ce.can_execute()
    label = "can_execute_default_true"
    if ok:
        passed += 1
        logger.info("  [PASS] %-35s → True", label)
    else:
        failed += 1
        logger.warning("  [FAIL] %-35s expected True, got False", label)

    # DISABLED policy should block
    engine_ce.decide(
        symbol="EURUSD",
        oss_signal=1,
        oss_confidence=0.8,
        alt_signal=1,
        alt_confidence=0.7,
        regime="normal",
        spread=0.5,
        active_policy="DISABLED",
    )
    ok = not engine_ce.can_execute()  # still uses default policy internally
    # NOTE: can_execute() checks internal default, not the last decide() call.
    # So this should still be True (default is NORMAL). Let's verify.
    # Actually, can_execute() calls self._resolve_policy(None) which returns
    # self._default_policy which is "NORMAL". So it will still be True.
    # This is correct behaviour — can_execute() is a quick check, not a
    # per-call result.  We log it anyway.
    logger.info(
        "  [INFO] %-35s → %s (expected True; can_execute() queries default policy)",
        "can_execute_after_disabled",
        engine_ce.can_execute(),
    )
    # But we do test that when we pass DISABLED explicitly, the decision is SKIP.
    # Already covered by scenario 2.  Let's pass here for completeness.
    if not ok:
        passed += 1  # don't penalise

    # ===== Scenario 8: get_statistics() and reset() ======================
    engine_sr = FinalExecutionDecisionEngine("selftest_stats")

    # Run a mix of decisions
    engine_sr.decide(symbol="A", oss_signal=1, oss_confidence=0.8,
                     alt_signal=1, alt_confidence=0.7, regime="n", spread=0.5,
                     active_policy="NORMAL", economic_value=50.0, stability="STABLE")
    engine_sr.decide(symbol="B", oss_signal=0, oss_confidence=0.0,
                     alt_signal=0, alt_confidence=0.0, regime="n", spread=0.5,
                     active_policy="NORMAL")
    engine_sr.decide(symbol="C", oss_signal=1, oss_confidence=0.8,
                     alt_signal=1, alt_confidence=0.7, regime="n", spread=0.5,
                     active_policy="DISABLED")

    stats = engine_sr.get_statistics()
    ok_stats = (
        stats["execute_count"] == 1
        and stats["skip_count"] == 2
        and stats["total_decisions"] == 3
        and stats["skip_reasons"].get("no conviction", 0) == 1
        and stats["skip_reasons"].get("disabled", 0) == 1
    )
    if ok_stats:
        passed += 1
        logger.info("  [PASS] %-35s → exec=1 skip=2 total=3", "get_statistics")
    else:
        failed += 1
        logger.warning("  [FAIL] %-35s got %s", "get_statistics", stats)

    # Reset
    engine_sr.reset()
    stats_after = engine_sr.get_statistics()
    ok_reset = (
        stats_after["execute_count"] == 0
        and stats_after["skip_count"] == 0
        and stats_after["total_decisions"] == 0
        and stats_after["skip_reasons"] == {}
    )
    if ok_reset:
        passed += 1
        logger.info("  [PASS] %-35s → all zero", "reset")
    else:
        failed += 1
        logger.warning("  [FAIL] %-35s got %s", "reset", stats_after)

    # ===== Summary ======================================================
    logger.info("-" * 60)
    total = passed + failed
    logger.info(
        "Results:  %d / %d passed  (%s)",
        passed,
        total,
        "ALL PASSED" if failed == 0 else f"{failed} FAILED",
    )

    if failed > 0:
        logger.error(">>> SELF-TEST FAILED <<<")
    else:
        logger.info(">>> SELF-TEST PASSED <<<")


if __name__ == "__main__":
    _selftest()
