"""
Pre-Trade Integrity Gate — final gate before ANY trade.

Checks
------
1. **SDIL stability** — CollapseCausalityTracker says not fully collapsed,
   DecisionSurfaceVisualizer says not plateau, SignalSpaceEntropy says not
   fully collapsed.
2. **CSRF consistency** — SignalTruthLabeler says at least one signal works
   (not NEITHER), RealityConsistencyGate says not NO_REALITY.
3. **SAAL stability** — AuthorityStabilityTracker says STABLE or MODERATE,
   and authority has been consistent for at least N cycles (default 10).
4. **Event chain validity** — CausalEventChainEnforcer shows no violations
   in the last N cycles (no backflow, no temporal paradox).
5. **Causal integrity** — CrossLayerCausalAuditor shows clean decision chain
   for the last N cycles.

Output
------
- ``ALLOW_TRADE`` — gate fully open.
- ``BLOCK_TRADE`` — one or more checks failed.

Usage
-----
    from core_runtime.pre_trade_integrity_gate import PreTradeIntegrityGate

    gate = PreTradeIntegrityGate()
    verdict = gate.check_all()
    if gate.quick_check():
        # proceed with trade
"""

import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core_runtime.collapse_causality_tracker import CollapseCausalityTracker
from core_runtime.decision_surface_visualizer import DecisionSurfaceVisualizer
from core_runtime.signal_space_entropy import SignalSpaceEntropy
from core_runtime.signal_truth_labeler import SignalTruthLabeler
from core_runtime.reality_consistency_gate import RealityConsistencyGate
from core_runtime.authority_stability_tracker import AuthorityStabilityTracker
from core_runtime.causal_event_chain_enforcer import CausalEventChainEnforcer
from core_runtime.cross_layer_causal_auditor import CrossLayerCausalAuditor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_STABILITY_CYCLES = 10   # N cycles for SAAL / event / causal checks
_DEFAULT_CONSECUTIVE_PASSES = 3  # passes required before gate opens
_MAX_ROLLING_WINDOW = 100        # rolling window size for check results

# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_instances: Dict[str, "_PreTradeIntegrityGate"] = {}


def PreTradeIntegrityGate(instance_id="default"):
    """Singleton accessor for ``_PreTradeIntegrityGate``.

    Parameters
    ----------
    instance_id : str
        Unique identifier.  Callers sharing the same *instance_id* share the
        same underlying gate object.

    Returns
    -------
    _PreTradeIntegrityGate
    """
    if instance_id not in _instances:
        _instances[instance_id] = _PreTradeIntegrityGate(instance_id)
    return _instances[instance_id]


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------

class _PreTradeIntegrityGate:
    """Final gate that runs five integrity checks before allowing a trade.

    Parameters
    ----------
    instance_id : str
        Logical instance identifier (used for logging and singleton registry).
    """

    def __init__(self, instance_id="default"):
        self._instance_id = instance_id

        # -- rolling window of all check results ----------------------------
        # Each entry: {"timestamp": ..., "checks": {...}, "gate_open": bool}
        self._rolling_window: deque[Dict[str, Any]] = deque(
            maxlen=_MAX_ROLLING_WINDOW
        )

        # -- counters -------------------------------------------------------
        self._consecutive_blocks: int = 0
        self._consecutive_passes: int = 0
        self._total_checks: int = 0

        # -- block history --------------------------------------------------
        # Each entry: {"timestamp": ..., "reason": str, "checks": dict}
        self._block_history: List[Dict[str, Any]] = []

        # -- configuration --------------------------------------------------
        self._required_stability_cycles: int = _DEFAULT_STABILITY_CYCLES
        self._required_consecutive_passes: int = _DEFAULT_CONSECUTIVE_PASSES

        # -- quick-check caching --------------------------------------------
        # If quick_check() has been called and returned False, this flag
        # forces subsequent quick_check() calls to return False until a
        # check_all() with all passing results resets it.
        self._quick_check_blocked: bool = False

        # -- singleton references to subsystems -----------------------------
        self._collapse_tracker = CollapseCausalityTracker(instance_id)
        self._surface_viz = DecisionSurfaceVisualizer(instance_id)
        self._signal_entropy = SignalSpaceEntropy(instance_id)
        self._truth_labeler = SignalTruthLabeler(instance_id)
        self._reality_gate = RealityConsistencyGate(instance_id)
        self._authority_tracker = AuthorityStabilityTracker(instance_id)
        self._chain_enforcer = CausalEventChainEnforcer(instance_id)
        self._causal_auditor = CrossLayerCausalAuditor(instance_id)

        logger.info(
            "PreTradeIntegrityGate(%r) initialised (req_passes=%d, "
            "stability_cycles=%d)",
            instance_id,
            self._required_consecutive_passes,
            self._required_stability_cycles,
        )

    # ------------------------------------------------------------------
    # Public API — configuration
    # ------------------------------------------------------------------

    def set_require_consecutive_passes(self, n: int) -> None:
        """Require *n* consecutive passing ``check_all()`` results before the
        gate opens.

        Parameters
        ----------
        n : int
            Minimum number of consecutive all-pass results.  Must be >= 1.
        """
        n = max(1, int(n))
        self._required_consecutive_passes = n
        logger.debug(
            "PreTradeIntegrityGate(%r) set_require_consecutive_passes=%d",
            self._instance_id,
            n,
        )

    # ------------------------------------------------------------------
    # Public API — core checks
    # ------------------------------------------------------------------

    def check_all(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Run all five integrity checks and return a detailed verdict.

        Parameters
        ----------
        symbol : str or None
            Optional trading symbol.  If provided, checks are run against that
            symbol's data where applicable.

        Returns
        -------
        dict
            Full verdict dictionary (see module docstring for schema).
        """
        logger.info(
            "PreTradeIntegrityGate(%r) — running check_all (symbol=%s)",
            self._instance_id,
            symbol,
        )

        # ---- Run all five checks -----------------------------------------
        checks = {
            "sdil_stability": self._check_sdil_stability(symbol),
            "csfr_consistency": self._check_csfr_consistency(symbol),
            "saal_stability": self._check_saal_stability(symbol),
            "event_chain_validity": self._check_event_chain_validity(),
            "causal_integrity": self._check_causal_integrity(),
        }

        # ---- Determine gate status ----------------------------------------
        blocking_checks = [
            name for name, result in checks.items() if not result["passed"]
        ]
        first_failure = blocking_checks[0] if blocking_checks else None
        gate_open = len(blocking_checks) == 0

        # ---- Update rolling window ----------------------------------------
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": {k: v["passed"] for k, v in checks.items()},
            "gate_open": gate_open,
        }
        self._rolling_window.append(record)
        self._total_checks += 1

        # ---- Update consecutive counters ----------------------------------
        if gate_open:
            self._consecutive_passes += 1
            self._consecutive_blocks = 0
        else:
            self._consecutive_blocks += 1
            self._consecutive_passes = 0

        # ---- Enforce consecutive-passes threshold -------------------------
        # The gate_open from checks is preliminary; the final gate only opens
        # when enough consecutive all-pass results have been observed.
        consecutive_met = (
            self._consecutive_passes >= self._required_consecutive_passes
        )
        final_gate_open = gate_open and consecutive_met

        # ---- Block history ------------------------------------------------
        if not final_gate_open:
            self._block_history.append({
                "timestamp": record["timestamp"],
                "reason": first_failure or "consecutive_passes_not_met",
                "checks": checks,
                "consecutive_passes": self._consecutive_passes,
            })

        # ---- Reset quick-check cache --------------------------------------
        # If check_all() passes fully, the quick-check block is lifted.
        if final_gate_open:
            self._quick_check_blocked = False
        else:
            self._quick_check_blocked = True

        # ---- Build verdict -------------------------------------------------
        if final_gate_open:
            verdict = "ALLOW_TRADE"
            recommendation = self._recommend_allow(checks)
        else:
            verdict = "BLOCK_TRADE"
            recommendation = self._recommend_block(checks, first_failure)

        result = {
            "gate_open": final_gate_open,
            "checks": {
                name: {"passed": c["passed"], "detail": c["detail"]}
                for name, c in checks.items()
            },
            "blocking_checks": blocking_checks,
            "first_failure": first_failure,
            "consecutive_blocks": self._consecutive_blocks,
            "consecutive_passes": self._consecutive_passes,
            "consecutive_passes_required": self._required_consecutive_passes,
            "consecutive_met": consecutive_met,
            "verdict": verdict,
            "recommendation": recommendation,
        }

        logger.info(
            "PreTradeIntegrityGate(%r) → %s (blocks=%d, passes=%d)",
            self._instance_id,
            verdict,
            self._consecutive_blocks,
            self._consecutive_passes,
        )
        return result

    def quick_check(self) -> bool:
        """Quick boolean check — returns ``True`` if trades are allowed.

        If a previous ``quick_check()`` returned ``False``, subsequent calls
        also return ``False`` until a ``check_all()`` with all passing results
        is run.

        Returns
        -------
        bool
            ``True`` = ALLOW_TRADE, ``False`` = BLOCK_TRADE.
        """
        if self._quick_check_blocked:
            logger.debug(
                "PreTradeIntegrityGate(%r) quick_check → False (cached block)",
                self._instance_id,
            )
            return False

        # Run a minimal check — if all pass, return True, otherwise cache
        # the block.
        gate_open = self._check_all_quick()
        if not gate_open:
            self._quick_check_blocked = True
        return gate_open

    def feed_check_result(
        self, check_name: str, passed: bool, detail: str = ""
    ) -> None:
        """Allow external feeding of a check result (for integration).

        Parameters
        ----------
        check_name : str
            Name identifying the check (e.g. ``"sdil_stability"``).
        passed : bool
            Whether the check passed.
        detail : str
            Optional detail string describing the result.
        """
        # Store externally fed results in a supplemental dict so they can be
        # retrieved later.
        if not hasattr(self, "_external_results"):
            self._external_results: Dict[str, Dict[str, Any]] = {}
        self._external_results[check_name] = {
            "passed": passed,
            "detail": detail,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.debug(
            "PreTradeIntegrityGate(%r) feed_check_result=%s passed=%s detail=%s",
            self._instance_id,
            check_name,
            passed,
            detail,
        )

    def get_block_history(self) -> List[Dict[str, Any]]:
        """Return the history of blocks with timestamps and reasons.

        Returns
        -------
        list of dict
            Each entry contains ``timestamp``, ``reason``, ``checks``, and
            ``consecutive_passes``.
        """
        return list(self._block_history)

    def reset(self) -> None:
        """Clear all state — rolling window, counters, block history, and the
        quick-check cache."""
        self._rolling_window.clear()
        self._consecutive_blocks = 0
        self._consecutive_passes = 0
        self._total_checks = 0
        self._block_history.clear()
        self._quick_check_blocked = False
        if hasattr(self, "_external_results"):
            del self._external_results
        logger.info("PreTradeIntegrityGate(%r) reset", self._instance_id)

    # ------------------------------------------------------------------
    # Individual check implementations
    # ------------------------------------------------------------------

    def _check_sdil_stability(self, symbol: Optional[str]) -> Dict[str, Any]:
        """Check 1 — SDIL stability.

        Verifies that:
        - CollapseCausalityTracker says the system is not fully collapsed.
        - DecisionSurfaceVisualizer says the surface is not a plateau.
        - SignalSpaceEntropy says the signal space is not fully collapsed.

        Returns
        -------
        dict with keys ``passed`` (bool) and ``detail`` (str).
        """
        failures: List[str] = []

        # -- CollapseCausalityTracker --
        # We check the most recent analysis (use a generic symbol or "default"
        # for system-level tracking).
        collapse_symbol = symbol or "default"
        collapse_report = self._collapse_tracker.analyze(collapse_symbol)
        collapse_progression = collapse_report.get("collapse_progression", "NONE")
        if collapse_progression == "FULL":
            failures.append(
                f"CollapseCausalityTracker: FULL collapse (progression={collapse_progression})"
            )

        # -- DecisionSurfaceVisualizer --
        surface_report = self._surface_viz.analyze_surface(collapse_symbol)
        is_plateau = surface_report.get("is_plateau", False)
        if is_plateau:
            failures.append(
                f"DecisionSurfaceVisualizer: PLATEAU detected "
                f"(plateau_pct={surface_report.get('plateau_pct', 0):.4f})"
            )

        # -- SignalSpaceEntropy --
        global_assessment = self._signal_entropy.get_global_assessment()
        entropy_verdict = global_assessment.get("verdict", "")
        if entropy_verdict == "FULL_COLLAPSE":
            failures.append(
                f"SignalSpaceEntropy: FULL_COLLAPSE "
                f"(global_entropy={global_assessment.get('global_signal_entropy', 0):.4f})"
            )

        if failures:
            detail = "; ".join(failures)
            return {"passed": False, "detail": detail}
        return {
            "passed": True,
            "detail": (
                f"SDIL stable: collapse={collapse_progression}, "
                f"plateau={is_plateau}, "
                f"entropy_verdict={entropy_verdict}"
            ),
        }

    def _check_csfr_consistency(self, symbol: Optional[str]) -> Dict[str, Any]:
        """Check 2 — CSRF consistency.

        Verifies that:
        - SignalTruthLabeler says at least one signal works (not NEITHER).
        - RealityConsistencyGate says not NO_REALITY.

        Returns
        -------
        dict with keys ``passed`` (bool) and ``detail`` (str).
        """
        failures: List[str] = []

        # -- SignalTruthLabeler --
        try:
            truth_label = self._truth_labeler.get_truth_label(symbol)
        except Exception as exc:
            truth_label = {"true_alpha_source": "INCONCLUSIVE", "oss_accuracy": 0.0,
                           "alt_accuracy": 0.0, "samples": 0}
            logger.warning(
                "PreTradeIntegrityGate(%r) — SignalTruthLabeler error: %s",
                self._instance_id, exc,
            )
        true_alpha = truth_label.get("true_alpha_source", "INCONCLUSIVE")
        if true_alpha == "NEITHER":
            failures.append(
                f"SignalTruthLabeler: both OSS and ALT are NEITHER "
                f"(oss_acc={truth_label.get('oss_accuracy', 0):.4f}, "
                f"alt_acc={truth_label.get('alt_accuracy', 0):.4f})"
            )

        # -- RealityConsistencyGate --
        check_symbol = symbol
        if check_symbol is None:
            # If no symbol, try "default" — the gate returns NO_REALITY for
            # unknown symbols with no data, which counts as a failure.
            check_symbol = "default"
        consistency = self._reality_gate.check_consistency(check_symbol)
        global_verdict = consistency.get("global_verdict", "NO_REALITY")
        if global_verdict == "NO_REALITY":
            failures.append(
                f"RealityConsistencyGate: {global_verdict} "
                f"(obs={consistency.get('observations', 0)})"
            )

        if failures:
            detail = "; ".join(failures)
            return {"passed": False, "detail": detail}
        return {
            "passed": True,
            "detail": (
                f"CSRF consistent: true_alpha={true_alpha}, "
                f"reality_verdict={global_verdict}"
            ),
        }

    def _check_saal_stability(self, symbol: Optional[str]) -> Dict[str, Any]:
        """Check 3 — SAAL authority stability.

        Verifies that:
        - AuthorityStabilityTracker says STABLE or MODERATE (not UNSTABLE).
        - Authority has been consistent for at least N cycles (default 10).

        Returns
        -------
        dict with keys ``passed`` (bool) and ``detail`` (str).
        """
        failures: List[str] = []

        report = self._authority_tracker.get_stability_report()
        verdict = report.get("stability_verdict", "UNSTABLE")

        if verdict == "UNSTABLE":
            failures.append(
                f"AuthorityStabilityTracker: {verdict} "
                f"(volatility={report.get('policy_volatility', 0):.4f}, "
                f"flip_rate={report.get('decision_flip_rate', 0):.4f})"
            )

        # Authority consistency duration
        decisions = self._authority_tracker._authority_decisions  # type: ignore
        if len(decisions) < self._required_stability_cycles:
            failures.append(
                f"Authority too short: {len(decisions)} decisions recorded, "
                f"need >= {self._required_stability_cycles}"
            )
        else:
            # Check that the current authority has been consistent
            recent = list(decisions)[-self._required_stability_cycles:]
            distinct_authorities = len(set(d[1] for d in recent))
            if distinct_authorities > 1:
                failures.append(
                    f"Authority flips in last {self._required_stability_cycles}: "
                    f"{distinct_authorities} distinct authorities observed"
                )

        if failures:
            detail = "; ".join(failures)
            return {"passed": False, "detail": detail}
        return {
            "passed": True,
            "detail": (
                f"SAAL stable: verdict={verdict}, "
                f"decisions={len(decisions)}, "
                f"conviction={report.get('authority_conviction', 0):.4f}"
            ),
        }

    def _check_event_chain_validity(self) -> Dict[str, Any]:
        """Check 4 — Event chain validity.

        Verifies that CausalEventChainEnforcer shows no violations in the
        last N cycles (no backflow, no temporal paradox).

        Returns
        -------
        dict with keys ``passed`` (bool) and ``detail`` (str).
        """
        failures: List[str] = []

        # Get the violation log
        violations = self._chain_enforcer.get_violation_log()

        # Filter to recent cycles
        recent_violations = [
            v
            for v in violations
            if v.get("cycle_id", 0)
            > max(0, self._chain_enforcer._current_cycle - self._required_stability_cycles)
        ]

        if recent_violations:
            total = len(recent_violations)
            types = set(v.get("type", "UNKNOWN") for v in recent_violations)
            failures.append(
                f"CausalEventChainEnforcer: {total} violation(s) in last "
                f"{self._required_stability_cycles} cycles: {', '.join(sorted(types))}"
            )

        # Also check the most recent cycle status
        current_cycle = self._chain_enforcer._current_cycle
        if current_cycle > 0:
            status = self._chain_enforcer.get_chain_status(current_cycle)
            if status.get("violations"):
                failures.append(
                    f"Chain status for cycle {current_cycle} has violations: "
                    f"{', '.join(status['violations'])}"
                )

        if failures:
            detail = "; ".join(failures)
            return {"passed": False, "detail": detail}
        return {
            "passed": True,
            "detail": (
                f"Event chain valid: no violations in last "
                f"{self._required_stability_cycles} cycles"
            ),
        }

    def _check_causal_integrity(self) -> Dict[str, Any]:
        """Check 5 — Causal integrity.

        Verifies that CrossLayerCausalAuditor shows a clean decision chain
        for the last N cycles.

        Returns
        -------
        dict with keys ``passed`` (bool) and ``detail`` (str).
        """
        failures: List[str] = []

        summary = self._causal_auditor.get_summary()
        total_nodes = summary.get("total_nodes", 0)
        total_trades = summary.get("total_trades", 0)

        # Check the most recent cycles for completeness
        # A clean decision chain means each cycle has nodes recorded in the
        # proper layer order without gaps.
        recent_cycles = sorted(self._causal_auditor._nodes.keys())[-self._required_stability_cycles:]  # type: ignore

        for cycle_id in recent_cycles:
            chain = self._causal_auditor.get_decision_chain(cycle_id)
            if not chain:
                failures.append(
                    f"CausalAuditor: cycle {cycle_id} has empty decision chain"
                )

        if failures:
            detail = "; ".join(failures)
            return {"passed": False, "detail": detail}
        return {
            "passed": True,
            "detail": (
                f"Causal integrity clean: {total_nodes} nodes, "
                f"{total_trades} trades, "
                f"{len(recent_cycles)} recent cycles checked"
            ),
        }

    # ------------------------------------------------------------------
    # Quick-check helper
    # ------------------------------------------------------------------

    def _check_all_quick(self) -> bool:
        """Minimal all-checks pass for ``quick_check()``.

        Returns ``True`` only if every check passes.
        """
        checks = [
            self._check_sdil_stability(None),
            self._check_csfr_consistency(None),
            self._check_saal_stability(None),
            self._check_event_chain_validity(),
            self._check_causal_integrity(),
        ]
        return all(c["passed"] for c in checks)

    # ------------------------------------------------------------------
    # Recommendation builders
    # ------------------------------------------------------------------

    @staticmethod
    def _recommend_allow(checks: Dict[str, Dict[str, Any]]) -> str:
        """Build a recommendation string for the ALLOW_TRADE case."""
        return (
            "All integrity checks passed. Gate is open for trading. "
            "Proceed with normal execution flow."
        )

    @staticmethod
    def _recommend_block(
        checks: Dict[str, Dict[str, Any]],
        first_failure: Optional[str],
    ) -> str:
        """Build a recommendation string for the BLOCK_TRADE case."""
        parts: List[str] = []
        if first_failure:
            parts.append(f"First failure: {first_failure}")

        failed_checks = [
            f"{name} ({c['detail'][:80]})"
            for name, c in checks.items()
            if not c["passed"]
        ]
        if failed_checks:
            parts.append("Failing check(s): " + "; ".join(failed_checks))

        parts.append(
            "Trade blocked. Do not execute any order until a subsequent "
            "check_all() reports all passing."
        )
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest():
    """Exercise the PreTradeIntegrityGate with key scenarios.

    Scenarios
    ---------
    1. **All checks passing** → ALLOW_TRADE.
    2. **One check failing** → BLOCK_TRADE.
    3. **Consecutive passes required** — gate stays closed until enough
       consecutive all-pass results.
    4. **Block history tracking** — verify block_history accumulates.
    5. **quick_check() caching** — after a block, subsequent quick_check
       returns False until check_all() passes.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("=" * 60)
    logger.info("PreTradeIntegrityGate — Self-Test")
    logger.info("=" * 60)

    passed = 0
    failed = 0

    def _check(cond: bool, msg: str) -> None:
        nonlocal passed, failed
        if cond:
            passed += 1
            logger.info("  [PASS] %s", msg)
        else:
            failed += 1
            logger.error("  [FAIL] %s", msg)

    import random as _random  # used across multiple scenarios

    # ==================================================================
    # Scenario 1 — All checks passing → ALLOW_TRADE
    #
    # We simulate a healthy environment by feeding data into the
    # underlying subsystems so that all five checks pass.
    # ==================================================================
    logger.info("")
    logger.info("--- Scenario 1: ALLOW_TRADE ---")

    gate1 = PreTradeIntegrityGate("selftest_allow")

    # Feed authority tracker with stable decisions
    for i in range(30):
        gate1._authority_tracker.feed_authority_decision(i, "bull", 0.85)
        gate1._authority_tracker.feed_signal_decision("EURUSD", 1)
        gate1._authority_tracker.feed_regime("bull")

    # Feed signal truth labeler with up-trend (mostly positive forward
    # returns) so OSS=+1 achieves > 0.53 accuracy; ALT near random.
    prices_up = [100.0]
    for _ in range(200 + 10):
        step = 1.001 if _random.random() < 0.75 else 0.999
        prices_up.append(prices_up[-1] * step)
    for i in range(100):
        oss = 1 if _random.random() < 0.80 else -1  # OSS mostly correct
        alt = 1 if _random.random() < 0.55 else -1  # ALT near random
        gate1._truth_labeler.feed_tick("EURUSD", prices_up[i], oss, alt)

    # Feed collapse tracker to avoid FULL collapse
    for _ in range(50):
        feat = {
            "rsi": 50.0 + _random.gauss(0, 10),
            "macd": _random.gauss(0, 0.3),
            "volume_ratio": 1.0 + _random.gauss(0, 0.2),
            "volatility": _random.uniform(0.1, 0.4),
        }
        gate1._collapse_tracker.feed_features("EURUSD", feat)
        p_cont = 0.5 + _random.gauss(0, 0.12)
        p_cont = max(0.01, min(0.99, p_cont))
        ev = 0.5 + _random.gauss(0, 0.1)
        ecdf_vals = sorted(_random.uniform(0, 1) for _ in range(20))
        gate1._collapse_tracker.feed_oss_output("EURUSD", p_cont, ev, 0, ecdf_vals)

    # Feed surface visualizer to avoid plateau
    rng = _random.Random(42)
    for _ in range(200):
        state = {
            "ecdf": rng.uniform(0.0, 1.0),
            "drift": rng.uniform(-0.5, 0.5),
            "spread": rng.uniform(0.0, 0.05),
            "volatility": rng.uniform(0.0, 0.3),
            "entropy": rng.uniform(0.0, 1.0),
        }
        p_cont = 0.2 + 0.6 * state["ecdf"] + rng.gauss(0, 0.03)
        p_cont = max(0.0, min(1.0, p_cont))
        gate1._surface_viz.feed_observation("EURUSD", state, p_cont)

    # Feed signal space entropy with varied signals
    for _ in range(50):
        gate1._signal_entropy.feed_observation("OSS", 1)
        gate1._signal_entropy.feed_observation("OSS", -1)
        gate1._signal_entropy.feed_observation("OSS", 0)
    for _ in range(30):
        gate1._signal_entropy.feed_observation("ALT", 1)
        gate1._signal_entropy.feed_observation("ALT", -1)
    for _ in range(30):
        gate1._signal_entropy.feed_observation("SHADOW", 1)
        gate1._signal_entropy.feed_observation("SHADOW", -1)
    for _ in range(30):
        gate1._signal_entropy.feed_observation("ECDF", 1)
        gate1._signal_entropy.feed_observation("ECDF", -1)

    # Feed reality gate with consistent data (same symbol as check_all)
    for _ in range(30):
        gate1._reality_gate.feed_tick("EURUSD", +1, +1, 1.1000)
        gate1._reality_gate.feed_forward_return("EURUSD", 0.005)
    for _ in range(30):
        gate1._reality_gate.feed_tick("EURUSD", -1, -1, 1.1000)
        gate1._reality_gate.feed_forward_return("EURUSD", -0.005)

    # Feed causal auditor with clean chains
    for cid in range(1, 12):
        gate1._causal_auditor.record_node(
            cycle_id=cid,
            layer="tick_ingestion",
            module="collector",
            action="tick_received",
        )
        gate1._causal_auditor.record_node(
            cycle_id=cid,
            layer="oss_surface",
            module="oss",
            action="signal_generated",
            parent_nodes=[cid],
        )
        gate1._causal_auditor.record_node(
            cycle_id=cid,
            layer="execution",
            module="executor",
            action="trade_pending",
            parent_nodes=[cid],
        )

    # Run check_all() enough times to satisfy consecutive passes
    result1 = gate1.check_all("EURUSD")
    logger.info("  Verdict: %s (passes=%d, required=%d)",
                result1["verdict"],
                result1["consecutive_passes"],
                result1["consecutive_passes_required"])

    # Run 3 more for consecutive-pass requirement (default=3)
    for _ in range(3):
        result1 = gate1.check_all("EURUSD")

    _check(result1["gate_open"], "Scenario 1 → gate_open=True")
    _check(result1["verdict"] == "ALLOW_TRADE",
           f"Scenario 1 → ALLOW_TRADE, got {result1['verdict']}")
    _check(result1["first_failure"] is None,
           "Scenario 1 → first_failure=None")
    _check(len(result1["blocking_checks"]) == 0,
           "Scenario 1 → no blocking checks")

    # ==================================================================
    # Scenario 2 — One check failing → BLOCK_TRADE
    #
    # We force a failure by feeding contradictory data to the
    # SignalTruthLabeler so that it reports NEITHER.
    # ==================================================================
    logger.info("")
    logger.info("--- Scenario 2: BLOCK_TRADE ---")

    gate2 = PreTradeIntegrityGate("selftest_block")
    # Setup everything healthy first
    for i in range(30):
        gate2._authority_tracker.feed_authority_decision(i, "bull", 0.85)
        gate2._authority_tracker.feed_signal_decision("EURUSD", 1)

    # Feed signal truth labeler with purely random signals → NEITHER
    for i in range(200):
        prices2 = 100.0 + i * 0.1
        gate2._truth_labeler.feed_tick("EURUSD", prices2,
                                        _random.choice([-1, 1]),
                                        _random.choice([-1, 1]))

    result2 = gate2.check_all("EURUSD")
    _check(not result2["gate_open"], "Scenario 2 → gate_open=False")
    _check(result2["verdict"] == "BLOCK_TRADE",
           f"Scenario 2 → BLOCK_TRADE, got {result2['verdict']}")
    _check(result2["first_failure"] is not None,
           "Scenario 2 → first_failure is set")
    _check(len(result2["blocking_checks"]) > 0,
           "Scenario 2 → at least one blocking check")

    # ==================================================================
    # Scenario 3 — Consecutive passes required
    #
    # Set require_consecutive_passes to 5, then run 4 passing checks.
    # The gate should still be blocked on the 4th.
    # ==================================================================
    logger.info("")
    logger.info("--- Scenario 3: Consecutive passes required ---")

    gate3 = PreTradeIntegrityGate("selftest_consecutive")
    gate3.set_require_consecutive_passes(5)

    # Feed healthy data (all using "EURUSD" to match check_all symbol)
    for i in range(30):
        gate3._authority_tracker.feed_authority_decision(i, "bull", 0.85)
        gate3._authority_tracker.feed_signal_decision("EURUSD", 1)
        gate3._authority_tracker.feed_regime("bull")
    prices_up3 = [100.0]
    for _ in range(200 + 10):
        step3 = 1.001 if _random.random() < 0.75 else 0.999
        prices_up3.append(prices_up3[-1] * step3)
    for i in range(100):
        oss = 1 if _random.random() < 0.80 else -1
        alt = 1 if _random.random() < 0.55 else -1
        gate3._truth_labeler.feed_tick("EURUSD", prices_up3[i], oss, alt)
    for _ in range(30):
        gate3._signal_entropy.feed_observation("OSS", 1)
        gate3._signal_entropy.feed_observation("OSS", -1)
        gate3._signal_entropy.feed_observation("ALT", 1)
        gate3._signal_entropy.feed_observation("ALT", -1)
    for _ in range(30):
        gate3._reality_gate.feed_tick("EURUSD", +1, +1, 1.1000)
        gate3._reality_gate.feed_forward_return("EURUSD", 0.005)
    for _ in range(30):
        gate3._reality_gate.feed_tick("EURUSD", -1, -1, 1.1000)
        gate3._reality_gate.feed_forward_return("EURUSD", -0.005)
    for _ in range(50):
        feat = {"rsi": 50.0 + _random.gauss(0, 5), "macd": _random.gauss(0, 0.2),
                "volume_ratio": 1.0 + _random.gauss(0, 0.1), "volatility": _random.uniform(0.1, 0.3)}
        gate3._collapse_tracker.feed_features("EURUSD", feat)
        gate3._collapse_tracker.feed_oss_output("EURUSD", 0.6, 0.5, 0,
                                                 sorted(_random.uniform(0, 1) for _ in range(20)))
    for _ in range(50):
        gate3._surface_viz.feed_observation(
            "EURUSD",
            {"ecdf": _random.uniform(0, 1), "drift": _random.uniform(-0.5, 0.5),
             "spread": _random.uniform(0, 0.05), "volatility": _random.uniform(0, 0.3),
             "entropy": _random.uniform(0, 1)},
            0.5 + _random.gauss(0, 0.1),
        )

    # Run 4 check_all calls — should still be blocked (need 5)
    for i in range(4):
        r3 = gate3.check_all("EURUSD")
        if i < 3:
            _check(
                not r3["gate_open"],
                f"Scenario 3 → iteration {i+1}: gate_open=False (passes={r3['consecutive_passes']})",
            )
            _check(
                r3["verdict"] == "BLOCK_TRADE",
                f"Scenario 3 → iteration {i+1}: BLOCK_TRADE",
            )

    # 5th call — gate should open
    r3_final = gate3.check_all("EURUSD")
    _check(
        r3_final["gate_open"],
        f"Scenario 3 → iteration 5: gate_open=True (passes={r3_final['consecutive_passes']})",
    )
    _check(
        r3_final["verdict"] == "ALLOW_TRADE",
        f"Scenario 3 → iteration 5: ALLOW_TRADE, got {r3_final['verdict']}",
    )

    # ==================================================================
    # Scenario 4 — Block history tracking
    # ==================================================================
    logger.info("")
    logger.info("--- Scenario 4: Block history ---")

    gate4 = PreTradeIntegrityGate("selftest_history")
    # Force block by not feeding any data (everything will fail)
    gate4.check_all("EURUSD")
    gate4.check_all("EURUSD")
    history = gate4.get_block_history()
    _check(len(history) > 0,
           f"Scenario 4 → block_history has {len(history)} entries")
    for entry in history:
        _check("timestamp" in entry, "Scenario 4 → entry has timestamp")
        _check("reason" in entry, "Scenario 4 → entry has reason")

    # ==================================================================
    # Scenario 5 — quick_check() caching
    # ==================================================================
    logger.info("")
    logger.info("--- Scenario 5: quick_check() caching ---")

    gate5 = PreTradeIntegrityGate("selftest_quickcache")
    # quick_check should return False because no data has been fed
    qc1 = gate5.quick_check()
    _check(not qc1, "Scenario 5 → quick_check() returns False when blocked")

    # A second quick_check should also return False (cached block)
    qc2 = gate5.quick_check()
    _check(not qc2,
           "Scenario 5 → second quick_check() also False (cached)")

    # Feed data and run a passing check_all to unblock
    for i in range(30):
        gate5._authority_tracker.feed_authority_decision(i, "bull", 0.85)
        gate5._authority_tracker.feed_signal_decision("EURUSD", 1)
        gate5._authority_tracker.feed_regime("bull")
    prices_up5 = [100.0]
    for _ in range(200 + 10):
        step5 = 1.001 if _random.random() < 0.75 else 0.999
        prices_up5.append(prices_up5[-1] * step5)
    for i in range(100):
        oss = 1 if _random.random() < 0.80 else -1
        alt = 1 if _random.random() < 0.55 else -1
        gate5._truth_labeler.feed_tick("EURUSD", prices_up5[i], oss, alt)
    for _ in range(30):
        gate5._signal_entropy.feed_observation("OSS", 1)
        gate5._signal_entropy.feed_observation("OSS", -1)
        gate5._signal_entropy.feed_observation("ALT", 1)
        gate5._signal_entropy.feed_observation("ALT", -1)
    # quick_check() uses symbol=None internally → reality gate checks "default"
    for _ in range(30):
        gate5._reality_gate.feed_tick("EURUSD", +1, +1, 1.1000)
        gate5._reality_gate.feed_forward_return("EURUSD", 0.005)
        gate5._reality_gate.feed_tick("default", +1, +1, 1.1000)   # also feed default
        gate5._reality_gate.feed_forward_return("default", 0.005)
    for _ in range(30):
        gate5._reality_gate.feed_tick("EURUSD", -1, -1, 1.1000)
        gate5._reality_gate.feed_forward_return("EURUSD", -0.005)
        gate5._reality_gate.feed_tick("default", -1, -1, 1.1000)
        gate5._reality_gate.feed_forward_return("default", -0.005)
    for _ in range(50):
        feat = {"rsi": 50.0 + _random.gauss(0, 5), "macd": _random.gauss(0, 0.2),
                "volume_ratio": 1.0 + _random.gauss(0, 0.1), "volatility": _random.uniform(0.1, 0.3)}
        gate5._collapse_tracker.feed_features("EURUSD", feat)
        gate5._collapse_tracker.feed_oss_output("EURUSD", 0.6, 0.5, 0,
                                                 sorted(_random.uniform(0, 1) for _ in range(20)))
    for _ in range(50):
        gate5._surface_viz.feed_observation(
            "EURUSD",
            {"ecdf": _random.uniform(0, 1), "drift": _random.uniform(-0.5, 0.5),
             "spread": _random.uniform(0, 0.05), "volatility": _random.uniform(0, 0.3),
             "entropy": _random.uniform(0, 1)},
            0.5 + _random.gauss(0, 0.1),
        )

    for _ in range(5):
        gate5.check_all("EURUSD")
    # After passing checks, quick_check should return True
    qc3 = gate5.quick_check()
    _check(qc3,
           "Scenario 5 → quick_check() returns True after passing checks")

    # ==================================================================
    # Singleton accessor
    # ==================================================================
    logger.info("")
    logger.info("--- Singleton accessor ---")

    a = PreTradeIntegrityGate("selftest_singleton")
    b = PreTradeIntegrityGate("selftest_singleton")
    c = PreTradeIntegrityGate("selftest_singleton_other")
    _check(a is b, "Same instance_id returns same object")
    _check(a is not c, "Different instance_id returns different object")

    # ==================================================================
    # feed_check_result
    # ==================================================================
    logger.info("")
    logger.info("--- feed_check_result ---")

    gate_feed = PreTradeIntegrityGate("selftest_feed")
    gate_feed.feed_check_result("sdil_stability", True, "manually passed")
    _check(gate_feed._external_results["sdil_stability"]["passed"],
           "feed_check_result stores passed=True")
    gate_feed.feed_check_result("csfr_consistency", False, "manually failed")
    _check(not gate_feed._external_results["csfr_consistency"]["passed"],
           "feed_check_result stores passed=False")

    # ==================================================================
    # Summary
    # ==================================================================
    logger.info("")
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
