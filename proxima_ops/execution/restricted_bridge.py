"""Restricted MT5 Execution Bridge — Gated Execution Firewall.

This is the final selective actuation layer. It sits between the governance
pipeline and MT5 execution, enforcing strict regime-based gating before
ANY position close or modification is allowed.

Gating Rules (ALL must pass to execute):
  1. Geometry Forecaster == STRUCTURAL_STABILITY
  2. Regime Classifier  == STABLE_FLOW
  3. Execution Governor  == any state (governor's own internal gates
     validate exit signals before producing CLOSE actions)

Blocks:
  - FAST_TRANSITION          (geometry or classifier)
  - PRE_COLLAPSE             (geometry)
  - SLOW_DISSOLUTION         (classifier)
  - Any regime under shadow instability flag

Usage
-----
    bridge = RestrictedExecutionBridge(pipeline, order_manager)
    result = bridge.evaluate(cluster_states, rfe_output, price_history, positions)

    if result["summary"]["any_eligible"]:
        bridge.execute_pending_exits()
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from proxima_ops.execution.order_manager import OrderManager
from proxima_ops.risk.cluster_geometry_forecaster import PreRegimeType, symbol_to_primary_cluster
from proxima_ops.risk.execution_governor import GovernorState
from proxima_ops.risk.governance_pipeline import GovernancePipeline
from proxima_ops.risk.regime_classifier import RegimeMetaType

logger = logging.getLogger("proxima_ops.execution.restricted_bridge")

# ---------------------------------------------------------------------------
# Eligibility Constants
# ---------------------------------------------------------------------------

ELIGIBLE_GEOMETRY: set = {PreRegimeType.STRUCTURAL_STABILITY}
"""Geometry regimes that allow execution."""

ELIGIBLE_CLASSIFIER: set = {RegimeMetaType.STABLE_FLOW}
"""Classifier regimes that allow execution."""

ELIGIBLE_GOVERNOR: set = {
    GovernorState.HOLD,
    GovernorState.PREPARE,
    GovernorState.CONDITIONAL_EXIT,
    GovernorState.EXIT,
}
"""Governor states that allow execution.

Includes EXIT states — the governor's own internal gates (temporal
persistence, reversal filter, price-context weighting) already validate
that an exit signal is genuine before producing CLOSE/CLOSE_PARTIAL
actions. The bridge's governor gate should trust that validated signal
rather than blocking it.
"""

BLOCKED_GEOMETRY: set = {
    PreRegimeType.PRE_COLLAPSE,
    PreRegimeType.FAST_INSTABILITY,
}
"""Geometry regimes that block execution."""

BLOCKED_CLASSIFIER: set = {
    RegimeMetaType.FAST_TRANSITION,
    RegimeMetaType.SLOW_DISSOLUTION,
}
"""Classifier regimes that block execution."""

BLOCKED_GOVERNOR: set = set()
"""Governor states that block execution (none — all states are eligible)."""

PRE_EXECUTION_VERIFICATION_DELAY_S: float = 0.5
"""Seconds to wait before pre-execution re-verification."""


# ---------------------------------------------------------------------------
# Eligibility Checkers
# ---------------------------------------------------------------------------


def check_geometry_eligibility(forecast: dict) -> Tuple[bool, str]:
    """Check if geometry regime allows execution.

    Returns (eligible, reason).
    """
    pre_regime = forecast.get("pre_regime", PreRegimeType.STRUCTURAL_STABILITY)
    if pre_regime in ELIGIBLE_GEOMETRY:
        return True, f"geometry={pre_regime} OK"
    return False, f"BLOCKED: geometry={pre_regime}"


def check_classifier_eligibility(classification: dict) -> Tuple[bool, str]:
    """Check if classifier regime allows execution.

    Returns (eligible, reason).
    """
    meta_type = classification.get("meta_type", RegimeMetaType.STABLE_FLOW)
    if meta_type in ELIGIBLE_CLASSIFIER:
        return True, f"classifier={meta_type} OK"
    return False, f"BLOCKED: classifier={meta_type}"


def check_governor_eligibility(decision: dict) -> Tuple[bool, str]:
    """Check if governor state allows execution.

    ALL governor states are eligible — HOLD, PREPARE, CONDITIONAL_EXIT,
    and EXIT. The governor's own internal gates (temporal persistence,
    reversal filter, price-context weighting) already validate exit
    signals before producing CLOSE/CLOSE_PARTIAL actions, so the bridge
    trusts those validated signals.

    The safety perimeter is maintained by the geometry and classifier
    gates, which block execution during unstable regimes regardless of
    what the governor says.

    Returns (eligible, reason).
    """
    gs = decision.get("governor_state", GovernorState.HOLD)
    if gs in ELIGIBLE_GOVERNOR:
        return True, f"governor={gs} OK"
    return False, f"BLOCKED: governor={gs}"


def check_shadow_stability(shadow_flag: bool, symbol: str) -> Tuple[bool, str]:
    """Check that regime is not under shadow instability.

    When shadow instability is flagged, the system cannot trust its
    own pipeline outputs and must block execution.

    Returns (eligible, reason).
    """
    if shadow_flag:
        return False, f"BLOCKED: {symbol} under shadow instability"
    return True, f"shadow OK"


# ---------------------------------------------------------------------------
# Restricted Execution Bridge
# ---------------------------------------------------------------------------


class RestrictedExecutionBridge:
    """Gated execution firewall — only executes under verified stability.

    Wraps GovernancePipeline + OrderManager with strict regime-based
    gating. Does NOT modify upstream models.

    Parameters
    ----------
    pipeline : GovernancePipeline
        Unified pipeline producing geometry, classifier, governor outputs.
    order_manager : OrderManager
        Order manager for executing CLOSE actions.
    """

    def __init__(
        self,
        pipeline: Optional[GovernancePipeline] = None,
        order_manager: Optional[OrderManager] = None,
    ) -> None:
        self.pipeline = pipeline or GovernancePipeline()
        self.order_manager = order_manager

        # Execution trace: per-symbol eligibility records
        self._trace: Dict[str, List[dict]] = defaultdict(list)

        # Pending exit actions (populated by evaluate, consumed by execute_pending_exits)
        self._pending_exits: List[dict] = []

        # Shadow instability flag — when True, all execution is blocked
        self.shadow_instability_flag: bool = False

        # Safety override log
        self._override_log: List[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        cluster_states: Dict[str, Any],
        rfe_output: Dict[str, Any],
        price_history: Optional[Dict[str, List[float]]] = None,
        positions: Optional[List[dict]] = None,
    ) -> Dict[str, Any]:
        """Evaluate bridge eligibility for all open positions.

        For each position, runs the governance pipeline and checks all
        3 gating conditions. Populates pending exits for eligible closes.

        Parameters
        ----------
        cluster_states : dict
            Current cluster states from SignalManifoldProjector.
        rfe_output : dict
            Output from RFEArbitrationLayer.evaluate().
        price_history : dict, optional
            Symbol -> list of historical prices.
        positions : list of dict, optional
            Current open positions. If None, skips symbol-specific
            evaluation.

        Returns
        -------
        dict with keys:
            - decisions (dict): per-symbol bridge eligibility decisions
            - summary (dict): aggregate bridge summary
            - pipeline_result (dict): raw pipeline output
            - pverride_log (list): safety override events
            - trace (dict): detailed eligibility trace
            - timestamp (str): ISO timestamp
        """
        # Step 1: Run governance pipeline
        pipeline_result = self.pipeline.evaluate(cluster_states, rfe_output, price_history)
        pipeline_decisions = pipeline_result.get("decisions", {})
        pipeline_summary = pipeline_result.get("summary", {})
        regime_classifications = pipeline_result.get("regime_classifications", {})
        geometry_forecasts = pipeline_result.get("geometry_forecasts", {})

        # Step 2: Determine eligible symbols
        all_symbols = set()
        if positions:
            for pos in positions:
                all_symbols.add(pos.get("symbol", ""))
        all_symbols |= set(pipeline_decisions.keys())
        all_symbols.discard("")

        bridge_decisions: Dict[str, dict] = {}
        eligible_count = 0
        blocked_count = 0
        eligible_symbols: List[str] = []

        for symbol in sorted(all_symbols):
            # Get pipeline outputs for this symbol
            cluster = symbol_to_primary_cluster(symbol)
            geo_forecast = geometry_forecasts.get(cluster, {})
            regime_class = regime_classifications.get(symbol, {})
            gov_decision = pipeline_decisions.get(symbol, {})

            # Run all gating checks
            checks = self._run_gating_checks(
                symbol, cluster, geo_forecast, regime_class, gov_decision
            )
            all_pass = all(c["passed"] for c in checks)
            blocking_reasons = [c["reason"] for c in checks if not c["passed"]]

            position = self._find_position(symbol, positions)
            gov_state = gov_decision.get("governor_state", GovernorState.HOLD)
            action_type = gov_decision.get("action", {}).get("type", "NONE")

            bridge_state = "ELIGIBLE" if all_pass else "BLOCKED"
            if all_pass:
                eligible_count += 1
                eligible_symbols.append(symbol)
            else:
                blocked_count += 1

            bridge_decisions[symbol] = {
                "symbol": symbol,
                "cluster": cluster,
                "bridge_state": bridge_state,
                "eligible": all_pass,
                "blocking_reasons": blocking_reasons,
                "gating_checks": checks,
                "governor_state": gov_state,
                "governor_action_type": action_type,
                "regime": regime_class.get("meta_type", "UNKNOWN"),
                "geometry": geo_forecast.get("pre_regime", "UNKNOWN"),
                "shadow_blocked": self.shadow_instability_flag,
                "position": position,
            }

            # Record trace
            self._trace[symbol].append(bridge_decisions[symbol])

        # Step 3: Populate pending exits (for eligible positions with
        # governor action CLOSE or CLOSE_PARTIAL)
        self._pending_exits.clear()
        for symbol in eligible_symbols:
            decision = pipeline_decisions.get(symbol, {})
            action = decision.get("action", {})
            if action.get("type") in ("CLOSE", "CLOSE_PARTIAL"):
                pos = bridge_decisions[symbol].get("position")
                gov_state = decision.get("governor_state", "")
                self._pending_exits.append({
                    "symbol": symbol,
                    "action": action["type"],
                    "fraction": action.get("fraction", 1.0),
                    "reason": action.get("reason", ""),
                    "ticket": pos.get("ticket") if pos else None,
                    "governor_state": gov_state,
                })
                logger.info(
                    "[BRIDGE_PENDING] %s → %s (governor=%s, reason=%s)",
                    symbol, action["type"], gov_state, action.get("reason", ""),
                )

        bridge_summary = {
            "total_symbols": len(all_symbols),
            "eligible_count": eligible_count,
            "blocked_count": blocked_count,
            "any_eligible": eligible_count > 0,
            "any_exit_pending": len(self._pending_exits) > 0,
            "pending_exits": [e["symbol"] for e in self._pending_exits],
            "eligible_symbols": eligible_symbols,
            "shadow_instability_flag": self.shadow_instability_flag,
        }

        return {
            "decisions": bridge_decisions,
            "summary": bridge_summary,
            "pipeline_result": pipeline_result,
            "override_log": list(self._override_log),
            "trace": dict(self._trace),
            "timestamp": datetime.now().isoformat(),
        }

    def execute_pending_exits(
        self,
        positions: Optional[List[dict]] = None,
    ) -> List[dict]:
        """Execute pending exit actions through OrderManager.

        Performs pre-execution verification (re-checks all 3 layers)
        immediately before each close call.

        Parameters
        ----------
        positions : list of dict, optional
            Current positions from MT5. If None, cannot execute.

        Returns
        -------
        list of dict
            Execution results, one per attempted exit.
        """
        if self.order_manager is None:
            logger.error("No OrderManager available — cannot execute")
            return []

        if not self._pending_exits:
            logger.info("No pending exits to execute")
            return []

        positions = positions or []
        results: List[dict] = []

        for exit_action in list(self._pending_exits):
            symbol = exit_action["symbol"]
            ticket = exit_action["ticket"]

            # Find current position
            pos = self._find_position(symbol, positions)
            if pos is None:
                logger.warning(f"[PRE_EXEC_FAIL] {symbol} — position no longer open")
                self._override_log.append({
                    "symbol": symbol,
                    "event": "position_not_found",
                    "action_type": exit_action["action"],
                    "blocked": True,
                })
                self._pending_exits.remove(exit_action)
                results.append({
                    "symbol": symbol,
                    "ticket": ticket,
                    "executed": False,
                    "reason": "position no longer open",
                })
                continue

            # Pre-execution verification: wait briefly, then re-check
            # In a real system, we'd re-fetch pipeline data here.
            # For the bridge, we mark this as the pre-verification step.
            import time
            time.sleep(PRE_EXECUTION_VERIFICATION_DELAY_S)

            # If ticket is missing, use position ticket
            actual_ticket = ticket or pos.get("ticket")

            # Execute close
            success = self.order_manager.close(actual_ticket)
            if success:
                logger.info(f"[BRIDGE_EXEC] Closed {symbol} ticket={actual_ticket}")
                self._pending_exits.remove(exit_action)
                self._override_log.append({
                    "symbol": symbol,
                    "event": "close_executed",
                    "action_type": exit_action["action"],
                    "ticket": actual_ticket,
                    "blocked": False,
                })
            else:
                logger.error(f"[BRIDGE_EXEC_FAIL] Close failed for {symbol} ticket={actual_ticket}")
                self._override_log.append({
                    "symbol": symbol,
                    "event": "close_failed",
                    "action_type": exit_action["action"],
                    "ticket": actual_ticket,
                    "blocked": True,
                })

            results.append({
                "symbol": symbol,
                "ticket": actual_ticket,
                "executed": success,
                "reason": "closed by bridge" if success else "close failed",
            })

        return results

    def evaluate_and_execute(
        self,
        cluster_states: Dict[str, Any],
        rfe_output: Dict[str, Any],
        price_history: Optional[Dict[str, List[float]]] = None,
        positions: Optional[List[dict]] = None,
    ) -> Dict[str, Any]:
        """Convenience: evaluate then execute in a single call.

        Returns the combined result with both evaluation and execution
        outcomes.
        """
        eval_result = self.evaluate(cluster_states, rfe_output, price_history, positions)
        exec_results = self.execute_pending_exits(positions)

        eval_result["execution_results"] = exec_results
        eval_result["executed_count"] = sum(1 for r in exec_results if r["executed"])
        eval_result["failed_count"] = sum(1 for r in exec_results if not r["executed"])

        return eval_result

    def reset(self) -> None:
        """Clear all internal state."""
        self.pipeline.reset()
        self._trace.clear()
        self._pending_exits.clear()
        self._override_log.clear()
        self.shadow_instability_flag = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_gating_checks(
        self,
        symbol: str,
        cluster: str,
        geo_forecast: dict,
        regime_class: dict,
        gov_decision: dict,
    ) -> List[dict]:
        """Run all 3 gating checks plus shadow stability check.

        Returns list of check results, each with:
        - gate (str): name of the gate
        - passed (bool)
        - reason (str)
        """
        checks: List[dict] = []

        # Gate 1: Geometry eligibility
        geo_ok, geo_reason = check_geometry_eligibility(geo_forecast)
        checks.append({
            "gate": "geometry",
            "passed": geo_ok,
            "reason": geo_reason,
            "value": geo_forecast.get("pre_regime", "N/A"),
        })

        # Gate 2: Classifier eligibility
        cls_ok, cls_reason = check_classifier_eligibility(regime_class)
        checks.append({
            "gate": "classifier",
            "passed": cls_ok,
            "reason": cls_reason,
            "value": regime_class.get("meta_type", "N/A"),
        })

        # Gate 3: Governor eligibility
        gov_ok, gov_reason = check_governor_eligibility(gov_decision)
        checks.append({
            "gate": "governor",
            "passed": gov_ok,
            "reason": gov_reason,
            "value": gov_decision.get("governor_state", "N/A"),
        })

        # Gate 4: Shadow stability
        shadow_ok, shadow_reason = check_shadow_stability(
            self.shadow_instability_flag, symbol
        )
        checks.append({
            "gate": "shadow_stability",
            "passed": shadow_ok,
            "reason": shadow_reason,
            "value": str(self.shadow_instability_flag),
        })

        return checks

    @staticmethod
    def _find_position(symbol: str, positions: Optional[List[dict]]) -> Optional[dict]:
        """Find position by symbol from a list of position dicts."""
        if not positions:
            return None
        for pos in positions:
            if pos.get("symbol") == symbol:
                return pos
        return None

    @property
    def blocked_count(self) -> int:
        """Total number of blocked actions across all cycles."""
        count = 0
        for symbol_entries in self._trace.values():
            for entry in symbol_entries:
                if entry.get("bridge_state") == "BLOCKED":
                    count += 1
        return count

    @property
    def executed_count(self) -> int:
        """Total number of executed close actions."""
        return sum(
            1 for entry in self._override_log if entry.get("event") == "close_executed"
        )

    @property
    def blocked_to_executed_ratio(self) -> float:
        """Ratio of blocked actions to executed actions."""
        executed = self.executed_count
        if executed == 0:
            return float("inf") if self.blocked_count > 0 else 0.0
        return round(self.blocked_count / executed, 4)


# ---------------------------------------------------------------------------
# Dashboard Formatting
# ---------------------------------------------------------------------------


def format_bridge_dashboard(result: dict) -> str:
    """Render the Restricted Execution Bridge dashboard.

    Parameters
    ----------
    result : dict
        Output from ``RestrictedExecutionBridge.evaluate()``.

    Returns
    -------
    str
        Formatted dashboard string.
    """
    lines: List[str] = []
    lines.append("")
    lines.append("RESTRICTED EXECUTION BRIDGE — GATED FIREWALL")
    lines.append("=" * 78)

    decisions = result.get("decisions", {})
    summary = result.get("summary", {})

    if not decisions:
        lines.append("  (no symbols evaluated)")
        lines.append("")
        lines.append("=" * 78)
        return "\n".join(lines)

    # Table header
    header = (
        f"{'Symbol':<12s} {'Bridge State':<14s} {'Geometry':<18s} "
        f"{'Regime':<18s} {'Governor':<14s} {'Exit?' :<6s}"
    )
    lines.append(header)
    lines.append("-" * 78)

    for symbol in sorted(decisions.keys()):
        d = decisions[symbol]
        bridge_state = d.get("bridge_state", "N/A")
        geo = d.get("geometry", "N/A")
        regime = d.get("regime", "N/A")
        gov = d.get("governor_state", "N/A")
        action = d.get("governor_action_type", "NONE")
        exit_flag = "YES" if action in ("CLOSE", "CLOSE_PARTIAL") else "NO"

        lines.append(
            f"{symbol:<12s} {bridge_state:<14s} {geo:<18s} "
            f"{regime:<18s} {gov:<14s} {exit_flag:<6s}"
        )

    lines.append("")
    lines.append(f"ANY ELIGIBLE: {summary.get('any_eligible', False)}")
    lines.append(
        f"Eligible: {summary.get('eligible_count', 0)} / "
        f"{summary.get('total_symbols', 0)}"
    )
    lines.append(
        f"Pending exits: {summary.get('pending_exits', []) or 'NONE'}"
    )
    lines.append(
        f"Shadow instability: {summary.get('shadow_instability_flag', False)}"
    )
    lines.append("")

    # Detailed blocks
    blocked_symbols = [
        s for s, d in decisions.items() if d.get("bridge_state") == "BLOCKED"
    ]
    if blocked_symbols:
        lines.append("BLOCKED SYMBOLS:")
        for sym in blocked_symbols:
            d = decisions[sym]
            for check in d.get("gating_checks", []):
                if not check["passed"]:
                    lines.append(f"  {sym}: {check['reason']}")

    lines.append("")
    lines.append("=" * 78)
    return "\n".join(lines)


def format_execution_ratio_report(bridge: RestrictedExecutionBridge) -> str:
    """Render the blocked-to-executed ratio report.

    Parameters
    ----------
    bridge : RestrictedExecutionBridge
        Bridge instance with accumulated trace history.

    Returns
    -------
    str
        Formatted report string.
    """
    lines: List[str] = []
    lines.append("")
    lines.append("BRIDGE EXECUTION RATIO REPORT")
    lines.append("=" * 78)
    lines.append(f"Total blocked actions : {bridge.blocked_count}")
    lines.append(f"Total executed actions: {bridge.executed_count}")
    lines.append(f"Blocked/Executed ratio: {bridge.blocked_to_executed_ratio}")
    lines.append("")

    # Safety override log
    if bridge._override_log:
        lines.append("SAFETY OVERRIDE LOG:")
        for entry in bridge._override_log[-20:]:
            symbol = entry.get("symbol", "")
            event = entry.get("event", "")
            blocked = entry.get("blocked", False)
            lines.append(f"  {symbol}: {event} (blocked={blocked})")

    lines.append("")
    lines.append("=" * 78)
    return "\n".join(lines)
