"""
Execution Policy Switcher — dynamically switches between OSS-driven, ALT-driven,
hybrid, or disabled execution modes.

Policy modes:
  OSS      — Only OSS signals are used for execution. ALT signals logged but ignored.
  ALT      — Only ALT signals are used. OSS logged but ignored.
  HYBRID   — Both signals feed into SignalConsensusModel for conflict resolution.
  DISABLED — No trading. All signals result in NO TRADE.
"""

import logging
import os
import sys
import time
from typing import Callable, Dict, List, Any, Optional

# Ensure project root is on path so proxima_x modules can be resolved.
# __file__ is .../proxima_x/core_runtime/execution_policy_switcher.py
# → go up 3 levels to reach C:\Trading\Agentic_Trading
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from proxima_x.core_runtime.signal_consensus_model import SignalConsensusModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------
class _ExecutionPolicySwitcher:
    """Dynamically switch execution policy at runtime."""

    VALID_POLICIES = frozenset({"OSS", "ALT", "HYBRID", "DISABLED"})

    def __init__(self, instance_id: str = "default") -> None:
        self._instance_id = instance_id
        self._active_policy: str = "OSS"
        self._policy_history: List[Dict[str, Any]] = []

        # — safety guard (anti-thrash) —
        self._min_ticks_between_switches: int = 50
        self._tick_counter: int = 0
        self._last_switch_tick: int = -self._min_ticks_between_switches  # allow first switch

        # — tracking —
        self._change_count: int = 0
        self._last_change_timestamp: Optional[float] = None

        # tick-based time tracking per policy
        self._policy_enter_tick: Dict[str, int] = {}
        self._policy_accumulated_ticks: Dict[str, int] = {}

        # initialise all policies
        for p in self.VALID_POLICIES:
            self._policy_accumulated_ticks[p] = 0
        self._policy_enter_tick[self._active_policy] = 0

        # record initial state
        self._policy_history.append({
            "timestamp": time.time(),
            "old_policy": None,
            "new_policy": self._active_policy,
            "reason": "initialisation",
        })

        logger.info(
            "ExecutionPolicySwitcher '%s' initialised (policy=%s, min_ticks=%d)",
            instance_id, self._active_policy, self._min_ticks_between_switches,
        )

    # -- Public config -------------------------------------------------------

    def set_policy(self, policy: str, reason: str = "") -> None:
        """Set the active execution policy.

        Parameters
        ----------
        policy : str
            One of ``OSS``, ``ALT``, ``HYBRID``, ``DISABLED``.
        reason : str
            Optional human-readable reason for the change.
        """
        policy = policy.upper()
        if policy not in self.VALID_POLICIES:
            raise ValueError(
                f"Unknown policy '{policy}'. Valid: {sorted(self.VALID_POLICIES)}"
            )

        if policy == self._active_policy:
            return  # no-op

        # safety guard — prevent thrashing
        if self._tick_counter - self._last_switch_tick < self._min_ticks_between_switches:
            logger.warning(
                "Switch to '%s' blocked by anti-thrash guard (tick %d, last switch at tick %d)",
                policy, self._tick_counter, self._last_switch_tick,
            )
            return

        old_policy = self._active_policy
        self._active_policy = policy
        self._last_switch_tick = self._tick_counter
        self._change_count += 1
        self._last_change_timestamp = time.time()

        # update tick-accumulation for the policy being left
        entered_at = self._policy_enter_tick.get(old_policy, self._tick_counter)
        elapsed = self._tick_counter - entered_at
        if elapsed > 0:
            self._policy_accumulated_ticks[old_policy] = \
                self._policy_accumulated_ticks.get(old_policy, 0) + elapsed

        # reset enter tick for new policy
        self._policy_enter_tick[policy] = self._tick_counter

        record = {
            "timestamp": self._last_change_timestamp,
            "old_policy": old_policy,
            "new_policy": policy,
            "reason": reason or "no reason provided",
        }
        self._policy_history.append(record)

        logger.info(
            "Policy changed: %s -> %s  (reason: %s)",
            old_policy, policy, record["reason"],
        )

    def get_policy(self) -> str:
        """Return the current active policy."""
        return self._active_policy

    def get_policy_history(self) -> List[Dict[str, Any]]:
        """Return the full history of policy changes.

        Each entry: ``{timestamp, old_policy, new_policy, reason}``.
        """
        return list(self._policy_history)

    # -- Conditional switching -----------------------------------------------

    def switch_if(
        self,
        condition: Callable[[], bool],
        new_policy: str,
        reason: str = "",
    ) -> bool:
        """Evaluate *condition* and switch to *new_policy* if it is truthy.

        The condition callable should close over any state it needs, matching
        the spec example::

            switcher.switch_if(
                condition=lambda: state["oss_score"] < 0.3,
                new_policy="ALT",
                reason="OSS score below threshold"
            )

        Parameters
        ----------
        condition : callable
            A no-argument callable that returns bool.
        new_policy : str
            Target policy if condition is met.
        reason : str
            Human-readable reason (auto-generated if empty).

        Returns
        -------
        bool
            ``True`` if a switch actually occurred.
        """
        if not condition():
            return False

        effective_reason = reason or f"Condition triggered: switch from {self._active_policy} to {new_policy}"
        self.set_policy(new_policy, effective_reason)
        return True

    # -- Arbiter-based evaluation --------------------------------------------

    def evaluate_switch(
        self,
        state_dict: Dict[str, Any],
        arbiter_result: Dict[str, Any],
    ) -> str:
        """Evaluate arbiter recommendation and conditionally switch policy.

        Rules
        -----
        - If arbiter says ``OSS`` and current is ``ALT`` → switch to ``OSS``
        - If arbiter says ``ALT`` and current is ``OSS`` → switch to ``ALT``
        - If arbiter says ``HYBRID`` → switch to ``HYBRID``
        - If arbiter says ``NONE`` → switch to ``DISABLED``
        - Only switch if ``arbiter_result["confidence"] > 0.6``

        Parameters
        ----------
        state_dict : Dict[str, Any]
            Current system state (unused internally, but available for subclasses).
        arbiter_result : Dict[str, Any]
            Must contain keys ``"arbiter_decision"`` (str: OSS/ALT/HYBRID/NONE)
            and ``"confidence"`` (float in [0, 1]).

        Returns
        -------
        str
            The policy in effect after evaluation (may be unchanged).
        """
        decision = arbiter_result.get("arbiter_decision", "").upper()
        confidence = arbiter_result.get("confidence", 0.0)

        if confidence <= 0.6:
            logger.debug(
                "Arbiter confidence %.2f <= 0.6; no switch triggered",
                confidence,
            )
            return self._active_policy

        current = self._active_policy
        target = None

        if decision == "OSS" and current == "ALT":
            target = "OSS"
        elif decision == "ALT" and current == "OSS":
            target = "ALT"
        elif decision == "HYBRID":
            target = "HYBRID"
        elif decision == "NONE":
            target = "DISABLED"

        if target is not None and target != current:
            self.set_policy(
                target,
                reason=f"Arbiter recommended {decision} (conf={confidence:.3f})",
            )

        return self._active_policy

    # -- Signal resolution ---------------------------------------------------

    def get_signals(
        self,
        symbol: str,
        oss_signal: int,
        oss_confidence: float,
        alt_signal: int,
        alt_confidence: float,
    ) -> Dict[str, Any]:
        """Resolve the effective execution signal under the current policy.

        Parameters
        ----------
        symbol : str
            Trading symbol (used for logging context).
        oss_signal : int
            OSS direction: -1, 0, or +1.
        oss_confidence : float
            OSS confidence in [0, 1].
        alt_signal : int
            ALT direction: -1, 0, or +1.
        alt_confidence : float
            ALT confidence in [0, 1].

        Returns
        -------
        dict with keys:
            policy : str
                The active policy used.
            final_signal : int
                -1, 0, or +1 — the signal to use for execution.
            oss_signal : int
                Original OSS signal.
            alt_signal : int
                Original ALT signal.
            consensus_result : dict or None
                If policy is HYBRID, the SignalConsensusModel result dict;
                otherwise ``None``.
        """
        policy = self._active_policy
        result: Dict[str, Any] = {
            "policy": policy,
            "final_signal": 0,
            "oss_signal": oss_signal,
            "alt_signal": alt_signal,
            "consensus_result": None,
        }

        if policy == "OSS":
            result["final_signal"] = oss_signal
            logger.debug("[%s] OSS policy → using OSS signal %d", symbol, oss_signal)

        elif policy == "ALT":
            result["final_signal"] = alt_signal
            logger.debug("[%s] ALT policy → using ALT signal %d", symbol, alt_signal)

        elif policy == "HYBRID":
            consensus = SignalConsensusModel()
            resolved = consensus.resolve(
                oss_signal, oss_confidence,
                alt_signal, alt_confidence,
            )
            result["final_signal"] = resolved["consensus_signal"]
            result["consensus_result"] = resolved
            logger.debug(
                "[%s] HYBRID policy → consensus signal %d",
                symbol, result["final_signal"],
            )

        elif policy == "DISABLED":
            result["final_signal"] = 0
            logger.debug("[%s] DISABLED policy → NO TRADE", symbol)

        return result

    # -- Tick management -----------------------------------------------------

    def tick(self) -> None:
        """Advance the internal tick counter (used for anti-thrash guard)."""
        self._tick_counter += 1

    def set_min_ticks_between_switches(self, n: int) -> None:
        """Set the minimum number of ticks that must elapse between switches."""
        if n < 1:
            raise ValueError("min_ticks_between_switches must be >= 1")
        self._min_ticks_between_switches = n
        logger.info("min_ticks_between_switches set to %d", n)

    # -- Tracking / statistics -----------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return accumulated policy-switching statistics."""
        # snapshot current tick counts
        current_policy = self._active_policy
        entered_at = self._policy_enter_tick.get(current_policy, self._tick_counter)
        elapsed_in_current = self._tick_counter - entered_at

        total_ticks_per_policy = dict(self._policy_accumulated_ticks)
        total_ticks_per_policy[current_policy] = \
            total_ticks_per_policy.get(current_policy, 0) + elapsed_in_current

        return {
            "active_policy": current_policy,
            "change_count": self._change_count,
            "last_change_timestamp": self._last_change_timestamp,
            "tick_counter": self._tick_counter,
            "min_ticks_between_switches": self._min_ticks_between_switches,
            "total_ticks_per_policy": total_ticks_per_policy,
        }


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------
_instances: Dict[str, _ExecutionPolicySwitcher] = {}


def ExecutionPolicySwitcher(instance_id: str = "default") -> _ExecutionPolicySwitcher:
    """Factory that returns a shared _ExecutionPolicySwitcher singleton per instance_id."""
    if instance_id not in _instances:
        _instances[instance_id] = _ExecutionPolicySwitcher(instance_id)
    return _instances[instance_id]


# ===========================================================================
# Self-test
# ===========================================================================
def _self_test() -> None:
    """Exercise all policy modes and conditional switching."""
    print("=" * 60)
    print("ExecutionPolicySwitcher :: Self-Test")
    print("=" * 60)

    # -- 1. Basic policy switching -------------------------------------------
    print("\n--- 1. Basic policy switching ---")
    sw = ExecutionPolicySwitcher("test_basic")
    sw.set_min_ticks_between_switches(1)  # relax guard for testing
    assert sw.get_policy() == "OSS", f"Expected OSS, got {sw.get_policy()}"
    print(f"  Initial policy: {sw.get_policy()}")

    sw.set_policy("ALT", "testing ALT mode")
    sw.tick()  # advance to satisfy min_ticks=1
    assert sw.get_policy() == "ALT"
    print(f"  After set_policy('ALT'): {sw.get_policy()}")

    sw.set_policy("HYBRID", "testing HYBRID mode")
    sw.tick()
    assert sw.get_policy() == "HYBRID"
    print(f"  After set_policy('HYBRID'): {sw.get_policy()}")

    sw.set_policy("DISABLED", "testing DISABLED mode")
    sw.tick()
    assert sw.get_policy() == "DISABLED"
    print(f"  After set_policy('DISABLED'): {sw.get_policy()}")

    # -- 2. Policy history ---------------------------------------------------
    print("\n--- 2. Policy history ---")
    history = sw.get_policy_history()
    print(f"  Total history entries: {len(history)}")
    for entry in history:
        print(f"    {entry['old_policy']} -> {entry['new_policy']}  |  {entry['reason']}")

    # -- 3. get_signals under each policy ------------------------------------
    print("\n--- 3. Signal resolution per policy ---")

    # OSS policy
    sw2 = ExecutionPolicySwitcher("test_signals")
    sw2.set_min_ticks_between_switches(1)
    # already starts at OSS
    r = sw2.get_signals("EURUSD", +1, 0.80, -1, 0.70)
    assert r["final_signal"] == +1, f"OSS policy should return OSS signal; got {r['final_signal']}"
    assert r["consensus_result"] is None
    print(f"  OSS policy: oss=+1 alt=-1 -> final={r['final_signal']}")

    # ALT policy
    sw2.set_policy("ALT")
    sw2.tick()
    r = sw2.get_signals("EURUSD", +1, 0.80, -1, 0.70)
    assert r["final_signal"] == -1, f"ALT policy should return ALT signal; got {r['final_signal']}"
    assert r["consensus_result"] is None
    print(f"  ALT policy: oss=+1 alt=-1 -> final={r['final_signal']}")

    # DISABLED policy
    sw2.set_policy("DISABLED")
    sw2.tick()
    r = sw2.get_signals("EURUSD", +1, 0.80, -1, 0.70)
    assert r["final_signal"] == 0, f"DISABLED policy should return 0; got {r['final_signal']}"
    print(f"  DISABLED policy: oss=+1 alt=-1 -> final={r['final_signal']}")

    # HYBRID policy
    sw2.set_policy("HYBRID")
    sw2.tick()
    r = sw2.get_signals("EURUSD", +1, 0.80, -1, 0.70)
    assert r["consensus_result"] is not None, "HYBRID should produce consensus_result"
    print(f"  HYBRID policy: oss=+1 alt=-1 -> final={r['final_signal']}  "
          f"(consensus={r['consensus_result']['consensus_signal']})")

    # HYBRID — agree
    r = sw2.get_signals("EURUSD", +1, 0.80, +1, 0.70)
    assert r["final_signal"] == +1
    print(f"  HYBRID policy: oss=+1 alt=+1 -> final={r['final_signal']}")

    # -- 4. Anti-thrash guard ------------------------------------------------
    print("\n--- 4. Anti-thrash guard ---")
    sw3 = ExecutionPolicySwitcher("test_thrash")
    sw3.set_min_ticks_between_switches(5)
    sw3.set_policy("ALT", "first switch")  # allowed
    assert sw3.get_policy() == "ALT"
    print(f"  After switch to ALT (tick 0): {sw3.get_policy()}")

    sw3.set_policy("HYBRID", "thrashing")  # blocked — only 0 ticks elapsed
    assert sw3.get_policy() == "ALT", f"Expected ALT (blocked), got {sw3.get_policy()}"
    print(f"  Thrash attempt blocked, still: {sw3.get_policy()}")

    # advance ticks and try again
    for _ in range(6):
        sw3.tick()
    sw3.set_policy("HYBRID", "after enough ticks")
    assert sw3.get_policy() == "HYBRID", f"Expected HYBRID, got {sw3.get_policy()}"
    print(f"  After {sw3.get_statistics()['min_ticks_between_switches']} ticks: allowed switch to {sw3.get_policy()}")

    # -- 5. switch_if (conditional switching) --------------------------------
    print("\n--- 5. Conditional switching (switch_if) ---")
    sw4 = ExecutionPolicySwitcher("test_conditional")
    sw4.set_min_ticks_between_switches(1)
    state = {"oss_score": 0.25, "alt_score": 0.80}
    switched = sw4.switch_if(
        condition=lambda: state["oss_score"] < 0.3,
        new_policy="ALT",
        reason="OSS score below threshold",
    )
    assert switched, "switch_if should return True when condition met"
    assert sw4.get_policy() == "ALT"
    print(f"  Condition oss_score < 0.3 → switched to {sw4.get_policy()} (switched={switched})")

    # test condition NOT met
    state["oss_score"] = 0.50
    switched2 = sw4.switch_if(
        condition=lambda: state["oss_score"] < 0.3,
        new_policy="OSS",
        reason="should not trigger",
    )
    assert not switched2
    assert sw4.get_policy() == "ALT", "Should remain ALT"
    print(f"  Condition oss_score=0.5 < 0.3 is False → no switch (switched={switched2})")

    # -- 6. evaluate_switch --------------------------------------------------
    print("\n--- 6. Arbiter-based evaluate_switch ---")
    sw5 = ExecutionPolicySwitcher("test_arbiter")
    sw5.set_min_ticks_between_switches(1)

    # Start as OSS, arbiter says ALT with high confidence → switch to ALT
    arbiter = {"arbiter_decision": "ALT", "confidence": 0.85}
    result = sw5.evaluate_switch({}, arbiter)
    assert result == "ALT", f"Expected ALT, got {result}"
    sw5.tick()
    print(f"  Arbiter ALT (conf=0.85) when OSS → {result}")

    # Arbiter says OSS with high confidence → switch back to OSS
    arbiter2 = {"arbiter_decision": "OSS", "confidence": 0.75}
    result = sw5.evaluate_switch({}, arbiter2)
    assert result == "OSS", f"Expected OSS, got {result}"
    sw5.tick()
    print(f"  Arbiter OSS (conf=0.75) when ALT → {result}")

    # Arbiter says HYBRID → switch to HYBRID
    arbiter3 = {"arbiter_decision": "HYBRID", "confidence": 0.90}
    result = sw5.evaluate_switch({}, arbiter3)
    assert result == "HYBRID", f"Expected HYBRID, got {result}"
    sw5.tick()
    print(f"  Arbiter HYBRID (conf=0.90) → {result}")

    # Arbiter says NONE → switch to DISABLED
    arbiter4 = {"arbiter_decision": "NONE", "confidence": 0.95}
    result = sw5.evaluate_switch({}, arbiter4)
    assert result == "DISABLED", f"Expected DISABLED, got {result}"
    sw5.tick()
    print(f"  Arbiter NONE (conf=0.95) → {result}")

    # Low confidence → no switch
    arbiter5 = {"arbiter_decision": "OSS", "confidence": 0.30}
    result = sw5.evaluate_switch({}, arbiter5)
    assert result == "DISABLED", f"Expected DISABLED (unchanged), got {result}"
    print(f"  Arbiter OSS (conf=0.30) → no switch, still: {result}")

    # -- 7. Statistics -------------------------------------------------------
    print("\n--- 7. Statistics ---")
    stats = sw5.get_statistics()
    print(f"  active_policy: {stats['active_policy']}")
    print(f"  change_count: {stats['change_count']}")
    print(f"  last_change_timestamp: {stats['last_change_timestamp']}")
    print(f"  tick_counter: {stats['tick_counter']}")
    print(f"  total_ticks_per_policy: {stats['total_ticks_per_policy']}")

    # -- 8. Invalid inputs ---------------------------------------------------
    print("\n--- 8. Invalid inputs ---")
    try:
        sw2.set_policy("INVALID")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  set_policy('INVALID') correctly raised: {e}")

    try:
        sw3.set_min_ticks_between_switches(0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  set_min_ticks_between_switches(0) correctly raised: {e}")

    # -- 9. Singleton pattern ------------------------------------------------
    print("\n--- 9. Singleton pattern ---")
    a = ExecutionPolicySwitcher("singleton_test")
    b = ExecutionPolicySwitcher("singleton_test")
    assert a is b
    c = ExecutionPolicySwitcher("other")
    assert c is not a
    print("  Singleton pattern verified")

    # -- 10. Graceful no-op when same policy ---------------------------------
    print("\n--- 10. No-op on same policy ---")
    sw6 = ExecutionPolicySwitcher("test_noop")
    assert sw6.get_policy() == "OSS"
    before_count = sw6.get_statistics()["change_count"]
    sw6.set_policy("OSS", "same policy")
    after_count = sw6.get_statistics()["change_count"]
    assert before_count == after_count, "Setting same policy should be a no-op"
    print(f"  set_policy('OSS') when already OSS → change_count unchanged ({after_count})")

    print("\n" + "=" * 60)
    print("All self-tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    _self_test()
