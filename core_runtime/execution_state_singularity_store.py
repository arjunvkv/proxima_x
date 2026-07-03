"""
Execution State Singularity Store — ONE canonical state object.

Replaces fragmented state (MT5 state, lifecycle state, SAAL state) with a
single authoritative dataclass.  No dual-state divergence possible.

Usage
-----
    from core_runtime.execution_state_singularity_store import (
        ExecutionStateSingularityStore,
        SystemState,
    )

    store = ExecutionStateSingularityStore()
    store.update({"mt5_connected": True, "cycle_count": 42})
    state: SystemState = store.get_state()
    print(state.mt5_connected)   # True
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton registry
# ---------------------------------------------------------------------------

_instances: Dict[str, "_ExecutionStateSingularityStore"] = {}


def ExecutionStateSingularityStore(instance_id="default"):
    """Singleton accessor — returns the same ``_ExecutionStateSingularityStore``
    for a given *instance_id*.

    Parameters
    ----------
    instance_id : str
        Logical identifier for the store instance (default ``"default"``).

    Returns
    -------
    _ExecutionStateSingularityStore
    """
    if instance_id not in _instances:
        logger.info("Creating new ExecutionStateSingularityStore instance '%s'", instance_id)
        _instances[instance_id] = _ExecutionStateSingularityStore(instance_id)
    return _instances[instance_id]


# ---------------------------------------------------------------------------
# The ONE canonical state
# ---------------------------------------------------------------------------


@dataclass
class SystemState:
    """Immutable-shaped dataclass representing the full canonical system state.

    All fields have default values so a fresh ``SystemState()`` is a valid
    "unknown / disconnected" state.
    """

    # ── Identity ──────────────────────────────────────────────────────────
    last_update: float = 0.0
    cycle_count: int = 0

    # ── MT5 connection ────────────────────────────────────────────────────
    mt5_connected: bool = False
    mt5_account_balance: float = 0.0
    mt5_account_equity: float = 0.0
    mt5_account_profit: float = 0.0

    # ── Positions ─────────────────────────────────────────────────────────
    open_positions: list = field(default_factory=list)
    open_positions_count: int = 0
    open_positions_pnl: float = 0.0

    # ── Current signal state (from SAAL) ──────────────────────────────────
    active_policy: str = "DISABLED"
    current_authority: str = "NONE"
    consensus_signal: int = 0
    final_execution_signal: int = 0
    execution_decision: str = "SKIP"
    execution_skip_reason: Optional[str] = None

    # ── System health ─────────────────────────────────────────────────────
    sdil_stable: bool = False
    csfr_consistent: bool = False
    saal_stable: bool = False
    event_chain_valid: bool = False
    system_ready: bool = False  # all checks pass

    # ── Trade history (summary) ───────────────────────────────────────────
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    last_trade_time: Optional[float] = None

    # Public constants for policy / authority strings
    POLICY_OSS = "OSS"
    POLICY_ALT = "ALT"
    POLICY_HYBRID = "HYBRID"
    POLICY_DISABLED = "DISABLED"

    AUTHORITY_OSS = "OSS"
    AUTHORITY_ALT = "ALT"
    AUTHORITY_HYBRID = "HYBRID"
    AUTHORITY_NONE = "NONE"

    SIGNAL_SELL = -1
    SIGNAL_NEUTRAL = 0
    SIGNAL_BUY = 1

    DECISION_EXECUTE = "EXECUTE"
    DECISION_SKIP = "SKIP"


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------


class _ExecutionStateSingularityStore:
    """Backing store for the singleton ``SystemState``.

    Parameters
    ----------
    instance_id : str
        Logical instance identifier (used for logging).
    """

    def __init__(self, instance_id: str):
        self._instance_id = instance_id
        self._state: SystemState = SystemState()
        logger.info(
            "ExecutionStateSingularityStore '%s' initialised with default state.",
            instance_id,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, state_update: dict) -> None:
        """Apply a partial update to the canonical state.

        Only the fields present in *state_update* are changed; all other
        fields retain their current value.

        Parameters
        ----------
        state_update : dict
            Dictionary of field_name → value pairs to apply.
        """
        if not isinstance(state_update, dict):
            logger.warning(
                "ExecutionStateSingularityStore '%s': update() called with "
                "non-dict argument (%s). Ignored.",
                self._instance_id,
                type(state_update).__name__,
            )
            return

        unknown = [k for k in state_update if not hasattr(self._state, k)]
        if unknown:
            logger.warning(
                "ExecutionStateSingularityStore '%s': update() received "
                "unknown field(s) %s. They will be stored as extra attributes.",
                self._instance_id,
                unknown,
            )

        for key, value in state_update.items():
            setattr(self._state, key, value)

        if "last_update" not in state_update:
            import time
            self._state.last_update = time.time()

        logger.debug(
            "ExecutionStateSingularityStore '%s': updated %d field(s).",
            self._instance_id,
            len(state_update),
        )

    def get_state(self) -> SystemState:
        """Return the full canonical ``SystemState`` object.

        Returns
        -------
        SystemState
        """
        return self._state

    def get_field(self, field_name: str) -> Any:
        """Get a specific field from the canonical state.

        Parameters
        ----------
        field_name : str
            Name of the field to retrieve.

        Returns
        -------
        Any
            The field value.

        Raises
        ------
        AttributeError
            If *field_name* does not exist on ``SystemState``.
        """
        return getattr(self._state, field_name)

    def snapshot(self) -> dict:
        """Return a frozen copy (shallow dict) of the current canonical state.

        Returns
        -------
        dict
        """
        return copy.copy(self._state.__dict__)

    def restore(self, snapshot_dict: dict) -> None:
        """Restore state from a snapshot (e.g. for replay).

        Parameters
        ----------
        snapshot_dict : dict
            A dictionary previously returned by ``snapshot()`` or a
            compatible dict where keys match ``SystemState`` fields.
        """
        if not isinstance(snapshot_dict, dict):
            logger.error(
                "ExecutionStateSingularityStore '%s': restore() requires a "
                "dict, got %s. State unchanged.",
                self._instance_id,
                type(snapshot_dict).__name__,
            )
            return

        # Create a fresh SystemState and overlay the snapshot values.
        restored = SystemState()
        for key, value in snapshot_dict.items():
            if hasattr(restored, key):
                setattr(restored, key, value)
        self._state = restored

        logger.info(
            "ExecutionStateSingularityStore '%s': state restored from "
            "snapshot (%d fields).",
            self._instance_id,
            len(snapshot_dict),
        )

    def get_diff(self, previous_snapshot: dict) -> dict:
        """Return a dict describing what changed between *previous_snapshot*
        and the current state.

        Parameters
        ----------
        previous_snapshot : dict
            A snapshot dict to compare against.

        Returns
        -------
        dict
            Keys are field names; values are ``(old_value, new_value)`` tuples
            for any field whose value changed.
        """
        current = self.snapshot()
        diff = {}
        all_keys = set(list(previous_snapshot.keys()) + list(current.keys()))
        for key in all_keys:
            old_val = previous_snapshot.get(key)
            new_val = current.get(key)
            if old_val != new_val:
                diff[key] = (old_val, new_val)
        return diff

    def validate_consistency(self) -> List[str]:
        """Run all consistency checks on the current canonical state.

        Returns
        -------
        list of str
            Human-readable descriptions of each consistency violation found.
            Empty list means everything is consistent.
        """
        violations: List[str] = []
        state = self._state

        # 1. equity vs. balance vs. unrealised PnL
        #    equity shouldn't exceed balance by more than unrealised PnL
        max_equity = state.mt5_account_balance + abs(state.mt5_account_profit)
        if state.mt5_account_equity > max_equity + 1e-6:
            msg = (
                f"mt5_account_equity ({state.mt5_account_equity}) exceeds "
                f"mt5_account_balance ({state.mt5_account_balance}) + "
                f"abs(unrealised PnL) ({abs(state.mt5_account_profit)}) by "
                f"{state.mt5_account_equity - max_equity:.2f}"
            )
            violations.append(msg)
            logger.warning("[%s] Consistency violation: %s", self._instance_id, msg)

        # 2. open_positions_count vs. len(open_positions)
        if state.open_positions_count != len(state.open_positions):
            msg = (
                f"open_positions_count ({state.open_positions_count}) does "
                f"not match len(open_positions) ({len(state.open_positions)})"
            )
            violations.append(msg)
            logger.warning("[%s] Consistency violation: %s", self._instance_id, msg)

        # 3. total_trades == winning_trades + losing_trades
        accounted = state.winning_trades + state.losing_trades
        if state.total_trades != accounted:
            msg = (
                f"total_trades ({state.total_trades}) != "
                f"winning_trades ({state.winning_trades}) + "
                f"losing_trades ({state.losing_trades}) = {accounted}"
            )
            violations.append(msg)
            logger.warning("[%s] Consistency violation: %s", self._instance_id, msg)

        # 4. system_ready only true if all sub-checks pass
        all_sub = (
            state.sdil_stable
            and state.csfr_consistent
            and state.saal_stable
            and state.event_chain_valid
        )
        if state.system_ready and not all_sub:
            msg = (
                f"system_ready is True but sub-checks: sdil_stable="
                f"{state.sdil_stable}, csfr_consistent={state.csfr_consistent}, "
                f"saal_stable={state.saal_stable}, "
                f"event_chain_valid={state.event_chain_valid}"
            )
            violations.append(msg)
            logger.warning("[%s] Consistency violation: %s", self._instance_id, msg)

        return violations

    def reset(self) -> None:
        """Reset the canonical state to factory defaults."""
        self._state = SystemState()
        logger.info(
            "ExecutionStateSingularityStore '%s': state reset to defaults.",
            self._instance_id,
        )

    def __repr__(self) -> str:
        return (
            f"ExecutionStateSingularityStore('{self._instance_id}', "
            f"cycle={self._state.cycle_count})"
        )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _run_self_test() -> None:
    """Run an exhaustive self-test of the store.

    Exercises:
      - update & retrieve
      - snapshots
      - restore
      - diff detection
      - consistency validation
      - multiple instance isolation
    """
    import time

    logger.info("=" * 60)
    logger.info("ExecutionStateSingularityStore — Self-test")
    logger.info("=" * 60)

    # --- 1.  Update and retrieve ------------------------------------------
    store = ExecutionStateSingularityStore("selftest")
    store.reset()

    assert store.get_state().cycle_count == 0
    assert store.get_state().mt5_connected is False

    store.update({"mt5_connected": True, "cycle_count": 5})
    assert store.get_state().mt5_connected is True
    assert store.get_state().cycle_count == 5
    logger.info("[PASS] update / get_state")

    # --- 2.  get_field ----------------------------------------------------
    assert store.get_field("mt5_connected") is True
    assert store.get_field("cycle_count") == 5
    assert store.get_field("active_policy") == "DISABLED"
    logger.info("[PASS] get_field")

    # --- 3.  last_update is auto-set --------------------------------------
    assert store.get_state().last_update > 0.0
    logger.info("[PASS] last_update auto-set on update")

    # --- 4.  Snapshot captures state --------------------------------------
    snap = store.snapshot()
    assert isinstance(snap, dict)
    assert snap["mt5_connected"] is True
    assert snap["cycle_count"] == 5
    logger.info("[PASS] snapshot")

    # --- 5.  Restore works ------------------------------------------------
    store.update({"cycle_count": 999})
    assert store.get_state().cycle_count == 999
    store.restore(snap)
    assert store.get_state().cycle_count == 5
    assert store.get_state().mt5_connected is True
    logger.info("[PASS] restore")

    # --- 6.  Diff detects changes -----------------------------------------
    snap_before = store.snapshot()
    store.update({"cycle_count": 42, "mt5_connected": False})
    snap_after = store.snapshot()
    diff = store.get_diff(snap_before)
    assert "cycle_count" in diff
    assert "mt5_connected" in diff
    assert diff["cycle_count"] == (5, 42)  # old, new
    assert diff["mt5_connected"] == (True, False)
    logger.info("[PASS] diff detection")

    # --- 7.  Consistency validation catches bad data ----------------------
    store.reset()
    store.update({"open_positions_count": 3, "open_positions": []})
    violations = store.validate_consistency()
    assert any("open_positions_count" in v for v in violations), violations
    logger.info("[PASS] validate_consistency — mismatch count vs list")

    store.reset()
    store.update({"total_trades": 10, "winning_trades": 7, "losing_trades": 2})
    violations = store.validate_consistency()
    assert any("total_trades" in v for v in violations), violations
    logger.info("[PASS] validate_consistency — trade arithmetic")

    store.reset()
    store.update({"system_ready": True})
    violations = store.validate_consistency()
    assert any("system_ready" in v for v in violations), violations
    logger.info("[PASS] validate_consistency — system_ready without sub-checks")

    store.reset()
    store.update({
        "system_ready": True,
        "sdil_stable": True,
        "csfr_consistent": True,
        "saal_stable": True,
        "event_chain_valid": True,
    })
    violations = store.validate_consistency()
    assert "system_ready" not in " ".join(violations), violations
    logger.info("[PASS] validate_consistency — clean state passes")

    # --- 8.  Equity / balance / PnL consistency ---------------------------
    store.reset()
    store.update({
        "mt5_account_balance": 10000.0,
        "mt5_account_equity": 15000.0,
        "mt5_account_profit": 500.0,
    })
    violations = store.validate_consistency()
    # equity (15000) > balance (10000) + abs(PnL) (500) = 10500 → violation
    assert any("equity" in v for v in violations)
    logger.info("[PASS] validate_consistency — equity vs balance vs PnL")

    # --- 9.  Multiple instances don't interfere ---------------------------
    store_a = ExecutionStateSingularityStore("multi_a")
    store_b = ExecutionStateSingularityStore("multi_b")
    store_a.reset()
    store_b.reset()

    store_a.update({"cycle_count": 111})
    store_b.update({"cycle_count": 222})

    assert store_a.get_field("cycle_count") == 111
    assert store_b.get_field("cycle_count") == 222
    assert store_a is not store_b
    logger.info("[PASS] multiple instance isolation")

    # --- 10.  reset -------------------------------------------------------
    store_a.update({"mt5_connected": True, "total_pnl": 1234.56})
    store_a.reset()
    assert store_a.get_state().mt5_connected is False
    assert store_a.get_state().total_pnl == 0.0
    logger.info("[PASS] reset")

    # --- 11.  update with unknown field (stored as extra attr) ------------
    store.update({"some_new_key": "hello"})
    assert store.get_field("some_new_key") == "hello"
    logger.info("[PASS] update with unknown field")

    # --- 12.  restore invalid argument ------------------------------------
    store.restore("not a dict")  # should not raise
    logger.info("[PASS] restore with invalid argument")

    logger.info("=" * 60)
    logger.info("ALL SELF-TESTS PASSED")
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _run_self_test()
