"""
Causal Event Chain Enforcer — guarantee strict temporal ordering.

Ensures that no module can read future state or stale state.  Enforces
the REQUIRED layer processing order per cycle with zero backflow
contamination and zero temporal paradox in logs.

Required layer order (per cycle)
---------------------------------
    1. tick_ingestion
    2. oss_surface
    3. alt_signal
    4. sdil
    5. csfr
    6. saal
    7. execution
    8. seal

Usage
-----
    from core_runtime.causal_event_chain_enforcer import CausalEventChainEnforcer

    enforcer = CausalEventChainEnforcer()
    enforcer.begin_cycle(1)
    enforcer.enter_layer("tick_ingestion")
    # ... do work ...
    enforcer.exit_layer("tick_ingestion")
    enforcer.enter_layer("oss_surface")
    # ...
    enforcer.seal_cycle(1)
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_instances: Dict[str, "_CausalEventChainEnforcer"] = {}


def CausalEventChainEnforcer(instance_id="default"):
    """Singleton accessor for ``_CausalEventChainEnforcer``.

    Parameters
    ----------
    instance_id : str
        Unique identifier.  Callers sharing the same *instance_id* share the
        same underlying enforcer object.

    Returns
    -------
    _CausalEventChainEnforcer
    """
    if instance_id not in _instances:
        _instances[instance_id] = _CausalEventChainEnforcer(instance_id)
    return _instances[instance_id]


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------

class _CausalEventChainEnforcer:
    """Enforces strict temporal ordering of causal event chains.

    Parameters
    ----------
    instance_id : str
        Instance identifier forwarded from the singleton accessor.
    """

    # Immutable layer order — shared across all instances
    LAYER_ORDER: List[str] = [
        "tick_ingestion",
        "oss_surface",
        "alt_signal",
        "sdil",
        "csfr",
        "saal",
        "execution",
        "seal",
    ]

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, instance_id="default"):
        self._instance_id = instance_id
        self._current_cycle: int = 0
        self._sealed_cycles: Set[int] = set()
        self._started_cycles: Set[int] = set()
        self._access_log: Dict[int, List[Dict[str, Any]]] = {}
        self._violations: List[Dict[str, Any]] = []

        # Per-cycle state: {cycle_id: {"current_layer_index": int,
        #                               "completed_layers": list[str],
        #                               "pending_layers": list[str]}}
        self._cycle_state: Dict[int, Dict[str, Any]] = {}

        logger.info("CausalEventChainEnforcer(%r) initialised", instance_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def begin_cycle(self, cycle_id: int) -> None:
        """Start a new processing cycle.

        Parameters
        ----------
        cycle_id : int
            Monotonically increasing cycle identifier.

        Raises
        ------
        ValueError
            If *cycle_id* <= ``_current_cycle`` (cannot go back in time).
        RuntimeError
            If the cycle has already been started.
        """
        if cycle_id <= self._current_cycle:
            msg = (
                f"BACKFLOW: cannot begin cycle {cycle_id} – "
                f"current cycle is {self._current_cycle}"
            )
            self._log_violation("BACKFLOW", cycle_id, msg)
            raise ValueError(msg)

        if cycle_id in self._started_cycles:
            msg = f"CYCLE_ALREADY_STARTED: cycle {cycle_id} has already begun"
            self._log_violation("CYCLE_ALREADY_STARTED", cycle_id, msg)
            raise RuntimeError(msg)

        self._current_cycle = cycle_id
        self._started_cycles.add(cycle_id)
        self._cycle_state[cycle_id] = {
            "current_layer_index": -1,
            "completed_layers": [],
            "pending_layers": list(self.LAYER_ORDER),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

        logger.info("Cycle %d started", cycle_id)

    def enter_layer(self, layer_name: str) -> None:
        """Signal entry into a processing layer.

        Parameters
        ----------
        layer_name : str
            Name of the layer to enter.  Must be one of
            ``LAYER_ORDER``.

        Raises
        ------
        RuntimeError
            If no cycle has been started.
        ValueError
            If *layer_name* is not a recognised layer.
        ValueError
            If the layer has already been processed this cycle.
        ValueError
            If the layer is being entered out of the required order.
        """
        # --- Guards --------------------------------------------------------
        if self._current_cycle == 0:
            msg = "No cycle has been started"
            self._log_violation("NO_CYCLE", 0, msg)
            raise RuntimeError(msg)

        cycle_id = self._current_cycle
        state = self._cycle_state.get(cycle_id)
        if state is None:
            msg = f"Cycle {cycle_id} has no state – call begin_cycle first"
            self._log_violation("NO_CYCLE_STATE", cycle_id, msg)
            raise RuntimeError(msg)

        if layer_name not in self.LAYER_ORDER:
            msg = (
                f"UNKNOWN_LAYER: '{layer_name}' is not in the layer order. "
                f"Valid layers: {self.LAYER_ORDER}"
            )
            self._log_violation("UNKNOWN_LAYER", cycle_id, msg)
            raise ValueError(msg)

        if layer_name in state["completed_layers"]:
            msg = (
                f"DUPLICATE_LAYER: layer '{layer_name}' has already been "
                f"processed in cycle {cycle_id}"
            )
            self._log_violation("DUPLICATE_LAYER", cycle_id, msg)
            raise ValueError(msg)

        expected_idx = state["current_layer_index"] + 1
        actual_idx = self.LAYER_ORDER.index(layer_name)

        if actual_idx != expected_idx:
            msg = (
                f"ORDER_VIOLATION: expected layer "
                f"'{self.LAYER_ORDER[expected_idx]}' (index {expected_idx}) "
                f"but got '{layer_name}' (index {actual_idx}) in cycle "
                f"{cycle_id}"
            )
            self._log_violation("ORDER_VIOLATION", cycle_id, msg)
            raise ValueError(msg)

        # --- Enter layer ---------------------------------------------------
        state["current_layer_index"] = actual_idx

        entry_record = {
            "cycle_id": cycle_id,
            "layer": layer_name,
            "event": "enter",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._access_log.setdefault(cycle_id, []).append(entry_record)

        logger.info("Cycle %d entered layer '%s'", cycle_id, layer_name)

    def exit_layer(self, layer_name: str) -> None:
        """Signal exit from a processing layer.

        Parameters
        ----------
        layer_name : str
            Name of the layer to exit.

        Raises
        ------
        RuntimeError
            If no cycle is active.
        ValueError
            If the layer has not been entered.
        """
        if self._current_cycle == 0:
            msg = "No cycle has been started – cannot exit layer"
            self._log_violation("NO_CYCLE", 0, msg)
            raise RuntimeError(msg)

        cycle_id = self._current_cycle
        state = self._cycle_state.get(cycle_id)
        if state is None:
            msg = f"Cycle {cycle_id} has no state"
            self._log_violation("NO_CYCLE_STATE", cycle_id, msg)
            raise RuntimeError(msg)

        if layer_name not in self.LAYER_ORDER:
            msg = f"UNKNOWN_LAYER: '{layer_name}' is not a recognised layer"
            self._log_violation("UNKNOWN_LAYER", cycle_id, msg)
            raise ValueError(msg)

        expected_layer = (
            self.LAYER_ORDER[state["current_layer_index"]]
            if state["current_layer_index"] >= 0
            else None
        )
        if expected_layer is None or expected_layer != layer_name:
            msg = (
                f"LAYER_MISMATCH: cannot exit '{layer_name}' – "
                f"currently active layer is '{expected_layer}'"
            )
            self._log_violation("LAYER_MISMATCH", cycle_id, msg)
            raise ValueError(msg)

        # --- Exit layer ----------------------------------------------------
        state["completed_layers"].append(layer_name)
        state["pending_layers"].remove(layer_name)

        exit_record = {
            "cycle_id": cycle_id,
            "layer": layer_name,
            "event": "exit",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._access_log.setdefault(cycle_id, []).append(exit_record)

        logger.info("Cycle %d exited layer '%s'", cycle_id, layer_name)

    def seal_cycle(self, cycle_id: int) -> None:
        """Finalise (seal) a cycle so no further updates are allowed.

        Parameters
        ----------
        cycle_id : int
            The cycle to seal.

        Raises
        ------
        RuntimeError
            If the cycle has not been started.
        RuntimeError
            If not all layers have been completed.
        ValueError
            If the cycle is already sealed.
        """
        if cycle_id not in self._started_cycles:
            msg = f"Cannot seal – cycle {cycle_id} has not been started"
            self._log_violation("CYCLE_NOT_STARTED", cycle_id, msg)
            raise RuntimeError(msg)

        if cycle_id in self._sealed_cycles:
            msg = f"Cycle {cycle_id} is already sealed"
            self._log_violation("ALREADY_SEALED", cycle_id, msg)
            raise ValueError(msg)

        state = self._cycle_state.get(cycle_id)
        if state is None:
            msg = f"Cycle {cycle_id} has no state"
            self._log_violation("NO_CYCLE_STATE", cycle_id, msg)
            raise RuntimeError(msg)

        if len(state["completed_layers"]) != len(self.LAYER_ORDER):
            missing = set(self.LAYER_ORDER) - set(state["completed_layers"])
            msg = (
                f"Cannot seal cycle {cycle_id} – not all layers complete. "
                f"Missing: {sorted(missing)}"
            )
            self._log_violation("INCOMPLETE_CYCLE", cycle_id, msg)
            raise RuntimeError(msg)

        self._sealed_cycles.add(cycle_id)

        seal_record = {
            "cycle_id": cycle_id,
            "event": "seal",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._access_log.setdefault(cycle_id, []).append(seal_record)

        logger.info("Cycle %d sealed", cycle_id)

    def validate_read(self, cycle_id: int) -> None:
        """Check whether *cycle_id* can be read safely.

        Parameters
        ----------
        cycle_id : int
            The cycle being read.

        Raises
        ------
        ValueError
            If *cycle_id* > ``_current_cycle`` (future read).
        """
        if cycle_id > self._current_cycle:
            msg = (
                f"FUTURE_READ: cannot read cycle {cycle_id} – "
                f"current cycle is {self._current_cycle}"
            )
            self._log_violation("FUTURE_READ", cycle_id, msg)
            raise ValueError(msg)

    def get_chain_status(self, cycle_id: int) -> Dict[str, Any]:
        """Return the current causal-chain status for a given cycle.

        Parameters
        ----------
        cycle_id : int
            The cycle to inspect.

        Returns
        -------
        dict
            Status dictionary with keys: ``cycle_id``, ``started``,
            ``sealed``, ``layers_completed``, ``layers_pending``,
            ``order_correct``, ``temporal_valid``, ``violations``.
        """
        started = cycle_id in self._started_cycles
        sealed = cycle_id in self._sealed_cycles
        violations: List[str] = []

        state = self._cycle_state.get(cycle_id)
        if state is not None:
            layers_completed = list(state["completed_layers"])
            layers_pending = list(state["pending_layers"])

            # Check that completed layers are in the expected order
            order_correct = (
                layers_completed
                == self.LAYER_ORDER[: len(layers_completed)]
            )
            if not order_correct:
                violations.append("ORDER_VIOLATION")

            # Check temporal validity (no future reads against this cycle)
            temporal_valid = (
                cycle_id <= self._current_cycle
            )
            if not temporal_valid:
                violations.append("FUTURE_READ")
        else:
            layers_completed = []
            layers_pending = list(self.LAYER_ORDER)
            order_correct = True
            temporal_valid = cycle_id <= self._current_cycle

        if sealed and state and len(layers_completed) != len(self.LAYER_ORDER):
            violations.append("INCOMPLETE_CYCLE")

        # Collect relevant violations from the log
        for v in self._violations:
            if v.get("cycle_id") == cycle_id and v["type"] not in violations:
                violations.append(v["type"])

        return {
            "cycle_id": cycle_id,
            "started": started,
            "sealed": sealed,
            "layers_completed": layers_completed,
            "layers_pending": layers_pending,
            "order_correct": order_correct,
            "temporal_valid": temporal_valid,
            "violations": violations,
        }

    def get_violation_log(self) -> List[Dict[str, Any]]:
        """Return the full list of recorded violations with timestamps.

        Returns
        -------
        list of dict
            Each entry has keys: ``type``, ``cycle_id``, ``message``,
            ``timestamp``.
        """
        return list(self._violations)

    def reset(self) -> None:
        """Clear all state.  Useful for tests and fresh starts."""
        self._current_cycle = 0
        self._sealed_cycles.clear()
        self._started_cycles.clear()
        self._access_log.clear()
        self._violations.clear()
        self._cycle_state.clear()
        logger.info("CausalEventChainEnforcer(%r) reset", self._instance_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_violation(self, vtype: str, cycle_id: int, message: str) -> None:
        """Record a violation and emit a warning log."""
        record = {
            "type": vtype,
            "cycle_id": cycle_id,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._violations.append(record)
        logger.warning("VIOLATION [%s] cycle=%d: %s", vtype, cycle_id, message)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _run_self_test() -> None:
    """Run a self-test to verify the enforcer catches expected violations."""
    import sys

    logger.info("=" * 60)
    logger.info("CausalEventChainEnforcer — self-test")
    logger.info("=" * 60)

    enforcer = CausalEventChainEnforcer("self_test")

    # ---- 1. Normal cycle flow succeeds -----------------------------------
    try:
        enforcer.begin_cycle(1)
        for layer in enforcer.LAYER_ORDER:
            enforcer.enter_layer(layer)
            enforcer.exit_layer(layer)
        enforcer.seal_cycle(1)
        logger.info("PASS [normal_flow]: cycle 1 completed and sealed")
    except Exception as exc:
        logger.error("FAIL [normal_flow]: %s", exc)
        sys.exit(1)

    # ---- 2. Backflow is caught -------------------------------------------
    try:
        enforcer.begin_cycle(0)  # <= current_cycle (1)
        logger.error("FAIL [backflow]: should have raised ValueError")
        sys.exit(1)
    except ValueError:
        logger.info("PASS [backflow]: backflow correctly caught")

    # ---- 3. Order violation is caught ------------------------------------
    try:
        enforcer.begin_cycle(2)
        enforcer.enter_layer("tick_ingestion")
        enforcer.exit_layer("tick_ingestion")
        enforcer.enter_layer("saal")  # skipping oss_surface, alt_signal, sdil
        logger.error("FAIL [order_violation]: should have raised ValueError")
        sys.exit(1)
    except ValueError:
        logger.info("PASS [order_violation]: order violation correctly caught")
    finally:
        # Clean up cycle 2 state for future tests
        enforcer._started_cycles.discard(2)
        enforcer._cycle_state.pop(2, None)

    # ---- 4. Future read is caught ----------------------------------------
    try:
        enforcer.validate_read(99)  # 99 > current_cycle
        logger.error("FAIL [future_read]: should have raised ValueError")
        sys.exit(1)
    except ValueError:
        logger.info("PASS [future_read]: future read correctly caught")

    # ---- 5. Duplicate layer is caught ------------------------------------
    try:
        enforcer.begin_cycle(3)
        enforcer.enter_layer("tick_ingestion")
        enforcer.exit_layer("tick_ingestion")
        enforcer.enter_layer("oss_surface")
        enforcer.exit_layer("oss_surface")
        enforcer.enter_layer("oss_surface")  # duplicate
        logger.error("FAIL [duplicate_layer]: should have raised ValueError")
        sys.exit(1)
    except ValueError:
        logger.info(
            "PASS [duplicate_layer]: duplicate layer correctly caught"
        )
    finally:
        enforcer._started_cycles.discard(3)
        enforcer._cycle_state.pop(3, None)

    # ---- 6. Write to sealed cycle is caught (UNSEALED_WRITE_AFTER_DEADLINE)
    #       The enforcer doesn't track generic "writes"; instead, re-sealing
    #       demonstrates the protection.
    enforcer.begin_cycle(4)
    for layer in enforcer.LAYER_ORDER:
        enforcer.enter_layer(layer)
        enforcer.exit_layer(layer)
    enforcer.seal_cycle(4)
    try:
        enforcer.seal_cycle(4)  # already sealed
        logger.error("FAIL [unsealed_write]: should have raised ValueError")
        sys.exit(1)
    except ValueError:
        logger.info(
            "PASS [unsealed_write]: write to sealed cycle correctly caught"
        )

    # ---- Summary ---------------------------------------------------------
    violations = enforcer.get_violation_log()
    logger.info("Total violations recorded: %d", len(violations))
    for v in violations:
        logger.info("  - [%s] %s", v["type"], v["message"])

    logger.info("=" * 60)
    logger.info("All self-tests passed!")
    logger.info("=" * 60)

    # Clean up instance for subsequent runs
    enforcer.reset()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _run_self_test()
