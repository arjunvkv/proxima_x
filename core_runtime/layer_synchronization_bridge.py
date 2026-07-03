"""
Layer Synchronization Bridge — guarantee SDIL, CSRF, and SAAL layers all
consume the SAME event object per tick cycle.

Not copies, not snapshots — a single shared reference that each layer's
registered callbacks mutate in place.

Layer processing order (per cycle, enforced by ``CausalEventChainEnforcer``):
    1. tick_ingestion  — verify raw data is populated
    2. oss_surface     — run registered OSS module callbacks
    3. alt_signal      — run registered ALT module callbacks
    4. sdil            — run registered SDIL callbacks
    5. csfr            — run registered CSRF callbacks
    6. saal            — run registered SAAL callbacks
    7. execution       — run registered execution callbacks
    8. seal            — finalise the cycle

Usage
-----
    from core_runtime.layer_synchronization_bridge import LayerSynchronizationBridge

    bridge = LayerSynchronizationBridge()
    bridge.register_oss_module("my_oss", my_callback)
    bridge.register_sdil_module("my_sdil", my_callback)
    # ...
    summary = bridge.process_cycle(42)
    status = bridge.get_layer_status()
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .event_canonical_kernel import (
    CanonicalEvent,
    EventCanonicalKernel,
)
from .causal_event_chain_enforcer import CausalEventChainEnforcer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton registry
# ---------------------------------------------------------------------------

_instances: Dict[str, "_LayerSynchronizationBridge"] = {}


def LayerSynchronizationBridge(
    instance_id: str = "default",
) -> "_LayerSynchronizationBridge":
    """Singleton accessor — returns the same ``_LayerSynchronizationBridge``
    for a given *instance_id*.

    Parameters
    ----------
    instance_id : str
        Logical identifier for the bridge instance (default ``"default"``).

    Returns
    -------
    _LayerSynchronizationBridge
    """
    if instance_id not in _instances:
        _instances[instance_id] = _LayerSynchronizationBridge(instance_id)
    return _instances[instance_id]


# ---------------------------------------------------------------------------
# Callback type alias
# ---------------------------------------------------------------------------

LayerCallback = Callable[[CanonicalEvent], Dict[str, Any]]
"""Signature: ``callback(event) -> dict`` where the returned dict contains
field-name → value updates that are merged back into the event via
``EventCanonicalKernel.update_event``.
"""


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------


class _LayerSynchronizationBridge:
    """Ensures every layer module sees the exact same event object within a
    tick cycle, with strict temporal ordering enforced by the causal chain.

    Parameters
    ----------
    instance_id : str
        Logical instance identifier (used for logging and singleton lookup).
    """

    # Ordered list of (layer_name, has_modules, handler_or_None) used by
    # ``process_cycle``.  The final "seal" step is handled by
    # ``enforcer.seal_cycle()`` rather than a layer callback.
    LAYER_PIPELINE: List[Tuple[str, bool, Optional[str]]] = [
        ("tick_ingestion", False, "verify_tick"),
        ("oss_surface", True, None),
        ("alt_signal", True, None),
        ("sdil", True, None),
        ("csfr", True, None),
        ("saal", True, None),
        ("execution", True, None),
        ("seal", False, None),  # enforcer requires entering/exiting "seal" layer
    ]

    def __init__(self, instance_id: str = "default") -> None:
        self._instance_id = instance_id

        # -- Dependencies (same instance_id for isolated testing) ----------
        self._kernel = EventCanonicalKernel(instance_id)
        self._enforcer = CausalEventChainEnforcer(instance_id)

        # -- The current event being processed (shared reference) ----------
        self._current_event: Optional[CanonicalEvent] = None

        # -- Module registries ---------------------------------------------
        self._oss_modules: Dict[str, LayerCallback] = {}
        self._alt_modules: Dict[str, LayerCallback] = {}
        self._sdil_modules: Dict[str, LayerCallback] = {}
        self._csfr_modules: Dict[str, LayerCallback] = {}
        self._saal_modules: Dict[str, LayerCallback] = {}
        self._execution_modules: Dict[str, LayerCallback] = {}

        # -- Processing history (cycle_id -> summary dict) -----------------
        self._processing_history: Dict[int, Dict[str, Any]] = {}

        logger.info("LayerSynchronizationBridge(%r) initialised", instance_id)

    # ------------------------------------------------------------------
    # Registration API
    # ------------------------------------------------------------------

    def register_oss_module(self, name: str, callback: LayerCallback) -> None:
        """Register an OSS-surface module callback.

        Parameters
        ----------
        name : str
            Unique module name.
        callback : LayerCallback
            ``callback(event) -> dict`` of field updates to merge.
        """
        self._oss_modules[name] = callback
        logger.debug("OSS module '%s' registered", name)

    def register_alt_module(self, name: str, callback: LayerCallback) -> None:
        """Register an ALT-signal module callback.

        Parameters
        ----------
        name : str
            Unique module name.
        callback : LayerCallback
            ``callback(event) -> dict`` of field updates to merge.
        """
        self._alt_modules[name] = callback
        logger.debug("ALT module '%s' registered", name)

    def register_sdil_module(self, name: str, callback: LayerCallback) -> None:
        """Register an SDIL (Signal Duality / Integrity Layer) module callback.

        Parameters
        ----------
        name : str
            Unique module name.
        callback : LayerCallback
            ``callback(event) -> dict`` of field updates to merge.
        """
        self._sdil_modules[name] = callback
        logger.debug("SDIL module '%s' registered", name)

    def register_csfr_module(self, name: str, callback: LayerCallback) -> None:
        """Register a CSRF (Cognitive Signal Reliability Framework) module
        callback.

        Parameters
        ----------
        name : str
            Unique module name.
        callback : LayerCallback
            ``callback(event) -> dict`` of field updates to merge.
        """
        self._csfr_modules[name] = callback
        logger.debug("CSRF module '%s' registered", name)

    def register_saal_module(self, name: str, callback: LayerCallback) -> None:
        """Register a SAAL (Signal Authority & Arbitration Layer) module
        callback.

        Parameters
        ----------
        name : str
            Unique module name.
        callback : LayerCallback
            ``callback(event) -> dict`` of field updates to merge.
        """
        self._saal_modules[name] = callback
        logger.debug("SAAL module '%s' registered", name)

    def register_execution_module(
        self, name: str, callback: LayerCallback
    ) -> None:
        """Register an Execution-layer module callback.

        Parameters
        ----------
        name : str
            Unique module name.
        callback : LayerCallback
            ``callback(event) -> dict`` of field updates to merge.
        """
        self._execution_modules[name] = callback
        logger.debug("Execution module '%s' registered", name)

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process_cycle(self, cycle_id: int) -> Dict[str, Any]:
        """Run one full processing cycle through all layers for *cycle_id*.

        The flow:
            1. Retrieve the :class:`CanonicalEvent` from the kernel.
            2. Ask the enforcer to ``begin_cycle``.
            3. Walk through each layer in the required order:
               ``enter_layer`` → run callbacks → merge updates → ``exit_layer``.
            4. ``seal_cycle`` on the enforcer.

        Parameters
        ----------
        cycle_id : int
            The cycle to process.  The event MUST already exist in the
            kernel (created via ``EventCanonicalKernel.create_event``).

        Returns
        -------
        dict
            Processing summary for this cycle (see ``get_processing_summary``).

        Raises
        ------
        ValueError
            If no event exists for *cycle_id*.
        """
        event = self._kernel.get_event(cycle_id)
        if event is None:
            raise ValueError(
                f"No canonical event found for cycle_id={cycle_id}. "
                "Call EventCanonicalKernel.create_event() first."
            )

        self._current_event = event
        enforcer = self._enforcer
        start_wall = time.time()

        # --- Begin the causal chain ---------------------------------------
        enforcer.begin_cycle(cycle_id)

        # --- Walk the layer pipeline --------------------------------------
        for layer_name, has_modules, handler_key in self.LAYER_PIPELINE:
            enforcer.enter_layer(layer_name)

            if handler_key == "verify_tick":
                self._ensure_tick_ingested(event)

            if has_modules:
                modules = self._get_modules_for_layer(layer_name)
                self._run_layer_modules(layer_name, modules, event)

            enforcer.exit_layer(layer_name)

        # --- Seal the cycle -----------------------------------------------
        enforcer.seal_cycle(cycle_id)

        elapsed = time.time() - start_wall
        self._current_event = None

        # --- Build & store summary ----------------------------------------
        summary = self._build_summary(cycle_id, elapsed)
        self._processing_history[cycle_id] = summary
        logger.info(
            "Cycle %d processed in %.4f s — %d total callback invocations",
            cycle_id,
            elapsed,
            summary.get("total_callbacks", 0),
        )

        return summary

    # ------------------------------------------------------------------
    # Status & introspection
    # ------------------------------------------------------------------

    def get_layer_status(self) -> Dict[str, Any]:
        """Return a snapshot of layer registration status.

        Returns
        -------
        dict
            Keys are layer names; each value is a dict with:
            ``"registered"`` count and ``"modules"`` (list of names).
        """
        return {
            "oss_surface": {
                "registered": len(self._oss_modules),
                "modules": sorted(self._oss_modules),
            },
            "alt_signal": {
                "registered": len(self._alt_modules),
                "modules": sorted(self._alt_modules),
            },
            "sdil": {
                "registered": len(self._sdil_modules),
                "modules": sorted(self._sdil_modules),
            },
            "csfr": {
                "registered": len(self._csfr_modules),
                "modules": sorted(self._csfr_modules),
            },
            "saal": {
                "registered": len(self._saal_modules),
                "modules": sorted(self._saal_modules),
            },
            "execution": {
                "registered": len(self._execution_modules),
                "modules": sorted(self._execution_modules),
            },
        }

    def get_processing_summary(self, cycle_id: int) -> Optional[Dict[str, Any]]:
        """Return the processing summary for a previously processed cycle.

        Parameters
        ----------
        cycle_id : int
            The cycle to inspect.

        Returns
        -------
        dict or None
            Summary dict if the cycle has been processed, else ``None``.
        """
        return self._processing_history.get(cycle_id)

    def get_current_event(self) -> Optional[CanonicalEvent]:
        """Return the event object currently being processed, or ``None``
        if no cycle is active."""
        return self._current_event

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all module registrations, processing history, and reset
        the enforcer.  Does **not** clear the event kernel — call
        ``EventCanonicalKernel.reset()`` separately if needed.
        """
        self._oss_modules.clear()
        self._alt_modules.clear()
        self._sdil_modules.clear()
        self._csfr_modules.clear()
        self._saal_modules.clear()
        self._execution_modules.clear()
        self._processing_history.clear()
        self._current_event = None
        self._enforcer.reset()
        logger.info("LayerSynchronizationBridge(%r) reset", self._instance_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_modules_for_layer(
        self, layer_name: str
    ) -> Dict[str, LayerCallback]:
        """Return the module registry dict for a given layer name."""
        mapping: Dict[str, Dict[str, LayerCallback]] = {
            "oss_surface": self._oss_modules,
            "alt_signal": self._alt_modules,
            "sdil": self._sdil_modules,
            "csfr": self._csfr_modules,
            "saal": self._saal_modules,
            "execution": self._execution_modules,
        }
        return mapping.get(layer_name, {})

    @staticmethod
    def _ensure_tick_ingested(event: CanonicalEvent) -> None:
        """Verify that raw tick data is populated for *event*.

        Logs a warning if critical fields appear to be zero / unset.
        """
        if event.bid == 0.0 and event.ask == 0.0:
            logger.warning(
                "Tick ingestion warning — cycle_id=%d symbol=%s "
                "has bid=0 and ask=0 (raw data may be missing)",
                event.cycle_id,
                event.symbol,
            )
        elif event.bid == 0.0:
            logger.warning(
                "Tick ingestion warning — cycle_id=%d symbol=%s bid=0",
                event.cycle_id,
                event.symbol,
            )
        elif event.ask == 0.0:
            logger.warning(
                "Tick ingestion warning — cycle_id=%d symbol=%s ask=0",
                event.cycle_id,
                event.symbol,
            )

    def _run_layer_modules(
        self,
        layer_name: str,
        modules: Dict[str, LayerCallback],
        event: CanonicalEvent,
    ) -> None:
        """Execute all registered callbacks for *layer_name*, catch
        exceptions, and merge returned updates into *event*.

        Each callback receives the **same** ``CanonicalEvent`` reference,
        not a copy.  Modifications made by early callbacks are visible to
        later callbacks within the same layer.
        """
        cycle_id = event.cycle_id
        for module_name, callback in modules.items():
            try:
                updates = callback(event)
                if updates and isinstance(updates, dict):
                    # Merge updates into the event kernel
                    self._kernel.update_event(cycle_id, **updates)
                elif updates is not None:
                    logger.warning(
                        "Layer '%s' module '%s' returned non-dict: %s",
                        layer_name,
                        module_name,
                        type(updates).__name__,
                    )
            except Exception:
                logger.exception(
                    "Layer '%s' module '%s' raised an exception (cycle=%d) — "
                    "continuing with next module",
                    layer_name,
                    module_name,
                    cycle_id,
                )

    def _build_summary(
        self, cycle_id: int, elapsed: float
    ) -> Dict[str, Any]:
        """Assemble a processing summary dict for *cycle_id*."""
        return {
            "cycle_id": cycle_id,
            "elapsed_seconds": round(elapsed, 6),
            "layers_processed": [l[0] for l in self.LAYER_PIPELINE],
            "total_callbacks": (
                len(self._oss_modules)
                + len(self._alt_modules)
                + len(self._sdil_modules)
                + len(self._csfr_modules)
                + len(self._saal_modules)
                + len(self._execution_modules)
            ),
            "layer_status": self.get_layer_status(),
            "chain_status": self._enforcer.get_chain_status(cycle_id),
        }


# ===========================================================================
# Self-test
# ===========================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("LayerSynchronizationBridge — Self Test")
    print("=" * 60)

    _state = {"passed": True}

    def _check(cond: bool, msg: str) -> None:
        if not cond:
            _state["passed"] = False
            print(f"  FAIL: {msg}")
        else:
            print(f"  PASS: {msg}")

    # ------------------------------------------------------------------
    # Helpers — mutable shared containers for cross-layer verification
    # ------------------------------------------------------------------

    # We use a plain list to track the *identity* of the event object each
    # callback receives.  This proves they all see the same reference.
    seen_ids: List[int] = []

    # Shared dict that modules write into; later modules can read prior writes.
    marker: Dict[str, Any] = {}

    def _make_oss_callback(name: str) -> LayerCallback:
        def _cb(event: CanonicalEvent) -> Dict[str, Any]:
            seen_ids.append(id(event))
            marker["oss_ran"] = True
            marker["oss_module"] = name
            # Verify tick layer already ran (bid/ask populated)
            assert event.bid > 0 or event.ask > 0
            return {"oss_signal": 1, "oss_confidence": 0.85}
        return _cb

    def _make_alt_callback(name: str) -> LayerCallback:
        def _cb(event: CanonicalEvent) -> Dict[str, Any]:
            seen_ids.append(id(event))
            marker["alt_ran"] = True
            marker["alt_module"] = name
            # OSS modifications must be visible
            assert event.oss_signal == 1, (
                f"OSS signal ({event.oss_signal}) should be 1"
            )
            assert event.oss_confidence == 0.85
            return {"alt_signal": -1, "alt_confidence": 0.72}
        return _cb

    def _make_sdil_callback(name: str) -> LayerCallback:
        def _cb(event: CanonicalEvent) -> Dict[str, Any]:
            seen_ids.append(id(event))
            marker["sdil_ran"] = True
            # Both OSS and ALT modifications must be visible
            assert event.oss_signal == 1
            assert event.alt_signal == -1
            return {
                "duality_verdict": "CONFLICT",
                "entropy_assessment": "MODERATE",
            }
        return _cb

    def _make_csfr_callback(name: str) -> LayerCallback:
        def _cb(event: CanonicalEvent) -> Dict[str, Any]:
            seen_ids.append(id(event))
            marker["csfr_ran"] = True
            # SDIL modifications must be visible
            assert event.duality_verdict == "CONFLICT"
            return {
                "collapse_verdict": "ACTIVE",
                "truth_label": "TRUTH",
            }
        return _cb

    def _make_saal_callback(name: str) -> LayerCallback:
        def _cb(event: CanonicalEvent) -> Dict[str, Any]:
            seen_ids.append(id(event))
            marker["saal_ran"] = True
            # CSRF modifications must be visible
            assert event.collapse_verdict == "ACTIVE"
            assert event.truth_label == "TRUTH"
            return {
                "authority": "OSS",
                "active_policy": "HYBRID",
                "consensus_signal": 1,
            }
        return _cb

    def _make_execution_callback(name: str) -> LayerCallback:
        def _cb(event: CanonicalEvent) -> Dict[str, Any]:
            seen_ids.append(id(event))
            marker["execution_ran"] = True
            # SAAL modifications must be visible
            assert event.authority == "OSS"
            assert event.active_policy == "HYBRID"
            assert event.consensus_signal == 1
            return {
                "execution_decision": "EXECUTE",
                "execution_signal": 1,
                "execution_reason": "All layers confirmed",
            }
        return _cb

    # ------------------------------------------------------------------
    # Scenario 1: Full cycle with all layers
    # ------------------------------------------------------------------

    print("\n--- SCE1: Full cycle with all 7 layers ---")

    bridge = LayerSynchronizationBridge("self_test")

    # Pre-seed an event in the kernel (normally done by tick ingestion)
    import time as _time

    kernel = EventCanonicalKernel("self_test")
    kernel.create_event(
        cycle_id=1,
        timestamp=_time.time(),
        symbol="EURUSD",
        bid=1.1050,
        ask=1.1052,
        spread=0.0002,
        volume=100.0,
        oss_p_cont=0.75,
        oss_ev=0.0025,
    )

    # Register one module per layer (each with cross-layer assertions)
    bridge.register_oss_module("oss_primary", _make_oss_callback("oss_primary"))
    bridge.register_alt_module("alt_primary", _make_alt_callback("alt_primary"))
    bridge.register_sdil_module("sdil_core", _make_sdil_callback("sdil_core"))
    bridge.register_csfr_module("csfr_core", _make_csfr_callback("csfr_core"))
    bridge.register_saal_module("saal_arbiter", _make_saal_callback("saal_arbiter"))
    bridge.register_execution_module(
        "exec_engine", _make_execution_callback("exec_engine")
    )

    # Run the cycle
    summary = bridge.process_cycle(1)

    print(f"  summary keys: {sorted(summary)}")
    print(f"  elapsed: {summary['elapsed_seconds']:.6f} s")
    print(f"  total_callbacks: {summary['total_callbacks']}")
    print(f"  layers_processed: {summary['layers_processed']}")

    _check(summary["cycle_id"] == 1, "Summary reports cycle_id=1")
    _check(
        summary["total_callbacks"] == 6,
        f"Expected 6 callbacks, got {summary['total_callbacks']}",
    )
    _check(
        len(summary["layers_processed"]) == 8,
        f"Expected 8 layers (incl. seal), got {len(summary['layers_processed'])}",
    )
    _check(
        summary["chain_status"]["sealed"],
        "Cycle 1 is sealed in enforcer",
    )

    # Verify the event object identity — all callbacks saw the SAME object
    first_id = seen_ids[0]
    all_same = all(eid == first_id for eid in seen_ids)
    _check(all_same, "All 6 callbacks received the SAME event object reference")

    # Verify the marker dict confirms every layer ran
    _check(marker.get("oss_ran"), "OSS layer ran")
    _check(marker.get("alt_ran"), "ALT layer ran")
    _check(marker.get("sdil_ran"), "SDIL layer ran")
    _check(marker.get("csfr_ran"), "CSRF layer ran")
    _check(marker.get("saal_ran"), "SAAL layer ran")
    _check(marker.get("execution_ran"), "Execution layer ran")

    # Verify the final event in the kernel has ALL fields populated correctly
    final_event = kernel.get_event(1)
    assert final_event is not None
    _check(
        final_event.oss_signal == 1,
        f"oss_signal == 1, got {final_event.oss_signal}",
    )
    _check(
        final_event.alt_signal == -1,
        f"alt_signal == -1, got {final_event.alt_signal}",
    )
    _check(
        final_event.duality_verdict == "CONFLICT",
        f"duality_verdict == 'CONFLICT', got {final_event.duality_verdict}",
    )
    _check(
        final_event.execution_decision == "EXECUTE",
        f"execution_decision == 'EXECUTE', got {final_event.execution_decision}",
    )
    _check(
        final_event.verify_integrity(),
        "Final event integrity hash is valid",
    )

    # ------------------------------------------------------------------
    # Scenario 2: get_layer_status()
    # ------------------------------------------------------------------

    print("\n--- SCE2: get_layer_status() ---")

    status = bridge.get_layer_status()
    _check(
        status["oss_surface"]["registered"] == 1,
        "1 OSS module registered",
    )
    _check(
        status["sdil"]["registered"] == 1,
        "1 SDIL module registered",
    )
    _check(
        status["csfr"]["modules"] == ["csfr_core"],
        "CSFR module name matches",
    )
    _check(
        status["execution"]["registered"] == 1,
        "1 Execution module registered",
    )

    # ------------------------------------------------------------------
    # Scenario 3: get_processing_summary()
    # ------------------------------------------------------------------

    print("\n--- SCE3: get_processing_summary() ---")

    retrieved = bridge.get_processing_summary(1)
    _check(retrieved is not None, "Processing summary exists for cycle 1")
    if retrieved:
        _check(retrieved["cycle_id"] == 1, "Summary cycle_id = 1")
        _check(retrieved["elapsed_seconds"] >= 0, "Elapsed time >= 0")

    # Non-existent cycle returns None
    _check(
        bridge.get_processing_summary(999) is None,
        "Non-existent cycle returns None",
    )

    # ------------------------------------------------------------------
    # Scenario 4: reset()
    # ------------------------------------------------------------------

    print("\n--- SCE4: reset() ---")

    bridge.reset()
    status_after = bridge.get_layer_status()
    _check(
        status_after["oss_surface"]["registered"] == 0,
        "0 OSS modules after reset",
    )
    _check(
        status_after["saal"]["registered"] == 0,
        "0 SAAL modules after reset",
    )
    _check(
        status_after["execution"]["registered"] == 0,
        "0 Execution modules after reset",
    )
    _check(
        bridge.get_processing_summary(1) is None,
        "Processing history cleared after reset",
    )
    _check(
        bridge.get_current_event() is None,
        "Current event is None after reset",
    )

    # ------------------------------------------------------------------
    # Scenario 5: Exception resilience
    # ------------------------------------------------------------------

    print("\n--- SCE5: Callback exception does not crash pipeline ---")

    bridge2 = LayerSynchronizationBridge("self_test_exc")

    def _failing_cb(event: CanonicalEvent) -> Dict[str, Any]:
        raise RuntimeError("Intentional failure for test")

    def _healthy_cb(event: CanonicalEvent) -> Dict[str, Any]:
        return {"oss_signal": 1, "oss_confidence": 0.99}

    bridge2.register_oss_module("failing", _failing_cb)
    bridge2.register_oss_module("healthy", _healthy_cb)

    kernel2 = EventCanonicalKernel("self_test_exc")
    kernel2.create_event(
        cycle_id=100,
        timestamp=_time.time(),
        symbol="GBPUSD",
        bid=1.2500,
        ask=1.2503,
        spread=0.0003,
    )

    try:
        summary2 = bridge2.process_cycle(100)
        _check(True, "process_cycle completed despite callback exception")
        ev = kernel2.get_event(100)
        _check(ev is not None, "Event still exists in kernel")
        if ev:
            _check(
                ev.oss_signal == 1,
                "Healthy callback updates applied despite failing sibling",
            )
            _check(
                ev.oss_confidence == 0.99,
                "oss_confidence from healthy callback applied",
            )
    except Exception as exc:
        _check(False, f"process_cycle raised: {exc}")

    # ------------------------------------------------------------------
    # Scenario 6: Singleton identity
    # ------------------------------------------------------------------

    print("\n--- SCE6: Singleton identity ---")

    bridge_default_1 = LayerSynchronizationBridge()
    bridge_default_2 = LayerSynchronizationBridge("default")
    bridge_other = LayerSynchronizationBridge("other")

    _check(
        bridge_default_1 is bridge_default_2,
        "Default singleton identity",
    )
    _check(
        bridge_other is not bridge_default_1,
        "Different instance_id returns different object",
    )
    _check(
        bridge is LayerSynchronizationBridge("self_test"),
        "Same instance_id returns same object after reset",
    )

    # ------------------------------------------------------------------
    # Scenario 7: Layer order enforcement
    # ------------------------------------------------------------------

    print("\n--- SCE7: Enforcer catches order violation ---")

    # Manually use the enforcer to verify that breaking order raises
    bridge3 = LayerSynchronizationBridge("self_test_order")
    enforcer3 = CausalEventChainEnforcer("self_test_order")

    try:
        enforcer3.begin_cycle(200)
        enforcer3.enter_layer("tick_ingestion")
        enforcer3.exit_layer("tick_ingestion")
        enforcer3.enter_layer("saal")  # should be oss_surface
        _check(False, "Order violation should have raised ValueError")
    except ValueError:
        _check(True, "Enforcer caught order violation: saal before oss_surface")
    except Exception as exc:
        _check(False, f"Unexpected exception: {exc}")

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------

    print()
    print("=" * 60)
    if _state["passed"]:
        print("ALL SELF-TESTS PASSED ✓")
    else:
        print("SOME SELF-TESTS FAILED ✗")
    print("=" * 60)

    sys.exit(0 if _state["passed"] else 1)
