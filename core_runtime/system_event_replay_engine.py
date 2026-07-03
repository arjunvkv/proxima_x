"""
System Event Replay Engine — Replay full system behaviour deterministically.

The replay engine loads events (from EventCanonicalKernel or a JSONL file),
re-feeds them through registered callbacks, compares outputs against original
outputs, and reports divergence.  It is the primary tool for debugging,
validation, and collapse detection.

Usage::

    from proxima_x.core_runtime.system_event_replay_engine import (
        SystemEventReplayEngine,
    )

    engine = SystemEventReplayEngine()

    # Register deterministic callbacks for each layer
    engine.register_callback("oss_surface", "p_cont", lambda e: {"p_cont": e["p_cont"]})
    engine.register_callback("signal_truth", "accuracy", lambda e: {"acc": e["oss_accuracy"]})

    # Register assertions
    engine.register_assertion("oss_surface", lambda orig, replay: orig == replay)

    # Replay specific cycles
    result = engine.replay([1, 2, 3])

    # Or replay all stored events
    result = engine.replay_all()

    # Save / load event files
    engine.save_events_to_file("events.jsonl")
    engine.load_events_from_file("events.jsonl")
"""

from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API — explicitly listed so ``from module import *`` is safe.
# ---------------------------------------------------------------------------
__all__ = [
    "SystemEventReplayEngine",
]

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

#: A replay callback receives the original event dict and returns a dict
#: of replayed outputs for that module.
ReplayCallback = Callable[[Dict[str, Any]], Dict[str, Any]]

#: An assertion function receives the original event and the replayed
#: output (returned by the callback) and returns True if they match.
AssertionFn = Callable[[Dict[str, Any], Dict[str, Any]], bool]


# ===================================================================
# Internal implementation class
# ===================================================================


class _SystemEventReplayEngine:
    """Replay engine that stores events and replays them through callbacks.

    This class should **not** be instantiated directly.  Use the module-level
    :func:`SystemEventReplayEngine` factory instead.
    """

    def __init__(self, instance_id: str = "default") -> None:
        self._instance_id: str = instance_id

        #: All stored events, keyed by cycle_id.
        #: Each value is a dict::
        #:   {
        #:       "cycle_id": int,
        #:       "layer_name": str,
        #:       "module_name": str,
        #:       "output": dict,       # original output
        #:       "timestamp": float,   # optional
        #:   }
        self._events: Dict[int, Dict[str, Any]] = {}

        #: Registered callbacks: layer_name -> {module_name -> callback}
        self._callbacks: Dict[str, Dict[str, ReplayCallback]] = {}

        #: Registered assertions: layer_name -> list of (check_fn)
        self._assertions: Dict[str, List[AssertionFn]] = {}

        #: Historical replay statistics
        self._replay_history: List[Dict[str, Any]] = []

        logger.info(
            "[SYSTEM_EVENT_REPLAY_ENGINE] Instance '%s' initialised.",
            instance_id,
        )

    # ------------------------------------------------------------------
    # Event ingestion
    # ------------------------------------------------------------------

    def ingest_event(
        self,
        cycle_id: int,
        layer_name: str,
        module_name: str,
        output: Dict[str, Any],
        timestamp: Optional[float] = None,
    ) -> None:
        """Store a single event for later replay.

        Parameters
        ----------
        cycle_id : int
            Unique cycle identifier (e.g. tick number or decision cycle).
        layer_name : str
            Logical layer name (e.g. ``"oss_surface"``, ``"signal_truth"``).
        module_name : str
            Module within the layer (e.g. ``"p_cont"``, ``"accuracy"``).
        output : dict
            The original output produced by this module for this cycle.
        timestamp : float, optional
            Optional UNIX timestamp for ordering.
        """
        self._events[cycle_id] = {
            "cycle_id": cycle_id,
            "layer_name": layer_name,
            "module_name": module_name,
            "output": deepcopy(output),
            "timestamp": timestamp,
        }
        logger.debug(
            "[SYSTEM_EVENT_REPLAY_ENGINE] Ingested event cycle_id=%d layer=%s module=%s",
            cycle_id,
            layer_name,
            module_name,
        )

    def ingest_events_batch(
        self,
        events: List[Dict[str, Any]],
    ) -> int:
        """Ingest multiple events at once.

        Parameters
        ----------
        events : list of dict
            Each dict must contain ``cycle_id``, ``layer_name``,
            ``module_name``, ``output``, and optionally ``timestamp``.

        Returns
        -------
        int
            Number of events ingested.
        """
        count = 0
        for ev in events:
            self.ingest_event(
                cycle_id=ev["cycle_id"],
                layer_name=ev["layer_name"],
                module_name=ev["module_name"],
                output=ev["output"],
                timestamp=ev.get("timestamp"),
            )
            count += 1
        logger.info(
            "[SYSTEM_EVENT_REPLAY_ENGINE] Batch ingested %d events.",
            count,
        )
        return count

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def register_callback(
        self,
        layer_name: str,
        module_name: str,
        callback: ReplayCallback,
    ) -> None:
        """Register a replay callback for a given layer and module.

        The callback signature is::

            callback(original_event: dict) -> dict

        It receives the *full* original event dict (including ``cycle_id``,
        ``layer_name``, ``module_name``, ``output``, etc.) and must return
        a dict representing the replayed output for that module.

        Parameters
        ----------
        layer_name : str
            Logical layer name.
        module_name : str
            Module name within the layer.
        callback : callable
            Replay callback function.
        """
        if layer_name not in self._callbacks:
            self._callbacks[layer_name] = {}
        self._callbacks[layer_name][module_name] = callback
        logger.debug(
            "[SYSTEM_EVENT_REPLAY_ENGINE] Registered callback: %s/%s",
            layer_name,
            module_name,
        )

    # ------------------------------------------------------------------
    # Assertion registration
    # ------------------------------------------------------------------

    def register_assertion(
        self,
        layer_name: str,
        check_fn: AssertionFn,
    ) -> None:
        """Register an assertion function for a given layer.

        The assertion signature is::

            check_fn(original_event: dict, replayed_output: dict) -> bool

        It receives the *original event dict* and the *replayed output dict*
        (the result of the callback).  Return ``True`` if the replayed output
        matches the original, ``False`` if divergence is detected.

        Parameters
        ----------
        layer_name : str
            Logical layer name.
        check_fn : callable
            Assertion function.
        """
        if layer_name not in self._assertions:
            self._assertions[layer_name] = []
        self._assertions[layer_name].append(check_fn)
        logger.debug(
            "[SYSTEM_EVENT_REPLAY_ENGINE] Registered assertion for layer: %s",
            layer_name,
        )

    # ------------------------------------------------------------------
    # Core replay logic
    # ------------------------------------------------------------------

    def _replay_single_event(
        self,
        event: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Replay a single event through its registered callbacks.

        Parameters
        ----------
        event : dict
            The stored event (includes ``cycle_id``, ``layer_name``,
            ``module_name``, ``output``, etc.).

        Returns
        -------
        dict
            The replayed output from the callback, or an error dict if no
            callback is registered.
        """
        layer = event["layer_name"]
        module = event["module_name"]

        layer_callbacks = self._callbacks.get(layer, {})
        callback = layer_callbacks.get(module)

        if callback is None:
            logger.warning(
                "[SYSTEM_EVENT_REPLAY_ENGINE] No callback registered for %s/%s "
                "(cycle_id=%d). Returning empty dict.",
                layer,
                module,
                event["cycle_id"],
            )
            return {}

        try:
            return callback(event)
        except Exception as exc:
            logger.error(
                "[SYSTEM_EVENT_REPLAY_ENGINE] Callback %s/%s raised an exception "
                "for cycle_id=%d: %s",
                layer,
                module,
                event["cycle_id"],
                exc,
            )
            return {"__error__": str(exc), "__exception_type__": type(exc).__name__}

    def _run_assertions_for_event(
        self,
        event: Dict[str, Any],
        replayed_output: Dict[str, Any],
    ) -> Tuple[int, int, Optional[str]]:
        """Run all registered assertions for the event's layer.

        Parameters
        ----------
        event : dict
            The original stored event.
        replayed_output : dict
            The output produced by the replay callback.

        Returns
        -------
        tuple of (passed, failed, first_diverging_field)
            ``first_diverging_field`` is ``None`` if all passed.
        """
        layer = event["layer_name"]
        checks = self._assertions.get(layer, [])
        passed = 0
        failed = 0
        first_diverging_field: Optional[str] = None

        if not checks:
            # No assertions registered — auto-pass
            return 1, 0, None

        for check_fn in checks:
            try:
                ok = check_fn(event, replayed_output)
            except Exception as exc:
                logger.warning(
                    "[SYSTEM_EVENT_REPLAY_ENGINE] Assertion %s raised: %s",
                    layer,
                    exc,
                )
                ok = False

            if ok:
                passed += 1
            else:
                failed += 1
                if first_diverging_field is None:
                    first_diverging_field = f"{layer}/assertion"

        return passed, failed, first_diverging_field

    # ------------------------------------------------------------------
    # Public replay API
    # ------------------------------------------------------------------

    def replay(self, cycle_ids: List[int]) -> Dict[str, Any]:
        """Replay specific cycles through registered callbacks.

        Parameters
        ----------
        cycle_ids : list of int
            Cycle IDs to replay.

        Returns
        -------
        dict
            Replay result containing summary statistics and per-layer detail.
        """
        cycles_replayed = 0
        total_assertions = 0
        total_passed = 0
        total_failed = 0
        first_divergence_cycle: Optional[int] = None
        per_layer: Dict[str, Dict[str, Any]] = {}

        for cid in cycle_ids:
            if cid not in self._events:
                logger.warning(
                    "[SYSTEM_EVENT_REPLAY_ENGINE] cycle_id=%d not found. Skipping.",
                    cid,
                )
                continue

            event = self._events[cid]
            replayed = self._replay_single_event(event)
            p, f, _ = self._run_assertions_for_event(event, replayed)
            total_passed += p
            total_failed += f
            total_assertions += p + f
            cycles_replayed += 1

            layer = event["layer_name"]
            if layer not in per_layer:
                per_layer[layer] = {"passed": 0, "failed": 0}
            per_layer[layer]["passed"] += p
            per_layer[layer]["failed"] += f

            if f > 0 and first_divergence_cycle is None:
                first_divergence_cycle = cid

        # Compute per-layer divergence rates
        for layer in per_layer:
            total = per_layer[layer]["passed"] + per_layer[layer]["failed"]
            per_layer[layer]["divergence_rate"] = (
                round(per_layer[layer]["failed"] / total, 4) if total > 0 else 0.0
            )

        total = total_passed + total_failed
        divergence_rate = round(total_failed / total, 4) if total > 0 else 0.0
        deterministic = divergence_rate == 0.0

        if deterministic:
            verdict = "DETERMINISTIC"
        elif divergence_rate < 0.5:
            verdict = "PARTIAL"
        else:
            verdict = "NON_DETERMINISTIC"

        result = {
            "cycles_replayed": cycles_replayed,
            "total_assertions": total_assertions,
            "passed": total_passed,
            "failed": total_failed,
            "divergence_rate": divergence_rate,
            "first_divergence_cycle": first_divergence_cycle,
            "per_layer": per_layer,
            "deterministic": deterministic,
            "verdict": verdict,
        }

        self._replay_history.append(result)

        logger.info(
            "[SYSTEM_EVENT_REPLAY_ENGINE] Replay result: %s verdict=%s "
            "passed=%d failed=%d divergence=%.4f",
            self._instance_id,
            verdict,
            total_passed,
            total_failed,
            divergence_rate,
        )

        return result

    def replay_all(self) -> Dict[str, Any]:
        """Replay all stored events.

        Returns
        -------
        dict
            Replay result with the same structure as :meth:`replay`.
        """
        cycle_ids = sorted(self._events.keys())
        if not cycle_ids:
            logger.warning(
                "[SYSTEM_EVENT_REPLAY_ENGINE] No events stored. replay_all() returns empty result.",
            )
            return {
                "cycles_replayed": 0,
                "total_assertions": 0,
                "passed": 0,
                "failed": 0,
                "divergence_rate": 0.0,
                "first_divergence_cycle": None,
                "per_layer": {},
                "deterministic": True,
                "verdict": "DETERMINISTIC",
            }
        return self.replay(cycle_ids)

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare(
        self,
        original_event: Dict[str, Any],
        replayed_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compare an original event with its replayed output field by field.

        Parameters
        ----------
        original_event : dict
            The original stored event (must contain an ``output`` key).
        replayed_output : dict
            The dict returned by the replay callback.

        Returns
        -------
        dict with keys:
            cycle_id : int
            fields_matching : list of str
            fields_diverging : list of str
            match_rate : float
            is_identical : bool
        """
        original_output = original_event.get("output", {})
        cycle_id = original_event.get("cycle_id", -1)

        # Flatten nested dicts to dotted paths for comparison
        def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
            result: Dict[str, Any] = {}
            for k, v in d.items():
                full_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    result.update(_flatten(v, full_key))
                else:
                    result[full_key] = v
            return result

        orig_flat = _flatten(original_output)
        replay_flat = _flatten(replayed_output)

        all_keys = set(orig_flat.keys()) | set(replay_flat.keys())
        fields_matching: List[str] = []
        fields_diverging: List[str] = []

        for key in sorted(all_keys):
            ov = orig_flat.get(key)
            rv = replay_flat.get(key)
            if ov == rv:
                fields_matching.append(key)
            else:
                fields_diverging.append(key)

        total = len(fields_matching) + len(fields_diverging)
        match_rate = round(len(fields_matching) / total, 4) if total > 0 else 1.0
        is_identical = len(fields_diverging) == 0

        return {
            "cycle_id": cycle_id,
            "fields_matching": fields_matching,
            "fields_diverging": fields_diverging,
            "match_rate": match_rate,
            "is_identical": is_identical,
        }

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def load_events_from_file(self, filepath: str) -> int:
        """Load events from a JSONL file (one JSON event per line).

        Parameters
        ----------
        filepath : str
            Path to the JSONL file.

        Returns
        -------
        int
            Number of events loaded.
        """
        count = 0
        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    ev = json.loads(line)
                    self.ingest_event(
                        cycle_id=ev["cycle_id"],
                        layer_name=ev["layer_name"],
                        module_name=ev["module_name"],
                        output=ev["output"],
                        timestamp=ev.get("timestamp"),
                    )
                    count += 1
        except FileNotFoundError:
            logger.error(
                "[SYSTEM_EVENT_REPLAY_ENGINE] File not found: %s",
                filepath,
            )
            return 0
        except (json.JSONDecodeError, KeyError) as exc:
            logger.error(
                "[SYSTEM_EVENT_REPLAY_ENGINE] Error parsing %s: %s",
                filepath,
                exc,
            )
            return count  # Return what was loaded so far

        logger.info(
            "[SYSTEM_EVENT_REPLAY_ENGINE] Loaded %d events from %s",
            count,
            filepath,
        )
        return count

    def save_events_to_file(self, filepath: str) -> int:
        """Save all stored events to a JSONL file.

        Parameters
        ----------
        filepath : str
            Path to the output JSONL file.

        Returns
        -------
        int
            Number of events saved.
        """
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        count = 0
        with open(filepath, "w", encoding="utf-8") as fh:
            for cid in sorted(self._events.keys()):
                ev = self._events[cid]
                fh.write(json.dumps(ev, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1

        logger.info(
            "[SYSTEM_EVENT_REPLAY_ENGINE] Saved %d events to %s",
            count,
            filepath,
        )
        return count

    # ------------------------------------------------------------------
    # Statistics & introspection
    # ------------------------------------------------------------------

    def get_replay_statistics(self) -> Dict[str, Any]:
        """Return historical replay statistics.

        Returns
        -------
        dict with keys:
            total_replays : int
            last_replay : dict or None
            all_verdicts : list of str
            total_divergences : int
            stored_events : int
            registered_layers : list of str
            registered_callbacks : int
            registered_assertions : int
        """
        total_divergences = sum(
            1 for r in self._replay_history if r["failed"] > 0
        )
        return {
            "total_replays": len(self._replay_history),
            "last_replay": self._replay_history[-1] if self._replay_history else None,
            "all_verdicts": [r["verdict"] for r in self._replay_history],
            "total_divergences": total_divergences,
            "stored_events": len(self._events),
            "registered_layers": list(self._callbacks.keys()),
            "registered_callbacks": sum(
                len(mods) for mods in self._callbacks.values()
            ),
            "registered_assertions": sum(
                len(checks) for checks in self._assertions.values()
            ),
        }

    def get_events(self) -> Dict[int, Dict[str, Any]]:
        """Return a shallow copy of all stored events.

        Returns
        -------
        dict
            Mapping ``{cycle_id: event_dict, ...}``.
        """
        return dict(self._events)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all stored events, callbacks, assertions, and history."""
        self._events.clear()
        self._callbacks.clear()
        self._assertions.clear()
        self._replay_history.clear()
        logger.info(
            "[SYSTEM_EVENT_REPLAY_ENGINE] Instance '%s' reset.",
            self._instance_id,
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<_SystemEventReplayEngine id='{self._instance_id}' "
            f"events={len(self._events)} "
            f"callbacks={sum(len(m) for m in self._callbacks.values())} "
            f"assertions={sum(len(a) for a in self._assertions.values())}>"
        )


# ===================================================================
# Singleton accessor pattern
# ===================================================================

_instances: Dict[str, _SystemEventReplayEngine] = {}


def SystemEventReplayEngine(
    instance_id: str = "default",
) -> _SystemEventReplayEngine:
    """Return a shared ``_SystemEventReplayEngine`` instance.

    This is the **only** way to obtain a system event replay engine.
    It implements a simple registry of singletons keyed by *instance_id*.

    Parameters
    ----------
    instance_id : str
        Identifier for the replay engine instance.  Use ``"default"`` (or
        omit) for the global singleton.  Pass a unique string to
        create/maintain an independent engine for a specific subsystem.

    Returns
    -------
    _SystemEventReplayEngine
        The shared instance for the given *instance_id*.

    Usage::

        from proxima_x.core_runtime.system_event_replay_engine import (
            SystemEventReplayEngine,
        )

        engine = SystemEventReplayEngine()
        engine.register_callback("layer", "module", my_callback)
        result = engine.replay_all()
    """
    if instance_id not in _instances:
        _instances[instance_id] = _SystemEventReplayEngine(
            instance_id=instance_id,
        )
    return _instances[instance_id]


# ===================================================================
# Self-test (only when run directly)
# ===================================================================

def _selftest() -> None:
    """Run comprehensive self-test covering all major features."""
    import tempfile

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("=" * 72)
    logger.info("SystemEventReplayEngine — Self-Test")
    logger.info("=" * 72)

    passed = 0
    failed = 0

    def _check(label: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if condition:
            passed += 1
            logger.info("  [PASS] %s", label)
        else:
            failed += 1
            msg = f"  [FAIL] {label}"
            if detail:
                msg += f" — {detail}"
            logger.warning(msg)

    # ------------------------------------------------------------------
    # Scenario 1: Ingest events and replay with deterministic callbacks
    # ------------------------------------------------------------------
    logger.info("-" * 72)
    logger.info("Scenario 1: Deterministic replay")
    logger.info("-" * 72)

    engine = SystemEventReplayEngine("selftest_deterministic")
    engine.reset()

    # Ingest events
    engine.ingest_event(
        cycle_id=1,
        layer_name="oss_surface",
        module_name="p_cont",
        output={"p_cont": 0.55, "direction": 1},
    )
    engine.ingest_event(
        cycle_id=2,
        layer_name="oss_surface",
        module_name="p_cont",
        output={"p_cont": 0.48, "direction": -1},
    )
    engine.ingest_event(
        cycle_id=3,
        layer_name="signal_truth",
        module_name="accuracy",
        output={"oss_accuracy": 0.62, "samples": 500},
    )

    _check("3 events ingested", len(engine._events) == 3)

    # Register deterministic callbacks (return same output as original)
    engine.register_callback(
        "oss_surface",
        "p_cont",
        lambda e: dict(e["output"]),
    )
    engine.register_callback(
        "signal_truth",
        "accuracy",
        lambda e: dict(e["output"]),
    )

    _check("callbacks registered", len(engine._callbacks) == 2)

    # Register assertions that check exact match
    engine.register_assertion(
        "oss_surface",
        lambda orig, replay: orig["output"] == replay,
    )
    engine.register_assertion(
        "signal_truth",
        lambda orig, replay: orig["output"] == replay,
    )

    _check("assertions registered", len(engine._assertions) == 2)

    # Replay specific cycles
    result = engine.replay([1, 2])

    _check("cycles_replayed == 2", result["cycles_replayed"] == 2)
    _check("total_assertions == 2", result["total_assertions"] == 2)
    _check("passed == 2", result["passed"] == 2)
    _check("failed == 0", result["failed"] == 0)
    _check("divergence_rate == 0.0", result["divergence_rate"] == 0.0)
    _check(
        "first_divergence_cycle is None",
        result["first_divergence_cycle"] is None,
    )
    _check("deterministic is True", result["deterministic"] is True)
    _check(
        "verdict == DETERMINISTIC",
        result["verdict"] == "DETERMINISTIC",
    )

    _check(
        "per_layer has oss_surface",
        "oss_surface" in result["per_layer"],
    )
    _check(
        "per_layer oss_surface passed == 2",
        result["per_layer"]["oss_surface"]["passed"] == 2,
    )
    _check(
        "per_layer oss_surface failed == 0",
        result["per_layer"]["oss_surface"]["failed"] == 0,
    )
    _check(
        "per_layer does NOT have signal_truth (not replayed)",
        "signal_truth" not in result["per_layer"],
    )

    # Compare individual event
    cmp = engine.compare(
        engine._events[1],
        {"p_cont": 0.55, "direction": 1},
    )
    _check("compare is_identical True", cmp["is_identical"] is True)
    _check(
        "compare match_rate == 1.0",
        cmp["match_rate"] == 1.0,
    )

    # ------------------------------------------------------------------
    # Scenario 2: Non-deterministic callback -> divergence detection
    # ------------------------------------------------------------------
    logger.info("-" * 72)
    logger.info("Scenario 2: Non-deterministic replay -> divergence detected")
    logger.info("-" * 72)

    engine2 = SystemEventReplayEngine("selftest_nondeterministic")
    engine2.reset()

    # Ingest a single event
    engine2.ingest_event(
        cycle_id=10,
        layer_name="random_layer",
        module_name="noise_generator",
        output={"value": 42},
    )

    # Register a callback that returns a DIFFERENT value (non-deterministic)
    engine2.register_callback(
        "random_layer",
        "noise_generator",
        lambda e: {"value": 99},  # deliberate mismatch
    )

    engine2.register_assertion(
        "random_layer",
        lambda orig, replay: orig["output"] == replay,
    )

    result2 = engine2.replay([10])

    _check("nondet: cycles_replayed == 1", result2["cycles_replayed"] == 1)
    _check("nondet: passed == 0", result2["passed"] == 0)
    _check("nondet: failed == 1", result2["failed"] == 1)
    _check(
        "nondet: divergence_rate == 1.0",
        result2["divergence_rate"] == 1.0,
    )
    _check(
        "nondet: first_divergence_cycle == 10",
        result2["first_divergence_cycle"] == 10,
    )
    _check("nondet: deterministic is False", result2["deterministic"] is False)
    _check(
        "nondet: verdict == NON_DETERMINISTIC",
        result2["verdict"] == "NON_DETERMINISTIC",
    )

    # Compare should show divergence
    cmp2 = engine2.compare(
        engine2._events[10],
        {"value": 99},
    )
    _check("nondet: compare is_identical False", cmp2["is_identical"] is False)
    _check(
        "nondet: fields_diverging contains 'value'",
        "value" in cmp2["fields_diverging"],
    )
    _check(
        "nondet: match_rate == 0.0",
        cmp2["match_rate"] == 0.0,
    )

    # ------------------------------------------------------------------
    # Scenario 3: Missed callback -> warning, not crash
    # ------------------------------------------------------------------
    logger.info("-" * 72)
    logger.info("Scenario 3: Missing callback handling")
    logger.info("-" * 72)

    engine3 = SystemEventReplayEngine("selftest_missing_cb")
    engine3.reset()
    engine3.ingest_event(
        cycle_id=100,
        layer_name="ghost_layer",
        module_name="nonexistent",
        output={"x": 1},
    )

    # No callback registered for ghost_layer/nonexistent
    result3 = engine3.replay([100])
    _check(
        "missing_cb: cycles_replayed == 1",
        result3["cycles_replayed"] == 1,
    )
    _check(
        "missing_cb: no assertion crash",
        result3["failed"] == 0,
    )

    # ------------------------------------------------------------------
    # Scenario 4: Save / load events round-trip
    # ------------------------------------------------------------------
    logger.info("-" * 72)
    logger.info("Scenario 4: Save / load events from JSONL")
    logger.info("-" * 72)

    engine4 = SystemEventReplayEngine("selftest_file_io")
    engine4.reset()

    engine4.ingest_event(
        cycle_id=1,
        layer_name="layer_a",
        module_name="mod_x",
        output={"a": 1, "b": 2},
    )
    engine4.ingest_event(
        cycle_id=2,
        layer_name="layer_b",
        module_name="mod_y",
        output={"c": 3},
        timestamp=12345.0,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".jsonl",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        tmp_path = tmp.name

    try:
        saved = engine4.save_events_to_file(tmp_path)
        _check("saved 2 events", saved == 2)

        # Create a fresh engine and load
        engine4b = SystemEventReplayEngine("selftest_file_io_loaded")
        engine4b.reset()
        loaded = engine4b.load_events_from_file(tmp_path)
        _check("loaded 2 events", loaded == 2)

        _check(
            "loaded events match keys",
            engine4b._events.keys() == engine4._events.keys(),
        )
        _check(
            "loaded event 1 output matches",
            engine4b._events[1]["output"] == engine4._events[1]["output"],
        )
        _check(
            "loaded event 2 timestamp matches",
            engine4b._events[2]["timestamp"] == engine4._events[2]["timestamp"],
        )

    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Scenario 5: replay_all on empty engine
    # ------------------------------------------------------------------
    logger.info("-" * 72)
    logger.info("Scenario 5: replay_all on empty engine")
    logger.info("-" * 72)

    engine5 = SystemEventReplayEngine("selftest_empty")
    engine5.reset()
    result5 = engine5.replay_all()
    _check("empty replay: cycles_replayed == 0", result5["cycles_replayed"] == 0)
    _check("empty replay: verdict == DETERMINISTIC", result5["verdict"] == "DETERMINISTIC")

    # ------------------------------------------------------------------
    # Scenario 6: Replay statistics
    # ------------------------------------------------------------------
    logger.info("-" * 72)
    logger.info("Scenario 6: get_replay_statistics")
    logger.info("-" * 72)

    stats = engine.get_replay_statistics()
    _check("stats: total_replays >= 1", stats["total_replays"] >= 1)
    _check("stats: last_replay is not None", stats["last_replay"] is not None)
    _check("stats: stored_events == 3", stats["stored_events"] == 3)
    _check(
        "stats: 'oss_surface' in registered_layers",
        "oss_surface" in stats["registered_layers"],
    )
    _check(
        "stats: registered_callbacks == 2",
        stats["registered_callbacks"] == 2,
    )
    _check(
        "stats: registered_assertions == 2",
        stats["registered_assertions"] == 2,
    )

    # ------------------------------------------------------------------
    # Scenario 7: Reset
    # ------------------------------------------------------------------
    logger.info("-" * 72)
    logger.info("Scenario 7: Reset")
    logger.info("-" * 72)

    engine.reset()
    _check("reset: events empty", len(engine._events) == 0)
    _check("reset: callbacks empty", len(engine._callbacks) == 0)
    _check("reset: assertions empty", len(engine._assertions) == 0)
    _check("reset: history empty", len(engine._replay_history) == 0)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("=" * 72)
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
