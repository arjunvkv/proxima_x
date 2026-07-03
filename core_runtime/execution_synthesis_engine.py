"""
Execution Synthesis Engine — takes the resolved decision vector (after conflict
resolution across SDIL, CSRF, SAAL, MRSRL) and produces ONE final decision:
BUY, SELL, HOLD, or SKIP.

NOT per-layer decisions. ONE final action.

This is the final gate of the Unified Execution Synthesis Layer (UESL) — the
system that resolves disagreement between all cognitive layers in real time.

Usage
-----
    from core_runtime.execution_synthesis_engine import ExecutionSynthesisEngine

    engine = ExecutionSynthesisEngine()
    result = engine.synthesize(resolved_vector, decision_metadata)
    # result["decision"] in ("BUY", "SELL", "HOLD", "SKIP")
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton registry
# ---------------------------------------------------------------------------

_instances: Dict[str, "_ExecutionSynthesisEngine"] = {}


def ExecutionSynthesisEngine(instance_id: str = "default"):
    """Singleton accessor — returns the same ``_ExecutionSynthesisEngine``
    for a given *instance_id*.

    Parameters
    ----------
    instance_id : str
        Logical identifier (default ``"default"``).

    Returns
    -------
    _ExecutionSynthesisEngine
    """
    if instance_id not in _instances:
        logger.info(
            "Creating new ExecutionSynthesisEngine instance '%s'", instance_id
        )
        _instances[instance_id] = _ExecutionSynthesisEngine(instance_id)
    return _instances[instance_id]


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------


def _make_decision_id() -> str:
    """Generate a simple unique decision ID (UUID-like)."""
    import uuid
    return str(uuid.uuid4())


class _ExecutionSynthesisEngine:
    """Synthesises a single final decision from the resolved multi-layer vector.

    Parameters
    ----------
    instance_id : str
        Logical instance identifier (used for logging and singleton lookup).
    """

    def __init__(self, instance_id: str = "default"):
        self._instance_id = instance_id

        # ── Thresholds ──────────────────────────────────────────────────────
        self._min_confidence: float = 0.4
        self._min_ev: float = 0.0
        self._max_flat_consecutive: int = 10

        # ── Statistics ──────────────────────────────────────────────────────
        self._decision_counts: Counter = Counter()       # BUY / SELL / HOLD / SKIP
        self._total_decisions: int = 0
        self._total_correct: int = 0
        self._total_incorrect: int = 0
        self._decision_history: List[Dict[str, Any]] = []

        # ── Consecutive flat tracking ───────────────────────────────────────
        self._consecutive_flat: int = 0

        logger.info(
            "ExecutionSynthesisEngine(%r) initialised: min_conf=%.2f "
            "min_ev=%.2f max_flat=%d",
            instance_id, self._min_confidence, self._min_ev,
            self._max_flat_consecutive,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(
        self,
        resolved_vector: dict,
        decision_metadata: dict,
    ) -> Dict[str, Any]:
        """Produce the ONE final decision from the resolved multi-layer vector.

        Parameters
        ----------
        resolved_vector : dict
            The resolved conflict output. Expected keys (at minimum):
                ``signal``          — int: -1, 0, or +1
                ``confidence``      — float [0, 1]
                ``authority``       — str: ``"OSS"``, ``"ALT"``, ``"HYBRID"``, ``"NONE"``
                ``conflict_matrix`` — dict of ``{layer_name: veto_info}``
                ``expected_value``  — float (optional, may be computed here)
            Additional keys are forwarded to the decision record.

        decision_metadata : dict
            Contextual metadata about this decision cycle. Expected keys:
                ``regime``          — str: current regime classification
                ``spread``          — float (optional)
                ``recent_accuracy`` — float [0, 1] (optional, default 0.5)
            Additional keys are forwarded to the decision record.

        Returns
        -------
        dict with keys:
            decision           — ``"BUY"`` | ``"SELL"`` | ``"HOLD"`` | ``"SKIP"``
            signal             — ``+1`` | ``-1`` | ``0``
            confidence         — float [0, 1]
            expected_value     — float
            reason             — str
            contributing_layers — list of str
            vetoing_layers     — list of str
            decision_id        — str (UUID)
        """
        # ── Extract resolved fields ─────────────────────────────────────
        resolved_signal = resolved_vector.get("signal", 0)
        resolved_confidence = resolved_vector.get("confidence", 0.0)
        resolved_authority = resolved_vector.get("authority", "NONE")
        conflict_matrix = resolved_vector.get("conflict_matrix", {})
        resolved_ev = resolved_vector.get("expected_value", None)

        # ── Extract metadata fields ─────────────────────────────────────
        regime = decision_metadata.get("regime", "unknown")
        recent_accuracy = decision_metadata.get("recent_accuracy", 0.5)

        # ── Determine vetoing / contributing layers ─────────────────────
        vetoing_layers: List[str] = []
        contributing_layers: List[str] = []

        if isinstance(conflict_matrix, dict):
            for layer_name, veto_info in conflict_matrix.items():
                if isinstance(veto_info, dict):
                    if veto_info.get("veto", False):
                        vetoing_layers.append(layer_name)
                    elif veto_info.get("contribute", False):
                        contributing_layers.append(layer_name)
                elif veto_info is True:
                    vetoing_layers.append(layer_name)
                elif veto_info is False:
                    contributing_layers.append(layer_name)

        # ── Decision ID ─────────────────────────────────────────────────
        decision_id = _make_decision_id()

        # ── Step 1: Any layer veto? ─────────────────────────────────────
        if vetoing_layers:
            return self._build_result(
                decision="SKIP",
                signal=0,
                confidence=0.0,
                expected_value=0.0,
                reason=f"Layer veto from: {', '.join(vetoing_layers)}",
                contributing_layers=contributing_layers,
                vetoing_layers=vetoing_layers,
                decision_id=decision_id,
                resolved_vector=resolved_vector,
                decision_metadata=decision_metadata,
            )

        # ── Step 2: No conviction (flat signal) ─────────────────────────
        if resolved_signal == 0:
            return self._build_result(
                decision="HOLD",
                signal=0,
                confidence=0.0,
                expected_value=0.0,
                reason="No conviction — resolved signal is neutral",
                contributing_layers=contributing_layers,
                vetoing_layers=vetoing_layers,
                decision_id=decision_id,
                resolved_vector=resolved_vector,
                decision_metadata=decision_metadata,
            )

        # ── Step 3: No authority ────────────────────────────────────────
        if resolved_authority == "NONE":
            return self._build_result(
                decision="SKIP",
                signal=0,
                confidence=0.0,
                expected_value=0.0,
                reason="No authority — resolved_authority is NONE",
                contributing_layers=contributing_layers,
                vetoing_layers=vetoing_layers,
                decision_id=decision_id,
                resolved_vector=resolved_vector,
                decision_metadata=decision_metadata,
            )

        # ── Step 4: Compute expected_value ──────────────────────────────
        # EV = confidence × recent_accuracy (always magnitude; signal
        # direction is carried separately).
        ev_explicitly_set = resolved_ev is not None
        if resolved_ev is None:
            resolved_ev = resolved_confidence * recent_accuracy
        # If the caller provided an explicit EV, honour its sign; otherwise
        # the computed EV is a magnitude (always non-negative).
        if not ev_explicitly_set:
            resolved_ev = abs(resolved_ev)

        # ── Step 5: Zero / negligible expected value → SKIP ──────────────
        if resolved_ev <= self._min_ev:
            return self._build_result(
                decision="SKIP",
                signal=resolved_signal,
                confidence=resolved_confidence,
                expected_value=resolved_ev,
                reason=(
                    f"Expected value {resolved_ev:.4f} <= min_ev "
                    f"({self._min_ev})"
                ),
                contributing_layers=contributing_layers,
                vetoing_layers=vetoing_layers,
                decision_id=decision_id,
                resolved_vector=resolved_vector,
                decision_metadata=decision_metadata,
            )

        # ── Step 6: Low conviction → HOLD ───────────────────────────────
        if resolved_confidence < self._min_confidence:
            return self._build_result(
                decision="HOLD",
                signal=resolved_signal,
                confidence=resolved_confidence,
                expected_value=resolved_ev,
                reason=(
                    f"Low conviction — confidence {resolved_confidence:.4f} "
                    f"< min_confidence ({self._min_confidence})"
                ),
                contributing_layers=contributing_layers,
                vetoing_layers=vetoing_layers,
                decision_id=decision_id,
                resolved_vector=resolved_vector,
                decision_metadata=decision_metadata,
            )

        # ── Step 7: Consecutive flat tracking ───────────────────────────
        decision: str
        if resolved_signal == 1:
            decision = "BUY"
            self._consecutive_flat = 0
        elif resolved_signal == -1:
            decision = "SELL"
            self._consecutive_flat = 0
        else:
            self._consecutive_flat += 1
            if self._consecutive_flat > self._max_flat_consecutive:
                # Reset confidence if we've been flat too long
                resolved_confidence = max(
                    0.0, resolved_confidence - 0.1
                )
                logger.warning(
                    "Exceeded max_flat_consecutive (%d) — "
                    "reducing confidence to %.4f",
                    self._max_flat_consecutive, resolved_confidence,
                )
            decision = "HOLD"

        # ── Step 8: Final decision ──────────────────────────────────────
        return self._build_result(
            decision=decision,
            signal=resolved_signal,
            confidence=resolved_confidence,
            expected_value=resolved_ev,
            reason=(
                f"{decision} signal={resolved_signal} "
                f"conf={resolved_confidence:.4f} ev={resolved_ev:.4f}"
            ),
            contributing_layers=contributing_layers,
            vetoing_layers=vetoing_layers,
            decision_id=decision_id,
            resolved_vector=resolved_vector,
            decision_metadata=decision_metadata,
        )

    # ------------------------------------------------------------------
    # Statistics & introspection
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return decision statistics and accuracy tracking.

        Returns
        -------
        dict with keys:
            decision_counts     — Counter of BUY/SELL/HOLD/SKIP
            total_decisions     — int
            accuracy            — float (correct / total, or 0.0)
            total_correct       — int
            total_incorrect     — int
            consecutive_flat    — int
        """
        total = self._total_correct + self._total_incorrect
        accuracy = (
            self._total_correct / total if total > 0 else 0.0
        )
        return {
            "decision_counts": dict(self._decision_counts),
            "total_decisions": self._total_decisions,
            "accuracy": round(accuracy, 4),
            "total_correct": self._total_correct,
            "total_incorrect": self._total_incorrect,
            "consecutive_flat": self._consecutive_flat,
        }

    def report_outcome(self, decision_id: str, was_correct: bool) -> None:
        """Report the outcome of a previous decision for accuracy tracking.

        Parameters
        ----------
        decision_id : str
            The decision_id returned by ``synthesize()``.
        was_correct : bool
            Whether the decision turned out to be correct.
        """
        if was_correct:
            self._total_correct += 1
        else:
            self._total_incorrect += 1
        logger.debug(
            "Outcome reported for %s: correct=%s", decision_id, was_correct,
        )

    def reset(self) -> None:
        """Clear all statistics, history, and flat tracking."""
        self._decision_counts.clear()
        self._total_decisions = 0
        self._total_correct = 0
        self._total_incorrect = 0
        self._decision_history.clear()
        self._consecutive_flat = 0
        logger.info(
            "ExecutionSynthesisEngine(%r) reset", self._instance_id
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_result(
        self,
        decision: str,
        signal: int,
        confidence: float,
        expected_value: float,
        reason: str,
        contributing_layers: List[str],
        vetoing_layers: List[str],
        decision_id: str,
        resolved_vector: dict,
        decision_metadata: dict,
    ) -> Dict[str, Any]:
        """Build the result dict and update statistics."""
        self._decision_counts[decision] += 1
        self._total_decisions += 1

        result: Dict[str, Any] = {
            "decision": decision,
            "signal": signal,
            "confidence": round(confidence, 4),
            "expected_value": round(expected_value, 4),
            "reason": reason,
            "contributing_layers": contributing_layers,
            "vetoing_layers": vetoing_layers,
            "decision_id": decision_id,
        }
        # Forward additional info for traceability
        result["_resolved_vector_keys"] = sorted(resolved_vector)
        result["_metadata_keys"] = sorted(decision_metadata)

        self._decision_history.append(result)
        return result

    def __repr__(self) -> str:
        return (
            f"ExecutionSynthesisEngine('{self._instance_id}', "
            f"decisions={self._total_decisions})"
        )


# ===========================================================================
# Self-test
# ===========================================================================

def _run_self_test() -> None:
    """Run 5+ scenarios to verify the synthesis logic."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("ExecutionSynthesisEngine — Self-Test")
    print("=" * 60)

    _state = {"passed": True}

    def _check(cond: bool, msg: str) -> None:
        if not cond:
            _state["passed"] = False
            print(f"  FAIL: {msg}")
        else:
            print(f"  PASS: {msg}")

    def _make_vector(
        signal: int = 0,
        confidence: float = 0.5,
        authority: str = "OSS",
        veto_layers: Optional[List[str]] = None,
        expected_value: Optional[float] = None,
        conflict_matrix: Optional[dict] = None,
    ) -> dict:
        """Helper to build a resolved_vector dict."""
        if conflict_matrix is None:
            conflict_matrix = {}
            if veto_layers:
                for layer in veto_layers:
                    conflict_matrix[layer] = {"veto": True, "contribute": False}
                # mark other layers as contributing
                all_layers = ["sdil", "csfr", "saal", "mrsrl"]
                for layer in all_layers:
                    if layer not in conflict_matrix:
                        conflict_matrix[layer] = {
                            "veto": False, "contribute": True
                        }
            else:
                for layer in ["sdil", "csfr", "saal", "mrsrl"]:
                    conflict_matrix[layer] = {
                        "veto": False, "contribute": True
                    }
        vec = {
            "signal": signal,
            "confidence": confidence,
            "authority": authority,
            "conflict_matrix": conflict_matrix,
        }
        if expected_value is not None:
            vec["expected_value"] = expected_value
        return vec

    def _default_metadata(**overrides) -> dict:
        md = {
            "regime": "normal",
            "spread": 0.5,
            "recent_accuracy": 0.65,
        }
        md.update(overrides)
        return md

    # ------------------------------------------------------------------
    # Scenario 1: SDIL veto → SKIP
    # ------------------------------------------------------------------
    print("\n--- SCE 1: SDIL veto → SKIP ---")
    engine1 = ExecutionSynthesisEngine("selftest_1")
    engine1.reset()

    vec1 = _make_vector(
        signal=1,
        confidence=0.8,
        authority="OSS",
        veto_layers=["sdil"],
    )
    res1 = engine1.synthesize(vec1, _default_metadata())
    print(f"  decision={res1['decision']} reason={res1['reason']}")
    _check(
        res1["decision"] == "SKIP",
        f"Expected SKIP, got {res1['decision']}",
    )
    _check(
        "sdil" in res1["vetoing_layers"],
        "sdil listed in vetoing_layers",
    )
    _check(
        "Layer veto" in res1["reason"],
        "Reason mentions layer veto",
    )

    # ------------------------------------------------------------------
    # Scenario 2: Strong buy signal → BUY
    # ------------------------------------------------------------------
    print("\n--- SCE 2: Strong buy signal → BUY ---")
    engine2 = ExecutionSynthesisEngine("selftest_2")
    engine2.reset()

    vec2 = _make_vector(
        signal=1,
        confidence=0.85,
        authority="OSS",
    )
    res2 = engine2.synthesize(vec2, _default_metadata(recent_accuracy=0.75))
    print(f"  decision={res2['decision']} signal={res2['signal']} "
          f"conf={res2['confidence']} ev={res2['expected_value']}")
    _check(
        res2["decision"] == "BUY",
        f"Expected BUY, got {res2['decision']}",
    )
    _check(res2["signal"] == 1, "Signal is +1 for BUY")
    _check(res2["confidence"] >= 0.4, "Confidence >= min_confidence")
    _check(res2["expected_value"] > 0, "Expected value > 0")

    # ------------------------------------------------------------------
    # Scenario 3: Low confidence → HOLD
    # ------------------------------------------------------------------
    print("\n--- SCE 3: Low confidence → HOLD ---")
    engine3 = ExecutionSynthesisEngine("selftest_3")
    engine3.reset()

    vec3 = _make_vector(
        signal=1,
        confidence=0.25,  # below min_confidence (0.4)
        authority="OSS",
    )
    res3 = engine3.synthesize(vec3, _default_metadata(recent_accuracy=0.5))
    print(f"  decision={res3['decision']} reason={res3['reason']}")
    _check(
        res3["decision"] == "HOLD",
        f"Expected HOLD, got {res3['decision']}",
    )
    _check(
        "confidence" in res3["reason"].lower(),
        "Reason mentions low confidence",
    )

    # ------------------------------------------------------------------
    # Scenario 4: Conflicting signals (resolved_signal=0) → HOLD
    # ------------------------------------------------------------------
    print("\n--- SCE 4: Flat/no conviction → HOLD ---")
    engine4 = ExecutionSynthesisEngine("selftest_4")
    engine4.reset()

    vec4 = _make_vector(
        signal=0,       # no conviction
        confidence=0.0,
        authority="OSS",
    )
    res4 = engine4.synthesize(vec4, _default_metadata())
    print(f"  decision={res4['decision']} reason={res4['reason']}")
    _check(
        res4["decision"] == "HOLD",
        f"Expected HOLD, got {res4['decision']}",
    )
    _check(
        "no conviction" in res4["reason"].lower(),
        "Reason mentions no conviction",
    )

    # ------------------------------------------------------------------
    # Scenario 5: NONE authority → SKIP
    # ------------------------------------------------------------------
    print("\n--- SCE 5: NONE authority → SKIP ---")
    engine5 = ExecutionSynthesisEngine("selftest_5")
    engine5.reset()

    vec5 = _make_vector(
        signal=1,
        confidence=0.8,
        authority="NONE",
    )
    res5 = engine5.synthesize(vec5, _default_metadata())
    print(f"  decision={res5['decision']} reason={res5['reason']}")
    _check(
        res5["decision"] == "SKIP",
        f"Expected SKIP, got {res5['decision']}",
    )
    _check(
        "authority" in res5["reason"].lower(),
        "Reason mentions authority",
    )

    # ------------------------------------------------------------------
    # Scenario 6: Negative expected value → SKIP
    # ------------------------------------------------------------------
    print("\n--- SCE 6: Negative expected value → SKIP ---")
    engine6 = ExecutionSynthesisEngine("selftest_6")
    engine6.reset()

    # Provide a negative EV explicitly
    vec6 = _make_vector(
        signal=1,
        confidence=0.8,
        authority="OSS",
        expected_value=-0.5,
    )
    res6 = engine6.synthesize(vec6, _default_metadata())
    print(f"  decision={res6['decision']} ev={res6['expected_value']} "
          f"reason={res6['reason']}")
    _check(
        res6["decision"] == "SKIP",
        f"Expected SKIP, got {res6['decision']}",
    )
    _check(
        res6["expected_value"] <= 0,
        "Expected value <= min_ev",
    )

    # ------------------------------------------------------------------
    # Scenario 7: SELL signal → SELL
    # ------------------------------------------------------------------
    print("\n--- SCE 7: Strong sell signal → SELL ---")
    engine7 = ExecutionSynthesisEngine("selftest_7")
    engine7.reset()

    vec7 = _make_vector(
        signal=-1,
        confidence=0.75,
        authority="OSS",
    )
    res7 = engine7.synthesize(vec7, _default_metadata(recent_accuracy=0.7))
    print(f"  decision={res7['decision']} signal={res7['signal']} "
          f"conf={res7['confidence']} ev={res7['expected_value']}")
    _check(
        res7["decision"] == "SELL",
        f"Expected SELL, got {res7['decision']}",
    )
    _check(res7["signal"] == -1, "Signal is -1 for SELL")
    _check(res7["expected_value"] > 0, "Expected value positive (magnitude) for SELL")
    _check(res7["confidence"] >= 0.4, "Confidence >= min_confidence")

    # ------------------------------------------------------------------
    # Scenario 8: Multiple layers veto → SKIP with all vetoing layers
    # ------------------------------------------------------------------
    print("\n--- SCE 8: Multiple layer vetos → SKIP ---")
    engine8 = ExecutionSynthesisEngine("selftest_8")
    engine8.reset()

    vec8 = _make_vector(
        signal=1,
        confidence=0.9,
        authority="OSS",
        veto_layers=["sdil", "saal"],
    )
    res8 = engine8.synthesize(vec8, _default_metadata())
    print(f"  decision={res8['decision']} vetoing={res8['vetoing_layers']}")
    _check(res8["decision"] == "SKIP", "Expected SKIP")
    _check(
        "sdil" in res8["vetoing_layers"] and "saal" in res8["vetoing_layers"],
        "Both vetoing layers listed",
    )

    # ------------------------------------------------------------------
    # Scenario 9: get_statistics() and report_outcome()
    # ------------------------------------------------------------------
    print("\n--- SCE 9: Statistics and accuracy tracking ---")
    engine9 = ExecutionSynthesisEngine("selftest_9")
    engine9.reset()

    r1 = engine9.synthesize(
        _make_vector(signal=1, confidence=0.85, authority="OSS"),
        _default_metadata(),
    )
    r2 = engine9.synthesize(
        _make_vector(signal=-1, confidence=0.75, authority="OSS"),
        _default_metadata(),
    )
    # Non-zero signal with NONE authority → SKIP (authority check)
    r3 = engine9.synthesize(
        _make_vector(signal=1, confidence=0.5, authority="NONE"),
        _default_metadata(),
    )
    r4 = engine9.synthesize(
        _make_vector(signal=1, confidence=0.2, authority="OSS"),
        _default_metadata(),
    )

    stats = engine9.get_statistics()
    print(f"  decision_counts={stats['decision_counts']}")
    print(f"  total_decisions={stats['total_decisions']}")

    _check(
        stats["decision_counts"].get("BUY", 0) == 1,
        "1 BUY decision",
    )
    _check(
        stats["decision_counts"].get("SELL", 0) == 1,
        "1 SELL decision",
    )
    _check(
        stats["decision_counts"].get("HOLD", 0) == 1,
        "1 HOLD decision (low confidence)",
    )
    _check(
        stats["decision_counts"].get("SKIP", 0) == 1,
        "1 SKIP decision (NONE authority)",
    )
    _check(
        stats["total_decisions"] == 4,
        "Total decisions = 4",
    )

    # Report outcomes
    engine9.report_outcome(r1["decision_id"], was_correct=True)
    engine9.report_outcome(r2["decision_id"], was_correct=True)
    engine9.report_outcome(r3["decision_id"], was_correct=False)
    stats2 = engine9.get_statistics()
    _check(
        stats2["total_correct"] == 2,
        f"2 correct, got {stats2['total_correct']}",
    )
    _check(
        stats2["total_incorrect"] == 1,
        f"1 incorrect, got {stats2['total_incorrect']}",
    )
    _check(
        abs(stats2["accuracy"] - 2.0 / 3.0) < 1e-4,
        f"Accuracy ~0.6667, got {stats2['accuracy']}",
    )

    # ------------------------------------------------------------------
    # Scenario 10: reset()
    # ------------------------------------------------------------------
    print("\n--- SCE 10: reset() ---")
    engine9.reset()
    stats_reset = engine9.get_statistics()
    _check(
        stats_reset["total_decisions"] == 0,
        "Total decisions = 0 after reset",
    )
    _check(
        stats_reset["decision_counts"] == {},
        "Decision counts empty after reset",
    )
    _check(
        stats_reset["accuracy"] == 0.0,
        "Accuracy = 0 after reset",
    )

    # ------------------------------------------------------------------
    # Scenario 11: Singleton identity
    # ------------------------------------------------------------------
    print("\n--- SCE 11: Singleton identity ---")
    engine_default_1 = ExecutionSynthesisEngine()
    engine_default_2 = ExecutionSynthesisEngine("default")
    engine_other = ExecutionSynthesisEngine("other")

    _check(
        engine_default_1 is engine_default_2,
        "Default singleton identity",
    )
    _check(
        engine_other is not engine_default_1,
        "Different instance_id returns different object",
    )
    _check(
        engine1 is ExecutionSynthesisEngine("selftest_1"),
        "Same instance_id returns same object",
    )

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    if _state["passed"]:
        print("ALL SELF-TESTS PASSED")
        sys.exit(0)
    else:
        print("SOME SELF-TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    _run_self_test()
