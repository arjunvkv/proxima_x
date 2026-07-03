"""
Execution Priority Arbiter — the final signal selection gate in the UESL.

When multiple valid signal-generating paths exist (e.g., OSS says BUY,
ALT says SELL, consensus says HOLD), this module picks the one with the
highest **expected value × confidence × stability**.

Priority scoring formula (for non-zero signals)::

    direction_score = 1.0                     (any direction is acceptable)
    ev_score        = max(0, expected_value) / max_abs_ev   (batch-normalised)
    priority_score  = confidence * ev_score * stability * direction_score

Only candidates whose ``priority_score >= min_priority_score`` are eligible
for selection.  Ties are broken by source authority:

    SAAL > CONSENSUS > OSS > ALT

Usage
-----
    from core_runtime.execution_priority_arbiter import (
        ExecutionPriorityArbiter,
    )

    arbiter = ExecutionPriorityArbiter()
    result = arbiter.arbitrate([
        {"source": "OSS", "signal": 1, "confidence": 0.8,
         "expected_value": 0.05, "stability": 0.9, "timestamp": 1.0},
        {"source": "ALT", "signal": -1, "confidence": 0.6,
         "expected_value": -0.02, "stability": 0.7, "timestamp": 1.0},
    ])
    print(result["selected_source"])  # "OSS"
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton registry
# ---------------------------------------------------------------------------

_instances: Dict[str, "_ExecutionPriorityArbiter"] = {}


def ExecutionPriorityArbiter(instance_id="default"):
    """Singleton accessor for ``_ExecutionPriorityArbiter``.

    Parameters
    ----------
    instance_id : str
        Logical identifier for the arbiter instance (default ``"default"``).

    Returns
    -------
    _ExecutionPriorityArbiter
    """
    if instance_id not in _instances:
        _instances[instance_id] = _ExecutionPriorityArbiter(instance_id)
    return _instances[instance_id]


# ---------------------------------------------------------------------------
# Source tie-breaking priority (lower = higher priority)
# ---------------------------------------------------------------------------

_SOURCE_PRIORITY = {
    "SAAL": 0,
    "CONSENSUS": 1,
    "OSS": 2,
    "ALT": 3,
}


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------


class _ExecutionPriorityArbiter:
    """Selects the highest-value signal from multiple candidates.

    Parameters
    ----------
    instance_id : str
        Logical instance identifier (used for logging).
    """

    DEFAULT_MIN_PRIORITY_SCORE = 0.2

    def __init__(self, instance_id="default"):
        self._instance_id = instance_id
        self._min_priority_score = self.DEFAULT_MIN_PRIORITY_SCORE

        # Selection history — last 100 entries.
        self._selection_history: List[Dict[str, Any]] = []

        logger.debug(
            "ExecutionPriorityArbiter(%r) initialised",
            instance_id,
        )

    # ------------------------------------------------------------------
    # Public API — arbitration
    # ------------------------------------------------------------------

    def arbitrate(self, signals: list) -> dict:
        """Select the highest-priority signal from *signals*.

        Parameters
        ----------
        signals : list of dict
            Each candidate dict has keys:

            ================  =====  ===================================
            Key               Type   Description
            ================  =====  ===================================
            ``source``        str    e.g. ``"OSS"``, ``"ALT"``,
                                     ``"CONSENSUS"``, ``"SAAL"``
            ``signal``        int    ``-1``, ``0``, or ``+1``
            ``confidence``    float  0–1
            ``expected_value`` float  can be negative
            ``stability``     float  0–1
            ``timestamp``     float  when the signal was produced
            ================  =====  ===================================

        Returns
        -------
        dict with keys:

            ``selected_signal``  int     The chosen signal (-1, 0, +1)
            ``selected_source``  str     The source of the chosen signal
            ``priority_score``   float   The score of the chosen signal
            ``all_candidates``   list    All scored candidates
            ``reason``           str     Human-readable explanation
            ``switch_risk``      float   0–1, high if frequently switching
        """
        if not signals:
            result = {
                "selected_signal": 0,
                "selected_source": "NONE",
                "priority_score": 0.0,
                "all_candidates": [],
                "reason": "No candidates provided",
                "switch_risk": 0.0,
            }
            self._track_selection(result)
            return result

        # Score all candidates with batch-normalised expected value.
        scored = self._score_candidates(signals)

        # Filter by minimum priority threshold.
        eligible = [
            c for c in scored
            if c["priority_score"] >= self._min_priority_score
        ]

        if not eligible:
            result = {
                "selected_signal": 0,
                "selected_source": "NONE",
                "priority_score": 0.0,
                "all_candidates": scored,
                "reason": "No eligible candidates (all below "
                          f"min_priority_score={self._min_priority_score:.2f})",
                "switch_risk": 0.0,
            }
            self._track_selection(result)
            return result

        # Sort: descending priority_score, then ascending source priority.
        eligible.sort(key=lambda c: (
            -c["priority_score"],
            _SOURCE_PRIORITY.get(c["source"], 99),
        ))

        selected = eligible[0]

        # Build a human-readable reason.
        if len(eligible) == 1:
            reason = (
                f"Single eligible candidate from {selected['source']} "
                f"(score={selected['priority_score']:.4f})"
            )
        else:
            runner = eligible[1]
            reason = (
                f"Selected {selected['source']} "
                f"(score={selected['priority_score']:.4f}) over "
                f"{runner['source']} (score={runner['priority_score']:.4f})"
            )

        switch_risk = self._compute_switch_risk(selected)

        result = {
            "selected_signal": selected["signal"],
            "selected_source": selected["source"],
            "priority_score": selected["priority_score"],
            "all_candidates": scored,
            "reason": reason,
            "switch_risk": switch_risk,
        }

        self._track_selection(result)
        return result

    # ------------------------------------------------------------------
    # Public API — history & reset
    # ------------------------------------------------------------------

    def get_selection_history(self) -> list:
        """Return the last 100 selection records.

        Each record contains ``selected_signal``, ``selected_source``,
        ``priority_score``, and ``timestamp``.
        """
        return list(self._selection_history)

    def get_switch_frequency(self) -> float:
        """How often the arbiter switches between sources.

        Returns
        -------
        float
            Value in [0, 1]:
            - 0 = never switches (same source every time)
            - 1 = switches on every consecutive selection
        """
        if len(self._selection_history) < 2:
            return 0.0

        switches = 0
        for i in range(1, len(self._selection_history)):
            prev = self._selection_history[i - 1]
            curr = self._selection_history[i]
            prev_src = prev["selected_source"]
            curr_src = curr["selected_source"]
            # Count transitions where both are real signal sources.
            if prev_src != curr_src and prev_src != "NONE" and curr_src != "NONE":
                switches += 1

        return round(switches / (len(self._selection_history) - 1), 4)

    def set_min_priority_score(self, score: float):
        """Override the minimum priority threshold.

        Parameters
        ----------
        score : float
            New threshold (default 0.2).
        """
        self._min_priority_score = score
        logger.debug(
            "set_min_priority_score(%.4f)", score,
        )

    def reset(self):
        """Clear all selection history and reset to defaults."""
        self._selection_history.clear()
        self._min_priority_score = self.DEFAULT_MIN_PRIORITY_SCORE
        logger.info(
            "ExecutionPriorityArbiter(%r) reset",
            self._instance_id,
        )

    # ------------------------------------------------------------------
    # Internal scoring
    # ------------------------------------------------------------------

    def _score_candidates(self, signals: list) -> list:
        """Score all candidates with batch-normalised expected value.

        Parameters
        ----------
        signals : list of dict
            Raw signal candidates.

        Returns
        -------
        list of dict
            Each input enriched with a ``priority_score`` key.
        """
        if not signals:
            return []

        # Determine max_abs_ev across the batch for normalisation.
        max_abs_ev = max(
            (abs(s.get("expected_value", 0.0)) for s in signals),
            default=1.0,
        )
        if max_abs_ev == 0.0:
            max_abs_ev = 1.0  # avoid division by zero

        scored = []
        for s in signals:
            signal = s.get("signal", 0)
            source = s.get("source", "UNKNOWN")
            confidence = s.get("confidence", 0.0)
            expected_value = s.get("expected_value", 0.0)
            stability = s.get("stability", 0.0)
            timestamp = s.get("timestamp", 0.0)

            if signal == 0:
                priority_score = 0.0
            else:
                direction_score = 1.0
                ev_score = max(0.0, expected_value) / max_abs_ev
                priority_score = (
                    confidence * ev_score * stability * direction_score
                )

            scored.append({
                "source": source,
                "signal": signal,
                "confidence": confidence,
                "expected_value": expected_value,
                "stability": stability,
                "timestamp": timestamp,
                "priority_score": round(priority_score, 6),
            })

        return scored

    # ------------------------------------------------------------------
    # Internal — history tracking
    # ------------------------------------------------------------------

    def _track_selection(self, result: dict):
        """Append the latest selection to the history ring buffer."""
        # Grab a representative timestamp from the candidates (if any).
        candidates = result.get("all_candidates", [])
        ts = candidates[0].get("timestamp", 0.0) if candidates else 0.0

        self._selection_history.append({
            "selected_signal": result["selected_signal"],
            "selected_source": result["selected_source"],
            "priority_score": result["priority_score"],
            "timestamp": ts,
        })

        # Keep at most 100 entries.
        if len(self._selection_history) > 100:
            self._selection_history.pop(0)

    def _compute_switch_risk(self, selected: dict) -> float:
        """Compute switch risk based on recent selection history.

        High switch frequency → elevated risk of oscillating between
        signal sources.
        """
        if len(self._selection_history) < 2:
            return 0.0
        return self.get_switch_frequency()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("Execution Priority Arbiter — Self Test")
    print("=" * 60)

    _state = {"passed": True}

    def _check(cond, msg):
        if not cond:
            _state["passed"] = False
            print(f"  FAIL: {msg}")
        else:
            print(f"  PASS: {msg}")

    # ==================================================================
    # Scenario 1: OSS is clearly the best
    #
    #   OSS: signal=+1, confidence=0.9, ev=0.10, stability=0.95
    #   ALT: signal=-1, confidence=0.4, ev=0.02, stability=0.50
    #
    #   max_abs_ev = max(0.10, 0.02) = 0.10
    #   OSS: ev_score = 0.10/0.10 = 1.0,  priority = 0.9*1.0*0.95 = 0.855
    #   ALT: ev_score = 0.02/0.10 = 0.2, priority = 0.4*0.2*0.5  = 0.04
    #   ALT falls below threshold (0.04 < 0.2) → only OSS eligible.
    # ==================================================================
    print("\n--- SCE1: OSS clearly best ---")
    arb1 = ExecutionPriorityArbiter("sce1")
    result1 = arb1.arbitrate([
        {
            "source": "OSS",
            "signal": 1,
            "confidence": 0.9,
            "expected_value": 0.10,
            "stability": 0.95,
            "timestamp": 1.0,
        },
        {
            "source": "ALT",
            "signal": -1,
            "confidence": 0.4,
            "expected_value": 0.02,
            "stability": 0.50,
            "timestamp": 1.0,
        },
    ])
    print(f"  selected_signal  = {result1['selected_signal']}")
    print(f"  selected_source  = {result1['selected_source']}")
    print(f"  priority_score   = {result1['priority_score']:.4f}")
    print(f"  reason           = {result1['reason']}")
    print(f"  switch_risk      = {result1['switch_risk']:.4f}")

    _check(result1["selected_signal"] == 1,
           f"OSS wins → signal=1, got {result1['selected_signal']}")
    _check(result1["selected_source"] == "OSS",
           f"OSS wins, got {result1['selected_source']}")
    _check(abs(result1["priority_score"] - 0.855) < 0.001,
           f"OSS priority ≈ 0.855, got {result1['priority_score']:.4f}")
    _check(result1["switch_risk"] == 0.0,
           "First selection → switch_risk = 0.0")

    # Verify ALT was ineligible.
    alt_candidate = [c for c in result1["all_candidates"]
                     if c["source"] == "ALT"][0]
    _check(alt_candidate["priority_score"] < 0.2,
           "ALT priority below threshold → ineligible")

    # ==================================================================
    # Scenario 2: ALT slightly better than OSS
    #
    #   OSS: signal=+1, confidence=0.7, ev=0.03, stability=0.75
    #   ALT: signal=-1, confidence=0.8, ev=0.05, stability=0.80
    #
    #   max_abs_ev = max(0.03, 0.05) = 0.05
    #   OSS: ev_score = 0.03/0.05 = 0.6, priority = 0.7*0.6*0.75 = 0.315
    #   ALT: ev_score = 0.05/0.05 = 1.0, priority = 0.8*1.0*0.80 = 0.640
    # ==================================================================
    print("\n--- SCE2: ALT slightly better ---")
    arb2 = ExecutionPriorityArbiter("sce2")
    result2 = arb2.arbitrate([
        {
            "source": "OSS",
            "signal": 1,
            "confidence": 0.7,
            "expected_value": 0.03,
            "stability": 0.75,
            "timestamp": 2.0,
        },
        {
            "source": "ALT",
            "signal": -1,
            "confidence": 0.8,
            "expected_value": 0.05,
            "stability": 0.80,
            "timestamp": 2.0,
        },
    ])
    print(f"  selected_signal  = {result2['selected_signal']}")
    print(f"  selected_source  = {result2['selected_source']}")
    print(f"  priority_score   = {result2['priority_score']:.4f}")
    print(f"  reason           = {result2['reason']}")

    _check(result2["selected_signal"] == -1,
           f"ALT wins → signal=-1, got {result2['selected_signal']}")
    _check(result2["selected_source"] == "ALT",
           f"ALT wins, got {result2['selected_source']}")
    _check(abs(result2["priority_score"] - 0.64) < 0.001,
           f"ALT priority ≈ 0.64, got {result2['priority_score']:.4f}")

    # ==================================================================
    # Scenario 3: All signals flat → NONE
    # ==================================================================
    print("\n--- SCE3: All flat → NONE ---")
    arb3 = ExecutionPriorityArbiter("sce3")
    result3 = arb3.arbitrate([
        {
            "source": "OSS", "signal": 0, "confidence": 0.8,
            "expected_value": 0.0, "stability": 0.9, "timestamp": 3.0,
        },
        {
            "source": "ALT", "signal": 0, "confidence": 0.7,
            "expected_value": 0.0, "stability": 0.8, "timestamp": 3.0,
        },
    ])
    print(f"  selected_signal  = {result3['selected_signal']}")
    print(f"  selected_source  = {result3['selected_source']}")
    print(f"  reason           = {result3['reason']}")

    _check(result3["selected_signal"] == 0,
           f"All flat → signal=0, got {result3['selected_signal']}")
    _check(result3["selected_source"] == "NONE",
           f"All flat → source=NONE, got {result3['selected_source']}")
    _check("No eligible" in result3["reason"],
           "Reason mentions no eligible candidates")

    # ==================================================================
    # Scenario 4: Empty signals list → NONE
    # ==================================================================
    print("\n--- SCE4: Empty list → NONE ---")
    arb4 = ExecutionPriorityArbiter("sce4")
    result4 = arb4.arbitrate([])
    _check(result4["selected_signal"] == 0,
           f"Empty list → signal=0, got {result4['selected_signal']}")
    _check(result4["selected_source"] == "NONE",
           f"Empty list → source=NONE, got {result4['selected_source']}")
    _check("No candidates provided" in result4["reason"],
           "Reason mentions no candidates provided")

    # ==================================================================
    # Scenario 5: Tie-breaking by source priority
    #
    #   Both candidates have identical scores.  SAAL should win over OSS.
    # ==================================================================
    print("\n--- SCE5: Tie-breaking by source priority ---")
    arb5 = ExecutionPriorityArbiter("sce5")
    result5 = arb5.arbitrate([
        {
            "source": "OSS", "signal": 1, "confidence": 0.8,
            "expected_value": 0.05, "stability": 0.8, "timestamp": 4.0,
        },
        {
            "source": "SAAL", "signal": -1, "confidence": 0.8,
            "expected_value": 0.05, "stability": 0.8, "timestamp": 4.0,
        },
    ])
    print(f"  selected_signal  = {result5['selected_signal']}")
    print(f"  selected_source  = {result5['selected_source']}")
    print(f"  priority_score   = {result5['priority_score']:.4f}")

    # Same confidence/ev/stability → same score. SAAL wins by source priority.
    _check(result5["selected_source"] == "SAAL",
           f"Tie → SAAL wins, got {result5['selected_source']}")

    # ==================================================================
    # Scenario 6: Switch frequency tracking
    #
    #   Make 3 selections that alternate sources → switch_frequency > 0.
    # ==================================================================
    print("\n--- SCE6: Switch frequency ---")
    arb6 = ExecutionPriorityArbiter("sce6")

    # Selection 1: OSS wins.
    arb6.arbitrate([
        {"source": "OSS", "signal": 1, "confidence": 0.9,
         "expected_value": 0.10, "stability": 0.9, "timestamp": 1.0},
        {"source": "ALT", "signal": -1, "confidence": 0.3,
         "expected_value": 0.01, "stability": 0.5, "timestamp": 1.0},
    ])

    # Selection 2: ALT wins (different make-up).
    arb6.arbitrate([
        {"source": "OSS", "signal": 1, "confidence": 0.3,
         "expected_value": 0.01, "stability": 0.5, "timestamp": 2.0},
        {"source": "ALT", "signal": -1, "confidence": 0.9,
         "expected_value": 0.10, "stability": 0.9, "timestamp": 2.0},
    ])

    # Selection 3: OSS wins again.
    arb6.arbitrate([
        {"source": "OSS", "signal": 1, "confidence": 0.9,
         "expected_value": 0.10, "stability": 0.9, "timestamp": 3.0},
        {"source": "ALT", "signal": -1, "confidence": 0.3,
         "expected_value": 0.01, "stability": 0.5, "timestamp": 3.0},
    ])

    freq6 = arb6.get_switch_frequency()
    hist6 = arb6.get_selection_history()
    print(f"  switch_frequency = {freq6}")
    print(f"  history length   = {len(hist6)}")
    print(f"  history sources  = {[h['selected_source'] for h in hist6]}")

    # 3 selections = 2 transitions.  OSS→ALT (switch), ALT→OSS (switch) = 2/2.
    _check(freq6 == 1.0,
           f"Switch freq = 1.0 (2/2), got {freq6}")

    # ==================================================================
    # Scenario 7: Selections with negative expected value
    #
    #   All candidates have negative expected_value → ev_score=0 for all
    #   → priority_score=0 for all → no eligible → NONE.
    # ==================================================================
    print("\n--- SCE7: Negative expected value only ---")
    arb7 = ExecutionPriorityArbiter("sce7")
    result7 = arb7.arbitrate([
        {"source": "OSS", "signal": 1, "confidence": 0.9,
         "expected_value": -0.05, "stability": 0.9, "timestamp": 5.0},
        {"source": "ALT", "signal": -1, "confidence": 0.8,
         "expected_value": -0.03, "stability": 0.8, "timestamp": 5.0},
    ])
    _check(result7["selected_source"] == "NONE",
           f"All negative ev → NONE, got {result7['selected_source']}")
    _check(result7["selected_signal"] == 0,
           "All negative ev → signal=0")

    # ==================================================================
    # Scenario 8: get_selection_history bounds
    # ==================================================================
    print("\n--- SCE8: History bounds (max 100) ---")
    arb8 = ExecutionPriorityArbiter("sce8")
    for i in range(150):
        arb8.arbitrate([
            {"source": "OSS", "signal": 1, "confidence": 0.8,
             "expected_value": 0.05, "stability": 0.8, "timestamp": float(i)},
        ])
    hist8 = arb8.get_selection_history()
    _check(len(hist8) == 100,
           f"History capped at 100, got {len(hist8)}")
    # Most recent timestamp should be 149.0 (last entry).
    _check(hist8[-1]["timestamp"] == 149.0,
           f"Last entry timestamp = 149.0, got {hist8[-1]['timestamp']}")

    # ==================================================================
    # Scenario 9: Custom min_priority_score
    # ==================================================================
    print("\n--- SCE9: Custom min_priority_score ---")
    arb9 = ExecutionPriorityArbiter("sce9")
    arb9.set_min_priority_score(0.5)
    # OSS has score = 0.8*1.0*0.8 = 0.64, eligible (>= 0.5).
    # ALT has score = 0.3*0.4*0.6 = 0.072, not eligible.
    result9 = arb9.arbitrate([
        {"source": "OSS", "signal": 1, "confidence": 0.8,
         "expected_value": 0.05, "stability": 0.8, "timestamp": 6.0},
        {"source": "ALT", "signal": -1, "confidence": 0.3,
         "expected_value": 0.02, "stability": 0.6, "timestamp": 6.0},
    ])
    _check(result9["selected_source"] == "OSS",
           f"OSS still eligible, got {result9['selected_source']}")
    _check(result9["priority_score"] >= 0.5,
           f"OSS priority >= 0.5, got {result9['priority_score']:.4f}")

    # Raise threshold above OSS's score → no eligible.
    arb9.set_min_priority_score(0.9)
    result9b = arb9.arbitrate([
        {"source": "OSS", "signal": 1, "confidence": 0.8,
         "expected_value": 0.05, "stability": 0.8, "timestamp": 7.0},
    ])
    _check(result9b["selected_source"] == "NONE",
           "Threshold 0.9 > 0.64 → NONE")

    # ==================================================================
    # Scenario 10: Reset
    # ==================================================================
    print("\n--- SCE10: Reset ---")
    arb5.reset()
    hist_reset = arb5.get_selection_history()
    _check(len(hist_reset) == 0,
           "After reset → history empty")
    freq_reset = arb5.get_switch_frequency()
    _check(freq_reset == 0.0,
           "After reset → switch_frequency = 0.0")
    _check(arb5._min_priority_score == 0.2,
           "After reset → min_priority_score = 0.2")

    # ==================================================================
    # Scenario 11: Singleton identity
    # ==================================================================
    print("\n--- SCE11: Singleton identity ---")
    a = ExecutionPriorityArbiter("sce1")
    b = ExecutionPriorityArbiter("sce1")
    c = ExecutionPriorityArbiter("other")
    d = ExecutionPriorityArbiter()
    e = ExecutionPriorityArbiter("default")
    _check(a is b, "Same instance_id returns same object")
    _check(a is not c, "Different instance_id returns different object")
    _check(d is e, "Default singleton identity")

    # ==================================================================
    # Scenario 12: Signal candidate with signal=0 gets score=0
    # ==================================================================
    print("\n--- SCE12: Zero-signal candidate always scores 0 ---")
    arb12 = ExecutionPriorityArbiter("sce12")
    result12 = arb12.arbitrate([
        {"source": "OSS", "signal": 0, "confidence": 0.99,
         "expected_value": 0.99, "stability": 0.99, "timestamp": 8.0},
        {"source": "ALT", "signal": 1, "confidence": 0.5,
         "expected_value": 0.01, "stability": 0.5, "timestamp": 8.0},
    ])
    # OSS has signal=0 → score=0.  ALT has score = 0.5*0.01*0.5/0.01 = 0.25
    # (max_abs_ev = 0.99, so ALT ev_score = 0.01/0.99 ≈ 0.0101, priority ≈ 0.5*0.0101*0.5 ≈ 0.0025)
    # Wait, 0.01/0.99 = 0.0101, times 0.5*0.5 = 0.0025, which is below 0.2.
    # So no one is eligible.
    _check(result12["selected_source"] == "NONE",
           "Zero-signal OSS score=0, ALT score below threshold → NONE")
    oss_cand = [c for c in result12["all_candidates"] if c["source"] == "OSS"][0]
    _check(oss_cand["priority_score"] == 0.0,
           "OSS signal=0 → priority_score=0.0")

    # ==================================================================
    # Scenario 13: Mixed source priority ordering (full tie)
    # ==================================================================
    print("\n--- SCE13: Full 4-way tie → SAAL > CONSENSUS > OSS > ALT ---")
    arb13 = ExecutionPriorityArbiter("sce13")
    result13 = arb13.arbitrate([
        {"source": "ALT", "signal": 1, "confidence": 0.7,
         "expected_value": 0.04, "stability": 0.7, "timestamp": 9.0},
        {"source": "OSS", "signal": 1, "confidence": 0.7,
         "expected_value": 0.04, "stability": 0.7, "timestamp": 9.0},
        {"source": "CONSENSUS", "signal": 1, "confidence": 0.7,
         "expected_value": 0.04, "stability": 0.7, "timestamp": 9.0},
        {"source": "SAAL", "signal": 1, "confidence": 0.7,
         "expected_value": 0.04, "stability": 0.7, "timestamp": 9.0},
    ])
    # All have identical scores.  Ties break: SAAL > CONSENSUS > OSS > ALT.
    _check(result13["selected_source"] == "SAAL",
           f"Tie → SAAL wins, got {result13['selected_source']}")

    # ==================================================================
    # Final result
    # ==================================================================
    print()
    print("=" * 60)
    if _state["passed"]:
        print("ALL SELF-TESTS PASSED ✓")
    else:
        print("SOME SELF-TESTS FAILED ✗")
    print("=" * 60)

    import sys
    sys.exit(0 if _state["passed"] else 1)
