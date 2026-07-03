"""
Resolution Authority Engine — final authority on what resolution level the
system should operate at.  Combines all MRSRL module outputs into a single
resolution authority decision.
"""

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton registry
# ---------------------------------------------------------------------------

_instances = {}


def ResolutionAuthorityEngine(instance_id="default"):
    """Singleton accessor — returns the same ``_ResolutionAuthorityEngine``
    for a given *instance_id*.

    Parameters
    ----------
    instance_id : str
        Logical identifier (default ``"default"``).

    Returns
    -------
    _ResolutionAuthorityEngine
    """
    if instance_id not in _instances:
        _instances[instance_id] = _ResolutionAuthorityEngine(instance_id)
    return _instances[instance_id]


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------

#: Which resolution level matches which regime classification.
_REGIME_LEVEL_MAP = {
    "NOISE": "TICK",
    "MICRO_STRUCTURE": "TICK",
    "MESO_STRUCTURE": "1M",
    "MACRO_TREND": "5M",
}

#: Level proximity for partial regime / structure matching.
_LEVEL_ORDER = ["TICK", "1M", "5M", "SESSION"]


def _clamp(value, lo=0.0, hi=1.0):
    return max(lo, min(hi, value))


def _level_index(level):
    """Return the index of *level* in the resolution hierarchy, or -1."""
    try:
        return _LEVEL_ORDER.index(level)
    except ValueError:
        return -1


def _regime_match_score(level, classification):
    """Compute regime-score: 1.0 for exact match, 0.5 for adjacent, 0.0 else."""
    ideal = _REGIME_LEVEL_MAP.get(classification)
    if ideal is None:
        return 0.0
    if level == ideal:
        return 1.0
    li = _level_index(level)
    ii = _level_index(ideal)
    if li >= 0 and ii >= 0 and abs(li - ii) == 1:
        return 0.5
    return 0.0


def _structure_match_score(level, structure_scale):
    """Compute structure-score: 1.0 for exact match, 0.5 for adjacent, 0.0 else."""
    if structure_scale is None:
        return 0.0
    if level == structure_scale:
        return 1.0
    li = _level_index(level)
    si = _level_index(structure_scale)
    if li >= 0 and si >= 0 and abs(li - si) == 1:
        return 0.5
    return 0.0


def _stability_score(mode_stability):
    """Convert mode_stability to a 0-1 score (higher stability = higher score)."""
    return _clamp(mode_stability / 10.0)


class _ResolutionAuthorityEngine:
    """Computes the authoritative resolution level from multi-source inputs.

    Parameters
    ----------
    instance_id : str
        Logical instance identifier.
    """

    def __init__(self, instance_id="default"):
        self._instance_id = instance_id

        # Resolution levels and authority weights
        self._resolution_levels = ["TICK", "1M", "5M", "SESSION"]
        self._authority_weights = {
            "entropy": 0.3,
            "regime": 0.3,
            "structure": 0.2,
            "stability": 0.2,
        }

        logger.debug(
            "ResolutionAuthorityEngine(%r) initialised levels=%s weights=%s",
            instance_id, self._resolution_levels, self._authority_weights,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, symbol, entropy_map, resolution_classification,
                 structure_scale, switching_modes):
        """Produce the final resolution authority decision.

        Parameters
        ----------
        symbol : str
            Instrument identifier.
        entropy_map : dict
            ``{level: entropy_value}`` — entropy per resolution level.
        resolution_classification : str
            Current regime classification (NOISE, MICRO_STRUCTURE,
            MESO_STRUCTURE, MACRO_TREND).
        structure_scale : str or None
            Detected structure scale (TICK, 1M, 5M, SESSION or None).
        switching_modes : dict
            Result from ``AdaptiveSignalSwitcher.switch_for_regime`` —
            must contain ``"mode_stability"``.

        Returns
        -------
        dict
            ``authoritative_resolution``  — final decision (e.g. ``"TICK"``).

            ``confidence``                — confidence in the decision (0–1).

            ``resolution_scores``         — dict level → composite score.

            ``reasoning``                 — human-readable explanation.

            ``should_escalate``           — whether to escalate resolution.

            ``escalation_target``         — recommended level if escalating.
        """
        w_entropy = self._authority_weights.get("entropy", 0.3)
        w_regime = self._authority_weights.get("regime", 0.3)
        w_structure = self._authority_weights.get("structure", 0.2)
        w_stability = self._authority_weights.get("stability", 0.2)

        mode_stability = switching_modes.get("mode_stability", 0)

        resolution_scores = {}
        component_details = {}

        for level in self._resolution_levels:
            # --- entropy_score ---
            raw_entropy = entropy_map.get(level, 0.5)
            entropy_score = 1.0 - _clamp(raw_entropy, 0.0, 1.0)

            # --- regime_score ---
            regime_score = _regime_match_score(level, resolution_classification)

            # --- structure_score ---
            struct_score = _structure_match_score(level, structure_scale)

            # --- stability_score ---
            stab_score = _stability_score(mode_stability)

            composite = (
                entropy_score * w_entropy
                + regime_score * w_regime
                + struct_score * w_structure
                + stab_score * w_stability
            )
            composite = _clamp(composite, 0.0, 1.0)
            resolution_scores[level] = composite

            component_details[level] = {
                "entropy_score": round(entropy_score, 4),
                "regime_score": round(regime_score, 4),
                "structure_score": round(struct_score, 4),
                "stability_score": round(stab_score, 4),
            }

        # ---- Pick best level ----
        best_level = max(resolution_scores, key=lambda lv: resolution_scores[lv])
        best_score = resolution_scores[best_level]

        # ---- Compute confidence ----
        # Normalised gap between best and second-best
        sorted_scores = sorted(resolution_scores.values(), reverse=True)
        gap = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 1.0
        confidence = _clamp(best_score * (0.5 + 0.5 * gap))

        # ---- Escalation logic ----
        should_escalate = best_level != "TICK" and confidence > 0.6
        escalation_target = None
        if should_escalate:
            bi = _level_index(best_level)
            if bi >= 0 and bi < len(self._resolution_levels) - 1:
                escalation_target = self._resolution_levels[bi + 1]
            else:
                escalation_target = best_level

        # ---- Reasoning ----
        detail_lines = []
        for lv in self._resolution_levels:
            cd = component_details[lv]
            detail_lines.append(
                f"{lv}: score={resolution_scores[lv]:.4f} "
                f"(ent={cd['entropy_score']:.2f} reg={cd['regime_score']:.2f} "
                f"str={cd['structure_score']:.2f} stab={cd['stability_score']:.2f})"
            )

        reasoning = (
            f"symbol={symbol} classification={resolution_classification} "
            f"structure={structure_scale} stability={mode_stability} | "
            + " | ".join(detail_lines)
            + f" | best={best_level} confidence={confidence:.4f}"
        )

        if should_escalate:
            reasoning += (
                f" | ESCALATE to {escalation_target} (best={best_level} "
                f"!= TICK, conf={confidence:.4f} > 0.6)"
            )

        logger.info(
            "evaluate %s -> best=%s conf=%.4f escalate=%s",
            symbol, best_level, confidence, should_escalate,
        )

        return {
            "authoritative_resolution": best_level,
            "confidence": round(confidence, 4),
            "resolution_scores": {
                lv: round(s, 4) for lv, s in resolution_scores.items()
            },
            "reasoning": reasoning,
            "should_escalate": should_escalate,
            "escalation_target": escalation_target,
        }

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_weights(self, entropy=None, regime=None, structure=None,
                    stability=None):
        """Override authority-score weights.

        Parameters
        ----------
        entropy : float or None
            Weight for entropy score (default 0.3).
        regime : float or None
            Weight for regime score (default 0.3).
        structure : float or None
            Weight for structure score (default 0.2).
        stability : float or None
            Weight for stability score (default 0.2).
        """
        if entropy is not None:
            self._authority_weights["entropy"] = entropy
        if regime is not None:
            self._authority_weights["regime"] = regime
        if structure is not None:
            self._authority_weights["structure"] = structure
        if stability is not None:
            self._authority_weights["stability"] = stability
        logger.debug("set_weights: %s", self._authority_weights)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("Resolution Authority Engine — Self Test")
    print("=" * 60)

    _state = {"passed": True}

    def _check(cond, msg):
        if not cond:
            _state["passed"] = False
            print(f"  FAIL: {msg}")
        else:
            print(f"  PASS: {msg}")

    # --------------------------------------------------------------
    # Scenario 1: NOISE regime — should favour TICK, no escalation
    # --------------------------------------------------------------
    print("\n--- SCE1: NOISE regime (low entropy on TICK) ---")
    rae1 = ResolutionAuthorityEngine("sce1")
    entropy_map1 = {
        "TICK": 0.15,
        "1M": 0.55,
        "5M": 0.70,
        "SESSION": 0.85,
    }
    switching_modes1 = {"mode_stability": 5}
    res1 = rae1.evaluate(
        symbol="EURUSD",
        entropy_map=entropy_map1,
        resolution_classification="NOISE",
        structure_scale="TICK",
        switching_modes=switching_modes1,
    )
    print(f"  authoritative_resolution = {res1['authoritative_resolution']}")
    print(f"  confidence               = {res1['confidence']:.4f}")
    print(f"  resolution_scores        = {res1['resolution_scores']}")
    print(f"  should_escalate          = {res1['should_escalate']}")
    print(f"  escalation_target        = {res1['escalation_target']}")
    print(f"  reasoning                = {res1['reasoning'][:120]}...")
    _check(res1["authoritative_resolution"] == "TICK",
           f"Expected TICK, got {res1['authoritative_resolution']}")
    _check(not res1["should_escalate"],
           f"Expected no escalation, got {res1['should_escalate']}")
    _check(res1["confidence"] > 0.5,
           f"Expected confidence > 0.5, got {res1['confidence']:.4f}")

    # --------------------------------------------------------------
    # Scenario 2: MESO_STRUCTURE regime — should favour 1M, may escalate
    # --------------------------------------------------------------
    print("\n--- SCE2: MESO_STRUCTURE regime (moderate entropy) ---")
    rae2 = ResolutionAuthorityEngine("sce2")
    entropy_map2 = {
        "TICK": 0.40,
        "1M": 0.30,
        "5M": 0.55,
        "SESSION": 0.75,
    }
    switching_modes2 = {"mode_stability": 8}
    res2 = rae2.evaluate(
        symbol="USDJPY",
        entropy_map=entropy_map2,
        resolution_classification="MESO_STRUCTURE",
        structure_scale="1M",
        switching_modes=switching_modes2,
    )
    print(f"  authoritative_resolution = {res2['authoritative_resolution']}")
    print(f"  confidence               = {res2['confidence']:.4f}")
    print(f"  resolution_scores        = {res2['resolution_scores']}")
    print(f"  should_escalate          = {res2['should_escalate']}")
    print(f"  escalation_target        = {res2['escalation_target']}")
    _check(res2["authoritative_resolution"] == "1M",
           f"Expected 1M, got {res2['authoritative_resolution']}")
    # If best != TICK and confidence > 0.6, should escalate
    if res2["should_escalate"]:
        _check(res2["escalation_target"] == "5M",
               f"Escalation target should be 5M, got {res2['escalation_target']}")

    # --------------------------------------------------------------
    # Scenario 3: MACRO_TREND regime — should favour 5M or SESSION, escalate
    # --------------------------------------------------------------
    print("\n--- SCE3: MACRO_TREND regime (trending, low entropy on 5M) ---")
    rae3 = ResolutionAuthorityEngine("sce3")
    entropy_map3 = {
        "TICK": 0.80,
        "1M": 0.60,
        "5M": 0.20,
        "SESSION": 0.35,
    }
    switching_modes3 = {"mode_stability": 12}
    res3 = rae3.evaluate(
        symbol="GBPUSD",
        entropy_map=entropy_map3,
        resolution_classification="MACRO_TREND",
        structure_scale="5M",
        switching_modes=switching_modes3,
    )
    print(f"  authoritative_resolution = {res3['authoritative_resolution']}")
    print(f"  confidence               = {res3['confidence']:.4f}")
    print(f"  resolution_scores        = {res3['resolution_scores']}")
    print(f"  should_escalate          = {res3['should_escalate']}")
    print(f"  escalation_target        = {res3['escalation_target']}")
    _check(res3["authoritative_resolution"] in ("5M", "SESSION"),
           f"Expected 5M or SESSION, got {res3['authoritative_resolution']}")
    _check(res3["should_escalate"],
           "Expected escalation for non-TICK resolution")

    # --------------------------------------------------------------
    # Scenario 4: No structure scale provided
    # --------------------------------------------------------------
    print("\n--- SCE4: No structure scale ---")
    rae4 = ResolutionAuthorityEngine("sce4")
    entropy_map4 = {
        "TICK": 0.50,
        "1M": 0.50,
        "5M": 0.50,
        "SESSION": 0.50,
    }
    switching_modes4 = {"mode_stability": 1}
    res4 = rae4.evaluate(
        symbol="AUDUSD",
        entropy_map=entropy_map4,
        resolution_classification="MICRO_STRUCTURE",
        structure_scale=None,
        switching_modes=switching_modes4,
    )
    print(f"  authoritative_resolution = {res4['authoritative_resolution']}")
    print(f"  confidence               = {res4['confidence']:.4f}")
    _check(res4["authoritative_resolution"] == "TICK",
           f"Expected TICK for MICRO with no structure, got "
           f"{res4['authoritative_resolution']}")

    # --------------------------------------------------------------
    # Scenario 5: Low stability penalises scores
    # --------------------------------------------------------------
    print("\n--- SCE5: Low stability (mode_stability=0) ---")
    rae5 = ResolutionAuthorityEngine("sce5")
    entropy_map5 = {
        "TICK": 0.30,
        "1M": 0.30,
        "5M": 0.30,
        "SESSION": 0.30,
    }
    switching_modes5 = {"mode_stability": 0}
    res5 = rae5.evaluate(
        symbol="NZDUSD",
        entropy_map=entropy_map5,
        resolution_classification="MESO_STRUCTURE",
        structure_scale="1M",
        switching_modes=switching_modes5,
    )
    print(f"  authoritative_resolution = {res5['authoritative_resolution']}")
    print(f"  confidence               = {res5['confidence']:.4f}")
    print(f"  resolution_scores        = {res5['resolution_scores']}")
    # Stability score = 0/10 = 0.0, but regime+structure both match 1M
    _check(res5["authoritative_resolution"] == "1M",
           f"Expected 1M (regime/structure match), got "
           f"{res5['authoritative_resolution']}")

    # --------------------------------------------------------------
    # Scenario 6: All levels equal — tie-break
    # --------------------------------------------------------------
    print("\n--- SCE6: All entropy equal, tie-break ---")
    rae6 = ResolutionAuthorityEngine("sce6")
    entropy_map6 = {
        "TICK": 0.50,
        "1M": 0.50,
        "5M": 0.50,
        "SESSION": 0.50,
    }
    switching_modes6 = {"mode_stability": 5}
    res6 = rae6.evaluate(
        symbol="EURJPY",
        entropy_map=entropy_map6,
        resolution_classification="NOISE",
        structure_scale="TICK",
        switching_modes=switching_modes6,
    )
    print(f"  authoritative_resolution = {res6['authoritative_resolution']}")
    print(f"  resolution_scores        = {res6['resolution_scores']}")
    # TICK should win because of regime (NOISE->TICK) and structure (TICK) match
    _check(res6["authoritative_resolution"] == "TICK",
           f"Expected TICK (regime+structure match), got "
           f"{res6['authoritative_resolution']}")

    # --------------------------------------------------------------
    # Scenario 7: Singleton identity
    # --------------------------------------------------------------
    print("\n--- SCE7: Singleton identity ---")
    rae1_again = ResolutionAuthorityEngine("sce1")
    _check(rae1_again is rae1, "Same instance_id returns same object")
    default_a = ResolutionAuthorityEngine()
    default_b = ResolutionAuthorityEngine("default")
    _check(default_a is default_b, "Default singleton identity")
    other = ResolutionAuthorityEngine("other")
    _check(other is not rae1, "Different instance_id returns different object")

    # --------------------------------------------------------------
    # Scenario 8: Custom weights
    # --------------------------------------------------------------
    print("\n--- SCE8: Custom weights ---")
    rae8 = ResolutionAuthorityEngine("sce8")
    rae8.set_weights(entropy=0.5, regime=0.3, structure=0.1, stability=0.1)
    entropy_map8 = {
        "TICK": 0.10,
        "1M": 0.60,
        "5M": 0.70,
        "SESSION": 0.90,
    }
    switching_modes8 = {"mode_stability": 3}
    res8 = rae8.evaluate(
        symbol="USDCAD",
        entropy_map=entropy_map8,
        resolution_classification="NOISE",
        structure_scale="TICK",
        switching_modes=switching_modes8,
    )
    print(f"  authoritative_resolution = {res8['authoritative_resolution']}")
    # With entropy weight=0.5 and TICK entropy=0.1, TICK should dominate
    _check(res8["authoritative_resolution"] == "TICK",
           f"Expected TICK with high entropy weight, got "
           f"{res8['authoritative_resolution']}")

    # --------------------------------------------------------------
    # Scenario 9: Escalation only when best != TICK AND confidence > 0.6
    # --------------------------------------------------------------
    print("\n--- SCE9: Low confidence blocks escalation ---")
    rae9 = ResolutionAuthorityEngine("sce9")
    entropy_map9 = {
        "TICK": 0.50,
        "1M": 0.50,
        "5M": 0.50,
        "SESSION": 0.50,
    }
    switching_modes9 = {"mode_stability": 0}
    res9 = rae9.evaluate(
        symbol="CHFJPY",
        entropy_map=entropy_map9,
        resolution_classification="MESO_STRUCTURE",
        structure_scale=None,
        switching_modes=switching_modes9,
    )
    print(f"  authoritative_resolution = {res9['authoritative_resolution']}")
    print(f"  confidence               = {res9['confidence']:.4f}")
    print(f"  should_escalate          = {res9['should_escalate']}")
    # With all entropy flat, no structure, stability=0, confidence should be low
    # Note: the best could still be 1M due to regime match, but confidence
    # could be below 0.6.
    if res9["authoritative_resolution"] != "TICK":
        _check(not res9["should_escalate"] or res9["confidence"] <= 0.6,
               "Escalation should require both best!=TICK and conf>0.6")

    # --------------------------------------------------------------
    # Final result
    # --------------------------------------------------------------
    print()
    print("=" * 60)
    if _state["passed"]:
        print("ALL SELF-TESTS PASSED ✓")
    else:
        print("SOME SELF-TESTS FAILED ✗")
    print("=" * 60)

    import sys
    sys.exit(0 if _state["passed"] else 1)
