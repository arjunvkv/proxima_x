"""
Signal Authority Arbiter — determines which signal system has authority over
execution: OSS, ALT, HYBRID, or NONE.

Based on predictive accuracy, stability, entropy contribution, and regime
robustness from the respective signal sources.
"""

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton registry
# ---------------------------------------------------------------------------

_instances = {}


def SignalAuthorityArbiter(instance_id="default"):
    """Singleton accessor — returns the same ``_SignalAuthorityArbiter``
    for a given *instance_id*.

    Parameters
    ----------
    instance_id : str
        Logical identifier for the arbiter instance (default ``"default"``).

    Returns
    -------
    _SignalAuthorityArbiter
    """
    if instance_id not in _instances:
        _instances[instance_id] = _SignalAuthorityArbiter(instance_id)
    return _instances[instance_id]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clamp(value, lo=0.0, hi=1.0):
    """Clamp *value* to the closed interval ``[lo, hi]``."""
    return max(lo, min(hi, value))


def _normalise(value):
    """Normalise an arbitrary float to ``[0, 1]``, assuming inputs are
    already roughly in a sensible range.

    If the value is outside ``[0, 1]``, we clamp it.  This handles the
    common case where inputs are percentages, fractions, or correlation-like
    values.
    """
    return _clamp(float(value), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------


class _SignalAuthorityArbiter:
    """Determines which signal system (OSS, ALT, both, or neither) should
    have authority over execution based on aggregate quality metrics.

    Parameters
    ----------
    instance_id : str
        Logical instance identifier (used for logging).
    """

    # Default weights: w1 = accuracy, w2 = entropy, w3 = stability, w4 =
    # surface_health (OSS) / validity (ALT).
    DEFAULT_W1 = 0.4
    DEFAULT_W2 = 0.2
    DEFAULT_W3 = 0.2
    DEFAULT_W4 = 0.2

    # Authority margin threshold
    AUTHORITY_MARGIN = 0.15

    # Minimum score for HYBRID consideration
    HYBRID_MIN_SCORE = 0.5

    def __init__(self, instance_id="default"):
        self._instance_id = instance_id

        # OSS metrics store
        self._oss_accuracy = 0.0
        self._oss_entropy = 0.0
        self._oss_surface = 0.0
        self._oss_stability = 0.0
        self._oss_has_data = False

        # ALT metrics store
        self._alt_accuracy = 0.0
        self._alt_entropy = 0.0
        self._alt_validity = 0.0
        self._alt_stability = 0.0
        self._alt_has_data = False

        # Weights
        self._w1 = self.DEFAULT_W1
        self._w2 = self.DEFAULT_W2
        self._w3 = self.DEFAULT_W3
        self._w4 = self.DEFAULT_W4

        # Latest authority decision
        self._latest_authority = "NONE"
        self._latest_confidence = 0.0
        self._latest_margin = 0.0
        self._latest_reasoning = "No data fed yet."

        logger.debug("SignalAuthorityArbiter(%r) initialised", instance_id)

    # ------------------------------------------------------------------
    # Public API — feeding metrics
    # ------------------------------------------------------------------

    def feed_oss_metrics(self, accuracy, entropy_contribution, surface_health,
                         stability):
        """Feed OSS signal metrics.

        Parameters
        ----------
        accuracy : float
            Predictive accuracy of OSS signal (expected range [0, 1]).
        entropy_contribution : float
            Entropy contribution from OSS in signal space (expected [0, 1]).
        surface_health : float
            Decision surface health score for OSS (expected [0, 1]).
        stability : float
            Stability score for OSS (expected [0, 1]).
        """
        self._oss_accuracy = _normalise(accuracy)
        self._oss_entropy = _normalise(entropy_contribution)
        self._oss_surface = _normalise(surface_health)
        self._oss_stability = _normalise(stability)
        self._oss_has_data = True
        logger.debug(
            "feed_oss_metrics accuracy=%.4f entropy=%.4f surface=%.4f "
            "stability=%.4f",
            self._oss_accuracy, self._oss_entropy,
            self._oss_surface, self._oss_stability,
        )

    def feed_alt_metrics(self, accuracy, entropy_contribution, validity,
                         stability):
        """Feed ALT signal metrics.

        Parameters
        ----------
        accuracy : float
            Predictive accuracy of ALT signal (expected range [0, 1]).
        entropy_contribution : float
            Entropy contribution from ALT in signal space (expected [0, 1]).
        validity : float
            Validity score for ALT signal (expected [0, 1]).
        stability : float
            Stability score for ALT (expected [0, 1]).
        """
        self._alt_accuracy = _normalise(accuracy)
        self._alt_entropy = _normalise(entropy_contribution)
        self._alt_validity = _normalise(validity)
        self._alt_stability = _normalise(stability)
        self._alt_has_data = True
        logger.debug(
            "feed_alt_metrics accuracy=%.4f entropy=%.4f validity=%.4f "
            "stability=%.4f",
            self._alt_accuracy, self._alt_entropy,
            self._alt_validity, self._alt_stability,
        )

    # ------------------------------------------------------------------
    # Public API — configuration
    # ------------------------------------------------------------------

    def set_weights(self, w1=None, w2=None, w3=None, w4=None):
        """Override default authority-score weights.

        Parameters
        ----------
        w1 : float or None
            Weight for accuracy (default 0.4).
        w2 : float or None
            Weight for entropy contribution (default 0.2).
        w3 : float or None
            Weight for stability (default 0.2).
        w4 : float or None
            Weight for surface_health / validity (default 0.2).
        """
        if w1 is not None:
            self._w1 = w1
        if w2 is not None:
            self._w2 = w2
        if w3 is not None:
            self._w3 = w3
        if w4 is not None:
            self._w4 = w4
        logger.debug(
            "set_weights w1=%.4f w2=%.4f w3=%.4f w4=%.4f",
            self._w1, self._w2, self._w3, self._w4,
        )

    # ------------------------------------------------------------------
    # Public API — scoring & arbitration
    # ------------------------------------------------------------------

    def _compute_oss_score(self):
        """Compute the aggregate OSS authority score."""
        if not self._oss_has_data:
            return 0.0
        score = (
            self._w1 * self._oss_accuracy
            + self._w2 * self._oss_entropy
            + self._w3 * self._oss_stability
            + self._w4 * self._oss_surface
        )
        return _clamp(score, 0.0, 1.0)

    def _compute_alt_score(self):
        """Compute the aggregate ALT authority score."""
        if not self._alt_has_data:
            return 0.0
        score = (
            self._w1 * self._alt_accuracy
            + self._w2 * self._alt_entropy
            + self._w3 * self._alt_stability
            + self._w4 * self._alt_validity
        )
        return _clamp(score, 0.0, 1.0)

    def arbitrate(self):
        """Compute authority scores and produce the arbitration decision.

        Returns
        -------
        dict
            ``oss_score``       — OSS aggregate score (0.0–1.0).

            ``alt_score``       — ALT aggregate score (0.0–1.0).

            ``oss_detail``      — dict of raw OSS component scores.

            ``alt_detail``      — dict of raw ALT component scores.

            ``authority``       — one of ``"OSS"``, ``"ALT"``, ``"HYBRID"``,
                                  ``"NONE"``.

            ``confidence``      — how confident the arbiter is (0.0–1.0).

            ``margin``          — ``oss_score - alt_score``.

            ``reasoning``       — human-readable explanation.
        """
        oss_score = self._compute_oss_score()
        alt_score = self._compute_alt_score()
        margin = oss_score - alt_score

        oss_detail = {
            "accuracy": self._oss_accuracy,
            "entropy": self._oss_entropy,
            "surface": self._oss_surface,
            "stability": self._oss_stability,
        }
        alt_detail = {
            "accuracy": self._alt_accuracy,
            "entropy": self._alt_entropy,
            "validity": self._alt_validity,
            "stability": self._alt_stability,
        }

        # ---- Authority decision logic ----
        #   If oss_score > alt_score + 0.15: "OSS"
        #   If alt_score > oss_score + 0.15: "ALT"
        #   If both >= 0.5 and within 0.15 of each other: "HYBRID"
        #   If both < 0.5: "NONE"

        abs_diff = abs(margin)

        if oss_score > alt_score + self.AUTHORITY_MARGIN:
            authority = "OSS"
            confidence = _clamp(oss_score / (oss_score + alt_score + 1e-12))
            reasoning = (
                f"OSS score ({oss_score:.4f}) exceeds ALT score ({alt_score:.4f}) "
                f"by more than {self.AUTHORITY_MARGIN:.2f} (margin={margin:+.4f}). "
                f"OSS demonstrated superior predictive accuracy, entropy "
                f"contribution, surface health, and/or stability."
            )
        elif alt_score > oss_score + self.AUTHORITY_MARGIN:
            authority = "ALT"
            confidence = _clamp(alt_score / (alt_score + oss_score + 1e-12))
            reasoning = (
                f"ALT score ({alt_score:.4f}) exceeds OSS score ({oss_score:.4f}) "
                f"by more than {self.AUTHORITY_MARGIN:.2f} (margin={margin:+.4f}). "
                f"ALT demonstrated superior predictive accuracy, entropy "
                f"contribution, validity, and/or stability."
            )
        elif oss_score >= self.HYBRID_MIN_SCORE and alt_score >= self.HYBRID_MIN_SCORE:
            authority = "HYBRID"
            confidence = _clamp(1.0 - abs_diff)
            reasoning = (
                f"Both OSS ({oss_score:.4f}) and ALT ({alt_score:.4f}) scores "
                f"are above {self.HYBRID_MIN_SCORE:.2f} and within "
                f"{self.AUTHORITY_MARGIN:.2f} of each other (margin={margin:+.4f}). "
                f"Neither signal source clearly dominates. HYBRID authority "
                f"is recommended to combine both sources."
            )
        else:
            authority = "NONE"
            max_score = max(oss_score, alt_score)
            confidence = _clamp(1.0 - max_score)
            if not self._oss_has_data and not self._alt_has_data:
                reasoning = (
                    f"No metrics have been fed for either signal source. "
                    f"Authority is NONE by default."
                )
            elif oss_score < self.HYBRID_MIN_SCORE and alt_score < self.HYBRID_MIN_SCORE:
                reasoning = (
                    f"Both OSS ({oss_score:.4f}) and ALT ({alt_score:.4f}) scores "
                    f"are below {self.HYBRID_MIN_SCORE:.2f}. Neither signal source "
                    f"demonstrates sufficient quality to warrant authority. "
                    f"Execution should be paused or fall back to a safe default."
                )
            else:
                reasoning = (
                    f"OSS ({oss_score:.4f}) and ALT ({alt_score:.4f}) are within "
                    f"{self.AUTHORITY_MARGIN:.2f} of each other but at least one "
                    f"is below {self.HYBRID_MIN_SCORE:.2f}. Authority is NONE."
                )

        self._latest_authority = authority
        self._latest_confidence = confidence
        self._latest_margin = margin
        self._latest_reasoning = reasoning

        logger.info(
            "arbitrate -> authority=%s oss=%.4f alt=%.4f margin=%+.4f "
            "confidence=%.4f",
            authority, oss_score, alt_score, margin, confidence,
        )

        return {
            "oss_score": oss_score,
            "alt_score": alt_score,
            "oss_detail": oss_detail,
            "alt_detail": alt_detail,
            "authority": authority,
            "confidence": confidence,
            "margin": margin,
            "reasoning": reasoning,
        }

    def get_authority(self):
        """Return the latest authority decision string.

        Returns
        -------
        str
            One of ``"OSS"``, ``"ALT"``, ``"HYBRID"``, ``"NONE"``.
        """
        return self._latest_authority

    # ------------------------------------------------------------------
    # Public API — reset
    # ------------------------------------------------------------------

    def reset(self):
        """Clear all stored metrics and reset the latest decision."""
        self._oss_accuracy = 0.0
        self._oss_entropy = 0.0
        self._oss_surface = 0.0
        self._oss_stability = 0.0
        self._oss_has_data = False

        self._alt_accuracy = 0.0
        self._alt_entropy = 0.0
        self._alt_validity = 0.0
        self._alt_stability = 0.0
        self._alt_has_data = False

        self._w1 = self.DEFAULT_W1
        self._w2 = self.DEFAULT_W2
        self._w3 = self.DEFAULT_W3
        self._w4 = self.DEFAULT_W4

        self._latest_authority = "NONE"
        self._latest_confidence = 0.0
        self._latest_margin = 0.0
        self._latest_reasoning = "Reset."

        logger.info("SignalAuthorityArbiter(%r) reset", self._instance_id)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("Signal Authority Arbiter — Self Test")
    print("=" * 60)

    # Use a mutable container for pass/fail state so the nested _check
    # function can update it.
    _state = {"passed": True}

    def _check(cond, msg):
        if not cond:
            _state["passed"] = False
            print(f"  FAIL: {msg}")
        else:
            print(f"  PASS: {msg}")

    # ------------------------------------------------------------------
    # Scenario 1: OSS is clearly better than ALT
    # ------------------------------------------------------------------
    print("\n--- SCE1: OSS is Better ---")
    arb1 = SignalAuthorityArbiter("sce1")
    arb1.feed_oss_metrics(
        accuracy=0.85,
        entropy_contribution=0.78,
        surface_health=0.82,
        stability=0.90,
    )
    arb1.feed_alt_metrics(
        accuracy=0.42,
        entropy_contribution=0.35,
        validity=0.38,
        stability=0.45,
    )
    result1 = arb1.arbitrate()
    print(f"  authority  = {result1['authority']}")
    print(f"  oss_score  = {result1['oss_score']:.4f}")
    print(f"  alt_score  = {result1['alt_score']:.4f}")
    print(f"  margin     = {result1['margin']:+.4f}")
    print(f"  confidence = {result1['confidence']:.4f}")
    print(f"  reasoning  = {result1['reasoning']}")
    _check(result1["authority"] == "OSS",
           f"Expected OSS, got {result1['authority']}")
    _check(result1["oss_score"] > result1["alt_score"],
           "OSS score > ALT score")
    _check(result1["margin"] > 0.15,
           f"Margin {result1['margin']:.4f} > 0.15")

    # ------------------------------------------------------------------
    # Scenario 2: ALT is clearly better than OSS
    # ------------------------------------------------------------------
    print("\n--- SCE2: ALT is Better ---")
    arb2 = SignalAuthorityArbiter("sce2")
    arb2.feed_oss_metrics(
        accuracy=0.30,
        entropy_contribution=0.25,
        surface_health=0.28,
        stability=0.35,
    )
    arb2.feed_alt_metrics(
        accuracy=0.88,
        entropy_contribution=0.82,
        validity=0.90,
        stability=0.85,
    )
    result2 = arb2.arbitrate()
    print(f"  authority  = {result2['authority']}")
    print(f"  oss_score  = {result2['oss_score']:.4f}")
    print(f"  alt_score  = {result2['alt_score']:.4f}")
    print(f"  margin     = {result2['margin']:+.4f}")
    print(f"  confidence = {result2['confidence']:.4f}")
    print(f"  reasoning  = {result2['reasoning']}")
    _check(result2["authority"] == "ALT",
           f"Expected ALT, got {result2['authority']}")
    _check(result2["alt_score"] > result2["oss_score"],
           "ALT score > OSS score")
    _check(result2["margin"] < -0.15,
           f"Margin {result2['margin']:.4f} < -0.15")

    # ------------------------------------------------------------------
    # Scenario 3: Both are good and close -> HYBRID
    # ------------------------------------------------------------------
    print("\n--- SCE3: Both Good -> HYBRID ---")
    arb3 = SignalAuthorityArbiter("sce3")
    arb3.feed_oss_metrics(
        accuracy=0.72,
        entropy_contribution=0.68,
        surface_health=0.70,
        stability=0.75,
    )
    arb3.feed_alt_metrics(
        accuracy=0.74,
        entropy_contribution=0.66,
        validity=0.71,
        stability=0.73,
    )
    result3 = arb3.arbitrate()
    print(f"  authority  = {result3['authority']}")
    print(f"  oss_score  = {result3['oss_score']:.4f}")
    print(f"  alt_score  = {result3['alt_score']:.4f}")
    print(f"  margin     = {result3['margin']:+.4f}")
    print(f"  confidence = {result3['confidence']:.4f}")
    print(f"  reasoning  = {result3['reasoning']}")
    _check(result3["authority"] == "HYBRID",
           f"Expected HYBRID, got {result3['authority']}")
    _check(result3["oss_score"] >= 0.5 and result3["alt_score"] >= 0.5,
           "Both scores >= 0.5")
    _check(abs(result3["margin"]) <= 0.15,
           f"Abs margin {abs(result3['margin']):.4f} <= 0.15")

    # ------------------------------------------------------------------
    # Scenario 4: Both are bad -> NONE
    # ------------------------------------------------------------------
    print("\n--- SCE4: Both Bad -> NONE ---")
    arb4 = SignalAuthorityArbiter("sce4")
    arb4.feed_oss_metrics(
        accuracy=0.25,
        entropy_contribution=0.20,
        surface_health=0.30,
        stability=0.22,
    )
    arb4.feed_alt_metrics(
        accuracy=0.18,
        entropy_contribution=0.15,
        validity=0.20,
        stability=0.17,
    )
    result4 = arb4.arbitrate()
    print(f"  authority  = {result4['authority']}")
    print(f"  oss_score  = {result4['oss_score']:.4f}")
    print(f"  alt_score  = {result4['alt_score']:.4f}")
    print(f"  margin     = {result4['margin']:+.4f}")
    print(f"  confidence = {result4['confidence']:.4f}")
    print(f"  reasoning  = {result4['reasoning']}")
    _check(result4["authority"] == "NONE",
           f"Expected NONE, got {result4['authority']}")
    _check(result4["oss_score"] < 0.5 and result4["alt_score"] < 0.5,
           "Both scores < 0.5")

    # ------------------------------------------------------------------
    # Scenario 5: No data fed -> NONE
    # ------------------------------------------------------------------
    print("\n--- SCE5: No Data -> NONE ---")
    arb5 = SignalAuthorityArbiter("sce5")
    result5 = arb5.arbitrate()
    print(f"  authority  = {result5['authority']}")
    print(f"  oss_score  = {result5['oss_score']:.4f}")
    print(f"  alt_score  = {result5['alt_score']:.4f}")
    print(f"  reasoning  = {result5['reasoning']}")
    _check(result5["authority"] == "NONE",
           f"Expected NONE, got {result5['authority']}")
    _check(result5["oss_score"] == 0.0 and result5["alt_score"] == 0.0,
           "Both scores are 0.0")
    _check("No metrics have been fed" in result5["reasoning"],
           "Reasoning mentions no metrics fed")

    # ------------------------------------------------------------------
    # Scenario 6: Custom weights
    # ------------------------------------------------------------------
    print("\n--- SCE6: Custom Weights ---")
    arb6 = SignalAuthorityArbiter("sce6")
    arb6.set_weights(w1=0.5, w2=0.3, w3=0.1, w4=0.1)
    arb6.feed_oss_metrics(
        accuracy=0.80,
        entropy_contribution=0.70,
        surface_health=0.60,
        stability=0.50,
    )
    arb6.feed_alt_metrics(
        accuracy=0.60,
        entropy_contribution=0.80,
        validity=0.70,
        stability=0.90,
    )
    result6 = arb6.arbitrate()
    print(f"  authority  = {result6['authority']}")
    print(f"  oss_score  = {result6['oss_score']:.4f}")
    print(f"  alt_score  = {result6['alt_score']:.4f}")
    print(f"  margin     = {result6['margin']:+.4f}")
    # With custom weights, accuracy gets higher weight, so OSS should still
    # lead but the margin might be different.
    _check(isinstance(result6["authority"], str),
           "Authority is a string")
    _check(result6["oss_detail"]["accuracy"] == 0.80,
           "OSS accuracy preserved in detail")
    _check(result6["alt_detail"]["validity"] == 0.70,
           "ALT validity preserved in detail")

    # ------------------------------------------------------------------
    # Scenario 7: get_authority() returns latest
    # ------------------------------------------------------------------
    print("\n--- SCE7: get_authority() ---")
    _check(arb1.get_authority() == "OSS",
           f"arb1.get_authority() == 'OSS', got {arb1.get_authority()}")
    _check(arb2.get_authority() == "ALT",
           f"arb2.get_authority() == 'ALT', got {arb2.get_authority()}")
    _check(arb3.get_authority() == "HYBRID",
           f"arb3.get_authority() == 'HYBRID', got {arb3.get_authority()}")
    _check(arb4.get_authority() == "NONE",
           f"arb4.get_authority() == 'NONE', got {arb4.get_authority()}")

    # ------------------------------------------------------------------
    # Scenario 8: Reset
    # ------------------------------------------------------------------
    print("\n--- SCE8: Reset ---")
    arb1.reset()
    result_reset = arb1.arbitrate()
    _check(result_reset["authority"] == "NONE",
           f"After reset, authority == 'NONE', got {result_reset['authority']}")
    _check(result_reset["oss_score"] == 0.0 and result_reset["alt_score"] == 0.0,
           "After reset, both scores are 0.0")
    _check(arb1.get_authority() == "NONE",
           "get_authority() returns 'NONE' after reset")
    print("  Reset verified")

    # ------------------------------------------------------------------
    # Scenario 9: Singleton identity
    # ------------------------------------------------------------------
    print("\n--- SCE9: Singleton identity ---")
    arb1_again = SignalAuthorityArbiter("sce1")
    _check(arb1_again is arb1,
           "Same instance_id returns same object")
    arb_default_1 = SignalAuthorityArbiter()
    arb_default_2 = SignalAuthorityArbiter("default")
    _check(arb_default_1 is arb_default_2,
           "Default singleton identity")
    arb_other = SignalAuthorityArbiter("other")
    _check(arb_other is not arb1,
           "Different instance_id returns different object")
    print("  Singleton verified")

    # ------------------------------------------------------------------
    # Scenario 10: Edge case — one source has data, other doesn't
    # ------------------------------------------------------------------
    print("\n--- SCE10: Only OSS has data ---")
    arb10 = SignalAuthorityArbiter("sce10")
    arb10.feed_oss_metrics(
        accuracy=0.75,
        entropy_contribution=0.70,
        surface_health=0.68,
        stability=0.72,
    )
    result10 = arb10.arbitrate()
    print(f"  authority  = {result10['authority']}")
    print(f"  oss_score  = {result10['oss_score']:.4f}")
    print(f"  alt_score  = {result10['alt_score']:.4f}")
    print(f"  reasoning  = {result10['reasoning']}")
    # With only OSS fed, alt_score = 0.0, so OSS should dominate
    _check(result10["authority"] == "OSS",
           f"Expected OSS, got {result10['authority']}")
    _check(result10["alt_score"] == 0.0,
           "ALT score is 0.0 when no ALT data fed")

    # ------------------------------------------------------------------
    # Scenario 11: Both above 0.5 but one slightly higher, within margin
    # ------------------------------------------------------------------
    print("\n--- SCE11: Both above 0.5, within margin -> HYBRID ---")
    arb11 = SignalAuthorityArbiter("sce11")
    arb11.feed_oss_metrics(
        accuracy=0.65,
        entropy_contribution=0.60,
        surface_health=0.62,
        stability=0.63,
    )
    arb11.feed_alt_metrics(
        accuracy=0.55,
        entropy_contribution=0.58,
        validity=0.56,
        stability=0.57,
    )
    result11 = arb11.arbitrate()
    print(f"  authority  = {result11['authority']}")
    print(f"  oss_score  = {result11['oss_score']:.4f}")
    print(f"  alt_score  = {result11['alt_score']:.4f}")
    print(f"  margin     = {result11['margin']:+.4f}")
    # OSS is higher but within margin, both >= 0.5 -> HYBRID
    _check(result11["authority"] == "HYBRID",
           f"Expected HYBRID, got {result11['authority']}")
    _check(result11["oss_score"] >= 0.5 and result11["alt_score"] >= 0.5,
           "Both scores >= 0.5")
    _check(abs(result11["margin"]) <= 0.15,
           f"Abs margin {abs(result11['margin']):.4f} <= 0.15")

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

    import sys
    sys.exit(0 if _state["passed"] else 1)
