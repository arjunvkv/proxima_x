"""
Authority Stability Tracker

Tracks whether authority flips too often. Measures policy volatility,
regime instability, and decision flip rate over a sliding window.
"""

import logging
from collections import Counter, deque
from statistics import mean
from typing import Any

logger = logging.getLogger(__name__)

_instances: dict[str, "_AuthorityStabilityTracker"] = {}


def AuthorityStabilityTracker(instance_id: str = "default") -> "_AuthorityStabilityTracker":
    """Singleton accessor for _AuthorityStabilityTracker.

    Parameters
    ----------
    instance_id : str
        Unique identifier for the tracker instance (default 'default').

    Returns
    -------
    _AuthorityStabilityTracker
        The shared tracker instance.
    """
    if instance_id not in _instances:
        _instances[instance_id] = _AuthorityStabilityTracker(instance_id)
    return _instances[instance_id]


class _AuthorityStabilityTracker:
    """Internal implementation of AuthorityStabilityTracker.

    Tracks authority decisions, signal decisions, and regime classifications
    over a sliding window and computes stability metrics on demand.
    """

    def __init__(self, instance_id: str = "default") -> None:
        self.instance_id = instance_id
        self.window_size = 100

        self._authority_decisions: deque[tuple[Any, str, float]] = deque(maxlen=self.window_size)
        self._signal_decisions: deque[tuple[str, int]] = deque(maxlen=self.window_size)
        self._regimes: deque[str] = deque(maxlen=self.window_size)

        logger.info("AuthorityStabilityTracker(%s) initialized", instance_id)

    # ------------------------------------------------------------------
    # Public feed methods
    # ------------------------------------------------------------------

    def feed_authority_decision(self, timestamp: Any, authority: str, confidence: float) -> None:
        """Record an authority decision.

        Parameters
        ----------
        timestamp : hashable
            Timestamp identifier for the decision.
        authority : str
            Authority label (e.g. 'bull', 'bear', 'neutral').
        confidence : float
            Confidence of the authority decision in [0, 1].
        """
        confidence = max(0.0, min(1.0, float(confidence)))
        self._authority_decisions.append((timestamp, authority, confidence))

    def feed_signal_decision(self, symbol: str, signal: int) -> None:
        """Record a final signal decision.

        Parameters
        ----------
        symbol : str
            Trading symbol identifier.
        signal : int
            Trading signal: -1, 0, or +1.

        Raises
        ------
        ValueError
            If signal is not in {-1, 0, 1}.
        """
        if signal not in (-1, 0, 1):
            raise ValueError(f"signal must be -1, 0, or 1; got {signal!r}")
        self._signal_decisions.append((symbol, signal))

    def feed_regime(self, regime: str) -> None:
        """Record a regime classification.

        Parameters
        ----------
        regime : str
            Regime label (e.g. 'bull', 'bear', 'range', 'volatile').
        """
        self._regimes.append(regime)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def get_stability_report(self) -> dict[str, Any]:
        """Produce a full stability report.

        Returns
        -------
        dict
            Stability report with keys:
            - window_size : int
            - policy_volatility : float  (changes per 100 decisions)
            - decision_flip_rate : float  (fraction of flips in window)
            - regime_stability : float  (fraction of time in same regime)
            - authority_conviction : float  (mean confidence)
            - stability_verdict : str  (STABLE | MODERATE | UNSTABLE)
            - recommendation : str
        """
        policy_volatility = self._compute_policy_volatility()
        decision_flip_rate = self._compute_decision_flip_rate()
        regime_stability = self._compute_regime_stability()
        authority_conviction = self._compute_authority_conviction()

        verdict = self._determine_verdict(
            policy_volatility, decision_flip_rate, authority_conviction
        )
        recommendation = self._generate_recommendation(
            policy_volatility, decision_flip_rate, regime_stability,
            authority_conviction, verdict
        )

        return {
            "window_size": self.window_size,
            "policy_volatility": round(policy_volatility, 4),
            "decision_flip_rate": round(decision_flip_rate, 4),
            "regime_stability": round(regime_stability, 4),
            "authority_conviction": round(authority_conviction, 4),
            "stability_verdict": verdict,
            "recommendation": recommendation,
        }

    def is_stable(self) -> bool:
        """Quick check if the system is currently stable.

        Returns
        -------
        bool
            True if the stability verdict is 'STABLE'.
        """
        return (
            self._determine_verdict(
                self._compute_policy_volatility(),
                self._compute_decision_flip_rate(),
                self._compute_authority_conviction(),
            )
            == "STABLE"
        )

    def reset(self) -> None:
        """Clear all stored data."""
        self._authority_decisions.clear()
        self._signal_decisions.clear()
        self._regimes.clear()
        logger.info("AuthorityStabilityTracker(%s) reset", self.instance_id)

    # ------------------------------------------------------------------
    # Internal metric computations
    # ------------------------------------------------------------------

    def _compute_policy_volatility(self) -> float:
        """Number of policy changes in the window, scaled to per-100-decisions.

        A policy change is a transition where the authority label differs
        from the previous entry.

        Returns
        -------
        float
            Changes per 100 decisions.  Returns 0.0 when fewer than 2 entries.
        """
        decisions = list(self._authority_decisions)
        if len(decisions) < 2:
            return 0.0

        changes = sum(
            1
            for i in range(1, len(decisions))
            if decisions[i][1] != decisions[i - 1][1]
        )
        # Normalise to per-100-decisions
        return (changes / (len(decisions) - 1)) * 100.0

    def _compute_decision_flip_rate(self) -> float:
        """Fraction of consecutive signal decisions where direction flips.

        A flip is a transition from +1 to -1 or -1 to +1.  Transitions
        involving 0 are never counted as flips.

        Returns
        -------
        float
            Flip rate in [0.0, 1.0].  Returns 0.0 when fewer than 2 entries.
        """
        signals = [s for _, s in self._signal_decisions]
        if len(signals) < 2:
            return 0.0

        flips = 0
        for i in range(1, len(signals)):
            if (signals[i - 1] == 1 and signals[i] == -1) or (
                signals[i - 1] == -1 and signals[i] == 1
            ):
                flips += 1
        return flips / (len(signals) - 1)

    def _compute_regime_stability(self) -> float:
        """Fraction of time spent in the most common regime.

        Returns
        -------
        float
            Regime stability in [0.0, 1.0].  Returns 0.0 when the window is
            empty.
        """
        regimes = list(self._regimes)
        if not regimes:
            return 0.0

        counter = Counter(regimes)
        most_common_count = counter.most_common(1)[0][1]
        return most_common_count / len(regimes)

    def _compute_authority_conviction(self) -> float:
        """Mean confidence of authority decisions in the window.

        Returns
        -------
        float
            Mean confidence in [0.0, 1.0].  Returns 0.0 when no decisions.
        """
        decisions = list(self._authority_decisions)
        if not decisions:
            return 0.0

        confidences = [d[2] for d in decisions]
        return mean(confidences)

    # ------------------------------------------------------------------
    # Verdict & recommendation
    # ------------------------------------------------------------------

    @staticmethod
    def _determine_verdict(
        policy_volatility: float,
        decision_flip_rate: float,
        authority_conviction: float,
    ) -> str:
        """Classify overall stability.

        Rules
        -----
        STABLE
            volatility < 5  AND  flip_rate < 0.2  AND  conviction > 0.7
        UNSTABLE
            volatility > 20  OR  flip_rate > 0.4
        MODERATE
            Everything else.
        """
        if policy_volatility < 5 and decision_flip_rate < 0.2 and authority_conviction > 0.7:
            return "STABLE"
        if policy_volatility > 20 or decision_flip_rate > 0.4:
            return "UNSTABLE"
        return "MODERATE"

    @staticmethod
    def _generate_recommendation(
        policy_volatility: float,
        decision_flip_rate: float,
        regime_stability: float,
        authority_conviction: float,
        verdict: str,
    ) -> str:
        """Generate a human-readable recommendation based on stability metrics."""
        if verdict == "STABLE":
            return (
                "Authority is stable. Continue with current "
                "decision-making process."
            )

        parts: list[str] = []

        if policy_volatility > 20:
            parts.append(
                f"Policy volatility is very high ({policy_volatility:.1f} "
                f"changes per 100 decisions). Consider reducing the frequency "
                f"of authority switches."
            )
        elif policy_volatility > 5:
            parts.append(
                f"Policy volatility is elevated ({policy_volatility:.1f} "
                f"changes per 100 decisions). Monitor authority transitions."
            )

        if decision_flip_rate > 0.4:
            parts.append(
                f"Decision flip rate is very high ({decision_flip_rate:.2f}). "
                f"Signals are erratic; consider raising signal confidence "
                f"thresholds."
            )
        elif decision_flip_rate > 0.2:
            parts.append(
                f"Decision flip rate is elevated ({decision_flip_rate:.2f}). "
                f"Monitor for signal instability."
            )

        if regime_stability < 0.5:
            parts.append(
                f"Regime classification is unstable "
                f"(stability={regime_stability:.2f}). Regime detection may "
                f"need re-calibration."
            )

        if authority_conviction <= 0.7:
            parts.append(
                f"Authority conviction is low ({authority_conviction:.2f}). "
                f"Consider requiring higher confidence for authority "
                f"decisions."
            )

        if not parts:
            parts.append(
                "System shows moderate stability. Continue monitoring."
            )

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Self-test
    # ------------------------------------------------------------------

    @staticmethod
    def _self_test() -> None:
        """Run self-tests with stable, unstable, and moderate scenarios."""
        import random

        print("=" * 60)
        print("AuthorityStabilityTracker \u2014 Self-Test")
        print("=" * 60)

        # ---- 1. Stable scenario ----
        print("\n--- Scenario 1: STABLE ---")
        s = AuthorityStabilityTracker("self_test_stable")
        for i in range(100):
            s.feed_authority_decision(i, "bull", 0.85 + random.uniform(-0.05, 0.05))
            s.feed_signal_decision("EURUSD", random.choice([0, 1]))
            s.feed_regime("bull")

        r = s.get_stability_report()
        print(f"  Policy volatility:      {r['policy_volatility']:.4f}")
        print(f"  Decision flip rate:     {r['decision_flip_rate']:.4f}")
        print(f"  Regime stability:       {r['regime_stability']:.4f}")
        print(f"  Authority conviction:   {r['authority_conviction']:.4f}")
        print(f"  Stability verdict:      {r['stability_verdict']}")
        assert r["stability_verdict"] in ("STABLE", "MODERATE"), (
            f"Expected STABLE or MODERATE, got {r['stability_verdict']}"
        )
        print(f"  \u2713 Stable scenario passed")

        # ---- 2. Unstable scenario (high volatility + flips + low conviction) ----
        print("\n--- Scenario 2: UNSTABLE ---")
        u = AuthorityStabilityTracker("self_test_unstable")
        authorities = ["bull", "bear", "neutral", "volatile"]
        for i in range(100):
            auth = authorities[i % len(authorities)]
            u.feed_authority_decision(i, auth, random.uniform(0.3, 0.6))
            signal = 1 if i % 2 == 0 else -1
            u.feed_signal_decision("EURUSD", signal)
            u.feed_regime(authorities[(i + 1) % len(authorities)])

        r2 = u.get_stability_report()
        print(f"  Policy volatility:      {r2['policy_volatility']:.4f}")
        print(f"  Decision flip rate:     {r2['decision_flip_rate']:.4f}")
        print(f"  Regime stability:       {r2['regime_stability']:.4f}")
        print(f"  Authority conviction:   {r2['authority_conviction']:.4f}")
        print(f"  Stability verdict:      {r2['stability_verdict']}")
        assert r2["stability_verdict"] == "UNSTABLE", (
            f"Expected UNSTABLE, got {r2['stability_verdict']}"
        )
        # Verify the flip rate is high (alternating +/-1)
        assert r2["decision_flip_rate"] > 0.9, (
            f"Expected near-1.0 flip rate, got {r2['decision_flip_rate']}"
        )
        print(f"  \u2713 Unstable scenario passed")

        # ---- 3. Moderate scenario ----
        print("\n--- Scenario 3: MODERATE ---")
        m = AuthorityStabilityTracker("self_test_moderate")
        for i in range(100):
            auth = "bull" if i < 50 else "bear"
            m.feed_authority_decision(i, auth, 0.65 + random.uniform(-0.05, 0.05))
            m.feed_signal_decision("EURUSD", random.choice([-1, 0, 1]))
            m.feed_regime("bull" if i < 50 else "bear")

        r3 = m.get_stability_report()
        print(f"  Policy volatility:      {r3['policy_volatility']:.4f}")
        print(f"  Decision flip rate:     {r3['decision_flip_rate']:.4f}")
        print(f"  Regime stability:       {r3['regime_stability']:.4f}")
        print(f"  Authority conviction:   {r3['authority_conviction']:.4f}")
        print(f"  Stability verdict:      {r3['stability_verdict']}")
        print(f"  Recommendation:         {r3['recommendation'][:80]}...")
        assert r3["stability_verdict"] == "MODERATE", (
            f"Expected MODERATE, got {r3['stability_verdict']}"
        )
        print(f"  \u2713 Moderate scenario passed")

        # ---- 4. is_stable() ----
        print("\n--- Scenario 4: is_stable() ---")
        assert s.is_stable() == (r["stability_verdict"] == "STABLE")
        assert not u.is_stable()
        print("  \u2713 is_stable() passes")

        # ---- 5. reset() ----
        print("\n--- Scenario 5: reset() ---")
        s.reset()
        assert len(s._authority_decisions) == 0
        assert len(s._signal_decisions) == 0
        assert len(s._regimes) == 0
        r_reset = s.get_stability_report()
        for key in ("policy_volatility", "decision_flip_rate", "regime_stability", "authority_conviction"):
            assert r_reset[key] == 0.0, f"{key} should be 0.0 after reset, got {r_reset[key]}"
        print("  \u2713 reset() passes")

        print("\n" + "=" * 60)
        print("All self-tests passed.")
        print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _AuthorityStabilityTracker._self_test()
