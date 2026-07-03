"""
Reality Consistency Gate — answers "is there a consistent market truth at all?"

Checks three pairwise consistency relationships:

  1. **OSS ↔ ALT**  — Do they agree on direction?
  2. **ALT ↔ Returns** — Does ALT predict forward returns?
  3. **OSS ↔ Returns** — Does OSS predict forward returns?

The global verdict summarises whether a consistent reality (signal truth)
exists in the observed data.

Usage
-----
    from core_runtime.reality_consistency_gate import RealityConsistencyGate

    gate = RealityConsistencyGate()
    gate.feed_tick("EURUSD", oss_signal=+1, alt_signal=+1, bid_price=1.1000)
    gate.feed_forward_return("EURUSD", 0.005)
    report = gate.check_consistency("EURUSD")
    print(report["global_verdict"])
"""

import logging
import math
from typing import Any, Dict, List, Set, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_instances: Dict[str, "_RealityConsistencyGate"] = {}


def RealityConsistencyGate(instance_id="default"):
    """Singleton accessor for ``_RealityConsistencyGate``.

    Parameters
    ----------
    instance_id : str
        Unique identifier.  Callers sharing the same *instance_id* share the
        same underlying gate object.

    Returns
    -------
    _RealityConsistencyGate
    """
    if instance_id not in _instances:
        _instances[instance_id] = _RealityConsistencyGate(instance_id)
    return _instances[instance_id]


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------

class _RealityConsistencyGate:
    """Tracks consistency between OSS signals, ALT signals, and forward returns.

    For each symbol, stores a sequence of ticks and their corresponding forward
    returns, then analyses the three pairwise relationships to determine whether
    a consistent market truth exists.
    """

    def __init__(self, instance_id="default"):
        self._instance_id = instance_id

        # Each symbol maps to a list of tick dicts:
        #   {"timestamp": ..., "oss_signal": int, "alt_signal": int,
        #    "bid_price": float, "forward_return": Optional[float]}
        self._ticks: Dict[str, List[Dict[str, Any]]] = {}

        logger.info(
            "RealityConsistencyGate(%r) initialised",
            instance_id,
        )

    # ------------------------------------------------------------------
    # Feed API
    # ------------------------------------------------------------------

    def feed_tick(
        self,
        symbol: str,
        oss_signal: int,
        alt_signal: int,
        bid_price: float,
        timestamp=None,
    ):
        """Record a tick observation for *symbol*.

        Parameters
        ----------
        symbol : str
            Instrument identifier.
        oss_signal : int
            OSS directional signal (-1, 0, or +1).
        alt_signal : int
            ALT directional signal (-1, 0, or +1).
        bid_price : float
            Current bid price.
        timestamp : optional
            Tick timestamp.  If ``None``, an auto-incrementing integer
            (current tick count for the symbol) is used.
        """
        if symbol not in self._ticks:
            self._ticks[symbol] = []

        if timestamp is None:
            timestamp = len(self._ticks[symbol])

        self._ticks[symbol].append({
            "timestamp": timestamp,
            "oss_signal": oss_signal,
            "alt_signal": alt_signal,
            "bid_price": bid_price,
            "forward_return": None,
        })
        logger.debug(
            "feed_tick(%s): oss=%d alt=%d price=%.5f ts=%s",
            symbol, oss_signal, alt_signal, bid_price, timestamp,
        )

    def feed_forward_return(self, symbol: str, forward_return: float):
        """Record the forward return for the most recent unmatched tick.

        The forward return is paired with the **last** tick that does not
        yet have a return assigned (LIFO matching).

        Parameters
        ----------
        symbol : str
            Instrument identifier.
        forward_return : float
            Realised forward return (e.g. percentage change or log return).
        """
        ticks = self._ticks.get(symbol, [])
        if not ticks:
            logger.warning(
                "feed_forward_return(%s): no ticks recorded — ignoring", symbol,
            )
            return

        # Find the most recent tick without a forward return (LIFO)
        for tick in reversed(ticks):
            if tick["forward_return"] is None:
                tick["forward_return"] = forward_return
                logger.debug(
                    "feed_forward_return(%s): return=%.6f matched to tick ts=%s",
                    symbol, forward_return, tick["timestamp"],
                )
                return

        logger.warning(
            "feed_forward_return(%s): no unmatched ticks — ignoring return=%.6f",
            symbol, forward_return,
        )

    # ------------------------------------------------------------------
    # Consistency analysis
    # ------------------------------------------------------------------

    def check_consistency(self, symbol: str) -> dict:
        """Analyse the three consistency relationships for *symbol*.

        Parameters
        ----------
        symbol : str
            Instrument identifier.

        Returns
        -------
        dict
            Full consistency report (see module docstring for schema).
        """
        ticks = self._ticks.get(symbol, [])
        paired = [t for t in ticks if t["forward_return"] is not None]
        n_observations = len(paired)

        if n_observations < 2:
            return self._empty_report(symbol, n_observations)

        # Extract aligned arrays
        oss_signals = [t["oss_signal"] for t in paired]
        alt_signals = [t["alt_signal"] for t in paired]
        forward_returns = [t["forward_return"] for t in paired]

        # -- OSS ↔ ALT consistency ------------------------------------------
        oss_alt_agreement = self._agreement_rate(oss_signals, alt_signals)
        oss_alt_kappa = self._cohens_kappa(oss_signals, alt_signals)

        if oss_alt_agreement > 0.5:
            oss_alt_verdict = "AGREE"
        elif oss_alt_agreement > 0.3:
            oss_alt_verdict = "NEUTRAL"
        else:
            oss_alt_verdict = "DISAGREE"

        # -- ALT ↔ Returns consistency ---------------------------------------
        alt_return_accuracy = self._directional_accuracy(alt_signals, forward_returns)
        alt_return_corr = self._pearson_r(alt_signals, forward_returns)

        if alt_return_accuracy > 0.5 and alt_return_corr > 0.0:
            alt_return_verdict = "PREDICTIVE"
        elif alt_return_accuracy < 0.5 and alt_return_corr < 0.0:
            alt_return_verdict = "ANTI_PREDICTIVE"
        else:
            alt_return_verdict = "NOISY"

        # -- OSS ↔ Returns consistency ---------------------------------------
        oss_return_accuracy = self._directional_accuracy(oss_signals, forward_returns)
        oss_return_corr = self._pearson_r(oss_signals, forward_returns)

        if oss_return_accuracy > 0.5 and oss_return_corr > 0.0:
            oss_return_verdict = "PREDICTIVE"
        elif oss_return_accuracy < 0.5 and oss_return_corr < 0.0:
            oss_return_verdict = "ANTI_PREDICTIVE"
        else:
            oss_return_verdict = "NOISY"

        # -- Global verdict --------------------------------------------------
        global_verdict = self._global_verdict(
            oss_alt_agreement, alt_return_accuracy, oss_return_accuracy,
        )

        return {
            "symbol": symbol,
            "observations": n_observations,
            "oss_alt_consistency": {
                "agreement_rate": round(oss_alt_agreement, 4),
                "cohens_kappa": round(oss_alt_kappa, 4),
                "verdict": oss_alt_verdict,
            },
            "alt_return_consistency": {
                "directional_accuracy": round(alt_return_accuracy, 4),
                "correlation": round(alt_return_corr, 4),
                "verdict": alt_return_verdict,
            },
            "oss_return_consistency": {
                "directional_accuracy": round(oss_return_accuracy, 4),
                "correlation": round(oss_return_corr, 4),
                "verdict": oss_return_verdict,
            },
            "global_verdict": global_verdict,
        }

    # ------------------------------------------------------------------
    # Global verdict logic
    # ------------------------------------------------------------------

    @staticmethod
    def _global_verdict(
        agreement: float,
        alt_accuracy: float,
        oss_accuracy: float,
    ) -> str:
        """Determine the global consistency verdict.

        Logic
        -----
        - **CONSISTENT_REALITY**:  agreement > 0.5 AND either accuracy > 0.5
        - **PARTIAL_REALITY**:     agreement > 0.3 OR any accuracy > 0.5
          (but not meeting the CONSISTENT_REALITY threshold)
        - **NO_REALITY**:          otherwise
        """
        if agreement > 0.5 and (alt_accuracy > 0.5 or oss_accuracy > 0.5):
            return "CONSISTENT_REALITY"
        if agreement > 0.3 or alt_accuracy > 0.5 or oss_accuracy > 0.5:
            return "PARTIAL_REALITY"
        return "NO_REALITY"

    # ------------------------------------------------------------------
    # Statistical helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _agreement_rate(a: List[int], b: List[int]) -> float:
        """Fraction of positions where *a* and *b* have the same value."""
        if not a:
            return 0.0
        matches = sum(1 for x, y in zip(a, b) if x == y)
        return matches / len(a)

    @staticmethod
    def _cohens_kappa(a: List[int], b: List[int]) -> float:
        """Cohen's kappa: chance-adjusted agreement between two raters.

        .. math::

            \\kappa = \\frac{p_o - p_e}{1 - p_e}

        where *p_o* is the observed agreement rate and *p_e* is the expected
        agreement by chance (sum of category-wise marginal products).

        Categories are the unique values appearing in *a* and *b* (typically
        -1, 0, +1).

        Returns
        -------
        float
            Value in [-1, 1]; 1 = perfect agreement, 0 = chance, -1 = total
            disagreement.
        """
        if not a or len(a) != len(b):
            return 0.0

        n = len(a)
        categories: Set[int] = set(a) | set(b)
        if len(categories) < 2:
            # Only one category used — perfect agreement by definition
            return 1.0

        # Observed agreement
        p_o = sum(1 for x, y in zip(a, b) if x == y) / n

        # Expected agreement by chance
        #   p_e = sum_{cat} P(cat in a) * P(cat in b)
        p_e = 0.0
        for cat in categories:
            p_a = sum(1 for x in a if x == cat) / n
            p_b = sum(1 for x in b if x == cat) / n
            p_e += p_a * p_b

        denominator = 1.0 - p_e
        if denominator == 0.0:
            return 1.0

        return (p_o - p_e) / denominator

    @staticmethod
    def _directional_accuracy(
        signals: List[int],
        returns: List[float],
    ) -> float:
        """Fraction of times the signal sign matches the return sign.

        Signals of 0 (flat) are excluded from both numerator and denominator
        because a flat signal makes no directional claim.

        Returns
        -------
        float
            0.5 if no directional signals exist (fallback to chance).
        """
        correct = 0
        total = 0
        for sig, ret in zip(signals, returns):
            if sig == 0:
                continue
            total += 1
            if (sig > 0 and ret > 0) or (sig < 0 and ret < 0):
                correct += 1
        return correct / total if total > 0 else 0.5

    @staticmethod
    def _pearson_r(x: List[Union[int, float]], y: List[float]) -> float:
        """Pearson correlation coefficient between *x* and *y*.

        Returns
        -------
        float
            Value in [-1, 1]; 0 if fewer than 2 observations or degenerate.
        """
        n = len(x)
        if n < 2:
            return 0.0

        x_bar = sum(x) / n
        y_bar = sum(y) / n

        num = 0.0
        den_x = 0.0
        den_y = 0.0
        for xi, yi in zip(x, y):
            dx = xi - x_bar
            dy = yi - y_bar
            num += dx * dy
            den_x += dx * dx
            den_y += dy * dy

        denom = math.sqrt(den_x * den_y)
        if denom == 0.0:
            return 0.0

        return num / denom

    # ------------------------------------------------------------------
    # Default / empty report
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_report(symbol: str, observations: int) -> dict:
        """Return a default report when there is insufficient data."""
        return {
            "symbol": symbol,
            "observations": observations,
            "oss_alt_consistency": {
                "agreement_rate": 0.0,
                "cohens_kappa": 0.0,
                "verdict": "NEUTRAL",
            },
            "alt_return_consistency": {
                "directional_accuracy": 0.0,
                "correlation": 0.0,
                "verdict": "NOISY",
            },
            "oss_return_consistency": {
                "directional_accuracy": 0.0,
                "correlation": 0.0,
                "verdict": "NOISY",
            },
            "global_verdict": "NO_REALITY",
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest():
    """Exercise the RealityConsistencyGate with three key scenarios.

    Scenarios
    ---------
    1. **CONSISTENT_REALITY** — OSS and ALT agree on direction and both
       predict forward returns.
    2. **NO_REALITY** — OSS and ALT always disagree and neither predicts
       returns.
    3. **PARTIAL_REALITY** — partial agreement and some predictive power,
       but insufficient for a consistent-reality verdict.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("=" * 60)
    logger.info("RealityConsistencyGate — Self-Test")
    logger.info("=" * 60)

    passed = 0
    failed = 0

    def _check(cond: bool, msg: str) -> None:
        nonlocal passed, failed
        if cond:
            passed += 1
            logger.info("  [PASS] %s", msg)
        else:
            failed += 1
            logger.error("  [FAIL] %s", msg)

    # ==================================================================
    # Scenario 1 — CONSISTENT_REALITY
    #   OSS and ALT always agree (+1 with positive returns, -1 with
    #   negative returns).  Both are perfectly predictive.
    # ==================================================================
    logger.info("")
    logger.info("--- Scenario 1: CONSISTENT_REALITY ---")

    gate1 = RealityConsistencyGate("selftest_consistent")

    # 30 ticks: OSS=ALT=+1, positive returns
    for _ in range(30):
        gate1.feed_tick("EURUSD", +1, +1, 1.1000)
        gate1.feed_forward_return("EURUSD", 0.005)

    # 30 ticks: OSS=ALT=-1, negative returns
    for _ in range(30):
        gate1.feed_tick("EURUSD", -1, -1, 1.1000)
        gate1.feed_forward_return("EURUSD", -0.005)

    r1 = gate1.check_consistency("EURUSD")
    logger.info("  Global verdict: %s", r1["global_verdict"])

    _check(r1["global_verdict"] == "CONSISTENT_REALITY",
           "Scenario 1 → CONSISTENT_REALITY")
    _check(r1["observations"] == 60,
           f"Scenario 1 observations=60, got {r1['observations']}")
    _check(r1["oss_alt_consistency"]["agreement_rate"] == 1.0,
           "Scenario 1 agreement_rate = 1.0")
    _check(r1["oss_alt_consistency"]["cohens_kappa"] == 1.0,
           "Scenario 1 cohens_kappa = 1.0")
    _check(r1["alt_return_consistency"]["directional_accuracy"] == 1.0,
           "Scenario 1 alt return accuracy = 1.0")
    _check(r1["oss_return_consistency"]["directional_accuracy"] == 1.0,
           "Scenario 1 oss return accuracy = 1.0")

    # ==================================================================
    # Scenario 2 — NO_REALITY
    #   OSS always +1, ALT always -1 (zero agreement).  Returns are
    #   evenly split, so both directional accuracies are ~0.5 (chance).
    # ==================================================================
    logger.info("")
    logger.info("--- Scenario 2: NO_REALITY ---")

    gate2 = RealityConsistencyGate("selftest_no_reality")

    for _ in range(30):
        gate2.feed_tick("EURUSD", +1, -1, 1.1000)
        gate2.feed_forward_return("EURUSD", 0.005)
    for _ in range(30):
        gate2.feed_tick("EURUSD", +1, -1, 1.1000)
        gate2.feed_forward_return("EURUSD", -0.005)

    r2 = gate2.check_consistency("EURUSD")
    logger.info("  Global verdict: %s", r2["global_verdict"])

    _check(r2["global_verdict"] == "NO_REALITY",
           "Scenario 2 → NO_REALITY")
    _check(r2["oss_alt_consistency"]["agreement_rate"] == 0.0,
           "Scenario 2 agreement_rate = 0.0")
    _check(r2["oss_return_consistency"]["directional_accuracy"] == 0.5,
           "Scenario 2 oss accuracy = 0.5 (chance)")
    _check(r2["alt_return_consistency"]["directional_accuracy"] == 0.5,
           "Scenario 2 alt accuracy = 0.5 (chance)")

    # ==================================================================
    # Scenario 3 — PARTIAL_REALITY
    #   OSS and ALT agree 40% of the time (agreement=0.4, which is
    #   > 0.3 but ≤ 0.5).  OSS always predicts correctly (accuracy=1.0),
    #   but the CONSISTENT_REALITY condition requires *both* agreement
    #   > 0.5 *and* accuracy > 0.5 — since agreement is only 0.4, the
    #   result falls to PARTIAL_REALITY.
    # ==================================================================
    logger.info("")
    logger.info("--- Scenario 3: PARTIAL_REALITY ---")

    gate3 = RealityConsistencyGate("selftest_partial")

    # 20 agreeing ticks: OSS=ALT=+1, positive returns
    for _ in range(20):
        gate3.feed_tick("EURUSD", +1, +1, 1.1000)
        gate3.feed_forward_return("EURUSD", 0.003)

    # 30 disagreeing ticks: OSS=+1, ALT=-1, all positive returns
    for _ in range(30):
        gate3.feed_tick("EURUSD", +1, -1, 1.1000)
        gate3.feed_forward_return("EURUSD", 0.003)

    r3 = gate3.check_consistency("EURUSD")
    logger.info("  Global verdict: %s", r3["global_verdict"])
    logger.info("  agreement_rate: %.4f", r3["oss_alt_consistency"]["agreement_rate"])
    logger.info("  oss accuracy:   %.4f", r3["oss_return_consistency"]["directional_accuracy"])
    logger.info("  alt accuracy:   %.4f", r3["alt_return_consistency"]["directional_accuracy"])

    _check(r3["global_verdict"] == "PARTIAL_REALITY",
           "Scenario 3 → PARTIAL_REALITY")
    _check(r3["observations"] == 50,
           f"Scenario 3 observations=50, got {r3['observations']}")
    _check(
        abs(r3["oss_alt_consistency"]["agreement_rate"] - 0.4) < 0.001,
        "Scenario 3 agreement_rate ≈ 0.4",
    )
    _check(r3["oss_return_consistency"]["directional_accuracy"] == 1.0,
           "Scenario 3 oss accuracy = 1.0")
    _check(
        abs(r3["alt_return_consistency"]["directional_accuracy"] - 0.4) < 0.001,
        "Scenario 3 alt accuracy ≈ 0.4",
    )

    # ==================================================================
    # Edge case: insufficient data (fewer than 2 observations)
    # ==================================================================
    logger.info("")
    logger.info("--- Edge case: insufficient data ---")

    gate4 = RealityConsistencyGate("selftest_insufficient")
    gate4.feed_tick("EURUSD", +1, +1, 1.1000)
    gate4.feed_forward_return("EURUSD", 0.005)
    r4 = gate4.check_consistency("EURUSD")
    _check(r4["global_verdict"] == "NO_REALITY",
           "Insufficient data → NO_REALITY")
    _check(r4["observations"] == 1,
           f"Insufficient data observations=1, got {r4['observations']}")

    # ==================================================================
    # Edge case: unmatched forward return
    # ==================================================================
    logger.info("")
    logger.info("--- Edge case: unmatched forward return (no ticks) ---")

    gate5 = RealityConsistencyGate("selftest_unmatched")
    gate5.feed_forward_return("EURUSD", 0.01)  # no ticks yet — should warn
    # Should not crash; feed a tick and check
    gate5.feed_tick("EURUSD", 0, 0, 1.1000)
    gate5.feed_forward_return("EURUSD", 0.01)
    r5 = gate5.check_consistency("EURUSD")
    _check(r5["observations"] == 1,
           "Unmatched warn + single tick → observations=1")

    # ==================================================================
    # Singleton accessor
    # ==================================================================
    logger.info("")
    logger.info("--- Singleton accessor ---")

    a = RealityConsistencyGate("selftest_singleton")
    b = RealityConsistencyGate("selftest_singleton")
    c = RealityConsistencyGate("selftest_singleton_other")
    _check(a is b, "Same instance_id returns same object")
    _check(a is not c, "Different instance_id returns different object")

    # ==================================================================
    # Summary
    # ==================================================================
    logger.info("")
    logger.info("-" * 60)
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
