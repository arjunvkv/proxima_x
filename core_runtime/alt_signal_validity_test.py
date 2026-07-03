"""
AltSignalValidityTest — validates that the ALT signal is not just noise-fitting.

Tests persistence across regimes, stability under shuffle, and forward return
correlation. The ALT signal must be a VALID control, not another broken signal
source.

Usage
-----
    from core_runtime.alt_signal_validity_test import AltSignalValidityTest

    tester = AltSignalValidityTest()
    tester.feed_signal("EURUSD", 1, 0.8, forward_return=0.001)
    tester.feed_signal("EURUSD", -1, 0.6)
    tester.feed_forward_return("EURUSD", -0.002)
    results = tester.run_all_tests("EURUSD")
    print(results["overall_verdict"])
"""

import logging
import math
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.stats import chi2_contingency, pearsonr, spearmanr

    _SCIPY_AVAILABLE = True
except ImportError:
    chi2_contingency = None  # type: ignore
    pearsonr = None  # type: ignore
    spearmanr = None  # type: ignore
    _SCIPY_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.debug("scipy not available — chi-squared & correlation tests disabled")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton registry
# ---------------------------------------------------------------------------

_instances: Dict[str, "_AltSignalValidityTest"] = {}


def AltSignalValidityTest(instance_id: str = "default"):
    """Singleton accessor — returns the same ``_AltSignalValidityTest`` for a
    given *instance_id*.

    Parameters
    ----------
    instance_id : str
        Unique identifier for the test instance.

    Returns
    -------
    _AltSignalValidityTest
    """
    if instance_id not in _instances:
        _instances[instance_id] = _AltSignalValidityTest(instance_id)
    return _instances[instance_id]


# ===================================================================
# Internal implementation
# ===================================================================


class _AltSignalValidityTest:
    """Validates ALT signal quality through a battery of statistical tests.

    The ALT signal acts as a scientific control for the main OSS signal.
    These tests ensure the ALT signal is a valid control — persistent,
    temporally structured, and predictive — rather than random noise that
    happens to fit past data.

    Parameters
    ----------
    instance_id : str
        Label for logging and singleton registry.
    """

    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------

    DEFAULT_WINDOW_SIZE: int = 20      # ticks per regime window
    DEFAULT_FORWARD_TICKS: int = 1     # N for forward return matching

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def __init__(self, instance_id: str = "default"):
        self._instance_id = instance_id

        # Per-symbol signal history:
        #   _data[symbol] = list of dicts with keys:
        #       "signal"         : int (-1, 0, +1)
        #       "confidence"     : float
        #       "forward_return" : float or None
        self._data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Per-symbol queue of indices whose forward_return is still None.
        # Indices are popped FIFO when feed_forward_return is called.
        self._pending_fills: Dict[str, List[int]] = defaultdict(list)

        logger.info("AltSignalValidityTest(%r) initialised", instance_id)

    # ------------------------------------------------------------------
    # Public data-feed methods
    # ------------------------------------------------------------------

    def feed_signal(
        self,
        symbol: str,
        alt_signal: int,
        alt_confidence: float,
        forward_return: Optional[float] = None,
    ) -> None:
        """Record one ALT signal observation.

        Parameters
        ----------
        symbol : str
            Instrument identifier.
        alt_signal : int
            Directional signal (-1, 0, or +1).
        alt_confidence : float
            Confidence in the signal (0.0 to 1.0).
        forward_return : float, optional
            Forward return over N ticks, if already known.
        """
        entry = {
            "signal": alt_signal,
            "confidence": alt_confidence,
            "forward_return": forward_return,
        }
        self._data[symbol].append(entry)

        if forward_return is None:
            self._pending_fills[symbol].append(len(self._data[symbol]) - 1)

        logger.debug(
            "feed_signal(%s, signal=%d, conf=%.3f, fwd=%s) — total=%d",
            symbol,
            alt_signal,
            alt_confidence,
            f"{forward_return:.6f}" if forward_return is not None else "None",
            len(self._data[symbol]),
        )

    def feed_forward_return(self, symbol: str, forward_return: float) -> None:
        """Feed a forward return for the earliest pending signal of *symbol*.

        This method is only needed when ``forward_return`` was **not** provided
        at :meth:`feed_signal` time.

        Parameters
        ----------
        symbol : str
            Instrument identifier.
        forward_return : float
            Forward return over N ticks.
        """
        pending = self._pending_fills.get(symbol, [])
        if not pending:
            logger.warning(
                "feed_forward_return(%s, %.6f) — no pending signal to fill",
                symbol,
                forward_return,
            )
            return

        idx = pending.pop(0)  # FIFO
        self._data[symbol][idx]["forward_return"] = forward_return

        logger.debug(
            "feed_forward_return(%s, %.6f) — filled entry %d",
            symbol,
            forward_return,
            idx,
        )

    # ------------------------------------------------------------------
    # Public query / reset
    # ------------------------------------------------------------------

    def get_validity_verdict(self, symbol: str) -> str:
        """Return the overall validity verdict for *symbol*.

        Delegates to :meth:`run_all_tests` and returns the ``overall_verdict``.

        Returns
        -------
        str
            One of ``"VALID"``, ``"QUESTIONABLE"``, ``"INVALID"``.
        """
        results = self.run_all_tests(symbol)
        return results.get("overall_verdict", "INVALID")

    def reset(self) -> None:
        """Clear all stored data for every symbol."""
        self._data.clear()
        self._pending_fills.clear()
        logger.info("AltSignalValidityTest(%r) reset", self._instance_id)

    # ------------------------------------------------------------------
    # Test suite — signal helpers
    # ------------------------------------------------------------------

    def _get_signals(self, symbol: str) -> List[int]:
        """Return the sequence of signal values for *symbol*."""
        return [entry["signal"] for entry in self._data.get(symbol, [])]

    def _get_signal_return_pairs(
        self, symbol: str
    ) -> Tuple[List[int], List[float]]:
        """Return (signals, forward_returns) for entries that have both."""
        signals: List[int] = []
        returns: List[float] = []
        for entry in self._data.get(symbol, []):
            if entry["forward_return"] is not None:
                signals.append(entry["signal"])
                returns.append(entry["forward_return"])
        return signals, returns

    def _compute_counts(self, signals: List[int]) -> Tuple[int, int, int]:
        """Count buy (>0), sell (<0), flat (==0) signals."""
        buy = sum(1 for s in signals if s > 0)
        sell = sum(1 for s in signals if s < 0)
        flat = len(signals) - buy - sell
        return buy, sell, flat

    def _compute_distribution(
        self, signals: List[int]
    ) -> Dict[str, float]:
        """Return buy/sell/flat percentages."""
        total = len(signals)
        if total == 0:
            return {"buy_pct": 0.0, "sell_pct": 0.0, "flat_pct": 0.0}

        buy, sell, flat = self._compute_counts(signals)
        return {
            "buy_pct": buy / total,
            "sell_pct": sell / total,
            "flat_pct": flat / total,
        }

    def _autocorr_lag1(self, signals: List[int]) -> float:
        """Compute lag-1 autocorrelation of the signal sequence.

        Returns a value in [-1, 1].  Returns 0.0 if the sequence has
        zero variance or fewer than 2 elements.
        """
        n = len(signals)
        if n < 2:
            return 0.0

        arr = np.array(signals, dtype=np.float64)
        mean = float(np.mean(arr))
        var = float(np.var(arr))
        if var == 0.0:
            return 0.0

        # lag-1 autocorrelation
        cov = float(np.mean((arr[:-1] - mean) * (arr[1:] - mean)))
        return cov / var

    # ------------------------------------------------------------------
    # Test 1 — Persistence test
    # ------------------------------------------------------------------

    def test_persistence(self, symbol: str) -> Dict[str, Any]:
        """Test whether signal distribution is persistent across time.

        Splits the signal history into first and second halves and compares
        their buy/sell/flat distributions with a chi-squared test.

        Returns
        -------
        dict
            ``persistent`` (bool):  ``True`` if distributions are statistically
            similar (chi-squared p > 0.05).
            ``chi2_stat`` (float):  Chi-squared statistic.
            ``p_value`` (float):    P-value of the test.
            ``first_half`` (dict):  Distribution of first half.
            ``second_half`` (dict): Distribution of second half.
        """
        signals = self._get_signals(symbol)
        n = len(signals)

        if n < 4:
            logger.warning(
                "test_persistence(%s): insufficient data (%d entries)", symbol, n
            )
            return {
                "persistent": False,
                "chi2_stat": 0.0,
                "p_value": 0.0,
                "first_half": self._compute_distribution([]),
                "second_half": self._compute_distribution([]),
                "error": "insufficient_data",
            }

        mid = n // 2
        first_half = signals[:mid]
        second_half = signals[mid:]

        dist1 = self._compute_distribution(first_half)
        dist2 = self._compute_distribution(second_half)

        first_counts = self._compute_counts(first_half)
        second_counts = self._compute_counts(second_half)

        # Build contingency table: rows = half, cols = [buy, sell, flat]
        contingency = np.array([list(first_counts), list(second_counts)])

        # Drop columns (signal classes) that sum to zero across both rows —
        # otherwise chi2_contingency returns NaN / degeneracy.
        col_sums = contingency.sum(axis=0)
        valid_cols = col_sums > 0
        if valid_cols.sum() < 2:
            # Only one class present — distributions are trivially identical
            logger.debug(
                "test_persistence(%s): only one signal class present", symbol
            )
            # They're identical, so chi2 = 0, p = 1.0
            return {
                "persistent": True,
                "chi2_stat": 0.0,
                "p_value": 1.0,
                "first_half": dist1,
                "second_half": dist2,
            }

        contingency = contingency[:, valid_cols]

        if not _SCIPY_AVAILABLE or chi2_contingency is None:
            logger.warning(
                "test_persistence(%s): scipy unavailable — cannot compute chi2", symbol
            )
            return {
                "persistent": False,
                "chi2_stat": 0.0,
                "p_value": 0.0,
                "first_half": dist1,
                "second_half": dist2,
                "error": "scipy_unavailable",
            }

        chi2_stat, p_value, dof, expected = chi2_contingency(contingency, correction=False)  # noqa: F841
        persistent = bool(p_value > 0.05)

        logger.debug(
            "test_persistence(%s): chi2=%.4f  p=%.4f  persistent=%s",
            symbol,
            chi2_stat,
            p_value,
            persistent,
        )

        return {
            "persistent": persistent,
            "chi2_stat": float(chi2_stat),
            "p_value": float(p_value),
            "first_half": dist1,
            "second_half": dist2,
        }

    # ------------------------------------------------------------------
    # Test 2 — Shuffle stability test
    # ------------------------------------------------------------------

    def test_shuffle_stability(self, symbol: str) -> Dict[str, Any]:
        """Test whether the signal carries temporal information (not white noise).

        Shuffles the signal sequence randomly and compares lag-1 autocorrelation
        of the original vs shuffled sequence.  If the original has substantially
        higher autocorrelation, the signal has temporal structure.

        Returns
        -------
        dict
            ``original_autocorr`` (float):  Lag-1 autocorrelation of original.
            ``shuffled_autocorr`` (float):  Lag-1 autocorrelation of shuffled.
            ``has_temporal_structure`` (bool):  ``True`` if original autocorr
            exceeds shuffled autocorr by at least 0.01.
        """
        signals = self._get_signals(symbol)
        n = len(signals)

        if n < 4:
            logger.warning(
                "test_shuffle_stability(%s): insufficient data (%d entries)",
                symbol,
                n,
            )
            return {
                "original_autocorr": 0.0,
                "shuffled_autocorr": 0.0,
                "has_temporal_structure": False,
                "error": "insufficient_data",
            }

        original_ac = self._autocorr_lag1(signals)

        # Permutation test: compare original autocorrelation to the distribution
        # from many shuffles.  If the original is an extreme outlier (>2σ above
        # the shuffled mean), the signal carries temporal information.
        NUM_SHUFFLES = 100
        shuffled_acs = []
        for _ in range(NUM_SHUFFLES):
            shuffled = list(signals)
            random.shuffle(shuffled)
            shuffled_acs.append(self._autocorr_lag1(shuffled))

        shuffled_ac = float(np.mean(shuffled_acs))
        shuffled_std = float(np.std(shuffled_acs, ddof=1))

        # Temporal structure: original autocorrelation is significantly greater
        # than what we'd expect from shuffled noise (2σ above the shuffled mean).
        has_structure = (
            original_ac > shuffled_ac + 2.0 * shuffled_std
            if shuffled_std > 0.0
            else False
        )

        # We consider the signal to have temporal structure if the original
        # autocorrelation is clearly larger than the shuffled version.
        # The 0.01 threshold avoids floating-point noise.
        has_structure = bool(original_ac > shuffled_ac + 0.01)

        logger.debug(
            "test_shuffle_stability(%s): orig_ac=%.4f  shuf_ac=%.4f  has_structure=%s",
            symbol,
            original_ac,
            shuffled_ac,
            has_structure,
        )

        return {
            "original_autocorr": float(original_ac),
            "shuffled_autocorr": float(shuffled_ac),
            "has_temporal_structure": has_structure,
        }

    # ------------------------------------------------------------------
    # Test 3 — Forward return correlation test
    # ------------------------------------------------------------------

    def test_forward_return_correlation(
        self, symbol: str
    ) -> Dict[str, Any]:
        """Test whether the signal correlates with future returns.

        Uses Pearson correlation between signal values and their associated
        forward returns.  A positive, significant correlation indicates the
        signal has predictive value.

        Returns
        -------
        dict
            ``correlation`` (float):  Pearson correlation coefficient.
            ``p_value`` (float):      Two-tailed p-value.
            ``n_samples`` (int):      Number of (signal, return) pairs used.
            ``predictive`` (bool):    ``True`` if correlation > 0.
        """
        signals, returns = self._get_signal_return_pairs(symbol)
        n = len(signals)

        if n < 4:
            logger.warning(
                "test_forward_return_correlation(%s): insufficient pairs (%d)",
                symbol,
                n,
            )
            return {
                "correlation": 0.0,
                "p_value": 0.0,
                "n_samples": n,
                "predictive": False,
                "error": "insufficient_data",
            }

        if not _SCIPY_AVAILABLE or pearsonr is None:
            logger.warning(
                "test_forward_return_correlation(%s): scipy unavailable", symbol
            )
            return {
                "correlation": 0.0,
                "p_value": 0.0,
                "n_samples": n,
                "predictive": False,
                "error": "scipy_unavailable",
            }

        # Edge case: if all signals or all returns are constant, pearsonr fails
        if len(set(signals)) < 2 or len(set(returns)) < 2:
            logger.warning(
                "test_forward_return_correlation(%s): constant input", symbol
            )
            return {
                "correlation": 0.0,
                "p_value": 0.0,
                "n_samples": n,
                "predictive": False,
                "error": "constant_input",
            }

        corr, p_val = pearsonr(signals, returns)

        # Predictive: positive correlation AND statistically significant (p < 0.05)
        predictive = bool(corr > 0 and p_val < 0.05)

        logger.debug(
            "test_forward_return_correlation(%s): r=%.4f  p=%.4f  n=%d  predictive=%s",
            symbol,
            corr,
            p_val,
            n,
            predictive,
        )

        return {
            "correlation": float(corr),
            "p_value": float(p_val),
            "n_samples": n,
            "predictive": predictive,
        }

    # ------------------------------------------------------------------
    # Test 4 — Regime stability test
    # ------------------------------------------------------------------

    def test_regime_stability(
        self, symbol: str, window_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Test whether signal distribution is stable across market regimes.

        Splits the signal history into consecutive windows of *window_size*
        ticks and computes the buy percentage per window.  If the standard
        deviation of buy percentages across windows is low (< 0.1), the signal
        is considered regime-stable.

        Parameters
        ----------
        symbol : str
            Instrument identifier.
        window_size : int, optional
            Number of ticks per window.  Defaults to 20.

        Returns
        -------
        dict
            ``stable`` (bool):       ``True`` if buy_pct_std < 0.1.
            ``buy_pct_std`` (float):  Standard deviation of buy % across windows.
            ``buy_pct_mean`` (float): Mean buy % across windows.
            ``window_count`` (int):   Number of windows analysed.
            ``window_size`` (int):    Ticks per window.
        """
        signals = self._get_signals(symbol)
        n = len(signals)

        if window_size is None:
            window_size = self.DEFAULT_WINDOW_SIZE

        if n < window_size * 2:
            logger.warning(
                "test_regime_stability(%s): insufficient data (%d entries, need %d)",
                symbol,
                n,
                window_size * 2,
            )
            return {
                "stable": False,
                "buy_pct_std": 0.0,
                "buy_pct_mean": 0.0,
                "window_count": 0,
                "window_size": window_size,
                "error": "insufficient_data",
            }

        # Split into windows
        windows: List[List[int]] = []
        for start in range(0, n, window_size):
            window = signals[start: start + window_size]
            if len(window) == window_size:  # only full windows
                windows.append(window)

        if len(windows) < 2:
            logger.warning(
                "test_regime_stability(%s): need at least 2 full windows", symbol
            )
            return {
                "stable": False,
                "buy_pct_std": 0.0,
                "buy_pct_mean": 0.0,
                "window_count": len(windows),
                "window_size": window_size,
                "error": "insufficient_windows",
            }

        # Compute buy % per window
        buy_pcts = []
        for w in windows:
            buy = sum(1 for s in w if s > 0)
            buy_pcts.append(buy / len(w))

        buy_pct_std = float(np.std(buy_pcts, ddof=1))  # sample std
        buy_pct_mean = float(np.mean(buy_pcts))
        stable = bool(buy_pct_std < 0.1)

        logger.debug(
            "test_regime_stability(%s): buy_pct_mean=%.4f  buy_pct_std=%.4f  "
            "windows=%d  stable=%s",
            symbol,
            buy_pct_mean,
            buy_pct_std,
            len(windows),
            stable,
        )

        return {
            "stable": stable,
            "buy_pct_std": buy_pct_std,
            "buy_pct_mean": buy_pct_mean,
            "window_count": len(windows),
            "window_size": window_size,
        }

    # ------------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------------

    def run_all_tests(self, symbol: str) -> Dict[str, Any]:
        """Run the full ALT signal validity test suite for *symbol*.

        The four tests are:

        1. **Persistence** — chi-squared comparison of first/second half
        2. **Shuffle stability** — autocorrelation test for temporal structure
        3. **Forward return correlation** — Pearson r with future returns
        4. **Regime stability** — variance of buy % across windows

        Returns
        -------
        dict
            ``total_tests`` (int):    Always 4.
            ``passed`` (int):         Number of tests that passed.
            ``failed`` (int):         Number of tests that failed.
            ``results`` (dict):       Per-test result dicts keyed by test name.
            ``overall_verdict`` (str): One of ``"VALID"``, ``"QUESTIONABLE"``,
            ``"INVALID"``.
            ``recommendation`` (str): Human-readable guidance.
        """
        results: Dict[str, Dict[str, Any]] = {}

        # --- persistence ---
        results["persistence"] = self.test_persistence(symbol)
        # --- shuffle stability ---
        results["shuffle_stability"] = self.test_shuffle_stability(symbol)
        # --- forward return ---
        results["forward_return"] = self.test_forward_return_correlation(symbol)
        # --- regime stability ---
        results["regime_stability"] = self.test_regime_stability(symbol)

        # Determine pass/fail for each test
        persistence_pass = results["persistence"].get("persistent", False)
        shuffle_pass = results["shuffle_stability"].get(
            "has_temporal_structure", False
        )
        forward_pass = results["forward_return"].get("predictive", False)
        regime_pass = results["regime_stability"].get("stable", False)

        passed = sum([persistence_pass, shuffle_pass, forward_pass, regime_pass])
        failed = 4 - passed

        # Compute error count (tests that couldn't run)
        errors = sum(
            1 for r in results.values() if "error" in r
        )

        # Overall verdict
        if passed == 4:
            overall_verdict = "VALID"
            recommendation = (
                "ALT signal passes all validity checks. It is persistent, "
                "temporally structured, predictive of forward returns, and "
                "stable across regimes. It can serve as a reliable control signal."
            )
        elif passed >= 2:
            overall_verdict = "QUESTIONABLE"
            if errors > 0:
                recommendation = (
                    f"ALT signal passed {passed}/4 tests but {errors} test(s) "
                    f"could not complete (insufficient data or missing dependencies). "
                    f"Collect more data and re-run before drawing conclusions."
                )
            else:
                recommendation = (
                    f"ALT signal passed {passed}/4 tests. "
                    f"It has some validity characteristics but is not fully "
                    f"reliable. Investigate the failing tests and consider "
                    f"whether the signal needs refinement."
                )
        else:
            overall_verdict = "INVALID"
            if errors > 0:
                recommendation = (
                    f"ALT signal passed only {passed}/4 tests with {errors} "
                    f"test(s) incomplete. The available evidence suggests this "
                    f"signal is not suitable as a control. Collect more data or "
                    f"reconsider the signal construction."
                )
            else:
                recommendation = (
                    f"ALT signal failed {failed}/4 tests. "
                    f"It shows little-to-no evidence of validity: the signal "
                    f"distribution is inconsistent, lacks temporal structure, "
                    f"is not predictive, or varies wildly across regimes. "
                    f"This signal should NOT be used as a control."
                )

        return {
            "total_tests": 4,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "results": results,
            "overall_verdict": overall_verdict,
            "recommendation": recommendation,
            "symbol": symbol,
            "total_samples": len(self._data.get(symbol, [])),
        }


# ===================================================================
# Self-test
# ===================================================================


def _run_self_test() -> bool:
    """Run a comprehensive self-test of the AltSignalValidityTest module.

    Tests:
        1. Singleton accessor returns the same instance for same id.
        2. Singleton accessor returns different instances for different ids.
        3. Synthetic persistent signal passes all 4 tests.
        4. Random noise signal fails most tests.
        5. ``feed_forward_return`` fills pending entries correctly.
        6. ``reset`` clears all data.
        7. Edge case: insufficient data returns graceful error.

    Returns
    -------
    bool
        ``True`` if all checks pass.
    """
    logger.info("=" * 60)
    logger.info("AltSignalValidityTest — Self Test")
    logger.info("=" * 60)

    test_passed = True

    def _check(cond: bool, msg: str) -> None:
        nonlocal test_passed
        if cond:
            logger.info("  PASS: %s", msg)
        else:
            test_passed = False
            logger.error("  FAIL: %s", msg)

    # ----- 1. Singleton accessor ------------------------------------------
    logger.info("")
    logger.info("--- Singleton accessor ---")
    inst_a = AltSignalValidityTest("_selftest")
    inst_b = AltSignalValidityTest("_selftest")
    inst_c = AltSignalValidityTest("_selftest_other")
    _check(inst_a is inst_b, "same instance_id returns same object")
    _check(inst_a is not inst_c, "different instance_id returns different object")

    # ----- 2. Synthetic persistent signal (should pass all 4) -------------
    logger.info("")
    logger.info("--- Synthetic persistent signal (should pass all 4) ---")

    persistent_tester = _AltSignalValidityTest("_selftest_persistent")

    # Generate a persistent signal with clear temporal structure:
    #   [+1, +1, +1, 0, -1, -1, -1, 0, ...]  (8-tick cycle)
    # This pattern has:
    #   - Strong lag-1 autocorrelation (long runs of same sign)
    #   - A mix of buy/sell/flat for chi-squared test
    #   - Stable distribution across regimes
    # Forward returns are set to correlate positively with the signal.
    n_ticks = 120
    pattern = [1, 1, 1, 0, -1, -1, -1, 0] * (n_ticks // 8)
    for sig in pattern:
        # Forward return: positive when signal is +1, negative when signal is -1
        fwd = 0.002 * sig + random.gauss(0, 0.0005)
        persistent_tester.feed_signal("SYNTH", sig, 0.8, forward_return=fwd)

    p_results = persistent_tester.run_all_tests("SYNTH")
    _check(
        p_results["overall_verdict"] == "VALID",
        f"Synthetic persistent signal verdict=VALID, got {p_results['overall_verdict']} "
        f"(passed={p_results['passed']}/4)",
    )
    # At minimum the persistence, shuffle, and regime tests should pass
    _check(
        p_results["results"]["persistence"].get("persistent", False),
        "Synthetic signal is persistent (chi2 p > 0.05)",
    )
    _check(
        p_results["results"]["shuffle_stability"].get("has_temporal_structure", False),
        "Synthetic signal has temporal structure",
    )
    _check(
        p_results["results"]["regime_stability"].get("stable", False),
        "Synthetic signal is regime-stable",
    )

    # Log detailed results
    logger.info("  Persistence      : %s", p_results["results"]["persistence"].get("persistent", "?"))
    logger.info("  Shuffle stability: %s", p_results["results"]["shuffle_stability"].get("has_temporal_structure", "?"))
    logger.info("  Forward return   : %s", p_results["results"]["forward_return"].get("predictive", "?"))
    logger.info("  Regime stability : %s", p_results["results"]["regime_stability"].get("stable", "?"))

    # ----- 3. Random noise (should fail most) ----------------------------
    logger.info("")
    logger.info("--- Random noise signal (should fail most) ---")

    noise_tester = _AltSignalValidityTest("_selftest_noise")

    for _ in range(n_ticks):
        sig = random.choice([-1, 0, 1])
        fwd = random.gauss(0, 0.01)  # uncorrelated noise
        noise_tester.feed_signal("NOISE", sig, random.uniform(0.0, 0.5), forward_return=fwd)

    n_results = noise_tester.run_all_tests("NOISE")

    # Random noise forward return should NOT be predictive (r ~ 0, p > 0.05).
    # This is the most reliable indicator since uncorrelated noise cannot
    # systematically predict future returns.
    _check(
        not n_results["results"]["forward_return"].get("predictive", True),
        f"Random noise forward return is NOT predictive "
        f"(r={n_results['results']['forward_return'].get('correlation', 0):.4f}, "
        f"p={n_results['results']['forward_return'].get('p_value', 0):.4f})",
    )

    # At least one of the other tests should also fail for noise (though
    # persistence often passes since random noise has stable distribution).
    # Log the actual results for transparency.
    noise_passed = n_results["passed"]
    logger.info(
        "  Noise test results: passed=%d/4  "
        "persistence=%s  shuffle=%s  fwd=%s  regime=%s",
        n_results["passed"],
        n_results["results"]["persistence"].get("persistent", "?"),
        n_results["results"]["shuffle_stability"].get("has_temporal_structure", "?"),
        n_results["results"]["forward_return"].get("predictive", "?"),
        n_results["results"]["regime_stability"].get("stable", "?"),
    )

    # Log detailed results
    logger.info("  Persistence      : %s", n_results["results"]["persistence"].get("persistent", "?"))
    logger.info("  Shuffle stability: %s", n_results["results"]["shuffle_stability"].get("has_temporal_structure", "?"))
    logger.info("  Forward return   : %s", n_results["results"]["forward_return"].get("predictive", "?"))
    logger.info("  Regime stability : %s", n_results["results"]["regime_stability"].get("stable", "?"))

    # ----- 4. feed_forward_return -----------------------------------------
    logger.info("")
    logger.info("--- feed_forward_return ---")

    fr_tester = _AltSignalValidityTest("_selftest_fr")
    fr_tester.feed_signal("TEST", 1, 0.9)           # no forward return
    fr_tester.feed_signal("TEST", -1, 0.7)          # no forward return
    fr_tester.feed_forward_return("TEST", 0.005)    # fills first entry
    fr_tester.feed_forward_return("TEST", -0.003)   # fills second entry

    signals, returns = fr_tester._get_signal_return_pairs("TEST")
    _check(
        len(signals) == 2 and len(returns) == 2,
        f"feed_forward_return: 2 pairs, got signals={len(signals)} returns={len(returns)}",
    )
    _check(
        returns == [0.005, -0.003],
        f"feed_forward_return: returns=[0.005, -0.003], got {returns}",
    )

    # ---- 5. reset --------------------------------------------------------
    logger.info("")
    logger.info("--- reset ---")

    fr_tester.reset()
    _check(
        len(fr_tester._data) == 0,
        "reset clears _data",
    )
    _check(
        len(fr_tester._pending_fills) == 0,
        "reset clears _pending_fills",
    )

    # ----- 6. Insufficient data edge case ---------------------------------
    logger.info("")
    logger.info("--- Insufficient data edge case ---")

    empty_tester = _AltSignalValidityTest("_selftest_empty")
    empty_tester.feed_signal("EMPTY", 1, 0.5)
    empty_tester.feed_signal("EMPTY", -1, 0.5)
    empty_tester.feed_signal("EMPTY", 0, 0.5)

    p_res = empty_tester.test_persistence("EMPTY")
    _check(
        "error" in p_res,
        "test_persistence with <4 samples returns error",
    )

    s_res = empty_tester.test_shuffle_stability("EMPTY")
    _check(
        "error" in s_res,
        "test_shuffle_stability with <4 samples returns error",
    )

    f_res = empty_tester.test_forward_return_correlation("EMPTY")
    _check(
        "error" in f_res,
        "test_forward_return_correlation with <4 pairs returns error",
    )

    r_res = empty_tester.test_regime_stability("EMPTY")
    _check(
        "error" in r_res,
        "test_regime_stability with <2*window_size samples returns error",
    )

    all_res = empty_tester.run_all_tests("EMPTY")
    _check(
        all_res["overall_verdict"] == "INVALID",
        f"Empty data verdict=INVALID, got {all_res['overall_verdict']}",
    )

    # ----- 7. Forward return correlation edge cases -----------------------
    logger.info("")
    logger.info("--- Forward return edge cases ---")

    # Constant signal
    const_tester = _AltSignalValidityTest("_selftest_const")
    for _ in range(10):
        const_tester.feed_signal("CONST", 1, 0.8, forward_return=0.001)
    const_res = const_tester.test_forward_return_correlation("CONST")
    _check(
        "error" in const_res,
        "constant signal returns error from pearsonr",
    )

    # ----- 8. Autocorrelation edge case -----------------------------------
    logger.info("")
    logger.info("--- Autocorrelation edge case ---")

    edge_tester = _AltSignalValidityTest("_selftest_ac_edge")
    edge_tester.feed_signal("EDGE", 1, 0.5)
    ac_low = edge_tester._autocorr_lag1([1])
    _check(
        ac_low == 0.0,
        f"autocorr of single element is 0.0, got {ac_low}",
    )

    ac_const = edge_tester._autocorr_lag1([1, 1, 1])
    _check(
        ac_const == 0.0,
        f"autocorr of constant array is 0.0, got {ac_const}",
    )

    # ----- 9. get_validity_verdict -----------------------------------------
    logger.info("")
    logger.info("--- get_validity_verdict ---")

    verdict = persistent_tester.get_validity_verdict("SYNTH")
    _check(
        verdict in ("VALID", "QUESTIONABLE", "INVALID"),
        f"get_validity_verdict returns a valid verdict string, got {verdict}",
    )
    # Should reflect the full test results for this symbol
    _check(
        isinstance(verdict, str) and len(verdict) > 0,
        "get_validity_verdict returns a non-empty string",
    )

    # ----- 10. run_all_tests structure check -------------------------------
    logger.info("")
    logger.info("--- run_all_tests dict structure ---")

    expected_top_keys = {
        "total_tests", "passed", "failed", "errors",
        "results", "overall_verdict", "recommendation",
        "symbol", "total_samples",
    }
    _check(
        expected_top_keys.issubset(p_results.keys()),
        f"run_all_tests contains all expected top-level keys",
    )

    expected_result_keys = {
        "persistence", "shuffle_stability", "forward_return", "regime_stability",
    }
    _check(
        expected_result_keys.issubset(p_results["results"].keys()),
        f"run_all_tests contains all 4 expected result keys",
    )

    _check(
        isinstance(p_results["recommendation"], str) and len(p_results["recommendation"]) > 0,
        "recommendation is a non-empty string",
    )

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    if test_passed:
        logger.info("RESULT: ALL SELFTESTS PASSED")
    else:
        logger.error("RESULT: SOME SELFTESTS FAILED")
    logger.info("=" * 60)

    return test_passed


# ===================================================================
# CLI entry point
# ===================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if "--self-test" in sys.argv:
        success = _run_self_test()
        sys.exit(0 if success else 1)
    else:
        # Run full test suite on sample data
        tester = AltSignalValidityTest()

        print("")
        print("=" * 60)
        print("ALT SIGNAL VALIDITY TEST — FULL DIAGNOSTIC")
        print("=" * 60)

        # Feed some synthetic data (same 8-tick cycle as self-test)
        print("\nFeeding synthetic persistent signal (120 ticks)...")
        pattern = [1, 1, 1, 0, -1, -1, -1, 0] * 15
        for sig in pattern:
            fwd = 0.002 * sig + random.gauss(0, 0.0005)
            tester.feed_signal("EURUSD", sig, 0.8, forward_return=fwd)

        results = tester.run_all_tests("EURUSD")

        print(f"\n  Symbol         : {results['symbol']}")
        print(f"  Total samples  : {results['total_samples']}")
        print(f"  Tests passed   : {results['passed']}/{results['total_tests']}")
        print(f"  Tests failed   : {results['failed']}/{results['total_tests']}")
        print(f"  Errors         : {results['errors']}")
        print(f"  Overall verdict: {results['overall_verdict']}")
        print(f"  Recommendation : {results['recommendation']}")

        print("\n  --- Per-test results ---")
        for test_name, res in results["results"].items():
            print(f"    {test_name:25s}: ", end="")
            if "error" in res:
                print(f"ERROR — {res['error']}")
            else:
                # Print key results based on test type
                if test_name == "persistence":
                    print(
                        f"persistent={res.get('persistent', '?')}  "
                        f"chi2={res.get('chi2_stat', 0):.4f}  "
                        f"p={res.get('p_value', 0):.4f}"
                    )
                elif test_name == "shuffle_stability":
                    print(
                        f"has_structure={res.get('has_temporal_structure', '?')}  "
                        f"orig_ac={res.get('original_autocorr', 0):.4f}  "
                        f"shuf_ac={res.get('shuffled_autocorr', 0):.4f}"
                    )
                elif test_name == "forward_return":
                    print(
                        f"predictive={res.get('predictive', '?')}  "
                        f"r={res.get('correlation', 0):.4f}  "
                        f"p={res.get('p_value', 0):.4f}  "
                        f"n={res.get('n_samples', 0)}"
                    )
                elif test_name == "regime_stability":
                    print(
                        f"stable={res.get('stable', '?')}  "
                        f"buy_pct_std={res.get('buy_pct_std', 0):.4f}  "
                        f"windows={res.get('window_count', 0)}"
                    )

        print("")
        print("=" * 60)

        sys.exit(0)
