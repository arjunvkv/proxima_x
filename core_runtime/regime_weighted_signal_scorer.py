"""
RegimeWeightedSignalScorer — scores OSS and ALT signals differently per market
regime.  Four regimes are recognised:

    * TRENDING         — strong directional movement
    * MEAN_REVERTING   — oscillating within a range
    * VOLATILITY_SPIKE — sudden expansion of volatility
    * LOW_LIQUIDITY    — wide spreads, low volume

OSS and ALT get different base scores per regime.  Scores can be overridden
at runtime.
"""

import logging
import statistics
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton registry
# ---------------------------------------------------------------------------

_instances = {}


def RegimeWeightedSignalScorer(instance_id="default"):
    """Singleton accessor — returns the same ``_RegimeWeightedSignalScorer``
    for a given *instance_id*.

    Parameters
    ----------
    instance_id : str
        Unique identifier.  Callers sharing the same id share state.

    Returns
    -------
    _RegimeWeightedSignalScorer
    """
    if instance_id not in _instances:
        _instances[instance_id] = _RegimeWeightedSignalScorer(instance_id)
    return _instances[instance_id]


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

#: Slope magnitude above which we consider the market **trending**.
TREND_SLOPE_THRESHOLD = 0.0002

#: Multiplier of median spread above which we flag low liquidity.
LOW_LIQ_SPREAD_MULTIPLIER = 2.0

#: Multiplier of rolling mean ATR above which we flag a volatility spike.
VOLATILITY_ATR_MULTIPLIER = 2.0

#: Default look-back window (number of observations).
ROLLING_WINDOW = 20

#: Base scores per regime — OSS and ALT get different weights.
REGIME_SCORES = {
    "TRENDING": {"oss": 0.3, "alt": 0.8},         # ALT (momentum) better in trends
    "MEAN_REVERTING": {"oss": 0.7, "alt": 0.4},   # OSS better in mean-reversion
    "VOLATILITY_SPIKE": {"oss": 0.2, "alt": 0.5}, # Both weak in volatility
    "LOW_LIQUIDITY": {"oss": 0.1, "alt": 0.3},    # Both weak in low liq
}

#: Detection priority (highest first).
REGIME_PRIORITY = ["LOW_LIQUIDITY", "VOLATILITY_SPIKE", "TRENDING", "MEAN_REVERTING"]

# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------


class _RegimeWeightedSignalScorer:
    """Scores OSS and ALT signals based on the detected market regime.

    Parameters
    ----------
    instance_id : str
        Label used in logging.
    """

    def __init__(self, instance_id="default"):
        self._instance_id = instance_id

        # Mutable copy of default scores (allows override_regime_scores).
        self._regime_scores = {
            regime: dict(scores) for regime, scores in REGIME_SCORES.items()
        }

        # Per-symbol price data:  symbol -> deque of (bid, spread, volume, timestamp)
        self._price_data = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW * 3))

        logger.debug("RegimeWeightedSignalScorer(%r) initialised", instance_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed_price_data(self, symbol, bid_price, spread, volume=None, timestamp=None):
        """Store a price observation that will be used for later regime detection.

        Parameters
        ----------
        symbol : str
            Instrument identifier (e.g. ``"EURUSD"``).
        bid_price : float
            Current bid price.
        spread : float
            Current spread (ask - bid) in price units.
        volume : float or None
            Optional volume / tick value.
        timestamp : optional
            Observation timestamp (ignored, present for interface compatibility).
        """
        self._price_data[symbol].append((bid_price, spread, volume, timestamp))
        logger.debug(
            "feed_price_data %s price=%.6f spread=%.6f",
            symbol, bid_price, spread,
        )

    def detect_regime(self, bid_prices, spreads, volumes=None):
        """Detect the current market regime from price/spread arrays.

        Parameters
        ----------
        bid_prices : list[float]
            Sequence of bid prices (most recent last).
        spreads : list[float]
            Sequence of spreads corresponding to *bid_prices*.
        volumes : list[float] or None
            Optional volume data (currently unused, present for interface
            compatibility).

        Returns
        -------
        str
            One of ``"TRENDING"``, ``"MEAN_REVERTING"``,
            ``"VOLATILITY_SPIKE"``, ``"LOW_LIQUIDITY"``.

        Notes
        -----
        Priority order (highest first):
        ``LOW_LIQUIDITY > VOLATILITY_SPIKE > TRENDING > MEAN_REVERTING``.
        """
        n = len(bid_prices)
        if n < 3:
            return "MEAN_REVERTING"

        # --- 1. LOW_LIQUIDITY -----------------------------------------------
        current_spread = spreads[-1]
        median_spread = statistics.median(spreads)
        if median_spread > 0 and current_spread > LOW_LIQ_SPREAD_MULTIPLIER * median_spread:
            return "LOW_LIQUIDITY"

        # --- 2. VOLATILITY_SPIKE --------------------------------------------
        price_changes = [abs(bid_prices[i] - bid_prices[i - 1]) for i in range(1, n)]
        if len(price_changes) >= 3:
            current_atr = price_changes[-1]
            # Rolling baseline: mean of changes excluding the current one, over
            # at most ROLLING_WINDOW observations.
            baseline = price_changes[-min(len(price_changes) - 1, ROLLING_WINDOW):-1]
            mean_atr = statistics.mean(baseline) if baseline else 0.0
            if mean_atr > 0 and current_atr > VOLATILITY_ATR_MULTIPLIER * mean_atr:
                return "VOLATILITY_SPIKE"

        # --- 3. TRENDING vs MEAN_REVERTING -----------------------------------
        slope = self._compute_slope(bid_prices)
        if abs(slope) > TREND_SLOPE_THRESHOLD:
            return "TRENDING"

        return "MEAN_REVERTING"

    def get_scores(self, symbol):
        """Return OSS and ALT scores for *symbol* based on the detected regime.

        Uses internally stored price data (previously fed via
        :meth:`feed_price_data`).

        Parameters
        ----------
        symbol : str

        Returns
        -------
        dict
            Keys:

            ``symbol``
                Instrument identifier.
            ``regime``
                Detected regime string.
            ``oss_score``
                Base OSS score for the regime.
            ``alt_score``
                Base ALT score for the regime.
            ``regime_confidence``
                How confident we are of the regime detection (0.0 – 1.0).
            ``adjusted_oss_score``
                ``oss_score * regime_confidence``.
            ``adjusted_alt_score``
                ``alt_score * regime_confidence``.
        """
        data = list(self._price_data.get(symbol, []))

        if not data:
            # No data yet — return default regime scores with zero confidence.
            regime = "MEAN_REVERTING"
            return {
                "symbol": symbol,
                "regime": regime,
                "oss_score": self._regime_scores[regime]["oss"],
                "alt_score": self._regime_scores[regime]["alt"],
                "regime_confidence": 0.0,
                "adjusted_oss_score": 0.0,
                "adjusted_alt_score": 0.0,
            }

        bid_prices = [d[0] for d in data]
        spreads = [d[1] for d in data]
        volumes = [d[2] for d in data]

        regime = self.detect_regime(bid_prices, spreads, volumes)
        confidence = self._compute_confidence(regime, bid_prices, spreads)

        base_oss = self._regime_scores[regime]["oss"]
        base_alt = self._regime_scores[regime]["alt"]

        return {
            "symbol": symbol,
            "regime": regime,
            "oss_score": base_oss,
            "alt_score": base_alt,
            "regime_confidence": confidence,
            "adjusted_oss_score": round(base_oss * confidence, 4),
            "adjusted_alt_score": round(base_alt * confidence, 4),
        }

    def override_regime_scores(self, regime, oss_score, alt_score):
        """Manually set OSS and ALT scores for a given regime.

        Parameters
        ----------
        regime : str
            One of ``"TRENDING"``, ``"MEAN_REVERTING"``,
            ``"VOLATILITY_SPIKE"``, ``"LOW_LIQUIDITY"``.
        oss_score : float
            New OSS base score.
        alt_score : float
            New ALT base score.
        """
        if regime not in self._regime_scores:
            logger.warning("override_regime_scores: unknown regime '%s'", regime)
            return
        self._regime_scores[regime] = {"oss": oss_score, "alt": alt_score}
        logger.info(
            "override_regime_scores: %s -> oss=%.2f alt=%.2f",
            regime, oss_score, alt_score,
        )

    def reset(self):
        """Clear all stored price data.  Regime scores are *not* reset."""
        self._price_data.clear()
        logger.debug("RegimeWeightedSignalScorer(%r) reset", self._instance_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_slope(prices):
        """Compute the linear regression slope over a price sequence.

        Returns the slope (price change per step).  A positive value
        indicates an upward drift, negative a downward drift.
        """
        n = len(prices)
        if n < 2:
            return 0.0

        xs = list(range(n))
        mean_x = (n - 1) / 2.0
        mean_y = sum(prices) / n

        num = 0.0
        den = 0.0
        for x, y in zip(xs, prices):
            dx = x - mean_x
            num += dx * (y - mean_y)
            den += dx * dx

        if den == 0.0:
            return 0.0
        return num / den

    def _compute_confidence(self, regime, bid_prices, spreads):
        """Return a confidence score in [0.0, 1.0] for the regime detection."""
        n = len(bid_prices)
        if n < 3:
            return 0.0

        # Base confidence scales with data quantity.
        base = min(1.0, n / ROLLING_WINDOW) * 0.8

        if regime == "TRENDING":
            slope = self._compute_slope(bid_prices)
            # Stronger slope -> higher confidence
            boost = min(0.2, abs(slope) / (TREND_SLOPE_THRESHOLD * 10))
            return round(min(1.0, base + boost), 4)

        if regime == "MEAN_REVERTING":
            # Very flat slope increases confidence
            slope = self._compute_slope(bid_prices)
            boost = max(0.0, 0.2 - abs(slope) / TREND_SLOPE_THRESHOLD)
            return round(min(1.0, base + boost), 4)

        if regime == "LOW_LIQUIDITY":
            # How extreme is the spread?
            current_spread = spreads[-1]
            window = min(len(spreads), ROLLING_WINDOW)
            median_spread = statistics.median(spreads[-window:])
            if median_spread > 0:
                ratio = current_spread / median_spread
                boost = min(0.2, ratio / 10.0)
                return round(min(1.0, base + boost), 4)
            return round(base, 4)

        if regime == "VOLATILITY_SPIKE":
            # How extreme is the latest price change vs its rolling baseline?
            all_changes = [abs(bid_prices[i] - bid_prices[i - 1]) for i in range(1, n)]
            if len(all_changes) >= 3:
                current_change = all_changes[-1]
                baseline = all_changes[-min(len(all_changes) - 1, ROLLING_WINDOW):-1]
                mean_change = statistics.mean(baseline) if baseline else 0.0
                if mean_change > 0:
                    ratio = current_change / mean_change
                    boost = min(0.2, ratio / 10.0)
                    return round(min(1.0, base + boost), 4)
            return round(base, 4)

        return round(base, 4)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest():
    """Feed synthetic data for each regime type and verify detection."""
    import random

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("=" * 60)
    logger.info("RegimeWeightedSignalScorer — Self Test")
    logger.info("=" * 60)

    random.seed(42)
    scorer = RegimeWeightedSignalScorer("selftest")

    # ------------------------------------------------------------------
    # 1. TRENDING — strong upward drift
    # ------------------------------------------------------------------
    symbol_tr = "SYNTH_TREND"
    price = 1.10000
    for i in range(100):
        price += 0.0003 + random.uniform(-0.0002, 0.0002)  # upward drift
        spread = 0.0002 + random.uniform(0, 0.0001)
        scorer.feed_price_data(symbol_tr, price, spread)

    scores_tr = scorer.get_scores(symbol_tr)
    logger.info("TRENDING test  -> regime=%s  oss=%.3f  alt=%.3f  conf=%.3f",
                scores_tr["regime"], scores_tr["oss_score"],
                scores_tr["alt_score"], scores_tr["regime_confidence"])
    assert scores_tr["regime"] == "TRENDING", (
        f"Expected TRENDING, got {scores_tr['regime']}"
    )

    # ------------------------------------------------------------------
    # 2. MEAN_REVERTING — oscillation within a tight range
    # ------------------------------------------------------------------
    scorer.reset()
    symbol_mr = "SYNTH_MEANREV"
    base = 1.10000
    for i in range(100):
        price = base + random.uniform(-0.0005, 0.0005)
        spread = 0.0002 + random.uniform(0, 0.0001)
        scorer.feed_price_data(symbol_mr, price, spread)

    scores_mr = scorer.get_scores(symbol_mr)
    logger.info("MEAN_REVERT test -> regime=%s  oss=%.3f  alt=%.3f  conf=%.3f",
                scores_mr["regime"], scores_mr["oss_score"],
                scores_mr["alt_score"], scores_mr["regime_confidence"])
    assert scores_mr["regime"] == "MEAN_REVERTING", (
        f"Expected MEAN_REVERTING, got {scores_mr['regime']}"
    )

    # ------------------------------------------------------------------
    # 3. VOLATILITY_SPIKE — sudden large price move as the last tick
    # ------------------------------------------------------------------
    scorer.reset()
    symbol_vs = "SYNTH_VOLSPIKE"
    price = 1.10000
    for i in range(50):
        price += random.uniform(-0.0001, 0.0001)  # calm
        spread = 0.0002 + random.uniform(0, 0.0001)
        scorer.feed_price_data(symbol_vs, price, spread)
    # End with a sudden large spike as the very last observation.
    price += 0.005  # large jump
    scorer.feed_price_data(symbol_vs, price, spread)

    scores_vs = scorer.get_scores(symbol_vs)
    logger.info("VOLATILITY SPIKE test -> regime=%s  oss=%.3f  alt=%.3f  conf=%.3f",
                scores_vs["regime"], scores_vs["oss_score"],
                scores_vs["alt_score"], scores_vs["regime_confidence"])
    assert scores_vs["regime"] == "VOLATILITY_SPIKE", (
        f"Expected VOLATILITY_SPIKE, got {scores_vs['regime']}"
    )

    # ------------------------------------------------------------------
    # 4. LOW_LIQUIDITY — wide spread
    # ------------------------------------------------------------------
    scorer.reset()
    symbol_ll = "SYNTH_LOWLIQ"
    price = 1.10000
    for i in range(50):
        price += random.uniform(-0.0002, 0.0002)
        spread = 0.0002 + random.uniform(0, 0.0001)
        scorer.feed_price_data(symbol_ll, price, spread)
    # suddenly wide spread
    spread = 0.0020  # 10x normal
    scorer.feed_price_data(symbol_ll, price, spread)
    for i in range(5):
        price += random.uniform(-0.0002, 0.0002)
        scorer.feed_price_data(symbol_ll, price, spread)

    scores_ll = scorer.get_scores(symbol_ll)
    logger.info("LOW LIQUIDITY test -> regime=%s  oss=%.3f  alt=%.3f  conf=%.3f",
                scores_ll["regime"], scores_ll["oss_score"],
                scores_ll["alt_score"], scores_ll["regime_confidence"])
    assert scores_ll["regime"] == "LOW_LIQUIDITY", (
        f"Expected LOW_LIQUIDITY, got {scores_ll['regime']}"
    )

    # ------------------------------------------------------------------
    # 5. Override regime scores
    # ------------------------------------------------------------------
    logger.info("--- override_regime_scores ---")
    # Feed fresh trending data so the TRENDING symbol has active data.
    scorer.reset()
    price = 1.20000
    for i in range(100):
        price += 0.0003 + random.uniform(-0.0002, 0.0002)
        spread = 0.0002 + random.uniform(0, 0.0001)
        scorer.feed_price_data(symbol_tr, price, spread)

    scorer.override_regime_scores("TRENDING", oss_score=0.5, alt_score=0.9)
    scores_tr2 = scorer.get_scores(symbol_tr)
    logger.info("After override: regime=%s oss=%.2f alt=%.2f",
                scores_tr2["regime"], scores_tr2["oss_score"], scores_tr2["alt_score"])
    assert scores_tr2["regime"] == "TRENDING", (
        f"Expected TRENDING regime, got {scores_tr2['regime']}"
    )
    assert scores_tr2["oss_score"] == 0.5, (
        f"Expected oss=0.5 after override, got {scores_tr2['oss_score']}"
    )
    assert scores_tr2["alt_score"] == 0.9, (
        f"Expected alt=0.9 after override, got {scores_tr2['alt_score']}"
    )

    # ------------------------------------------------------------------
    # 6. No-data edge case
    # ------------------------------------------------------------------
    scores_empty = scorer.get_scores("NONEXISTENT")
    logger.info("No-data case -> regime=%s conf=%.3f oss=%.3f alt=%.3f",
                scores_empty["regime"], scores_empty["regime_confidence"],
                scores_empty["oss_score"], scores_empty["alt_score"])
    assert scores_empty["regime"] == "MEAN_REVERTING"
    assert scores_empty["regime_confidence"] == 0.0

    # ------------------------------------------------------------------
    # Verify singleton
    # ------------------------------------------------------------------
    same = RegimeWeightedSignalScorer("selftest")
    assert same is scorer, "Singleton pattern broken!"

    logger.info("=" * 60)
    logger.info(">>> ALL SELFTESTS PASSED <<<")
    logger.info("=" * 60)


if __name__ == "__main__":
    _selftest()
