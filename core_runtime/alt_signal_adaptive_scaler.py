"""AltSignalAdaptiveScaler — adaptive signal generation across volatility regimes.

Replaces fixed EMA crossover with z-score breakout, volatility-adjusted
momentum, and adaptive thresholding built on ATR.
"""

import logging
import math
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton registry
# ---------------------------------------------------------------------------

_instances = {}


def AltSignalAdaptiveScaler(instance_id="default"):
    """Singleton accessor — returns the same ``_AltSignalAdaptiveScaler``
    for a given *instance_id*.

    Parameters
    ----------
    instance_id : str
        Unique identifier.  Callers sharing the same id share state.

    Returns
    -------
    _AltSignalAdaptiveScaler
    """
    if instance_id not in _instances:
        _instances[instance_id] = _AltSignalAdaptiveScaler(instance_id)
    return _instances[instance_id]


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------


class _AltSignalAdaptiveScaler:
    """Adaptive signal generator that works across volatility regimes.

    Uses three complementary signal sources in priority order:
      1. ZSCORE_BREAKOUT — z-score of log returns exceeds 1.5
      2. MOMENTUM       — cumulative log return exceeds adaptive threshold
      3. ADAPTIVE_EMA   — fast/slow EMA separation exceeds adaptive threshold

    The adaptive threshold is based on ATR, making the generator sensitive
    to the current volatility regime.

    Parameters
    ----------
    instance_id : str
        Label used in logging.
    """

    def __init__(self, instance_id="default"):
        self._instance_id = instance_id

        # Configuration
        self._max_buffer = 200
        self._window = 20             # rolling mean/std window for z-score
        self._atr_window = 14         # ATR look-back
        self._momentum_window = 10    # log-return periods
        self._min_threshold = 1e-5

        # Fast / slow EMA periods for adaptive EMA signal
        self._ema_fast_period = 5
        self._ema_slow_period = 20

        # Per-symbol state
        self._prices = defaultdict(lambda: deque(maxlen=self._max_buffer))
        self._returns = defaultdict(lambda: deque(maxlen=self._window))
        self._changes = defaultdict(lambda: deque(maxlen=self._atr_window))
        self._tick_count = defaultdict(int)
        self._ema_fast = {}
        self._ema_slow = {}

        logger.debug("AltSignalAdaptiveScaler(%r) initialised", instance_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed_tick(self, symbol, bid, ask, timestamp=None):
        """Feed a tick and update rolling statistics.

        Parameters
        ----------
        symbol : str
            Instrument identifier (e.g. ``"EURUSD"``).
        bid : float
            Current bid price.
        ask : float
            Current ask price (stored but not used in signal generation).
        timestamp : optional
            Ignored; present for interface compatibility.
        """
        self._prices[symbol].append(bid)
        self._tick_count[symbol] += 1
        count = self._tick_count[symbol]

        # Update EMAs (seeded on first tick)
        if count == 1:
            self._ema_fast[symbol] = bid
            self._ema_slow[symbol] = bid
        else:
            alpha_fast = 2.0 / (self._ema_fast_period + 1)
            alpha_slow = 2.0 / (self._ema_slow_period + 1)
            self._ema_fast[symbol] = (
                bid * alpha_fast
                + self._ema_fast.get(symbol, bid) * (1.0 - alpha_fast)
            )
            self._ema_slow[symbol] = (
                bid * alpha_slow
                + self._ema_slow.get(symbol, bid) * (1.0 - alpha_slow)
            )

        # Compute log return over momentum_window
        prices = list(self._prices[symbol])
        if len(prices) >= self._momentum_window + 1:
            r = math.log(prices[-1] / prices[-(self._momentum_window + 1)])
            self._returns[symbol].append(r)

        # Compute single-step absolute change for ATR
        if len(prices) >= 2:
            change = abs(prices[-1] - prices[-2])
            self._changes[symbol].append(change)

    def get_signal(self, symbol):
        """Generate the current adaptive signal for *symbol*.

        Parameters
        ----------
        symbol : str
            Instrument identifier.

        Returns
        -------
        dict
            Keys:
            ``signal``      — -1 (sell), 0 (flat), +1 (buy).
            ``confidence``  — 0.0 – 1.0.
            ``source``      — one of "ZSCORE_BREAKOUT", "MOMENTUM",
                              "ADAPTIVE_EMA", "NO_SIGNAL".
            ``zscore``      — current z-score of the log return.
            ``threshold``   — adaptive threshold used.
            ``atr``         — current ATR value.
        """
        prices = list(self._prices.get(symbol, []))
        returns = list(self._returns.get(symbol, []))
        changes = list(self._changes.get(symbol, []))

        # --------------------------------------------------------------
        # Cold start — not enough data to compute any signal
        # --------------------------------------------------------------
        min_required = self._momentum_window + self._window
        if len(prices) < min_required or len(returns) < 2:
            return {
                "signal": 0,
                "confidence": 0.0,
                "source": "NO_SIGNAL",
                "zscore": 0.0,
                "threshold": 0.0,
                "atr": 0.0,
            }

        # --------------------------------------------------------------
        # 1. Compute log return over momentum_window
        # --------------------------------------------------------------
        current_return = returns[-1]

        # --------------------------------------------------------------
        # 2. Rolling z-score of returns
        # --------------------------------------------------------------
        window_returns = returns[-self._window:]
        n = len(window_returns)
        mean_ret = sum(window_returns) / n
        var_ret = sum((r - mean_ret) ** 2 for r in window_returns) / n
        std_ret = math.sqrt(var_ret) if var_ret > 0 else 1e-12
        zscore = (current_return - mean_ret) / std_ret

        # --------------------------------------------------------------
        # 3. ATR (Average True Range) over atr_window
        # --------------------------------------------------------------
        window_changes = changes[-self._atr_window:]
        atr = sum(window_changes) / len(window_changes) if window_changes else 0.0

        # --------------------------------------------------------------
        # 5. Adaptive threshold
        # --------------------------------------------------------------
        threshold = max(self._min_threshold, atr * 0.5)

        # --------------------------------------------------------------
        # 6-10. Signal logic with priority
        # --------------------------------------------------------------
        signal = 0
        source = "NO_SIGNAL"

        # Priority 1: ZSCORE_BREAKOUT
        if abs(zscore) > 1.5:
            signal = 1 if zscore > 0 else -1
            source = "ZSCORE_BREAKOUT"

        # Priority 2: MOMENTUM
        elif abs(current_return) > threshold:
            signal = 1 if current_return > 0 else -1
            source = "MOMENTUM"

        # Priority 3: ADAPTIVE_EMA
        else:
            fast_ema = self._ema_fast.get(symbol)
            slow_ema = self._ema_slow.get(symbol)
            if fast_ema is not None and slow_ema is not None:
                diff = fast_ema - slow_ema
                if abs(diff) > threshold:
                    signal = 1 if diff > 0 else -1
                    source = "ADAPTIVE_EMA"

        # --------------------------------------------------------------
        # 11. Confidence
        # --------------------------------------------------------------
        conf_z = abs(zscore) / 3.0
        conf_ret = abs(current_return) / (threshold if threshold > 0 else 1e-12)
        confidence = min(1.0, conf_z, conf_ret)

        return {
            "signal": signal,
            "confidence": round(confidence, 6),
            "source": source,
            "zscore": round(zscore, 6),
            "threshold": round(threshold, 8),
            "atr": round(atr, 8),
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest():
    """Feed synthetic trending data and verify non-zero signals."""
    import random

    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s | %(message)s",
    )
    logger.info("=" * 60)
    logger.info("AltSignalAdaptiveScaler — Self Test")
    logger.info("=" * 60)

    random.seed(42)
    scaler = AltSignalAdaptiveScaler("selftest")
    symbol = "SYNTH_USD"

    # ------------------------------------------------------------------
    # Feed 100 synthetic ticks with known upward drift + noise
    # ------------------------------------------------------------------
    price = 1.10000
    for i in range(100):
        trend = 0.0003
        noise = random.uniform(-0.0005, 0.0005)
        price = price + trend + noise
        price = max(price, 0.0001)
        scaler.feed_tick(symbol, bid=price, ask=price + 0.0002)

    # ------------------------------------------------------------------
    # Collect signals (skip cold-start ticks)
    # ------------------------------------------------------------------
    signals = []
    sources = {}
    confidences = []
    for i in range(100):
        sig = scaler.get_signal(symbol)
        signals.append(sig["signal"])
        sources[sig["source"]] = sources.get(sig["source"], 0) + 1
        confidences.append(sig["confidence"])

    # ------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------
    total = len(signals)
    buys = sum(1 for s in signals if s == 1)
    sells = sum(1 for s in signals if s == -1)
    flats = sum(1 for s in signals if s == 0)

    assert all(s in (-1, 0, 1) for s in signals), "Signal values out of range!"
    assert flats < total, "All signals are flat — no signals generated!"
    assert buys > sells, "Expected more buy than sell signals in uptrend!"

    logger.info("Total signals: %d", total)
    logger.info("  buy=%.1f%%  sell=%.1f%%  flat=%.1f%%",
                100.0 * buys / total, 100.0 * sells / total,
                100.0 * flats / total)
    logger.info("Source distribution: %s", sources)
    logger.info("Mean confidence: %.4f",
                sum(confidences) / len(confidences) if confidences else 0.0)

    # ------------------------------------------------------------------
    # Verify singleton
    # ------------------------------------------------------------------
    same = AltSignalAdaptiveScaler("selftest")
    assert same is scaler, "Singleton pattern broken!"
    logger.info("Singleton verified.")

    logger.info("=" * 60)
    logger.info(">>> ALL SELFTESTS PASSED <<<")
    logger.info("=" * 60)


if __name__ == "__main__":
    _selftest()
