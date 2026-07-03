"""AltSignalGenerator — scientific control baseline signal generator.

Produces simple directional signals (EMA crossover, momentum slope, z-score
deviation) as a control to compare against OSS signal quality. This is NOT a
production signal source — it is a scientific control to determine whether the
signal space itself is degenerate.
"""

import logging
import math
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_instances = {}


def AltSignalGenerator(instance_id="default", **kwargs):
    """Singleton accessor for _AltSignalGenerator instances.

    Parameters
    ----------
    instance_id : str
        Unique identifier for the generator instance.  Multiple callers
    sharing the same *instance_id* share the same underlying object.
    **kwargs
        Forwarded to ``_AltSignalGenerator.__init__`` (see class docstring).

    Returns
    -------
    _AltSignalGenerator
    """
    if instance_id not in _instances:
        _instances[instance_id] = _AltSignalGenerator(instance_id, **kwargs)
    return _instances[instance_id]


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------
class _AltSignalGenerator:
    """Simple directional signal generator used as a control baseline.

    Three modes are supported (selectable via the *mode* constructor kwarg):

    ``"ema_cross"`` (default)
        Fast EMA (default 5) minus slow EMA (default 20) on bid prices.
        Signal is +1 when fast > slow + threshold, -1 when fast < slow -
        threshold, and 0 otherwise.
    ``"momentum"``
        Rate of change (ROC) over *momentum_periods*.  Signal is +1 when
        ROC > +threshold, -1 when ROC < -threshold, 0 otherwise.
    ``"zscore"``
        Z-score of the current price vs a rolling mean/std.  Signal is +1
        when z < -threshold (oversold), -1 when z > +threshold (overbought),
        0 otherwise.

    Parameters
    ----------
    instance_id : str
        Label used in logging.
    mode : str
        One of ``"ema_cross"``, ``"momentum"``, ``"zscore"``.
    **kwargs
        *ema_fast* (int, default 5) — fast EMA period.
        *ema_slow* (int, default 20) — slow EMA period.
        *ema_threshold* (float, default 0.0001) — minimum absolute price
            difference for a non-flat EMA-cross signal.
        *momentum_periods* (int, default 10) — ROC look-back.
        *momentum_threshold* (float, default 0.001) — minimum relative price
            change for a non-flat momentum signal.
        *zscore_window* (int, default 20) — rolling window for mean/std.
        *zscore_threshold* (float, default 1.5) — z-score magnitude above
            which the signal becomes non-flat.
    """

    def __init__(self, instance_id="default", mode="ema_cross", **kwargs):
        self._instance_id = instance_id
        self._mode = mode

        # Config with sensible defaults ---------------------------------
        self._config = {
            "ema_fast": int(kwargs.get("ema_fast", 5)),
            "ema_slow": int(kwargs.get("ema_slow", 20)),
            "ema_threshold": float(kwargs.get("ema_threshold", 0.0001)),
            "momentum_periods": int(kwargs.get("momentum_periods", 10)),
            "momentum_threshold": float(kwargs.get("momentum_threshold", 0.001)),
            "zscore_window": int(kwargs.get("zscore_window", 20)),
            "zscore_threshold": float(kwargs.get("zscore_threshold", 1.5)),
        }

        # Maximum data depth needed by any mode
        max_window = max(
            self._config["ema_slow"],
            self._config["momentum_periods"],
            self._config["zscore_window"],
        ) + 5

        # Per-symbol price history (ring buffer)
        self._prices = defaultdict(lambda: deque(maxlen=max_window))

        # Per-symbol EMA state
        self._ema_fast = {}  # symbol -> float
        self._ema_slow = {}  # symbol -> float

        # Per-symbol tick counter (determines when EMAs are seeded)
        self._tick_count = defaultdict(int)

        # Per-symbol signal history for statistics: list of (signal, confidence)
        self._signal_history = defaultdict(lambda: deque(maxlen=5000))

        logger.info(
            "AltSignalGenerator initialized | instance=%s mode=%s config=%s",
            instance_id,
            mode,
            self._config,
        )

    # -- public api ---------------------------------------------------------

    def feed_tick(self, symbol, bid, ask, timestamp=None):
        """Store incoming tick data and update internal state.

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
        price = bid
        self._prices[symbol].append(price)
        self._tick_count[symbol] += 1

        # Update exponential moving averages
        cfg = self._config
        count = self._tick_count[symbol]

        if count == 1:
            self._ema_fast[symbol] = price
            self._ema_slow[symbol] = price
        else:
            alpha_fast = 2.0 / (cfg["ema_fast"] + 1)
            alpha_slow = 2.0 / (cfg["ema_slow"] + 1)
            self._ema_fast[symbol] = (
                price * alpha_fast
                + self._ema_fast.get(symbol, price) * (1.0 - alpha_fast)
            )
            self._ema_slow[symbol] = (
                price * alpha_slow
                + self._ema_slow.get(symbol, price) * (1.0 - alpha_slow)
            )

    def get_signal(self, symbol):
        """Generate the current directional signal for *symbol*.

        Returns
        -------
        dict
            Keys: ``signal`` (-1/0/+1), ``confidence`` (0.0-1.0),
            ``source`` (str), ``value`` (raw computed value).
        """
        prices = self._prices.get(symbol, [])
        if len(prices) < self._config["ema_slow"]:
            return {
                "signal": 0,
                "confidence": 0.0,
                "source": "insufficient_data",
                "value": 0.0,
            }

        if self._mode == "ema_cross":
            result = self._signal_ema_cross(symbol)
        elif self._mode == "momentum":
            result = self._signal_momentum(symbol)
        elif self._mode == "zscore":
            result = self._signal_zscore(symbol)
        else:
            result = {
                "signal": 0,
                "confidence": 0.0,
                "source": "unknown_mode",
                "value": 0.0,
            }

        # Record for statistics
        self._signal_history[symbol].append((result["signal"], result["confidence"]))
        return result

    def get_all_signals(self):
        """Return signal dict for every symbol that has sufficient data.

        Returns
        -------
        dict
            ``{symbol: signal_dict, ...}``
        """
        return {sym: self.get_signal(sym) for sym in list(self._prices.keys())}

    def get_statistics(self, symbol):
        """Compute summary statistics for a symbol's signal history.

        Parameters
        ----------
        symbol : str

        Returns
        -------
        dict
            Keys: ``total_signals``, ``buy_pct``, ``sell_pct``, ``flat_pct``,
            ``mean_confidence``, ``signal_std``.
        """
        history = list(self._signal_history.get(symbol, []))
        total = len(history)

        if total == 0:
            return {
                "total_signals": 0,
                "buy_pct": 0.0,
                "sell_pct": 0.0,
                "flat_pct": 100.0,
                "mean_confidence": 0.0,
                "signal_std": 0.0,
            }

        signals = [h[0] for h in history]
        confidences = [h[1] for h in history]

        buys = sum(1 for s in signals if s == 1)
        sells = sum(1 for s in signals if s == -1)
        flats = sum(1 for s in signals if s == 0)

        # Mean confidence from stored values
        mean_conf = sum(confidences) / total

        # Population standard deviation of signal values
        mean_sig = sum(signals) / total
        var_sig = sum((s - mean_sig) ** 2 for s in signals) / total
        std_sig = math.sqrt(var_sig)

        return {
            "total_signals": total,
            "buy_pct": 100.0 * buys / total,
            "sell_pct": 100.0 * sells / total,
            "flat_pct": 100.0 * flats / total,
            "mean_confidence": round(mean_conf, 4),
            "signal_std": round(std_sig, 4),
        }

    # -- internal signal generators -----------------------------------------

    def _signal_ema_cross(self, symbol):
        """EMA crossover: +1 (bullish), -1 (bearish), 0 (neutral)."""
        fast = self._ema_fast.get(symbol)
        slow = self._ema_slow.get(symbol)
        if fast is None or slow is None:
            return {"signal": 0, "confidence": 0.0, "source": "ema_cross", "value": 0.0}

        diff = fast - slow
        threshold = self._config["ema_threshold"]

        if diff > threshold:
            signal = 1
        elif diff < -threshold:
            signal = -1
        else:
            signal = 0

        # Confidence scales with how decisively the EMAs have separated
        # relative to the threshold.
        price = self._prices[symbol][-1]
        rel_diff = abs(diff) / price if price != 0.0 else 0.0
        confidence = min(1.0, rel_diff / (threshold * 10)) if threshold > 0 else 0.0

        return {"signal": signal, "confidence": round(confidence, 4), "source": "ema_cross", "value": round(diff, 8)}

    def _signal_momentum(self, symbol):
        """Rate-of-change momentum: +1 (up), -1 (down), 0 (flat)."""
        prices = self._prices[symbol]
        n = self._config["momentum_periods"]

        if len(prices) < n + 1:
            return {"signal": 0, "confidence": 0.0, "source": "momentum", "value": 0.0}

        current = prices[-1]
        past = prices[-(n + 1)]
        roc = (current - past) / past if past != 0.0 else 0.0

        threshold = self._config["momentum_threshold"]

        if roc > threshold:
            signal = 1
        elif roc < -threshold:
            signal = -1
        else:
            signal = 0

        confidence = min(1.0, abs(roc) / (threshold * 5)) if threshold > 0 else 0.0

        return {"signal": signal, "confidence": round(confidence, 4), "source": "momentum", "value": round(roc, 8)}

    def _signal_zscore(self, symbol):
        """Z-score deviation: +1 (oversold bounce), -1 (overbought), 0."""
        prices = list(self._prices[symbol])
        window = self._config["zscore_window"]

        if len(prices) < window + 1:
            return {"signal": 0, "confidence": 0.0, "source": "zscore", "value": 0.0}

        recent = prices[-window:]
        current = prices[-1]

        mean = sum(recent) / window
        var = sum((p - mean) ** 2 for p in recent) / window
        std = math.sqrt(var) if var > 0 else 1e-12

        z = (current - mean) / std
        threshold = self._config["zscore_threshold"]

        if z < -threshold:
            signal = 1  # oversold → buy
        elif z > threshold:
            signal = -1  # overbought → sell
        else:
            signal = 0

        confidence = min(1.0, abs(z) / (threshold * 3))

        return {"signal": signal, "confidence": round(confidence, 4), "source": "zscore", "value": round(z, 4)}

# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _selftest():
    """Feed synthetic trending data and verify non-flat signal distribution."""
    import random

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    logger.info("=== AltSignalGenerator self-test ===")

    gen = AltSignalGenerator("selftest")
    symbol = "SYNTH_USD"

    # -- generate 1000 ticks with a gentle upward trend + noise -----------
    price = 1.10000
    for i in range(1000):
        # Upward drift + random walk
        trend = 0.00002  # small positive drift per tick
        noise = random.uniform(-0.0005, 0.0005)
        price = price + trend + noise
        price = max(price, 0.0001)  # keep positive
        gen.feed_tick(symbol, bid=price, ask=price + 0.0002)

    # -- collect signals (skip first 50 ticks to allow EMA warm-up) -------
    signals = []
    for i in range(1000):
        sig = gen.get_signal(symbol)
        signals.append(sig["signal"])

    # -- verify -----------------------------------------------------------
    total = len(signals)
    buys = sum(1 for s in signals if s == 1)
    sells = sum(1 for s in signals if s == -1)
    flats = sum(1 for s in signals if s == 0)

    # All values are {-1, 0, +1}
    assert all(s in (-1, 0, 1) for s in signals), "Signal values out of range!"

    # Distribution must not be *all* flat
    assert flats < total, "All signals are flat — control is degenerate!"

    stats = gen.get_statistics(symbol)

    logger.info("Signals: total=%d  buy=%.1f%%  sell=%.1f%%  flat=%.1f%%", total, stats["buy_pct"], stats["sell_pct"], stats["flat_pct"])
    logger.info("Confidence: mean=%.4f  std=%.4f", stats["mean_confidence"], stats["signal_std"])
    logger.info("Non-flat ratio: %.1f%%", 100.0 * (total - flats) / total)
    logger.info(">>> SELFTEST PASSED <<<")


if __name__ == "__main__":
    _selftest()
