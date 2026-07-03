"""
Signal Truth Labeler — determines whether OSS or ALT signal (or neither)
actually correlates with forward return.

Tracks signal-return pairs over time and computes correlation / accuracy
metrics to label the true alpha source (OSS, ALT, NEITHER, or INCONCLUSIVE).
"""

import math
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton registry
# ---------------------------------------------------------------------------

_instances = {}


def SignalTruthLabeler(instance_id="default", **kwargs):
    """Singleton accessor — returns the same _SignalTruthLabeler for a given id.

    Parameters
    ----------
    instance_id : str
        Unique identifier for this labeler instance.
    **kwargs
        Additional arguments forwarded to ``_SignalTruthLabeler.__init__``
        (e.g. ``forward_horizon=10``).  These are only used when the instance
        is first created; subsequent calls with the same *instance_id* ignore
        them.
    """
    if instance_id not in _instances:
        _instances[instance_id] = _SignalTruthLabeler(instance_id, **kwargs)
    return _instances[instance_id]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pearson(xs, ys):
    """Pearson product-moment correlation coefficient."""
    n = len(xs)
    if n < 3:
        return 0.0
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xy = sum(a * b for a, b in zip(xs, ys))
    sum_x2 = sum(a * a for a in xs)
    sum_y2 = sum(b * b for b in ys)
    num = n * sum_xy - sum_x * sum_y
    den = math.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
    if den == 0.0:
        return 0.0
    return max(-1.0, min(1.0, num / den))


def _sign(x):
    """Return -1, 0, or +1 for negative, zero, positive input."""
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------

class _SignalTruthLabeler:
    """Tracks signal-return pairs per symbol and determines the true alpha source.

    Parameters
    ----------
    instance_id : str
        Identifier for this labeler instance (used by singleton registry).
    forward_horizon : int
        Number of ticks ahead used to compute forward return (default 5).
    """

    def __init__(self, instance_id="default", forward_horizon=5):
        self._instance_id = instance_id
        self._forward_horizon = int(forward_horizon)
        # symbol -> list of (price, oss_signal, alt_signal, timestamp)
        # oss_signal / alt_signal are None for price-only entries (feed_close)
        self._ticks = defaultdict(list)
        logger.debug(
            "SignalTruthLabeler(%r) initialised, forward_horizon=%d",
            instance_id, self._forward_horizon,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def forward_horizon(self):
        """Number of ticks ahead used to compute forward return."""
        return self._forward_horizon

    @forward_horizon.setter
    def forward_horizon(self, value):
        """Set the forward horizon (applied to future computations)."""
        self._forward_horizon = int(value)
        logger.debug("forward_horizon set to %d", self._forward_horizon)

    # ------------------------------------------------------------------
    # Public API — data ingestion
    # ------------------------------------------------------------------

    def feed_tick(self, symbol, bid, oss_signal, alt_signal, timestamp=None):
        """Store a tick with associated OSS and ALT signals.

        Parameters
        ----------
        symbol : str
            Instrument / ticker identifier.
        bid : float
            Current bid price.
        oss_signal : int
            OSS signal value (typically -1, 0, or +1).
        alt_signal : int
            ALT signal value (typically -1, 0, or +1).
        timestamp : optional
            Timestamp for the observation.
        """
        self._ticks[symbol].append((bid, oss_signal, alt_signal, timestamp))
        logger.debug(
            "feed_tick %s price=%.5f oss=%s alt=%s ts=%s",
            symbol, bid, oss_signal, alt_signal, timestamp,
        )

    def feed_close(self, symbol, bid):
        """Store a close price (no signals) to extend the price series.

        This enables forward-return computation for recent ticks that do not
        yet have ``forward_horizon`` future ticks.

        Parameters
        ----------
        symbol : str
            Instrument / ticker identifier.
        bid : float
            Close / settlement price.
        """
        self._ticks[symbol].append((bid, None, None, None))
        logger.debug("feed_close %s price=%.5f", symbol, bid)

    # ------------------------------------------------------------------
    # Correlation & accuracy computation
    # ------------------------------------------------------------------

    def _get_records(self, symbol=None):
        """Build (oss_signal, alt_signal, forward_return) triples.

        For each tick entry with valid signals, compute the forward return
        if enough future ticks exist in the buffer.

        Parameters
        ----------
        symbol : str or None
            If None, aggregate across all tracked symbols.

        Returns
        -------
        list of (oss, alt, forward_return) tuples.
        """
        symbols = [symbol] if symbol is not None else list(self._ticks.keys())
        records = []

        for sym in symbols:
            ticks = self._ticks.get(sym, [])
            for i, (price, oss, alt, ts) in enumerate(ticks):
                # Skip price-only entries (from feed_close)
                if oss is None or alt is None:
                    continue
                future_idx = i + self._forward_horizon
                if future_idx >= len(ticks):
                    continue  # not enough future data yet
                future_price = ticks[future_idx][0]
                forward_return = (future_price - price) / price
                records.append((oss, alt, forward_return))

        return records

    def compute_correlations(self, symbol=None):
        """Compute Pearson correlation and directional accuracy metrics.

        Parameters
        ----------
        symbol : str or None
            If None, aggregate across all tracked symbols.

        Returns
        -------
        dict with keys:
            oss_return_corr, alt_return_corr, oss_accuracy, alt_accuracy,
            samples, forward_horizon
        """
        records = self._get_records(symbol)
        n = len(records)

        if n < 3:
            return {
                "oss_return_corr": 0.0,
                "alt_return_corr": 0.0,
                "oss_accuracy": 0.0,
                "alt_accuracy": 0.0,
                "samples": n,
                "forward_horizon": self._forward_horizon,
            }

        oss_signals = [r[0] for r in records]
        alt_signals = [r[1] for r in records]
        returns = [r[2] for r in records]

        oss_corr = _pearson(oss_signals, returns)
        alt_corr = _pearson(alt_signals, returns)

        # Directional accuracy: % of ticks where signal sign matches return sign
        oss_correct = 0
        alt_correct = 0

        for oss, alt, ret in records:
            ret_s = _sign(ret)
            if _sign(oss) == ret_s:
                oss_correct += 1
            if _sign(alt) == ret_s:
                alt_correct += 1

        return {
            "oss_return_corr": round(oss_corr, 4),
            "alt_return_corr": round(alt_corr, 4),
            "oss_accuracy": round(oss_correct / n, 4),
            "alt_accuracy": round(alt_correct / n, 4),
            "samples": n,
            "forward_horizon": self._forward_horizon,
        }

    # ------------------------------------------------------------------
    # Truth label decision
    # ------------------------------------------------------------------

    def get_truth_label(self, symbol=None):
        """Return a dict identifying the true alpha source.

        Logic
        -----
        - ``oss_accuracy > alt_accuracy + 0.05`` AND ``oss_accuracy > 0.5``  →  "OSS"
        - ``alt_accuracy > oss_accuracy + 0.05`` AND ``alt_accuracy > 0.5``  →  "ALT"
        - Both accuracies > 0.5 but within 0.05 of each other                →  "INCONCLUSIVE"
        - Both accuracies ≤ 0.53 (near baseline)                             →  "NEITHER"
        - Fewer than 10 samples / mixed signals                              →  "INCONCLUSIVE"

        Parameters
        ----------
        symbol : str or None
            If None, aggregate across all tracked symbols.

        Returns
        -------
        dict with keys:
            true_alpha_source, oss_accuracy, alt_accuracy,
            oss_return_corr, alt_return_corr, samples, forward_horizon
        """
        corr = self.compute_correlations(symbol)
        samples = corr["samples"]
        oss_acc = corr["oss_accuracy"]
        alt_acc = corr["alt_accuracy"]

        if samples < 10:
            label = "INCONCLUSIVE"
        elif oss_acc > alt_acc + 0.05 and oss_acc > 0.5:
            label = "OSS"
        elif alt_acc > oss_acc + 0.05 and alt_acc > 0.5:
            label = "ALT"
        elif oss_acc > 0.5 and alt_acc > 0.5:
            # Both above 0.5 but within 0.05 of each other → inconclusive
            label = "INCONCLUSIVE"
        elif max(oss_acc, alt_acc) <= 0.53:
            # Both at or near the 0.5 baseline (allow small statistical
            # fluctuation around random chance) → no predictive signal
            label = "NEITHER"
        else:
            # One above 0.5, the other at or below 0.5, but margin
            # less than 0.05 → inconclusive (needs more data)
            label = "INCONCLUSIVE"

        return {
            "true_alpha_source": label,
            "oss_accuracy": oss_acc,
            "alt_accuracy": alt_acc,
            "oss_return_corr": corr["oss_return_corr"],
            "alt_return_corr": corr["alt_return_corr"],
            "samples": samples,
            "forward_horizon": self._forward_horizon,
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self):
        """Clear all tracked tick data."""
        self._ticks.clear()
        logger.debug("SignalTruthLabeler(%r) reset", self._instance_id)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    random.seed(42)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    HORIZON = 5
    NUM_TICKS = 2000

    print("=" * 60)
    print("Signal Truth Labeler — Self Test")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Scenario 1: ALT signal is better than OSS
    # ------------------------------------------------------------------
    labeler1 = SignalTruthLabeler("test_alt_better", forward_horizon=HORIZON)

    # Build a price series first (random walk)
    prices = [100.0]
    for _ in range(NUM_TICKS + HORIZON):
        prices.append(prices[-1] * (1 + random.choice([-0.001, 0.001])))

    for i in range(NUM_TICKS):
        forward_ret = (prices[i + HORIZON] - prices[i]) / prices[i]
        ret_sign = _sign(forward_ret)

        # ALT signal matches forward-return sign with 70% probability
        if random.random() < 0.70:
            alt_signal = ret_sign
        else:
            alt_signal = random.choice([-1, 1])

        # OSS signal is random ±1 (50% directional accuracy expected)
        oss_signal = random.choice([-1, 1])

        labeler1.feed_tick("SCE1_ALT_BETTER", prices[i], oss_signal, alt_signal)

    report1 = labeler1.get_truth_label("SCE1_ALT_BETTER")
    print("\n--- SCE1: ALT is Better ---")
    for k, v in report1.items():
        print(f"  {k:25s} = {v}")
    assert report1["true_alpha_source"] == "ALT", (
        f"Expected ALT, got {report1['true_alpha_source']}"
    )
    print("  >>> PASS")

    # ------------------------------------------------------------------
    # Scenario 2: OSS signal is better than ALT
    # ------------------------------------------------------------------
    labeler2 = SignalTruthLabeler("test_oss_better", forward_horizon=HORIZON)

    prices2 = [100.0]
    for _ in range(NUM_TICKS + HORIZON):
        prices2.append(prices2[-1] * (1 + random.choice([-0.001, 0.001])))

    for i in range(NUM_TICKS):
        forward_ret = (prices2[i + HORIZON] - prices2[i]) / prices2[i]
        ret_sign = _sign(forward_ret)

        # OSS signal matches forward-return sign with 70% probability
        if random.random() < 0.70:
            oss_signal = ret_sign
        else:
            oss_signal = random.choice([-1, 1])

        # ALT signal is random ±1 (50% directional accuracy expected)
        alt_signal = random.choice([-1, 1])

        labeler2.feed_tick("SCE2_OSS_BETTER", prices2[i], oss_signal, alt_signal)

    report2 = labeler2.get_truth_label("SCE2_OSS_BETTER")
    print("\n--- SCE2: OSS is Better ---")
    for k, v in report2.items():
        print(f"  {k:25s} = {v}")
    assert report2["true_alpha_source"] == "OSS", (
        f"Expected OSS, got {report2['true_alpha_source']}"
    )
    print("  >>> PASS")

    # ------------------------------------------------------------------
    # Scenario 3: Both signals are noise (no predictive power)
    # ------------------------------------------------------------------
    labeler3 = SignalTruthLabeler("test_noise", forward_horizon=HORIZON)

    prices3 = [100.0]
    for _ in range(NUM_TICKS + HORIZON):
        prices3.append(prices3[-1] * (1 + random.choice([-0.001, 0.001])))

    for i in range(NUM_TICKS):
        # Both signals are random ±1 — no correlation with forward return
        oss_signal = random.choice([-1, 1])
        alt_signal = random.choice([-1, 1])
        labeler3.feed_tick("SCE3_NOISE", prices3[i], oss_signal, alt_signal)

    report3 = labeler3.get_truth_label("SCE3_NOISE")
    print("\n--- SCE3: Both are Noise ---")
    for k, v in report3.items():
        print(f"  {k:25s} = {v}")
    assert report3["true_alpha_source"] == "NEITHER", (
        f"Expected NEITHER, got {report3['true_alpha_source']}"
    )
    print("  >>> PASS")

    # ------------------------------------------------------------------
    # Scenario 4: Both signals are close / inconclusive
    # (both > 0.5 but within 0.05 of each other)
    # ------------------------------------------------------------------
    labeler4 = SignalTruthLabeler("test_inconclusive", forward_horizon=HORIZON)

    prices4 = [100.0]
    for _ in range(NUM_TICKS + HORIZON):
        prices4.append(prices4[-1] * (1 + random.choice([-0.001, 0.001])))

    for i in range(NUM_TICKS):
        forward_ret = (prices4[i + HORIZON] - prices4[i]) / prices4[i]
        ret_sign = _sign(forward_ret)

        # Both signals predict direction ~55% of the time (both > 0.5,
        # close to each other within 0.05)
        if random.random() < 0.55:
            oss_signal = ret_sign
        else:
            oss_signal = random.choice([-1, 1])

        if random.random() < 0.55:
            alt_signal = ret_sign
        else:
            alt_signal = random.choice([-1, 1])

        labeler4.feed_tick("SCE4_INCONCLUSIVE", prices4[i], oss_signal, alt_signal)

    report4 = labeler4.get_truth_label("SCE4_INCONCLUSIVE")
    print("\n--- SCE4: Inconclusive (both close) ---")
    for k, v in report4.items():
        print(f"  {k:25s} = {v}")
    assert report4["true_alpha_source"] == "INCONCLUSIVE", (
        f"Expected INCONCLUSIVE, got {report4['true_alpha_source']}"
    )
    print("  >>> PASS")

    # ------------------------------------------------------------------
    # Scenario 5: Fewer than 10 samples → INCONCLUSIVE
    # ------------------------------------------------------------------
    labeler5 = SignalTruthLabeler("test_few_samples", forward_horizon=HORIZON)
    # Only feed 3 ticks (not enough future data for any record)
    for i in range(3):
        labeler5.feed_tick("SCE5_FEW", 100.0 + i, 1, 1)

    report5 = labeler5.get_truth_label("SCE5_FEW")
    print("\n--- SCE5: Too few samples ---")
    for k, v in report5.items():
        print(f"  {k:25s} = {v}")
    assert report5["true_alpha_source"] == "INCONCLUSIVE", (
        f"Expected INCONCLUSIVE, got {report5['true_alpha_source']}"
    )
    assert report5["samples"] == 0, f"Expected 0 samples, got {report5['samples']}"
    print("  >>> PASS")

    # ------------------------------------------------------------------
    # Reset test
    # ------------------------------------------------------------------
    labeler1.reset()
    assert len(labeler1._ticks) == 0, "Reset should clear all tick data"
    print("\n--- Reset test ---")
    print("  Reset verified >>> PASS")

    # ------------------------------------------------------------------
    # Singleton test
    # ------------------------------------------------------------------
    same = SignalTruthLabeler("test_alt_better")
    assert same is labeler1, "Singleton should return the same instance"
    print("\n--- Singleton test ---")
    print("  Singleton verified >>> PASS")

    print("\n" + "=" * 60)
    print("All self-tests PASSED.")
    print("=" * 60)
