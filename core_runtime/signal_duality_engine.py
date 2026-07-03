"""
Signal Duality Engine — compares OSS signals vs ALT signals to measure
correlation, divergence rate, and conflict frequency.

This module determines whether OSS is uniquely flat or whether the entire
signal space is flat.
"""

import math
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton registry
# ---------------------------------------------------------------------------

_instances = {}


def SignalDualityEngine(instance_id="default"):
    """Singleton accessor — returns the same _SignalDualityEngine for a given id."""
    if instance_id not in _instances:
        _instances[instance_id] = _SignalDualityEngine(instance_id)
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


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------

class _SignalDualityEngine:
    """Tracks signal pairs (oss, alt) per symbol and computes duality metrics."""

    def __init__(self, instance_id="default"):
        self._instance_id = instance_id
        # symbol -> list of (oss_signal, alt_signal) tuples
        self._data = defaultdict(list)
        logger.debug("SignalDualityEngine(%r) initialised", instance_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, symbol, oss_signal, alt_signal, oss_confidence=None, alt_confidence=None):
        """Record a signal-pair observation for *symbol*.

        Parameters
        ----------
        symbol : str
            Instrument / ticker identifier.
        oss_signal : int
            OSS signal value (typically -1, 0, or +1).
        alt_signal : int
            ALT signal value (typically -1, 0, or +1).
        oss_confidence : float or None
            Optional confidence score for the OSS signal.
        alt_confidence : float or None
            Optional confidence score for the ALT signal.
        """
        # Normalise to int (allow floats that represent discrete values)
        oss = int(oss_signal)
        alt = int(alt_signal)
        self._data[symbol].append((oss, alt))
        if oss_confidence is not None or alt_confidence is not None:
            logger.debug(
                "feed %s oss=%s alt=%s oss_conf=%s alt_conf=%s",
                symbol, oss, alt, oss_confidence, alt_confidence,
            )
        else:
            logger.debug("feed %s oss=%s alt=%s", symbol, oss, alt)

    # ------------------------------------------------------------------
    # Per-symbol report
    # ------------------------------------------------------------------

    def get_duality_report(self, symbol):
        """Return a dict of duality metrics for *symbol*.

        Returns
        -------
        dict with keys:
            symbol, sample_count,
            correlation, divergence_rate, conflict_frequency, agreement_rate,
            oss_flat_rate, alt_flat_rate
        """
        pairs = self._data.get(symbol, [])
        n = len(pairs)
        if n == 0:
            return {
                "symbol": symbol,
                "sample_count": 0,
                "correlation": 0.0,
                "divergence_rate": 0.0,
                "conflict_frequency": 0.0,
                "agreement_rate": 0.0,
                "oss_flat_rate": 0.0,
                "alt_flat_rate": 0.0,
            }

        oss_seq = [p[0] for p in pairs]
        alt_seq = [p[1] for p in pairs]

        # Correlation
        corr = _pearson(oss_seq, alt_seq)

        # Counters
        agree = 0
        disagree = 0          # any mismatch
        conflict = 0          # opposite signs: +1 vs -1
        oss_flat = 0
        alt_flat = 0

        for oss, alt in pairs:
            if oss == alt:
                agree += 1
            else:
                disagree += 1

            if (oss == 1 and alt == -1) or (oss == -1 and alt == 1):
                conflict += 1

            if oss == 0:
                oss_flat += 1
            if alt == 0:
                alt_flat += 1

        return {
            "symbol": symbol,
            "sample_count": n,
            "correlation": round(corr, 4),
            "divergence_rate": round(disagree / n, 4),
            "conflict_frequency": round(conflict / n, 4),
            "agreement_rate": round(agree / n, 4),
            "oss_flat_rate": round(oss_flat / n, 4),
            "alt_flat_rate": round(alt_flat / n, 4),
        }

    # ------------------------------------------------------------------
    # Batch reports
    # ------------------------------------------------------------------

    def get_all_reports(self):
        """Return dict mapping each symbol to its duality report."""
        return {sym: self.get_duality_report(sym) for sym in self._data}

    # ------------------------------------------------------------------
    # Aggregated summary
    # ------------------------------------------------------------------

    def get_summary(self):
        """Return an aggregated summary across all tracked symbols.

        Returns
        -------
        dict with keys:
            total_observations, global_agreement_rate, global_divergence_rate,
            global_oss_flat_rate, global_alt_flat_rate, verdict
        """
        total = 0
        total_agree = 0
        total_disagree = 0
        total_oss_flat = 0
        total_alt_flat = 0

        for symbol, pairs in self._data.items():
            n = len(pairs)
            total += n
            for oss, alt in pairs:
                if oss == alt:
                    total_agree += 1
                else:
                    total_disagree += 1
                if oss == 0:
                    total_oss_flat += 1
                if alt == 0:
                    total_alt_flat += 1

        if total == 0:
            return {
                "total_observations": 0,
                "global_agreement_rate": 0.0,
                "global_divergence_rate": 0.0,
                "global_oss_flat_rate": 0.0,
                "global_alt_flat_rate": 0.0,
                "verdict": "CONFLICTING",
            }

        global_agreement_rate = total_agree / total
        global_divergence_rate = total_disagree / total
        global_oss_flat_rate = total_oss_flat / total
        global_alt_flat_rate = total_alt_flat / total

        # Verdict logic
        if global_oss_flat_rate > 0.80 and global_alt_flat_rate < 0.30:
            verdict = "OSS_UNIQUELY_FLAT"
        elif global_oss_flat_rate > 0.80 and global_alt_flat_rate > 0.80:
            verdict = "BOTH_FLAT"
        elif global_oss_flat_rate < 0.30 and global_alt_flat_rate < 0.30:
            verdict = "BOTH_ACTIVE"
        else:
            verdict = "CONFLICTING"

        return {
            "total_observations": total,
            "global_agreement_rate": round(global_agreement_rate, 4),
            "global_divergence_rate": round(global_divergence_rate, 4),
            "global_oss_flat_rate": round(global_oss_flat_rate, 4),
            "global_alt_flat_rate": round(global_alt_flat_rate, 4),
            "verdict": verdict,
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    engine = SignalDualityEngine()

    # ---- Scenario 1: OSS flat, ALT active (OSS_UNIQUELY_FLAT) ----
    for _ in range(100):
        engine.feed("SCE1", 0, 1)
    for _ in range(100):
        engine.feed("SCE1", 0, -1)

    # ---- Scenario 2: both flat (BOTH_FLAT) ----
    for _ in range(200):
        engine.feed("SCE2", 0, 0)

    # ---- Scenario 3: both active + correlated (BOTH_ACTIVE) ----
    for _ in range(100):
        engine.feed("SCE3", 1, 1)
    for _ in range(100):
        engine.feed("SCE3", -1, -1)

    # ---- Scenario 4: mixed / conflicting ----
    for _ in range(50):
        engine.feed("SCE4", 1, 1)
    for _ in range(50):
        engine.feed("SCE4", 1, -1)
    for _ in range(50):
        engine.feed("SCE4", 1, 0)
    for _ in range(50):
        engine.feed("SCE4", 0, 0)

    # ---- Scenario 5: correlated with some noise ----
    import random
    random.seed(42)
    for _ in range(200):
        base = random.choice([-1, 0, 1])
        noise = random.choice([-1, 0, 1])
        engine.feed("SCE5", base, base if random.random() > 0.3 else noise)

    print("=" * 60)
    print("Signal Duality Engine — Self Test")
    print("=" * 60)

    for sym in ["SCE1", "SCE2", "SCE3", "SCE4", "SCE5"]:
        report = engine.get_duality_report(sym)
        print(f"\n--- {sym} ---")
        for k, v in report.items():
            print(f"  {k:25s} = {v}")

    print("\n" + "=" * 60)
    print("AGGREGATED SUMMARY")
    print("=" * 60)
    summary = engine.get_summary()
    for k, v in summary.items():
        print(f"  {k:25s} = {v}")
