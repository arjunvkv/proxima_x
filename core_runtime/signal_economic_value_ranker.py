"""
Signal Economic Value Ranker — rank signals by expected economic value instead
of treating them as binary.

Expected value formula::

    expected_value = direction * abs(confidence) * regime_multiplier
                     - spread_cost - latency_penalty

This replaces naive signal-vs-no-signal thinking with a continuous valuation
that accounts for market regime, transaction costs, and latency.
"""

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton registry
# ---------------------------------------------------------------------------

_instances = {}


def SignalEconomicValueRanker(instance_id="default"):
    """Accessor / singleton factory for ``_SignalEconomicValueRanker``."""
    if instance_id not in _instances:
        _instances[instance_id] = _SignalEconomicValueRanker(instance_id)
    return _instances[instance_id]


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

REGIME_MULTIPLIERS = {
    "TRENDING": 1.2,
    "MEAN_REVERTING": 0.8,
    "VOLATILITY_SPIKE": 0.5,
    "LOW_LIQUIDITY": 0.3,
    "UNKNOWN": 0.6,
}

_DEFAULT_SPREAD_COST_FN = lambda spread: spread * 100.0   # noqa: E731
_DEFAULT_LATENCY_PENALTY_FN = lambda ms: ms * 0.001       # noqa: E731


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------

class _SignalEconomicValueRanker:
    """Rank signals by expected economic value.

    Parameters
    ----------
    instance_id : str
        Arbitrary label for this instance (used in logging).
    """

    def __init__(self, instance_id="default"):
        self._instance_id = instance_id

        self._regime_multipliers = dict(REGIME_MULTIPLIERS)
        self._spread_cost_fn = _DEFAULT_SPREAD_COST_FN
        self._latency_penalty_fn = _DEFAULT_LATENCY_PENALTY_FN

        logger.info("SignalEconomicValueRanker(%s) initialised", instance_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_value(self, signal, confidence, regime, spread, latency_ms=0):
        """Compute the expected economic value of a single signal.

        Parameters
        ----------
        signal : int
            Signal direction: ``+1`` (buy) or ``-1`` (sell).
        confidence : float
            Signal confidence in the range ``[0.0, 1.0]``.
        regime : str
            Market regime label (e.g. ``"TRENDING"``).  Case-insensitive
            lookup; falls back to ``"UNKNOWN"`` if not recognised.
        spread : float
            Current bid-ask spread in price units.
        latency_ms : float
            Round-trip latency in milliseconds (default ``0``).

        Returns
        -------
        float
            Expected economic value.  Positive means the signal is worth
            acting on; negative or zero means it is not.
        """
        regime_upper = regime.upper() if isinstance(regime, str) else "UNKNOWN"
        regime_mult = self._regime_multipliers.get(regime_upper, 0.6)

        spread_cost = self._spread_cost_fn(spread)
        latency_penalty = self._latency_penalty_fn(latency_ms)

        expected = (
            signal * abs(confidence) * regime_mult
            - spread_cost
            - latency_penalty
        )
        return expected

    def compute_oss_value(self, oss_signal, oss_confidence, regime, spread,
                          latency_ms=0):
        """Convenience wrapper — compute expected value for an OSS signal.

        Parameters
        ----------
        oss_signal : int
            OSS signal direction: ``+1`` or ``-1``.
        oss_confidence : float
            OSS signal confidence ``[0.0, 1.0]``.

        Returns
        -------
        float
        """
        return self.compute_value(
            oss_signal, oss_confidence, regime, spread, latency_ms,
        )

    def compute_alt_value(self, alt_signal, alt_confidence, regime, spread,
                          latency_ms=0):
        """Convenience wrapper — compute expected value for an ALT signal.

        Parameters
        ----------
        alt_signal : int
            ALT signal direction: ``+1`` or ``-1``.
        alt_confidence : float
            ALT signal confidence ``[0.0, 1.0]``.

        Returns
        -------
        float
        """
        return self.compute_value(
            alt_signal, alt_confidence, regime, spread, latency_ms,
        )

    def rank(self, oss_signal, oss_confidence, alt_signal, alt_confidence,
             regime, spread, latency_ms=0):
        """Rank OSS versus ALT signals and return a detailed comparison dict.

        Parameters
        ----------
        oss_signal : int
            OSS signal direction: ``+1`` or ``-1``.
        oss_confidence : float
            OSS signal confidence ``[0.0, 1.0]``.
        alt_signal : int
            ALT signal direction: ``+1`` or ``-1``.
        alt_confidence : float
            ALT signal confidence ``[0.0, 1.0]``.
        regime : str
            Market regime label.
        spread : float
            Bid-ask spread in price units.
        latency_ms : float
            Round-trip latency in milliseconds (default ``0``).

        Returns
        -------
        dict
            Keys::

                oss_value           float
                alt_value           float
                best_source         str   ("OSS" | "ALT" | "NONE")
                best_value          float
                spread_cost         float
                latency_penalty     float
                regime_multiplier   float
                executable          bool  (True if best_value > 0)
        """
        oss_val = self.compute_oss_value(
            oss_signal, oss_confidence, regime, spread, latency_ms,
        )
        alt_val = self.compute_alt_value(
            alt_signal, alt_confidence, regime, spread, latency_ms,
        )

        regime_upper = regime.upper() if isinstance(regime, str) else "UNKNOWN"
        regime_mult = self._regime_multipliers.get(regime_upper, 0.6)
        spread_cost = self._spread_cost_fn(spread)
        latency_penalty = self._latency_penalty_fn(latency_ms)

        # Determine best source
        if oss_val > 0 and oss_val >= alt_val:
            best_source = "OSS"
            best_value = oss_val
        elif alt_val > 0 and alt_val >= oss_val:
            best_source = "ALT"
            best_value = alt_val
        elif oss_val <= 0 and alt_val <= 0:
            best_source = "NONE"
            best_value = max(oss_val, alt_val)
        else:
            # Both ≤ 0 — should be caught above; fallback
            best_source = "NONE"
            best_value = max(oss_val, alt_val)

        executable = best_value > 0

        return {
            "oss_value": oss_val,
            "alt_value": alt_val,
            "best_source": best_source,
            "best_value": best_value,
            "spread_cost": spread_cost,
            "latency_penalty": latency_penalty,
            "regime_multiplier": regime_mult,
            "executable": executable,
        }

    def set_regime_multiplier(self, regime, multiplier):
        """Override the default multiplier for a given regime.

        Parameters
        ----------
        regime : str
            Regime label (case-insensitive).
        multiplier : float
            New multiplier value.
        """
        self._regime_multipliers[regime.upper()] = multiplier
        logger.info(
            "SignalEconomicValueRanker(%s): regime %s multiplier set to %.4f",
            self._instance_id, regime.upper(), multiplier,
        )

    def set_spread_cost_fn(self, fn):
        """Set a custom spread cost function.

        Parameters
        ----------
        fn : callable
            Callable that accepts ``spread`` (float) and returns a cost
            (float).  Default: ``cost = spread * 100``.
        """
        self._spread_cost_fn = fn
        logger.info(
            "SignalEconomicValueRanker(%s): spread cost function replaced",
            self._instance_id,
        )

    def set_latency_penalty_fn(self, fn):
        """Set a custom latency penalty function.

        Parameters
        ----------
        fn : callable
            Callable that accepts ``latency_ms`` (float) and returns a
            penalty (float).  Default: ``penalty = ms * 0.001``.
        """
        self._latency_penalty_fn = fn
        logger.info(
            "SignalEconomicValueRanker(%s): latency penalty function replaced",
            self._instance_id,
        )

    def reset(self):
        """Reset all custom settings back to defaults."""
        self._regime_multipliers = dict(REGIME_MULTIPLIERS)
        self._spread_cost_fn = _DEFAULT_SPREAD_COST_FN
        self._latency_penalty_fn = _DEFAULT_LATENCY_PENALTY_FN
        logger.info("SignalEconomicValueRanker(%s) reset to defaults",
                     self._instance_id)


# ===================================================================
# Self-test
# ===================================================================

def _selftest():
    """Run a quick sanity check to verify the module works correctly."""
    ranker = SignalEconomicValueRanker("selftest")

    # ---- 1. Strong trending buy signal, tight spread, no latency ----
    val = ranker.compute_value(+1, 0.9, "TRENDING", 0.0001, latency_ms=0)
    expected = 1.0 * 0.9 * 1.2 - (0.0001 * 100.0) - 0.0
    assert abs(val - expected) < 1e-12, (
        f"Expected {expected:.6f}, got {val:.6f}"
    )
    print(f"[SELFTEST] Trending buy (tight spread):  value={val:.6f}  "
          f"(expected {expected:.6f})  \u2713")

    # ---- 2. Negative (sell) signal ----
    val = ranker.compute_value(-1, 0.8, "TRENDING", 0.0002, latency_ms=0)
    expected = -1.0 * 0.8 * 1.2 - (0.0002 * 100.0) - 0.0
    assert abs(val - expected) < 1e-12, (
        f"Expected {expected:.6f}, got {val:.6f}"
    )
    print(f"[SELFTEST] Trending sell (moderate spread):  value={val:.6f}  "
          f"(expected {expected:.6f})  \u2713")

    # ---- 3. High latency destroys value ----
    val_low_lat = ranker.compute_value(+1, 0.5, "MEAN_REVERTING", 0.0001,
                                        latency_ms=0)
    val_high_lat = ranker.compute_value(+1, 0.5, "MEAN_REVERTING", 0.0001,
                                         latency_ms=500)
    assert val_high_lat < val_low_lat, (
        f"High latency should reduce value: {val_high_lat} >= {val_low_lat}"
    )
    print(f"[SELFTEST] Latency penalty:  "
          f"0ms={val_low_lat:.6f}  500ms={val_high_lat:.6f}  \u2713")

    # ---- 4. Volatility spike regime reduces value ----
    val_trend = ranker.compute_value(+1, 1.0, "TRENDING", 0.0005, latency_ms=0)
    val_vol = ranker.compute_value(+1, 1.0, "VOLATILITY_SPIKE", 0.0005,
                                    latency_ms=0)
    assert val_vol < val_trend, (
        f"Volatility spike regime should reduce value: {val_vol} >= {val_trend}"
    )
    print(f"[SELFTEST] Volatility spike vs trending:  "
          f"{val_vol:.6f} < {val_trend:.6f}  \u2713")

    # ---- 5. Rank — OSS wins ----
    r = ranker.rank(+1, 0.9, -1, 0.3, "TRENDING", 0.0001, latency_ms=0)
    assert r["best_source"] == "OSS", (
        f"Expected OSS to win, got {r['best_source']}"
    )
    assert r["executable"] is True
    assert r["oss_value"] > r["alt_value"]
    print(f"[SELFTEST] Rank — OSS wins:  OSS={r['oss_value']:.6f}  "
          f"ALT={r['alt_value']:.6f}  best={r['best_source']}  \u2713")

    # ---- 6. Rank — ALT wins ----
    r = ranker.rank(+1, 0.3, +1, 0.9, "TRENDING", 0.0001, latency_ms=0)
    assert r["best_source"] == "ALT", (
        f"Expected ALT to win, got {r['best_source']}"
    )
    assert r["executable"] is True
    print(f"[SELFTEST] Rank — ALT wins:  OSS={r['oss_value']:.6f}  "
          f"ALT={r['alt_value']:.6f}  best={r['best_source']}  \u2713")

    # ---- 7. Rank — neither executable (spread too wide) ----
    r = ranker.rank(+1, 0.3, -1, 0.2, "LOW_LIQUIDITY", 0.05, latency_ms=0)
    assert r["executable"] is False, (
        f"Expected not executable with wide spread, got executable=True"
    )
    assert r["best_source"] == "NONE"
    print(f"[SELFTEST] Rank — not executable (wide spread):  "
          f"OSS={r['oss_value']:.6f}  ALT={r['alt_value']:.6f}  "
          f"executable={r['executable']}  \u2713")

    # ---- 8. set_regime_multiplier override ----
    ranker.set_regime_multiplier("TRENDING", 2.0)
    val_after = ranker.compute_value(+1, 0.5, "TRENDING", 0.0001, latency_ms=0)
    expected_after = 1.0 * 0.5 * 2.0 - (0.0001 * 100.0) - 0.0
    assert abs(val_after - expected_after) < 1e-12, (
        f"Expected {expected_after:.6f} after override, got {val_after:.6f}"
    )
    print(f"[SELFTEST] Regime multiplier override:  value={val_after:.6f}  "
          f"(expected {expected_after:.6f})  \u2713")

    # ---- 9. set_spread_cost_fn override ----
    ranker.set_spread_cost_fn(lambda s: s * 50.0)
    val_new_cost = ranker.compute_value(+1, 1.0, "TRENDING", 0.01, latency_ms=0)
    # multiplier is still 2.0 from step 8
    expected_new = 1.0 * 1.0 * 2.0 - (0.01 * 50.0) - 0.0
    assert abs(val_new_cost - expected_new) < 1e-12, (
        f"Expected {expected_new:.6f}, got {val_new_cost:.6f}"
    )
    print(f"[SELFTEST] Spread cost fn override:  value={val_new_cost:.6f}  "
          f"(expected {expected_new:.6f})  \u2713")

    # ---- 10. reset ----
    ranker.reset()
    # After reset, multiplier should be back to 1.2
    val_reset = ranker.compute_value(+1, 0.5, "TRENDING", 0.0001, latency_ms=0)
    expected_reset = 1.0 * 0.5 * 1.2 - (0.0001 * 100.0) - 0.0
    assert abs(val_reset - expected_reset) < 1e-12, (
        f"Expected {expected_reset:.6f} after reset, got {val_reset:.6f}"
    )
    print(f"[SELFTEST] Reset:  value={val_reset:.6f}  "
          f"(expected {expected_reset:.6f})  \u2713")

    # ---- 11. Unknown regime falls back to 0.6 ----
    val_unknown = ranker.compute_value(+1, 1.0, "BOGUS_REGIME", 0.0001,
                                        latency_ms=0)
    expected_unknown = 1.0 * 1.0 * 0.6 - (0.0001 * 100.0) - 0.0
    assert abs(val_unknown - expected_unknown) < 1e-12, (
        f"Expected {expected_unknown:.6f} for unknown regime, "
        f"got {val_unknown:.6f}"
    )
    print(f"[SELFTEST] Unknown regime fallback:  value={val_unknown:.6f}  "
          f"(expected {expected_unknown:.6f})  \u2713")

    # ---- 12. Compute OSS/ALT convenience wrappers ----
    oss = ranker.compute_oss_value(+1, 0.7, "TRENDING", 0.0002, latency_ms=10)
    alt = ranker.compute_alt_value(-1, 0.6, "TRENDING", 0.0002, latency_ms=10)
    expected_oss = 1.0 * 0.7 * 1.2 - (0.0002 * 100.0) - (10 * 0.001)
    expected_alt = -1.0 * 0.6 * 1.2 - (0.0002 * 100.0) - (10 * 0.001)
    assert abs(oss - expected_oss) < 1e-12
    assert abs(alt - expected_alt) < 1e-12
    print(f"[SELFTEST] Convenience wrappers:  OSS={oss:.6f}  ALT={alt:.6f}  "
          f"\u2713")

    # ---- 13. Rank dict structure ----
    r = ranker.rank(+1, 0.8, -1, 0.7, "TRENDING", 0.0001, latency_ms=5)
    expected_keys = {
        "oss_value", "alt_value", "best_source", "best_value",
        "spread_cost", "latency_penalty", "regime_multiplier", "executable",
    }
    assert set(r.keys()) == expected_keys, (
        f"Key mismatch: extra={set(r.keys()) - expected_keys}, "
        f"missing={expected_keys - set(r.keys())}"
    )
    print(f"[SELFTEST] Rank dict structure correct  \u2713")

    print("\nAll self-tests passed.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _selftest()
