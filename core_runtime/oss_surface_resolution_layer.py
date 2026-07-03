"""OSSSurfaceResolutionLayer — multi-scale normalisation before ECDF computation.

Prevents rank saturation at 0 or 1 in low-volatility regimes by decompressing
the ECDF rank toward 0.5 when volatility is below the minimum threshold.
"""

import logging
import math
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton registry
# ---------------------------------------------------------------------------

_instances = {}


def OSSSurfaceResolutionLayer(instance_id="default"):
    """Singleton accessor — returns the same ``_OSSSurfaceResolutionLayer``
    for a given *instance_id*.

    Parameters
    ----------
    instance_id : str
        Unique identifier.  Callers sharing the same id share state.

    Returns
    -------
    _OSSSurfaceResolutionLayer
    """
    if instance_id not in _instances:
        _instances[instance_id] = _OSSSurfaceResolutionLayer(instance_id)
    return _instances[instance_id]


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------


class _OSSSurfaceResolutionLayer:
    """Multi-scale normalisation layer that prevents ECDF rank saturation.

    Maintains price buffers at three time scales (fast=10, medium=50,
    slow=200).  When volatility at all scales drops below *min_volatility*,
    the raw ECDF rank is decompressed toward 0.5 so that OSS surface lookups
    do not collapse.

    Parameters
    ----------
    instance_id : str
        Label used in logging.
    """

    def __init__(self, instance_id="default"):
        self._instance_id = instance_id

        # Configuration
        self._min_volatility = 0.0001
        self._smoothing_factor = 0.3

        # Three time scales: name -> buffer size
        self._scale_sizes = {
            "fast": 10,
            "medium": 50,
            "slow": 200,
        }

        # Per-symbol, per-scale price buffers
        self._buffers = defaultdict(
            lambda: {
                name: deque(maxlen=size)
                for name, size in self._scale_sizes.items()
            }
        )

        logger.debug("OSSSurfaceResolutionLayer(%r) initialised", instance_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed_price(self, symbol, price):
        """Feed a price observation and update multi-scale buffers.

        Parameters
        ----------
        symbol : str
            Instrument identifier (e.g. ``"EURUSD"``).
        price : float
            Observed price.
        """
        buffers = self._buffers[symbol]
        for name in self._scale_sizes:
            buffers[name].append(price)

    def get_normalized_rank(self, symbol, ecdf_rank):
        """Take a raw ECDF rank and return a de-saturated normalised rank.

        Parameters
        ----------
        symbol : str
            Instrument identifier.
        ecdf_rank : float
            Raw ECDF rank in [0, 1].

        Returns
        -------
        dict
            Keys:
            ``normalized_rank``    — de-saturated rank in [0, 1].
            ``scale_weights``      — dict mapping scale name to weight.
            ``is_saturated``       — True if raw rank is 0 or 1 and vol is low.
            ``effective_resolution`` — "TICK", "1M", or "5M".
            ``adjusted_rank``      — final rank after all adjustments.
        """
        buffers = self._buffers.get(symbol, {})
        ecdf_rank = float(ecdf_rank)

        # --------------------------------------------------------------
        # 1. Compute rolling mean and std at each scale
        # --------------------------------------------------------------
        scale_stats = {}
        for name in ["fast", "medium", "slow"]:
            buf = list(buffers.get(name, []))
            if len(buf) >= 2:
                mean = sum(buf) / len(buf)
                var = sum((p - mean) ** 2 for p in buf) / len(buf)
                std = math.sqrt(var)
            else:
                std = 0.0
            scale_stats[name] = std

        # --------------------------------------------------------------
        # 3. Detect LOW_VOL regime
        # --------------------------------------------------------------
        std_fast = scale_stats["fast"]
        std_medium = scale_stats["medium"]
        std_slow = scale_stats["slow"]

        low_vol = (
            std_fast < self._min_volatility
            and std_medium < self._min_volatility
            and std_slow < self._min_volatility
        )

        # --------------------------------------------------------------
        # 5. Scale weights
        # --------------------------------------------------------------
        total_std = std_fast + std_medium + std_slow
        if total_std > 0:
            weights = {
                "fast": std_fast / total_std,
                "medium": std_medium / total_std,
                "slow": std_slow / total_std,
            }
        else:
            weights = {"fast": 1.0 / 3, "medium": 1.0 / 3, "slow": 1.0 / 3}

        # --------------------------------------------------------------
        # 3. LOW_VOL regime — decompress ECDF rank
        # --------------------------------------------------------------
        if low_vol:
            effective_std = max(std_fast, std_medium, std_slow)
            decompress_factor = min(1.0, effective_std / self._min_volatility)
            normalized_rank = 0.5 + (ecdf_rank - 0.5) * decompress_factor
        else:
            # ----------------------------------------------------------
            # 4. Normal regime — apply smoothing
            # ----------------------------------------------------------
            normalized_rank = (
                self._smoothing_factor * ecdf_rank
                + (1.0 - self._smoothing_factor) * 0.5
            )

        # --------------------------------------------------------------
        # 6. Saturation flag
        # --------------------------------------------------------------
        is_saturated = (ecdf_rank <= 0.0 or ecdf_rank >= 1.0) and low_vol

        # --------------------------------------------------------------
        # 7. Effective resolution (scale with highest weight)
        # --------------------------------------------------------------
        max_scale = max(weights, key=weights.get)
        resolution_map = {"fast": "TICK", "medium": "1M", "slow": "5M"}
        effective_resolution = resolution_map[max_scale]

        # --------------------------------------------------------------
        # 8. Adjusted rank — clamped to [0, 1]
        # --------------------------------------------------------------
        adjusted_rank = max(0.0, min(1.0, normalized_rank))

        return {
            "normalized_rank": round(normalized_rank, 6),
            "scale_weights": {
                k: round(v, 6) for k, v in weights.items()
            },
            "is_saturated": is_saturated,
            "effective_resolution": effective_resolution,
            "adjusted_rank": round(adjusted_rank, 6),
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _selftest():
    """Feed synthetic prices in high- and low-vol regimes and verify
    that ranks are decompressed in low-vol."""
    import random

    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s | %(message)s",
    )
    logger.info("=" * 60)
    logger.info("OSSSurfaceResolutionLayer — Self Test")
    logger.info("=" * 60)

    random.seed(42)
    layer = OSSSurfaceResolutionLayer("selftest")
    symbol = "SYNTH_USD"

    # ------------------------------------------------------------------
    # Phase 1 — LOW VOLATILITY: tiny random changes
    # ------------------------------------------------------------------
    logger.info("--- Phase 1: Low Volatility ---")
    price = 1.10000
    for i in range(200):
        price += random.uniform(-0.00001, 0.00001)
        layer.feed_price(symbol, price)

    # Test extreme ECDF ranks — should be decompressed toward 0.5
    result_low_0 = layer.get_normalized_rank(symbol, 0.0)
    result_low_1 = layer.get_normalized_rank(symbol, 1.0)
    result_low_mid = layer.get_normalized_rank(symbol, 0.5)

    logger.info("  ecdf_rank=0.0  -> normalized=%.6f  saturated=%s",
                result_low_0["normalized_rank"], result_low_0["is_saturated"])
    logger.info("  ecdf_rank=1.0  -> normalized=%.6f  saturated=%s",
                result_low_1["normalized_rank"], result_low_1["is_saturated"])
    logger.info("  ecdf_rank=0.5  -> normalized=%.6f",
                result_low_mid["normalized_rank"])
    logger.info("  scale_weights: %s", result_low_0["scale_weights"])
    logger.info("  effective_resolution: %s", result_low_0["effective_resolution"])

    # Verify decompression (0 and 1 should be pulled toward 0.5)
    assert result_low_0["normalized_rank"] > 0.0, (
        f"Expected rank > 0 for low-vol decompression, got "
        f"{result_low_0['normalized_rank']}"
    )
    assert result_low_1["normalized_rank"] < 1.0, (
        f"Expected rank < 1 for low-vol decompression, got "
        f"{result_low_1['normalized_rank']}"
    )
    assert result_low_0["is_saturated"], (
        "Expected is_saturated=True for ecdf_rank=0.0 in low vol"
    )
    assert result_low_1["is_saturated"], (
        "Expected is_saturated=True for ecdf_rank=1.0 in low vol"
    )
    # Mid rank should stay at 0.5
    assert abs(result_low_mid["normalized_rank"] - 0.5) < 0.01, (
        f"Expected mid rank ~0.5, got {result_low_mid['normalized_rank']}"
    )
    logger.info("  LOW VOL decompression: PASS")

    # ------------------------------------------------------------------
    # Phase 2 — HIGH VOLATILITY: large random changes
    # ------------------------------------------------------------------
    logger.info("--- Phase 2: High Volatility ---")
    # Reset by creating a new symbol
    symbol_hv = "SYNTH_USD_HV"
    price = 1.10000
    for i in range(200):
        price += random.uniform(-0.01, 0.01)
        layer.feed_price(symbol_hv, price)

    result_high_0 = layer.get_normalized_rank(symbol_hv, 0.0)
    result_high_1 = layer.get_normalized_rank(symbol_hv, 1.0)

    logger.info("  ecdf_rank=0.0  -> normalized=%.6f  saturated=%s",
                result_high_0["normalized_rank"], result_high_0["is_saturated"])
    logger.info("  ecdf_rank=1.0  -> normalized=%.6f  saturated=%s",
                result_high_1["normalized_rank"], result_high_1["is_saturated"])
    logger.info("  scale_weights: %s", result_high_0["scale_weights"])
    logger.info("  effective_resolution: %s", result_high_0["effective_resolution"])

    # In high vol, smoothing is applied (ecdf_rank=0.0 -> 0.35 with
    # smoothing_factor=0.3). Rank should NOT be saturated.
    assert 0.3 <= result_high_0["normalized_rank"] <= 0.5, (
        f"Expected rank in [0.3, 0.5] in high vol, got "
        f"{result_high_0['normalized_rank']}"
    )
    assert 0.5 <= result_high_1["normalized_rank"] <= 0.7, (
        f"Expected rank in [0.5, 0.7] in high vol, got "
        f"{result_high_1['normalized_rank']}"
    )
    assert not result_high_0["is_saturated"], (
        "Expected is_saturated=False for ecdf_rank=0.0 in high vol"
    )
    # Low-vol decompression should pull ranks more toward 0.5 than
    # high-vol smoothing does.
    low_decompress = abs(result_low_0["normalized_rank"] - 0.0)
    high_smooth = abs(result_high_0["normalized_rank"] - 0.0)
    assert low_decompress > high_smooth, (
        f"Low-vol decompression ({low_decompress:.4f}) should be stronger "
        f"than high-vol smoothing ({high_smooth:.4f})"
    )
    logger.info("  HIGH VOL no-saturation: PASS")

    # ------------------------------------------------------------------
    # Phase 3 — Verify resolution and weights are well-formed
    # ------------------------------------------------------------------
    logger.info("--- Phase 3: Resolution and weights ---")
    res_low = result_low_0["effective_resolution"]
    res_high = result_high_0["effective_resolution"]
    logger.info("  Low-vol resolution: %s", res_low)
    logger.info("  High-vol resolution: %s", res_high)
    # In low-vol, slow scale should have the most weight (most history)
    low_weights = result_low_0["scale_weights"]
    assert low_weights["slow"] >= low_weights["medium"], (
        f"Expected slow >= medium in low vol, got {low_weights}"
    )
    # All weights should sum to ~1.0
    for w in [result_low_0["scale_weights"], result_high_0["scale_weights"]]:
        total = sum(w.values())
        assert abs(total - 1.0) < 0.001, f"Weights sum to {total}, expected 1.0"
    # Effective resolution should be one of the valid values
    assert res_low in ("TICK", "1M", "5M"), f"Invalid resolution: {res_low}"
    assert res_high in ("TICK", "1M", "5M"), f"Invalid resolution: {res_high}"
    logger.info("  Low-vol weights: %s", low_weights)
    logger.info("  High-vol weights: %s", result_high_0["scale_weights"])
    logger.info("  Resolution and weights: PASS")

    # ------------------------------------------------------------------
    # Phase 4 — Verify singleton
    # ------------------------------------------------------------------
    logger.info("--- Phase 4: Singleton ---")
    same = OSSSurfaceResolutionLayer("selftest")
    assert same is layer, "Singleton pattern broken!"
    logger.info("  Singleton: PASS")

    logger.info("=" * 60)
    logger.info(">>> ALL SELFTESTS PASSED <<<")
    logger.info("=" * 60)


if __name__ == "__main__":
    _selftest()
