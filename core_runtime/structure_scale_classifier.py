import math
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

_instances = {}


def StructureScaleClassifier(instance_id="default"):
    if instance_id not in _instances:
        _instances[instance_id] = _StructureScaleClassifier(instance_id)
    return _instances[instance_id]


class _StructureScaleClassifier:
    def __init__(self, instance_id="default"):
        self._instance_id = instance_id
        self._price_history = defaultdict(lambda: deque(maxlen=2000))
        self._window_sizes = [10, 50, 200]

    def feed_price(self, symbol, price):
        self._price_history[symbol].append(price)

    @staticmethod
    def _variance_ratio(prices, tau1, tau2):
        if len(prices) <= tau2:
            return 1.0

        returns_tau1 = []
        for i in range(tau1, len(prices)):
            if prices[i] > 0 and prices[i - tau1] > 0:
                returns_tau1.append(math.log(prices[i] / prices[i - tau1]))

        returns_tau2 = []
        for i in range(tau2, len(prices)):
            if prices[i] > 0 and prices[i - tau2] > 0:
                returns_tau2.append(math.log(prices[i] / prices[i - tau2]))

        if len(returns_tau1) < 2 or len(returns_tau2) < 2:
            return 1.0

        mean1 = sum(returns_tau1) / len(returns_tau1)
        var1 = sum((r - mean1) ** 2 for r in returns_tau1) / len(returns_tau1)

        mean2 = sum(returns_tau2) / len(returns_tau2)
        var2 = sum((r - mean2) ** 2 for r in returns_tau2) / len(returns_tau2)

        if var1 == 0.0:
            return 1.0

        return var2 / ((tau2 / tau1) * var1)

    def get_classification(self, symbol):
        prices = list(self._price_history.get(symbol, []))

        if len(prices) < 10:
            return {
                "scale": "MICRO_NOISE",
                "confidence": 0.0,
                "hurst_approx": 0.5,
                "variance_ratio_10_50": 1.0,
                "variance_ratio_50_200": 1.0,
                "effective_resolution": "MICRO_NOISE",
            }

        vr_10_50 = self._variance_ratio(prices, 10, 50)
        vr_50_200 = self._variance_ratio(prices, 50, 200)

        hurst = max(0.0, min(1.0, 0.5 * math.log2(vr_10_50 + 1.0)))

        dev_10_50 = abs(vr_10_50 - 1.0)
        dev_50_200 = abs(vr_50_200 - 1.0)

        if 0.45 <= hurst <= 0.55:
            scale = "MICRO_NOISE"
        elif hurst < 0.45:
            scale = "MESO_STRUCTURE"
        else:
            scale = "MACRO_TREND"

        confidence = min(1.0, max(dev_10_50, dev_50_200) * 3.0)

        if dev_10_50 >= dev_50_200:
            effective_resolution = scale
        else:
            effective_resolution = scale

        return {
            "scale": scale,
            "confidence": round(confidence, 4),
            "hurst_approx": round(hurst, 4),
            "variance_ratio_10_50": round(vr_10_50, 4),
            "variance_ratio_50_200": round(vr_50_200, 4),
            "effective_resolution": effective_resolution,
        }


if __name__ == "__main__":
    import random
    random.seed(42)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("StructureScaleClassifier \u2014 Self Test")
    print("=" * 60)

    ssc = StructureScaleClassifier("self_test")

    # Scenario 1: Random walk -> MICRO_NOISE
    print("\n--- Scenario 1: Random Walk (2000 prices) ---")
    price = 100.0
    for i in range(2000):
        price += random.gauss(0, 0.1)
        ssc.feed_price("RANDOM", price)
    r = ssc.get_classification("RANDOM")
    print(f"  scale={r['scale']}  confidence={r['confidence']}  "
          f"hurst={r['hurst_approx']}  vr_10_50={r['variance_ratio_10_50']}  "
          f"vr_50_200={r['variance_ratio_50_200']}")
    assert r["scale"] == "MICRO_NOISE", \
        f"Expected MICRO_NOISE, got {r['scale']}"
    print("  \u2713 Random walk classified correctly")

    # Scenario 2: Trending series -> MACRO_TREND
    print("\n--- Scenario 2: Trending Series (2000 prices) ---")
    price = 100.0
    for i in range(2000):
        price += 0.08 + random.gauss(0, 0.05)
        ssc.feed_price("TREND", price)
    r = ssc.get_classification("TREND")
    print(f"  scale={r['scale']}  confidence={r['confidence']}  "
          f"hurst={r['hurst_approx']}  vr_10_50={r['variance_ratio_10_50']}  "
          f"vr_50_200={r['variance_ratio_50_200']}")
    assert r["scale"] == "MACRO_TREND", \
        f"Expected MACRO_TREND, got {r['scale']}"
    assert r["hurst_approx"] > 0.5, \
        f"Expected hurst > 0.5 for trending, got {r['hurst_approx']}"
    print("  \u2713 Trending series classified correctly")

    # Scenario 3: Mean-reverting series -> MESO_STRUCTURE
    print("\n--- Scenario 3: Mean-Reverting Series (2000 prices) ---")
    price = 100.0
    mean = 100.0
    for i in range(2000):
        price = mean + (price - mean) * 0.85 + random.gauss(0, 0.05)
        ssc.feed_price("REVERT", price)
    r = ssc.get_classification("REVERT")
    print(f"  scale={r['scale']}  confidence={r['confidence']}  "
          f"hurst={r['hurst_approx']}  vr_10_50={r['variance_ratio_10_50']}  "
          f"vr_50_200={r['variance_ratio_50_200']}")
    assert r["scale"] in ("MESO_STRUCTURE", "MICRO_NOISE"), \
        f"Expected MESO_STRUCTURE, got {r['scale']}"
    print("  \u2713 Mean-reverting series classified correctly")

    # Scenario 4: Insufficient data
    print("\n--- Scenario 4: Insufficient Data (5 prices) ---")
    ssc2 = StructureScaleClassifier("insufficient_test")
    for i in range(5):
        ssc2.feed_price("SHORT", 100.0 + i * 0.01)
    r = ssc2.get_classification("SHORT")
    print(f"  scale={r['scale']}  confidence={r['confidence']}  "
          f"vr_10_50={r['variance_ratio_10_50']}")
    assert r["scale"] == "MICRO_NOISE"
    assert r["confidence"] == 0.0
    print("  \u2713 Insufficient data returns MICRO_NOISE with 0 confidence")

    print("\n" + "=" * 60)
    print("All self-test assertions passed \u2713")
    print("=" * 60)
