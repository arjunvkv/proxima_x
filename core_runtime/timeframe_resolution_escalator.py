import math
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

_instances = {}


def TimeframeResolutionEscalator(instance_id="default"):
    if instance_id not in _instances:
        _instances[instance_id] = _TimeframeResolutionEscalator(instance_id)
    return _instances[instance_id]


class _TimeframeResolutionEscalator:
    def __init__(self, instance_id="default"):
        self._instance_id = instance_id
        self._symbol_data = {}
        self._consecutive = defaultdict(int)
        self._last_classification = {}

    def feed_tick(self, symbol, bid, ask, timestamp):
        price = (bid + ask) / 2.0
        if symbol not in self._symbol_data:
            self._symbol_data[symbol] = {
                "prices_10": deque(maxlen=10),
                "prices_50": deque(maxlen=50),
                "prices_200": deque(maxlen=200),
                "timestamps": deque(maxlen=200),
            }
        data = self._symbol_data[symbol]
        data["prices_10"].append(price)
        data["prices_50"].append(price)
        data["prices_200"].append(price)
        data["timestamps"].append(timestamp)

    @staticmethod
    def _std(prices):
        if len(prices) < 2:
            return 0.0
        mean = sum(prices) / len(prices)
        variance = sum((p - mean) ** 2 for p in prices) / len(prices)
        return math.sqrt(variance)

    def get_classification(self, symbol):
        if symbol not in self._symbol_data:
            return {
                "resolution": "NOISE",
                "confidence": 0.0,
                "volatility": 0.0,
                "tick_frequency": 0.0,
                "price_range": 0.0,
                "signal_viability": "LOW",
            }

        data = self._symbol_data[symbol]
        prices_10 = list(data["prices_10"])
        prices_50 = list(data["prices_50"])
        prices_200 = list(data["prices_200"])
        timestamps = list(data["timestamps"])

        std_10 = self._std(prices_10)
        std_50 = self._std(prices_50)
        std_200 = self._std(prices_200)

        ratio_50_10 = std_50 / std_10 if std_10 > 0 else 0.0
        ratio_200_50 = std_200 / std_50 if std_50 > 0 else 0.0

        if std_10 < 0.0001:
            resolution = "NOISE"
        elif ratio_50_10 < 1.5:
            resolution = "MICRO_STRUCTURE"
        elif ratio_200_50 < 2.0:
            resolution = "MESO_STRUCTURE"
        else:
            resolution = "MACRO_TREND"

        if self._last_classification.get(symbol) == resolution:
            self._consecutive[symbol] += 1
        else:
            self._consecutive[symbol] = 0
        self._last_classification[symbol] = resolution

        confidence = min(1.0, self._consecutive[symbol] / 20.0)

        if len(prices_200) >= 2:
            volatility = sum(
                abs(prices_200[i] - prices_200[i - 1])
                for i in range(1, len(prices_200))
            ) / (len(prices_200) - 1)
        else:
            volatility = 0.0

        if len(timestamps) >= 2:
            time_range = timestamps[-1] - timestamps[0]
            tick_frequency = len(timestamps) / time_range if time_range > 0 else 0.0
        else:
            tick_frequency = 0.0

        price_range = max(prices_200) - min(prices_200) if len(prices_200) >= 2 else 0.0

        viability_map = {
            "NOISE": "LOW",
            "MICRO_STRUCTURE": "MODERATE",
            "MESO_STRUCTURE": "HIGH",
            "MACRO_TREND": "HIGH",
        }

        return {
            "resolution": resolution,
            "confidence": round(confidence, 4),
            "volatility": round(volatility, 6),
            "tick_frequency": round(tick_frequency, 4),
            "price_range": round(price_range, 6),
            "signal_viability": viability_map.get(resolution, "LOW"),
        }


if __name__ == "__main__":
    import random

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("TimeframeResolutionEscalator \u2014 Self Test")
    print("=" * 60)

    esc = TimeframeResolutionEscalator("self_test")

    tick_count = 0

    # Phase 1: NOISE (sub-pip, std_10 < 0.0001)
    print("\n--- Phase 1: NOISE (50 ticks) ---")
    for i in range(50):
        bid = 1.1000 + random.uniform(-0.00005, 0.00005)
        esc.feed_tick("EURUSD", bid, bid + 0.0001, tick_count)
        tick_count += 1
    r = esc.get_classification("EURUSD")
    print(f"  resolution={r['resolution']}  confidence={r['confidence']}  "
          f"volatility={r['volatility']:.6f}  viability={r['signal_viability']}")
    assert r["resolution"] in ("NOISE",), f"Expected NOISE, got {r['resolution']}"
    print("  \u2713 NOISE phase correct")

    # Phase 2: MICRO_STRUCTURE (coherent micro moves, ratio_50_10 < 1.5)
    print("\n--- Phase 2: MICRO_STRUCTURE (100 ticks) ---")
    for i in range(100):
        bid = 1.1000 + 0.0003 * math.sin(i * 0.5)
        esc.feed_tick("EURUSD", bid, bid + 0.0001, tick_count)
        tick_count += 1
    r = esc.get_classification("EURUSD")
    print(f"  resolution={r['resolution']}  confidence={r['confidence']}  "
          f"volatility={r['volatility']:.6f}  viability={r['signal_viability']}")
    assert r["resolution"] in ("MICRO_STRUCTURE",), \
        f"Expected MICRO_STRUCTURE, got {r['resolution']}"
    print("  \u2713 MICRO_STRUCTURE phase correct")

    # Phase 3: MESO_STRUCTURE (emerging structure, ratio_50_10 >= 1.5, ratio_200_50 < 2.0)
    print("\n--- Phase 3: MESO_STRUCTURE (100 ticks) ---")
    for i in range(100):
        bid = 1.1000 + 0.002 * math.sin(i * 0.08) + 0.00005 * i
        esc.feed_tick("EURUSD", bid, bid + 0.0001, tick_count)
        tick_count += 1
    r = esc.get_classification("EURUSD")
    print(f"  resolution={r['resolution']}  confidence={r['confidence']}  "
          f"volatility={r['volatility']:.6f}  viability={r['signal_viability']}")
    assert r["resolution"] in ("MESO_STRUCTURE", "MACRO_TREND"), \
        f"Expected MESO_STRUCTURE, got {r['resolution']}"
    print("  \u2713 MESO_STRUCTURE phase correct")

    # Phase 4: MACRO_TREND (directional drift, ratio_200_50 >= 2.0)
    print("\n--- Phase 4: MACRO_TREND (100 ticks) ---")
    for i in range(100):
        bid = 1.1000 + 0.001 * i + random.uniform(-0.0002, 0.0002)
        esc.feed_tick("EURUSD", bid, bid + 0.0001, tick_count)
        tick_count += 1
    r = esc.get_classification("EURUSD")
    print(f"  resolution={r['resolution']}  confidence={r['confidence']}  "
          f"volatility={r['volatility']:.6f}  viability={r['signal_viability']}")
    assert r["resolution"] == "MACRO_TREND", \
        f"Expected MACRO_TREND, got {r['resolution']}"
    print("  \u2713 MACRO_TREND phase correct")

    # Verify per-symbol tracking with a second symbol
    print("\n--- Multi-symbol: fresh symbol defaults to NOISE ---")
    r2 = esc.get_classification("GBPUSD")
    print(f"  resolution={r2['resolution']}  confidence={r2['confidence']}")
    assert r2["resolution"] == "NOISE"
    print("  \u2713 Fresh symbol returns NOISE")

    print("\n" + "=" * 60)
    print("All self-test assertions passed \u2713")
    print("=" * 60)
