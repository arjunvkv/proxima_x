"""
Signal Entropy Timeframe Map — measures signal entropy across multiple timeframes
to find the optimal resolution band for prediction. Uses approximate entropy (ApEn)
and sample entropy approximations.
"""

import math
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

_instances = {}


def SignalEntropyTimeframeMap(instance_id="default"):
    if instance_id not in _instances:
        _instances[instance_id] = _SignalEntropyTimeframeMap(instance_id)
    return _instances[instance_id]


class _SignalEntropyTimeframeMap:
    def __init__(self, instance_id="default"):
        self._instance_id = instance_id
        self._price_history = defaultdict(lambda: {"tick_counter": 0, "prices": {tf: [] for tf in ["TICK", "1M", "5M", "15M", "1H"]}})
        self._max_per_symbol = 500
        self._timeframe_labels = ["TICK", "1M", "5M", "15M", "1H"]
        self._entropy_windows = {"TICK": 20, "1M": 30, "5M": 40, "15M": 50, "1H": 60}
        self._subsample_rates = {"TICK": 1, "1M": 60, "5M": 300, "15M": 900, "1H": 3600}

    def feed_price(self, symbol, price, timeframe="TICK"):
        data = self._price_history.setdefault(symbol, {"tick_counter": 0, "prices": {tf: [] for tf in self._timeframe_labels}})
        data["tick_counter"] += 1
        counter = data["tick_counter"]

        if timeframe != "TICK":
            rate = self._subsample_rates.get(timeframe, 1)
            if counter % rate != 0:
                return

        for tf in self._timeframe_labels:
            rate = self._subsample_rates[tf]
            if counter % rate == 0:
                series = data["prices"][tf]
                series.append(price)
                if len(series) > self._max_per_symbol:
                    series.pop(0)

        logger.debug("feed_price %s %s price=%.6f counter=%d", symbol, timeframe, price, counter)

    @staticmethod
    def _std(values):
        n = len(values)
        if n < 2:
            return 0.0
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        return math.sqrt(variance)

    @staticmethod
    def _approximate_entropy(series, m=2, r_factor=0.2):
        n = len(series)
        if n < m + 2:
            return 1.0

        r = r_factor * _SignalEntropyTimeframeMap._std(series)
        if r == 0.0:
            return 0.0

        def _count_matches(length):
            count = 0
            k = n - length + 1
            for i in range(k):
                for j in range(k):
                    if i == j:
                        continue
                    max_diff = 0.0
                    for t in range(length):
                        diff = abs(series[i + t] - series[j + t])
                        if diff > max_diff:
                            max_diff = diff
                    if max_diff <= r:
                        count += 1
            return count

        count_m = _count_matches(m)
        count_m1 = _count_matches(m + 1)

        if count_m == 0:
            return 1.0

        phi_m = math.log(count_m / (n - m + 1))
        phi_m1 = math.log(count_m1 / (n - m + 1 - 1)) if count_m1 > 0 else 0.0

        if count_m1 == 0:
            return 1.0

        ap_en = phi_m - phi_m1
        return max(0.0, min(2.0, ap_en))

    def get_entropy_map(self, symbol):
        data = self._price_history.get(symbol)
        if data is None:
            return {
                "entropies": {},
                "optimal_timeframe": "TICK",
                "predictability_horizon": 0,
                "signal_viability": "LOW",
                "entropy_gradient": [],
            }

        entropies = {}
        for tf in self._timeframe_labels:
            series = data["prices"][tf]
            window = self._entropy_windows[tf]
            if len(series) >= window:
                segment = series[-window:]
                entropies[tf] = round(self._approximate_entropy(segment), 6)

        if not entropies:
            return {
                "entropies": {},
                "optimal_timeframe": "TICK",
                "predictability_horizon": 0,
                "signal_viability": "LOW",
                "entropy_gradient": [],
            }

        optimal_tf = min(entropies, key=entropies.get)
        min_entropy = entropies[optimal_tf]

        if min_entropy > 0:
            horizon = int(1.0 / min_entropy)
        else:
            horizon = self._max_per_symbol // 2
        horizon = min(horizon, self._max_per_symbol // 2)

        if min_entropy < 0.5:
            viability = "HIGH"
        elif min_entropy < 1.0:
            viability = "MODERATE"
        else:
            viability = "LOW"

        tf_order = {tf: i for i, tf in enumerate(self._timeframe_labels)}
        gradient = sorted([[tf, entropies[tf]] for tf in entropies], key=lambda x: tf_order[x[0]])

        return {
            "entropies": entropies,
            "optimal_timeframe": optimal_tf,
            "predictability_horizon": horizon,
            "signal_viability": viability,
            "entropy_gradient": gradient,
        }


if __name__ == "__main__":
    import random

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("Signal Entropy Timeframe Map — Self Test")
    print("=" * 60)

    random.seed(42)

    # Scenario 1: Random walk (high entropy)
    print("\n--- Scenario 1: Random Walk (high entropy) ---")
    ent1 = SignalEntropyTimeframeMap("test_random")
    price = 100.0
    for i in range(5000):
        price += random.gauss(0, 0.1)
        ent1.feed_price("RANDOM", price)
    r1 = ent1.get_entropy_map("RANDOM")
    print(f"  Entropies: {r1['entropies']}")
    print(f"  Optimal timeframe: {r1['optimal_timeframe']}")
    print(f"  Predictability horizon: {r1['predictability_horizon']}")
    print(f"  Signal viability: {r1['signal_viability']}")
    print(f"  Entropy gradient: {r1['entropy_gradient']}")
    avg_entropy = sum(r1['entropies'].values()) / len(r1['entropies']) if r1['entropies'] else 0
    assert avg_entropy > 0.3, f"Expected high entropy for random walk, got avg={avg_entropy}"
    print("  >>> PASS")

    # Scenario 2: Trending series (low entropy)
    print("\n--- Scenario 2: Trending Series (low entropy) ---")
    ent2 = SignalEntropyTimeframeMap("test_trend")
    price = 100.0
    for i in range(5000):
        price += 0.02 + random.gauss(0, 0.01)
        ent2.feed_price("TREND", price)
    r2 = ent2.get_entropy_map("TREND")
    print(f"  Entropies: {r2['entropies']}")
    print(f"  Optimal timeframe: {r2['optimal_timeframe']}")
    print(f"  Predictability horizon: {r2['predictability_horizon']}")
    print(f"  Signal viability: {r2['signal_viability']}")
    print(f"  Entropy gradient: {r2['entropy_gradient']}")
    assert r2['signal_viability'] in ("HIGH", "MODERATE"), \
        f"Expected HIGH or MODERATE viability for trending, got {r2['signal_viability']}"
    print("  >>> PASS")

    # Scenario 3: No data edge case
    print("\n--- Scenario 3: No data ---")
    r3 = ent2.get_entropy_map("NONEXISTENT")
    print(f"  Entropies: {r3['entropies']}")
    print(f"  Optimal timeframe: {r3['optimal_timeframe']}")
    print(f"  Signal viability: {r3['signal_viability']}")
    assert r3['entropies'] == {}, f"Expected empty entropies, got {r3['entropies']}"
    assert r3['signal_viability'] == "LOW", f"Expected LOW viability, got {r3['signal_viability']}"
    print("  >>> PASS")

    # Singleton test
    print("\n--- Singleton test ---")
    same = SignalEntropyTimeframeMap("test_random")
    assert same is ent1, "Singleton should return the same instance"
    print("  >>> PASS")

    print("\n" + "=" * 60)
    print("All self-tests PASSED.")
    print("=" * 60)
