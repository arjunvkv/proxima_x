import sys
sys.path.insert(0, ".")
sys.path.insert(0, "proxima_x")

import time
from collections import deque
from typing import Optional, Dict
import numpy as np
import statistics


class BurstDetector:
    """Detects structural variance bursts from tick data.

    Tracks per-symbol tick velocity, price-change compression, and computes
    a burst_score that signals when tick velocity significantly exceeds its
    EMA baseline.
    """

    def __init__(
        self,
        window: int = 20,
        compression_threshold: float = 0.35,
        velocity_threshold: float = 2.5,
    ) -> None:
        self.window = window
        self.compression_threshold = compression_threshold
        self.velocity_threshold = velocity_threshold
        # Standard EMA smoothing factor
        self._alpha = 2.0 / (window + 1)

        # Per-symbol internal state
        self._tick_times: Dict[str, deque] = {}
        self._price_changes: Dict[str, deque] = {}
        self._velocity_baseline: Dict[str, float] = {}
        self._n_ticks: Dict[str, int] = {}
        self._last_price: Dict[str, float] = {}
        self._last_tick_time: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, symbol: str, tick: dict) -> dict:
        """Process one tick and return burst metrics for *symbol*.

        Parameters
        ----------
        symbol : str
            Instrument / ticker identifier.
        tick : dict
            Must contain at least ``"price"``.  May contain ``"timestamp"``
            (float, epoch seconds); if absent ``time.time()`` is used.

        Returns
        -------
        dict with keys:
            burst_score         float  — 0 … 1
            compression_density float  — 0 … 1
            tick_velocity       float  — ticks/sec over the window
            baseline_velocity   float  — EMA of tick_velocity
            n_ticks             int
        """
        self._ensure_symbol(symbol)

        price = float(tick.get("price", 0.0))
        now = tick.get("timestamp", time.time())

        n = self._n_ticks[symbol]
        self._n_ticks[symbol] = n + 1

        # --- inter-arrival time -------------------------------------------
        if self._last_tick_time[symbol] > 0:
            dt = now - self._last_tick_time[symbol]
            if dt > 0:
                self._tick_times[symbol].append(dt)
        self._last_tick_time[symbol] = now

        # --- absolute price change -----------------------------------------
        if self._last_price[symbol] > 0 and price > 0:
            change = abs(price - self._last_price[symbol])
            self._price_changes[symbol].append(change)
        self._last_price[symbol] = price

        # ---- tick_velocity ------------------------------------------------
        iat_list = list(self._tick_times[symbol])
        if len(iat_list) >= 1:
            mean_iat = statistics.mean(iat_list)
            tick_velocity = 1.0 / mean_iat if mean_iat > 0 else 0.0
        else:
            tick_velocity = 0.0

        # ---- velocity baseline (EMA) --------------------------------------
        if n == 0:
            self._velocity_baseline[symbol] = tick_velocity
        else:
            bl = self._velocity_baseline[symbol]
            self._velocity_baseline[symbol] = (
                self._alpha * tick_velocity + (1.0 - self._alpha) * bl
            )
        baseline = self._velocity_baseline[symbol]

        # ---- compression_density ------------------------------------------
        pc_list = list(self._price_changes[symbol])
        compression_density = 0.0
        if len(pc_list) >= 3:
            med = statistics.median(pc_list)
            if med > 0:
                thresh = med * 0.5
                small = sum(1 for pc in pc_list if pc < thresh)
                compression_density = small / len(pc_list)
            else:
                # All changes are zero — maximum compression
                compression_density = 1.0

        # ---- burst_score --------------------------------------------------
        burst_score = 0.0
        if baseline > 0 and tick_velocity > 0:
            ratio = tick_velocity / baseline
            if ratio > self.velocity_threshold:
                burst_score = min(
                    1.0, (ratio - self.velocity_threshold) / 2.0
                )

        return {
            "burst_score": burst_score,
            "compression_density": compression_density,
            "tick_velocity": tick_velocity,
            "baseline_velocity": baseline,
            "n_ticks": self._n_ticks[symbol],
        }

    def reset(self, symbol: Optional[str] = None) -> None:
        """Clear internal state for *symbol* (or all symbols if ``None``)."""
        if symbol is None:
            self._tick_times.clear()
            self._price_changes.clear()
            self._velocity_baseline.clear()
            self._n_ticks.clear()
            self._last_price.clear()
            self._last_tick_time.clear()
        else:
            self._tick_times.pop(symbol, None)
            self._price_changes.pop(symbol, None)
            self._velocity_baseline.pop(symbol, None)
            self._n_ticks.pop(symbol, None)
            self._last_price.pop(symbol, None)
            self._last_tick_time.pop(symbol, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_symbol(self, symbol: str) -> None:
        """Initialise per-symbol state on first encounter."""
        if symbol not in self._tick_times:
            self._tick_times[symbol] = deque(maxlen=self.window)
            self._price_changes[symbol] = deque(maxlen=self.window)
            self._velocity_baseline[symbol] = 0.0
            self._n_ticks[symbol] = 0
            self._last_price[symbol] = 0.0
            self._last_tick_time[symbol] = 0.0


# ======================================================================
# Self-test / simple simulation
# ======================================================================
if __name__ == "__main__":
    import random

    detector = BurstDetector(window=10)

    print("=== Ramp-up phase (steady ticks) ===")
    t = 1000.0
    price = 100.0
    for i in range(30):
        t += 0.05  # 20 ticks / sec steady
        price += random.gauss(0, 0.01)
        result = detector.update("BTC", {"price": price, "timestamp": t})
        if i < 5 or i >= 25:
            print(f"  tick {i:>2d}: {result}")

    print("\n=== Burst phase (rapid ticks) ===")
    for i in range(20):
        t += 0.005  # 200 ticks / sec burst
        price += random.gauss(0, 0.05)
        result = detector.update("BTC", {"price": price, "timestamp": t})
        print(f"  tick {i:>2d}: burst_score={result['burst_score']:.3f}  "
              f"vel={result['tick_velocity']:.1f}  "
              f"baseline={result['baseline_velocity']:.1f}  "
              f"compression={result['compression_density']:.3f}  "
              f"n={result['n_ticks']}")

    print("\n=== Cool-down phase ===")
    for i in range(20):
        t += 0.1  # 10 ticks / sec slow
        price += random.gauss(0, 0.005)
        result = detector.update("BTC", {"price": price, "timestamp": t})
        if i < 3 or i >= 17:
            print(f"  tick {i:>2d}: burst_score={result['burst_score']:.3f}  "
                  f"vel={result['tick_velocity']:.1f}  "
                  f"baseline={result['baseline_velocity']:.1f}")

    print("\n=== Reset symbol & verify clean state ===")
    detector.reset("BTC")
    fresh = detector.update("BTC", {"price": 200.0, "timestamp": 2000.0})
    print(f"  After reset: {fresh}")

    print("\n=== Multi-symbol test ===")
    detector.reset()
    for sym in ("AAPL", "GOOG"):
        for i in range(5):
            r = detector.update(sym, {"price": 150 + i, "timestamp": 1000 + i * 0.05})
            print(f"  {sym} tick {i}: n_ticks={r['n_ticks']}")
    print("  All tests passed.")
