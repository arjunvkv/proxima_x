from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np


class WeakDayDetector:
    __slots__ = (
        "ecf_window", "pf_window", "burst_window", "regime_window",
        "_spread_m2", "spread_mean", "spread_count", "_spread_deque",
        "_ecf_events", "_pf_events", "_burst_total", "_burst_fail",
        "_regime_flips", "_regime_prev", "_regime_count",
        "_tick_count",
    )

    def __init__(self) -> None:
        self.ecf_window: List[float] = []
        self.pf_window: List[float] = []
        self.burst_window: List[float] = []
        self.regime_window: List[float] = []
        self._spread_m2: float = 0.0
        self.spread_mean: float = 0.0
        self.spread_count: int = 0
        self._spread_deque: List[float] = []
        self._ecf_events: List[Dict] = []
        self._pf_events: List[Dict] = []
        self._burst_total: int = 0
        self._burst_fail: int = 0
        self._regime_flips: int = 0
        self._regime_prev: Optional[int] = None
        self._regime_count: int = 0
        self._tick_count: int = 0

    def record_entropy_compression(self, compressed: bool, breakout_success: bool) -> None:
        self._ecf_events.append({"compressed": compressed, "breakout": breakout_success})
        if len(self._ecf_events) > 500:
            self._ecf_events.pop(0)

    def record_persistence_event(self, persistence_streak: int, collapsed: bool) -> None:
        self._pf_events.append({"streak": persistence_streak, "collapsed": collapsed})
        if len(self._pf_events) > 500:
            self._pf_events.pop(0)

    def record_burst(self, burst_density: float, price_followed: bool) -> None:
        self._burst_total += 1
        if not price_followed:
            self._burst_fail += 1

    def record_regime(self, regime: int) -> None:
        if self._regime_prev is not None and regime != self._regime_prev:
            self._regime_flips += 1
        self._regime_prev = regime
        self._regime_count += 1

    def record_spread(self, spread: float) -> None:
        self._spread_deque.append(spread)
        if len(self._spread_deque) > 500:
            self._spread_deque.pop(0)
        n = len(self._spread_deque)
        if n == 1:
            self.spread_mean = spread
            self._spread_m2 = 0.0
            self.spread_count = n
            return
        old_mean = self.spread_mean
        self.spread_mean += (spread - old_mean) / n
        self._spread_m2 += (spread - old_mean) * (spread - self.spread_mean)
        self.spread_count = n

    def tick(self) -> None:
        self._tick_count += 1

    def compute(self) -> Dict:
        ecf = self._compute_ecf()
        pf = self._compute_pf()
        se = self._compute_se()
        bfr = self._compute_bfr()
        rc = self._compute_rc()

        wds = 0.30 * ecf + 0.20 * pf + 0.20 * se + 0.20 * bfr + 0.10 * rc

        if wds < 0.35:
            fragility = "HEALTHY"
            mult = 1.0
        elif wds < 0.60:
            fragility = "FRAGILE"
            mult = 0.5
        else:
            fragility = "WEAK"
            mult = 0.0

        reasons = []
        if ecf > 0.5:
            reasons.append(f"ECF={ecf:.2f}")
        if pf > 0.5:
            reasons.append(f"PF={pf:.2f}")
        if se > 0.5:
            reasons.append(f"SE={se:.2f}")
        if bfr > 0.5:
            reasons.append(f"BFR={bfr:.2f}")
        if rc > 0.5:
            reasons.append(f"RC={rc:.2f}")

        return {
            "weak_day_score": float(wds),
            "fragility_class": fragility,
            "trade_multiplier": mult,
            "components": {
                "entropy_compression_failure": float(ecf),
                "persistence_fragility": float(pf),
                "spread_elasticity": float(se),
                "burst_failure_rate": float(bfr),
                "regime_churn": float(rc),
            },
            "suppression_reason": ";".join(reasons) if reasons else "",
            "suppressed": fragility != "HEALTHY",
        }

    def _compute_ecf(self) -> float:
        if len(self._ecf_events) < 10:
            return 0.3
        compressed = sum(1 for e in self._ecf_events if e["compressed"])
        breakout_ok = sum(1 for e in self._ecf_events if e["compressed"] and e["breakout"])
        if compressed < 5:
            return 0.3
        return 1.0 - (breakout_ok / max(compressed, 1))

    def _compute_pf(self) -> float:
        if len(self._pf_events) < 5:
            return 0.3
        total = len(self._pf_events)
        collapsed = sum(1 for e in self._pf_events if e["collapsed"])
        return collapsed / max(total, 1)

    def _compute_se(self) -> float:
        if self.spread_count < 10:
            return 0.3
        var = self._spread_m2 / max(self.spread_count - 1, 1)
        std = np.sqrt(max(var, 1e-10))
        return min(1.0, std / max(self.spread_mean, 1e-10))

    def _compute_bfr(self) -> float:
        if self._burst_total < 5:
            return 0.3
        return self._burst_fail / max(self._burst_total, 1)

    def _compute_rc(self) -> float:
        if self._regime_count < 10:
            return 0.3
        return min(1.0, self._regime_flips / max(self._regime_count, 1))

    def reset(self) -> None:
        self.ecf_window.clear()
        self.pf_window.clear()
        self.burst_window.clear()
        self.regime_window.clear()
        self._spread_m2 = 0.0
        self.spread_mean = 0.0
        self.spread_count = 0
        self._spread_deque.clear()
        self._ecf_events.clear()
        self._pf_events.clear()
        self._burst_total = 0
        self._burst_fail = 0
        self._regime_flips = 0
        self._regime_prev = None
        self._regime_count = 0
        self._tick_count = 0
