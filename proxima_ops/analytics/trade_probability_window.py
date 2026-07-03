"""Trade Probability Window — Estimate likelihood of trade occurring in next N cycles.

Uses RSI distribution across universe, SIL volatility ranking, and historical
activation patterns to compute probability windows for trade occurrence.

This is a pure analytics module — reads data, computes estimates, never modifies state.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

logger = logging.getLogger("proxima_ops.analytics.trade_probability_window")

# ---------------------------------------------------------------------------
# Threshold Constants
# ---------------------------------------------------------------------------

# RSI levels used for activation-potential tracking (extreme conditions)
RSI_EXTREME_OVERSOLD = 30
RSI_EXTREME_OVERBOUGHT = 70

# RSI levels used for regime classification
RSI_DEAD_LOWER = 40
RSI_DEAD_UPPER = 60
RSI_ACTIVE_LOW = 35
RSI_ACTIVE_HIGH = 65

# Volatility thresholds
ATR_RATIO_VOLATILE = 1.3   # ATR / baseline above this = high vol
ATR_RATIO_ACTIVATION = 1.3  # threshold for activation-potential check

# Regime boost multipliers applied to base probability
REGIME_BOOST: dict[str, float] = {
    "dead": 0.5,
    "neutral": 1.0,
    "active": 1.3,
    "volatile": 1.5,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_atr(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14
) -> np.ndarray:
    """Compute Average True Range."""
    n = min(len(highs), len(lows), len(closes))
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
    atr = np.full(n, np.nan)
    atr[period] = np.mean(tr[:period])
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


# ---------------------------------------------------------------------------
# TradeProbabilityWindow
# ---------------------------------------------------------------------------


class TradeProbabilityWindow:
    """Estimate likelihood of trade occurring in the next N cycles.

    Maintains a rolling window of historical market conditions and uses
    RSI distribution, SIL volatility scores, and past activation patterns
    to compute the probability of a trade within 10 and 50 cycles.

    This is a pure analytics module — it reads data and computes estimates.
    It never modifies external state.

    Parameters
    ----------
    history_size : int
        Number of past cycles to retain for historical pattern analysis.
        Default is 100. Minimum is 10.
    """

    def __init__(self, history_size: int = 100):
        self._history_size = max(10, history_size)

        # Rolling buffers (FIFO, bounded by history_size)
        self._regime_history: deque[str] = deque(maxlen=self._history_size)
        self._rsi_records: deque[dict[str, float]] = deque(maxlen=self._history_size)
        self._sil_records: deque[dict[str, float]] = deque(maxlen=self._history_size)
        self._activation_flags: deque[bool] = deque(maxlen=self._history_size)

        # Latest raw inputs
        self._current_rsi_dict: dict[str, float] = {}
        self._current_sil_dict: dict[str, float] = {}
        self._current_cycle: int = 0
        self._total_updates: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, md: dict, rsi_dict: dict, sil_scores: dict, current_cycle: int) -> None:
        """Record a new data point for the current cycle.

        Parameters
        ----------
        md : dict
            Market data dict. Expected to contain (at least one of)
            ``closes``, ``highs``, ``lows`` mapping symbol -> array-like.
        rsi_dict : dict
            Symbol -> current RSI value (float 0-100).
        sil_scores : dict
            Symbol -> SIL volatility ranking score (float).
        current_cycle : int
            Current cycle number.
        """
        try:
            self._current_rsi_dict = dict(rsi_dict)
            self._current_sil_dict = dict(sil_scores)
            self._current_cycle = current_cycle
            self._total_updates += 1

            # Classify regime for this cycle
            regime = self._classify_regime(rsi_dict, md)
            self._regime_history.append(regime)
            self._rsi_records.append(dict(rsi_dict))
            self._sil_records.append(dict(sil_scores))

            # Determine if this cycle had extreme RSI / volatility conditions
            # that COULD have triggered a trade
            had_activation = self._check_activation_potential(rsi_dict, md)
            self._activation_flags.append(had_activation)

        except Exception as exc:
            logger.warning("TradeProbabilityWindow.update failed: %s", exc)

    def estimate(self) -> dict[str, Any]:
        """Compute trade probability window estimates.

        Returns
        -------
        dict
            Probability estimates with regime classification and driver breakdown::

                {
                  "p_trade_next_10_cycles": float,   # 0.0-1.0
                  "p_trade_next_50_cycles": float,   # 0.0-1.0
                  "current_regime_class": str,       # "dead"|"neutral"|"active"|"volatile"
                  "confidence": float,               # 0.0-1.0
                  "drivers": {
                    "rsi_extremity": float,
                    "volatility_readiness": float,
                    "historical_pattern": float,
                  }
                }
        """
        result: dict[str, Any] = {
            "p_trade_next_10_cycles": 0.0,
            "p_trade_next_50_cycles": 0.0,
            "current_regime_class": "neutral",
            "confidence": 0.0,
            "drivers": {
                "rsi_extremity": 0.0,
                "volatility_readiness": 0.0,
                "historical_pattern": 0.0,
            },
        }

        try:
            # ---- compute driver contributions ----
            rsi_extremity = self._compute_rsi_extremity()
            volatility_readiness = self._compute_volatility_readiness()
            historical_pattern = self._compute_historical_pattern()

            # ---- use the most recent regime classification ----
            # (this was already computed during update() with full market data)
            regime = self.current_regime
            result["current_regime_class"] = regime

            # ---- blend drivers into base probability ----
            # Each driver is 0.0 (no signal) to 1.0 (strong signal).
            base_prob = (
                0.35 * rsi_extremity
                + 0.35 * volatility_readiness
                + 0.30 * historical_pattern
            )

            # Apply regime boost
            boost = REGIME_BOOST.get(regime, 1.0)
            boosted = min(1.0, base_prob * boost)

            # 10-cycle probability = boosted blended signal
            p_10 = boosted

            # 50-cycle extrapolation: p_50 = 1 - (1 - p_10) ^ (50/10)
            # This assumes independent trials across windows.
            if p_10 >= 1.0:
                p_50 = 1.0
            else:
                p_50 = 1.0 - (1.0 - p_10) ** 5.0

            result["p_trade_next_10_cycles"] = round(p_10, 4)
            result["p_trade_next_50_cycles"] = round(p_50, 4)

            # Confidence in the estimate
            result["confidence"] = self._compute_confidence(
                rsi_extremity, volatility_readiness, historical_pattern
            )

            result["drivers"] = {
                "rsi_extremity": round(rsi_extremity, 4),
                "volatility_readiness": round(volatility_readiness, 4),
                "historical_pattern": round(historical_pattern, 4),
            }

        except Exception as exc:
            logger.warning("TradeProbabilityWindow.estimate failed: %s", exc)

        return result

    # ------------------------------------------------------------------
    # Internal: Regime Classification
    # ------------------------------------------------------------------

    def _classify_regime(self, rsi_dict: dict, md: dict) -> str:
        """Classify current market regime.

        Rules
        -----
        * ``"dead"`` — all symbols RSI in 40-60 range, low volatility
        * ``"neutral"`` — mixed RSI, moderate volatility
        * ``"active"`` — 1+ symbols with RSI < 35 or > 65
        * ``"volatile"`` — 2+ symbols extreme + high ATR

        Parameters
        ----------
        rsi_dict : dict
            Symbol -> RSI value.
        md : dict
            Market data (used for ATR-based volatility assessment).

        Returns
        -------
        str
            One of ``"dead"``, ``"neutral"``, ``"active"``, ``"volatile"``.
        """
        if not rsi_dict:
            return "neutral"

        rsi_values = list(rsi_dict.values())
        if not rsi_values:
            return "neutral"

        # Count symbols in each zone
        extreme_count = sum(
            1 for r in rsi_values
            if r < RSI_EXTREME_OVERSOLD or r > RSI_EXTREME_OVERBOUGHT
        )
        active_count = sum(
            1 for r in rsi_values
            if r < RSI_ACTIVE_LOW or r > RSI_ACTIVE_HIGH
        )
        dead_count = sum(
            1 for r in rsi_values
            if RSI_DEAD_LOWER <= r <= RSI_DEAD_UPPER
        )

        # Volatility assessment from market data
        high_vol_count = self._count_high_vol_symbols(rsi_dict, md)

        # Apply classification rules
        if extreme_count >= 2 and high_vol_count >= 1:
            return "volatile"
        if active_count >= 1:
            return "active"
        if dead_count == len(rsi_values) and high_vol_count == 0:
            return "dead"

        return "neutral"

    def _count_high_vol_symbols(self, rsi_dict: dict, md: dict) -> int:
        """Count symbols with ATR ratio above the volatile threshold."""
        count = 0
        try:
            closes = md.get("closes", {})
            highs = md.get("highs", {})
            lows = md.get("lows", {})
            for sym in rsi_dict:
                c = closes.get(sym)
                h = highs.get(sym)
                l_ = lows.get(sym)
                if c is None or h is None or l_ is None:
                    continue
                if not hasattr(c, "__len__") or len(c) < 16:
                    continue
                c_arr = np.asarray(c, dtype=np.float64)
                h_arr = np.asarray(h, dtype=np.float64)
                l_arr = np.asarray(l_, dtype=np.float64)
                atr = _compute_atr(h_arr, l_arr, c_arr)
                valid = atr[~np.isnan(atr)]
                if len(valid) < 20:
                    continue
                current_atr = valid[-1]
                baseline = float(np.nanmean(valid[-50:])) if len(valid) >= 50 else float(np.nanmean(valid))
                if baseline > 0 and (current_atr / baseline) > ATR_RATIO_VOLATILE:
                    count += 1
        except Exception:
            pass
        return count

    # ------------------------------------------------------------------
    # Internal: Driver Computations
    # ------------------------------------------------------------------

    def _compute_rsi_extremity(self) -> float:
        """Contribution from RSI extremes, 0.0 (none) to 1.0 (strong).

        Higher values indicate more symbols are far from neutral (RSI=50),
        suggesting mean-reversion trade opportunities.
        """
        if not self._current_rsi_dict:
            return 0.0

        rsi_values = list(self._current_rsi_dict.values())
        if not rsi_values:
            return 0.0

        # Average distance from RSI=50, normalized to [0, 1]
        extremity_scores = [abs(r - 50.0) / 50.0 for r in rsi_values]
        mean_extremity = float(np.mean(extremity_scores))

        # Proportion of symbols that cross the active thresholds
        threshold_count = sum(
            1 for r in rsi_values if r < RSI_ACTIVE_LOW or r > RSI_ACTIVE_HIGH
        )
        threshold_ratio = threshold_count / max(len(rsi_values), 1)

        # Blend: 60 % mean extremity + 40 % threshold ratio
        return min(1.0, 0.6 * mean_extremity + 0.4 * threshold_ratio)

    def _compute_volatility_readiness(self) -> float:
        """Contribution from SIL volatility scores, 0.0 to 1.0.

        Measures how "ready" the market is for volatility-driven trades
        based on SIL ranking scores.  Both the mean level and the
        cross-symbol dispersion contribute.
        """
        if not self._current_sil_dict:
            return 0.0

        sil_values = list(self._current_sil_dict.values())
        if not sil_values:
            return 0.0

        # Normalise SIL scores (assume typical range 0-100)
        normalized = [min(1.0, max(0.0, v / 100.0)) for v in sil_values]

        mean_sil = float(np.mean(normalized))

        # Dispersion — more spread means more opportunity for relative-value
        sil_std = float(np.std(normalized)) if len(normalized) > 1 else 0.0

        # Blend: 70 % mean level + 30 % dispersion (capped at 1.0)
        return min(1.0, 0.7 * mean_sil + 0.3 * min(1.0, sil_std * 2.0))

    def _compute_historical_pattern(self) -> float:
        """Contribution from historical activation frequency, 0.0 to 1.0.

        Uses the rolling buffer of past activation flags.  Recent cycles
        are weighted more heavily via exponential weighting.
        """
        if not self._activation_flags:
            return 0.0

        total = len(self._activation_flags)
        if total < 2:
            return 0.0

        # Exponential weighting: more recent cycles get higher weight
        weights = np.exp(np.linspace(0, 1, total))
        weights = weights / weights.sum()

        weighted_activated = sum(
            w for f, w in zip(self._activation_flags, weights) if f
        )
        return min(1.0, float(weighted_activated))

    def _check_activation_potential(self, rsi_dict: dict, md: dict) -> bool:
        """Check whether current conditions COULD trigger a trade.

        Returns ``True`` if at least one symbol has extreme RSI (potential
        mean-reversion trade) or extreme volatility conditions (ATR spike).
        """
        # RSI-based activation
        for r in rsi_dict.values():
            if r < RSI_EXTREME_OVERSOLD or r > RSI_EXTREME_OVERBOUGHT:
                return True

        # Volatility-based activation (ATR ratio above threshold)
        try:
            closes = md.get("closes", {})
            highs = md.get("highs", {})
            lows = md.get("lows", {})
            for sym in rsi_dict:
                c = closes.get(sym)
                h = highs.get(sym)
                l_ = lows.get(sym)
                if c is None or h is None or l_ is None:
                    continue
                if not hasattr(c, "__len__") or len(c) < 16:
                    continue
                c_arr = np.asarray(c, dtype=np.float64)
                h_arr = np.asarray(h, dtype=np.float64)
                l_arr = np.asarray(l_, dtype=np.float64)
                atr = _compute_atr(h_arr, l_arr, c_arr)
                valid = atr[~np.isnan(atr)]
                if len(valid) < 10:
                    continue
                current_atr = valid[-1]
                baseline = float(np.nanmean(valid[-50:])) if len(valid) >= 50 else float(np.nanmean(valid))
                if baseline > 0 and (current_atr / baseline) > ATR_RATIO_ACTIVATION:
                    return True
        except Exception:
            pass

        return False

    def _compute_confidence(
        self,
        rsi_extremity: float,
        volatility_readiness: float,
        historical_pattern: float,
    ) -> float:
        """Confidence in the probability estimate, 0.0 (low) to 1.0 (high).

        Factors considered:
            * **Data sufficiency** — how many cycles have been observed.
            * **Driver consistency** — low variance across the three drivers.
            * **Signal strength** — how strong the drivers collectively are.
        """
        # Data sufficiency: need at least 50 cycles for full confidence
        data_ratio = min(1.0, self._total_updates / 50.0)

        # Driver consistency (low std => high consistency)
        driver_values = [rsi_extremity, volatility_readiness, historical_pattern]
        driver_mean = float(np.mean(driver_values))
        driver_std = float(np.std(driver_values)) if len(driver_values) > 1 else 0.0
        consistency = 1.0 - min(1.0, driver_std * 2.0)

        # Signal strength
        signal_strength = driver_mean

        # Blend
        confidence = 0.30 * data_ratio + 0.35 * consistency + 0.35 * signal_strength
        return round(min(1.0, confidence), 4)

    # ------------------------------------------------------------------
    # Introspection / Properties
    # ------------------------------------------------------------------

    @property
    def total_updates(self) -> int:
        """Number of times :meth:`update` has been called."""
        return self._total_updates

    @property
    def current_regime(self) -> str:
        """Most recent regime classification."""
        if self._regime_history:
            return self._regime_history[-1]
        return "neutral"

    @property
    def activation_rate(self) -> float:
        """Fraction of observed cycles that had activation potential."""
        if not self._activation_flags:
            return 0.0
        return sum(1 for f in self._activation_flags if f) / max(len(self._activation_flags), 1)

    def get_summary(self) -> dict[str, Any]:
        """Return a concise snapshot of current state and estimates."""
        return {
            "total_updates": self._total_updates,
            "history_size": self._history_size,
            "current_cycle": self._current_cycle,
            "current_regime": self.current_regime,
            "activation_rate": round(self.activation_rate, 4),
            "symbols_in_view": list(self._current_rsi_dict.keys()),
            "estimate": self.estimate(),
        }
