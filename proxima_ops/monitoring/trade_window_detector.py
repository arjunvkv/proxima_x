"""
TradeWindowDetector
===================
Predict the NEXT 5-20 cycle window with highest trade probability.
Uses SIL volatility spikes, RSI convergence, spread compression,
and regime activation score to generate a composite probability.

No state changes, no execution interference. All calculations are wrapped in
try/except so a failure never crashes the calling cycle.
"""

import logging
from collections import deque

import numpy as np

logger = logging.getLogger("proxima_ops.monitoring.trade_window_detector")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RSI_LOWER_EXTREME = 35.0
RSI_UPPER_EXTREME = 65.0
RSI_NEUTRAL = 50.0
RSI_EXTREMITY_THRESHOLD = 15.0  # abs(rsi - 50) at or above this is "extreme"

SURGE_MODERATE = 0.15  # > 15 % increase → 0.3 score
SURGE_HIGH = 0.30     # > 30 % increase → 0.6 score

# Weights for composite
WEIGHT_RSI = 0.30
WEIGHT_VOLATILITY = 0.30
WEIGHT_SPREAD = 0.20
WEIGHT_REGIME = 0.20

TRADE_WINDOW_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class TradeWindowDetector:
    """Predict the next trade window based on multi-signal composite.

    Parameters
    ----------
    window_lookback : int
        Rolling window size for SIL volatility baseline (default 50).
    window_forecast : int
        Forecast horizon in cycles (default 20).
    """

    def __init__(self, window_lookback: int = 50, window_forecast: int = 20):
        self.window_lookback = window_lookback
        self.window_forecast = window_forecast

        # Rolling average of top-3 SIL scores (volatility spike baseline)
        self._sil_top3_baseline: deque[float] = deque(maxlen=window_lookback)

        # Stored inputs (set by update(), consumed by detect_window())
        self._md: dict = {}
        self._rsi_dict: dict = {}
        self._sil_scores: dict = {}
        self._activation: dict = {}
        self._cycle: int = 0

        # Cached result (computations are stateful — the baseline deque
        # advances on every call; caching ensures detect_window() is
        # idempotent between update() calls).
        self._last_result: dict | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, md: dict, rsi_dict: dict, sil_scores: dict,
               activation: dict, cycle: int) -> dict:
        """Accept latest market data and return the trade-window detection dict.

        Parameters
        ----------
        md : dict
            Market-data container with keys ``closes``, ``highs``, ``lows``,
            ``prices``, each being ``{symbol: np.ndarray}``.
        rsi_dict : dict
            ``{symbol: float}`` of latest RSI values.
        sil_scores : dict
            ``{symbol: float}`` from the Symbol Intelligence Layer.
        activation : dict
            Output from ``RegimeActivationWatch.update()`` with keys
            ``regime_pressure_score``, ``pre_activation``, ``triggers``,
            ``dominant_symbols``.
        cycle : int
            Current cycle number.

        Returns
        -------
        dict
            See ``detect_window()`` for output schema.
        """
        self._md = md
        self._rsi_dict = rsi_dict
        self._sil_scores = sil_scores
        self._activation = activation
        self._cycle = cycle

        self._last_result = self._compute()
        return self._last_result

    def detect_window(self) -> dict:
        """Return the last computed trade-window result.

        If ``update()`` has never been called a default zero-result is
        returned so the caller never has to handle ``None``.
        """
        if self._last_result is not None:
            return self._last_result

        return {
            "trade_window_open": False,
            "probability_next_10_cycles": 0.0,
            "expected_symbols": [],
            "confidence": 0.0,
            "window_details": {
                "rsi_convergence_score": 0.0,
                "volatility_spike_score": 0.0,
                "spread_compression_score": 0.0,
                "regime_pressure_score": 0.0,
            },
        }

    # ------------------------------------------------------------------
    # Internal computation (stateful — advances the baseline deque)
    # ------------------------------------------------------------------

    def _compute(self) -> dict:
        """Compute trade-window probability from stored inputs.

        Returns
        -------
        dict with keys:
            trade_window_open        : bool   True if composite > 0.5
            probability_next_10_cycles : float 0.0-1.0 composite
            expected_symbols         : list   most likely trigger symbols
            confidence               : float  same as composite (0.0-1.0)
            window_details           : dict   per-component scores
        """
        # --- 1. RSI convergence score ------------------------------------
        try:
            rsi_convergence = self._compute_rsi_convergence(self._rsi_dict)
        except Exception as exc:
            logger.warning("TradeWindowDetector RSI convergence failed: %s", exc)
            rsi_convergence = 0.0

        # --- 2. Volatility spike score ----------------------------------
        try:
            volatility_spike = self._compute_volatility_spike(self._sil_scores)
        except Exception as exc:
            logger.warning(
                "TradeWindowDetector volatility spike failed: %s", exc
            )
            volatility_spike = 0.0

        # --- 3. Spread compression score ---------------------------------
        try:
            spread_compression = self._compute_spread_compression(
                self._activation
            )
        except Exception as exc:
            logger.warning(
                "TradeWindowDetector spread compression failed: %s", exc
            )
            spread_compression = 0.0

        # --- 4. Regime pressure score ------------------------------------
        try:
            regime_pressure = self._compute_regime_pressure(self._activation)
        except Exception as exc:
            logger.warning(
                "TradeWindowDetector regime pressure failed: %s", exc
            )
            regime_pressure = 0.0

        # --- Composite weighted average ----------------------------------
        try:
            composite = (
                rsi_convergence * WEIGHT_RSI
                + volatility_spike * WEIGHT_VOLATILITY
                + spread_compression * WEIGHT_SPREAD
                + regime_pressure * WEIGHT_REGIME
            )
            composite = min(max(composite, 0.0), 1.0)
        except Exception as exc:
            logger.warning("TradeWindowDetector composite failed: %s", exc)
            composite = 0.0

        # --- Expected symbols --------------------------------------------
        try:
            expected_symbols = self._resolve_expected_symbols(
                self._activation, self._rsi_dict
            )
        except Exception as exc:
            logger.warning(
                "TradeWindowDetector expected symbols failed: %s", exc
            )
            expected_symbols = []

        trade_window_open = composite > TRADE_WINDOW_THRESHOLD

        result = {
            "trade_window_open": trade_window_open,
            "probability_next_10_cycles": round(composite, 4),
            "expected_symbols": expected_symbols,
            "confidence": round(composite, 4),
            "window_details": {
                "rsi_convergence_score": round(rsi_convergence, 4),
                "volatility_spike_score": round(volatility_spike, 4),
                "spread_compression_score": round(spread_compression, 4),
                "regime_pressure_score": round(regime_pressure, 4),
            },
        }

        logger.debug(
            "TradeWindowDetector cycle=%d open=%s prob=%.4f symbols=%s",
            self._cycle,
            trade_window_open,
            composite,
            expected_symbols,
        )

        return result

    # ------------------------------------------------------------------
    # Internal score methods  (package-visible for unit testing)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_rsi_convergence(rsi_dict: dict) -> float:
        """Fraction of symbols approaching RSI extreme (< 35 or > 65).

        Returns
        -------
        float
            0.0 if no symbols, otherwise fraction of symbols at extreme.
        """
        if not rsi_dict:
            return 0.0

        extreme_count = 0
        total = 0

        for sym, rsi in rsi_dict.items():
            if rsi is None:
                continue
            try:
                rv = float(rsi)
                total += 1
                if rv < RSI_LOWER_EXTREME or rv > RSI_UPPER_EXTREME:
                    extreme_count += 1
            except (TypeError, ValueError):
                continue

        if total == 0:
            return 0.0

        return extreme_count / total

    def _compute_volatility_spike(self, sil_scores: dict) -> float:
        """Score based on SIL top-3 average surge over rolling baseline.

        Compares the current average of the top-3 SIL scores against a
        rolling baseline.  A surge > 15 % yields 0.3, > 30 % yields 0.6,
        capped at 1.0.

        Returns
        -------
        float
            0.0, 0.3, or 0.6 based on surge magnitude.
        """
        if not sil_scores:
            return 0.0

        # Top 3 by score
        try:
            sorted_pairs = sorted(
                sil_scores.items(), key=lambda x: float(x[1]), reverse=True
            )
        except (TypeError, ValueError):
            return 0.0

        top3 = sorted_pairs[:3]
        if not top3:
            return 0.0

        top3_avg = float(np.mean([float(v) for _, v in top3]))

        self._sil_top3_baseline.append(top3_avg)

        if len(self._sil_top3_baseline) < 2:
            return 0.0

        baseline_avg = float(np.mean(self._sil_top3_baseline))
        if baseline_avg <= 0.0:
            return 0.0

        surge_pct = (top3_avg - baseline_avg) / baseline_avg

        if surge_pct > SURGE_HIGH:
            score = 0.6
        elif surge_pct > SURGE_MODERATE:
            score = 0.3
        else:
            score = 0.0

        return min(score, 1.0)

    @staticmethod
    def _compute_spread_compression(activation: dict) -> float:
        """Return 0.3 if spread compression trigger is active, else 0.0."""
        triggers = activation.get("triggers", {})
        if triggers.get("spread_compression", False):
            return 0.3
        return 0.0

    @staticmethod
    def _compute_regime_pressure(activation: dict) -> float:
        """Regime pressure score directly from activation dict."""
        return float(activation.get("regime_pressure_score", 0.0))

    @staticmethod
    def _resolve_expected_symbols(activation: dict,
                                  rsi_dict: dict) -> list[str]:
        """Return dominant symbols, or symbols with highest RSI extremity.

        Uses ``activation["dominant_symbols"]`` when non-empty.
        Falls back to symbols whose RSI is farthest from 50 (most extreme).
        """
        dominant = activation.get("dominant_symbols", [])
        if dominant:
            return sorted(dominant)

        # Fall back to symbols with RSI farthest from 50
        if not rsi_dict:
            return []

        sym_extremity: list[tuple[str, float]] = []
        for sym, rsi in rsi_dict.items():
            if rsi is None:
                continue
            try:
                rv = float(rsi)
                extremity = abs(rv - RSI_NEUTRAL)
                sym_extremity.append((sym, extremity))
            except (TypeError, ValueError):
                continue

        # Sort by extremity descending (most extreme first)
        sym_extremity.sort(key=lambda x: x[1], reverse=True)

        # Return symbols at or above the extremity threshold
        extreme_syms = [
            s for s, e in sym_extremity if e >= RSI_EXTREMITY_THRESHOLD
        ]
        if extreme_syms:
            return extreme_syms

        # If none are truly extreme, return top 3 closest to extreme
        return [s for s, _ in sym_extremity[:3]]
