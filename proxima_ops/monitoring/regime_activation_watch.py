"""
RegimeActivationWatch
=====================
Pre-activation detection — alert when the system is *about* to enter a tradable
regime, not just when it already has.

Triggers (composite watch):
  1. RSI extreme spread   — 2+ symbols RSI < 40 or > 60 simultaneously
  2. Volatility surge     — SIL top-3 average volatility score jumps > 20 %
                           over its rolling baseline
  3. Spread compression   — (high-low)/close proxy compresses on >= 3 symbols
  4. ATR percentile shift — absolute percentile change > 0.15 within the
                           rolling window on any symbol

No state changes, no execution interference.  All calculations are wrapped in
try/except so a failure never crashes the calling cycle.
"""

import logging
from collections import deque
import numpy as np

logger = logging.getLogger("proxima_ops.monitoring.regime_activation_watch")

PERIOD = 14
LOOKBACK = 100


# ---------------------------------------------------------------------------
# Pure helpers (stateless, can be tested in isolation)
# ---------------------------------------------------------------------------

def _compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                 period: int = PERIOD) -> np.ndarray:
    """Return the ATR series (same length as the shortest input)."""
    n = min(len(highs), len(lows), len(closes))
    if n < 2:
        return np.array([])
    tr = np.zeros(n)
    tr[0] = float(highs[0]) - float(lows[0])
    for i in range(1, n):
        tr[i] = max(
            float(highs[i]) - float(lows[i]),
            abs(float(highs[i]) - float(closes[i - 1])),
            abs(float(lows[i]) - float(closes[i - 1])),
        )
    atr = np.full(n, np.nan)
    if n <= period:
        atr[0] = float(np.mean(tr))
        return atr
    atr[period] = float(np.mean(tr[:period]))
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _atr_current_percentile(highs: np.ndarray, lows: np.ndarray,
                            closes: np.ndarray, period: int = PERIOD,
                            lookback: int = LOOKBACK) -> float:
    """Return current ATR percentile as a float in [0.0, 1.0]."""
    n = min(len(highs), len(lows), len(closes))
    if n < period + 1:
        return 0.5
    atr_series = _compute_atr(highs, lows, closes, period)
    valid = atr_series[~np.isnan(atr_series)]
    if len(valid) < 2:
        return 0.5
    current = float(valid[-1])
    window = valid[-min(lookback, len(valid)):]
    count_below = np.sum(window <= current)
    return float(count_below) / float(len(window))


# ---------------------------------------------------------------------------
# Watch class
# ---------------------------------------------------------------------------

class RegimeActivationWatch:
    """Monitor for pre-regime-activation signals.

    Parameters
    ----------
    window : int
        Rolling window size for baselines and shift detection (default 10).
    """

    def __init__(self, window: int = 10):
        self.window = window

        # Rolling ATR-percentile history per symbol
        self._atr_pctile_buf: dict[str, deque[float]] = {}

        # Rolling average of top-3 SIL scores (baseline for surge detection)
        self._sil_top3_baseline: deque[float] = deque(maxlen=window)

        # Rolling normalised-range history per symbol  (high-low)/close
        self._range_buf: dict[str, deque[float]] = {}

        self._cycle = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, md: dict, sil_scores: dict, rsi_dict: dict) -> dict:
        """Evaluate all four triggers and return a composite alert dict.

        Parameters
        ----------
        md : dict
            Market-data container with keys ``closes``, ``highs``, ``lows``,
            ``prices``, each being ``{symbol: np.ndarray}``.
        sil_scores : dict
            ``{symbol: float}`` from the Symbol Intelligence Layer.
        rsi_dict : dict
            ``{symbol: float}`` of latest RSI values (may be pre-computed
            upstream).

        Returns
        -------
        dict with keys:
            regime_pressure_score : float 0.0 – 1.0 composite
            pre_activation        : bool   any trigger fired
            dominant_symbols      : list   symbols contributing to the score
            triggers              : dict   per-trigger boolean
        """
        self._cycle += 1

        triggers: dict[str, bool] = {
            "rsi_extreme": False,
            "volatility_surge": False,
            "spread_compression": False,
            "atr_shift": False,
        }
        dominant: set[str] = set()

        # --- 1. RSI extreme ---------------------------------------------------
        try:
            rsi_syms, rsi_fired = self._check_rsi_extreme(rsi_dict)
            triggers["rsi_extreme"] = rsi_fired
            dominant.update(rsi_syms)
        except Exception as exc:
            logger.warning("RegimeActivationWatch RSI check failed: %s", exc)

        # --- 2. Volatility surge (SIL) ----------------------------------------
        try:
            vol_syms, vol_fired = self._check_volatility_surge(sil_scores)
            triggers["volatility_surge"] = vol_fired
            dominant.update(vol_syms)
        except Exception as exc:
            logger.warning("RegimeActivationWatch SIL surge check failed: %s", exc)

        # --- 3. Spread compression --------------------------------------------
        try:
            spread_syms, spread_fired = self._check_spread_compression(md)
            triggers["spread_compression"] = spread_fired
            dominant.update(spread_syms)
        except Exception as exc:
            logger.warning("RegimeActivationWatch spread check failed: %s", exc)

        # --- 4. ATR percentile shift ------------------------------------------
        try:
            atr_syms, atr_fired = self._check_atr_shift(md)
            triggers["atr_shift"] = atr_fired
            dominant.update(atr_syms)
        except Exception as exc:
            logger.warning("RegimeActivationWatch ATR shift check failed: %s", exc)

        dominant_list = sorted(dominant)

        regime_pressure_score = self._compute_pressure_score(triggers, dominant_list)
        pre_activation = any(triggers.values())

        logger.debug(
            "RegimeActivationWatch cycle=%d pressure=%.4f pre_activation=%s "
            "triggers=%s dominant=%s",
            self._cycle, regime_pressure_score, pre_activation,
            triggers, dominant_list,
        )

        return {
            "regime_pressure_score": regime_pressure_score,
            "pre_activation": pre_activation,
            "dominant_symbols": dominant_list,
            "triggers": triggers,
        }

    # ------------------------------------------------------------------
    # Individual trigger checks  (package-visible for unit testing)
    # ------------------------------------------------------------------

    def _check_rsi_extreme(self, rsi_dict: dict) -> tuple[list[str], bool]:
        """Trigger when 2+ symbols have RSI < 40 or > 60."""
        extreme: list[str] = []
        for sym, rsi in rsi_dict.items():
            if rsi is None:
                continue
            try:
                rv = float(rsi)
                if rv < 40.0 or rv > 60.0:
                    extreme.append(sym)
            except (TypeError, ValueError):
                continue
        return extreme, len(extreme) >= 2

    def _check_volatility_surge(self, sil_scores: dict) -> tuple[list[str], bool]:
        """Trigger when average of top-3 SIL scores rises > 20 % over baseline."""
        if not sil_scores:
            return [], False

        # Top 3 by score
        try:
            sorted_pairs = sorted(
                sil_scores.items(), key=lambda x: float(x[1]), reverse=True
            )
        except (TypeError, ValueError):
            return [], False

        top3 = sorted_pairs[:3]
        top3_syms = [s for s, _ in top3]
        top3_avg = float(np.mean([float(v) for _, v in top3]))

        self._sil_top3_baseline.append(top3_avg)

        if len(self._sil_top3_baseline) < 2:
            return top3_syms, False

        baseline_avg = float(np.mean(self._sil_top3_baseline))
        if baseline_avg <= 0.0:
            return top3_syms, False

        surge_pct = (top3_avg - baseline_avg) / baseline_avg
        return top3_syms, surge_pct > 0.20

    def _check_spread_compression(self, md: dict) -> tuple[list[str], bool]:
        """Trigger when >= 3 symbols show range compression.

        Uses normalised (high - low) / close as a spread proxy.
        Compression = current value < 50 % of the rolling average.
        """
        closes = md.get("closes", {})
        highs = md.get("highs", {})
        lows = md.get("lows", {})

        compressing: list[str] = []

        for sym in closes:
            try:
                c_arr = closes[sym]
                h_arr = highs.get(sym)
                l_arr = lows.get(sym)

                if c_arr is None or h_arr is None or l_arr is None:
                    continue
                if len(c_arr) < 1 or len(h_arr) < 1 or len(l_arr) < 1:
                    continue

                c_arr = np.asarray(c_arr, dtype=np.float64)
                h_arr = np.asarray(h_arr, dtype=np.float64)
                l_arr = np.asarray(l_arr, dtype=np.float64)

                close_price = float(c_arr[-1])
                if close_price == 0.0:
                    continue

                current_range = float((h_arr[-1] - l_arr[-1]) / close_price)

                if sym not in self._range_buf:
                    self._range_buf[sym] = deque(maxlen=self.window)
                self._range_buf[sym].append(current_range)

                if len(self._range_buf[sym]) < self.window // 2:
                    continue

                baseline = float(np.mean(self._range_buf[sym]))
                if baseline <= 0.0:
                    continue

                if current_range < 0.50 * baseline:
                    compressing.append(sym)
            except Exception:
                continue

        return compressing, len(compressing) >= 3

    def _check_atr_shift(self, md: dict) -> tuple[list[str], bool]:
        """Trigger when any symbol has ATR percentile shift > 0.15."""
        closes = md.get("closes", {})
        highs = md.get("highs", {})
        lows = md.get("lows", {})

        shifted: list[str] = []

        for sym in closes:
            try:
                c_arr = closes[sym]
                h_arr = highs.get(sym)
                l_arr = lows.get(sym)

                if c_arr is None or h_arr is None or l_arr is None:
                    continue
                if len(c_arr) < PERIOD + 1:
                    continue

                c_arr = np.asarray(c_arr, dtype=np.float64)
                h_arr = np.asarray(h_arr, dtype=np.float64)
                l_arr = np.asarray(l_arr, dtype=np.float64)

                pctile = _atr_current_percentile(h_arr, l_arr, c_arr)

                if sym not in self._atr_pctile_buf:
                    self._atr_pctile_buf[sym] = deque(maxlen=self.window)
                self._atr_pctile_buf[sym].append(pctile)

                if len(self._atr_pctile_buf[sym]) < 2:
                    continue

                buf = list(self._atr_pctile_buf[sym])
                shift = abs(buf[-1] - buf[0])

                if shift > 0.15:
                    shifted.append(sym)
            except Exception:
                continue

        return shifted, len(shifted) > 0

    # ------------------------------------------------------------------
    # Composite score
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_pressure_score(triggers: dict[str, bool],
                                dominant_symbols: list[str]) -> float:
        """Compute a 0.0 – 1.0 composite pressure score.

        Formula:
          - Each fired trigger               = +0.20
          - Each dominant symbol beyond the
            minimum needed for any trigger   = +0.05  (capped)
          - Result clamped to [0.0, 1.0].
        """
        score = 0.0
        for fired in triggers.values():
            if fired:
                score += 0.20

        # Symbols add extra pressure proportional to breadth
        if dominant_symbols:
            score += min(len(dominant_symbols) * 0.05, 0.20)

        return round(min(score, 1.0), 4)
