"""
SYMBOL INTELLIGENCE LAYER — Computes a tradability score per symbol using live MT5 data.

Scores are weighted composite of liquidity, spread quality, volatility usability,
and regime activity. Run as __main__ to test against a static symbol set.
"""

import numpy as np
import MetaTrader5 as mt5
import logging
from typing import List

logger = logging.getLogger("proxima_ops.intelligence.sil")


class SymbolIntelligenceLayer:
    """Computes tradability scores for a list of symbols using live MT5 data."""

    def __init__(self, min_spread: float = 0.0, max_spread: float = 50.0):
        self.min_spread = min_spread
        self.max_spread = max_spread

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_symbols(self, symbols: List[str]) -> List[dict]:
        """Return a list of per-symbol score dicts, sorted by total_score descending."""
        results = []
        for symbol in symbols:
            try:
                liquidity = self._compute_liquidity(symbol)
                spread = self._compute_spread_quality(symbol)
                volatility = self._compute_volatility_usability(symbol)
                activity = self._compute_regime_activity(symbol)

                total = (
                    liquidity * 0.40
                    + spread * 0.25
                    + volatility * 0.20
                    + activity * 0.15
                )

                results.append(
                    {
                        "symbol": symbol,
                        "scores": {
                            "liquidity": round(liquidity, 4),
                            "spread_quality": round(spread, 4),
                            "volatility_usability": round(volatility, 4),
                            "regime_activity": round(activity, 4),
                        },
                        "total_score": round(total, 4),
                        "meta": self._collect_meta(symbol),
                    }
                )
            except Exception as exc:
                logger.warning("Failed to score symbol %s: %s", symbol, exc)
                results.append(
                    {
                        "symbol": symbol,
                        "scores": {},
                        "total_score": 0.0,
                        "meta": {},
                        "error": str(exc),
                    }
                )

        return sorted(results, key=lambda x: x["total_score"], reverse=True)

    # ------------------------------------------------------------------
    # Per-component helpers
    # ------------------------------------------------------------------

    def _compute_liquidity(self, symbol: str) -> float:
        """Liquidity score based on tick frequency over the last 60 M1 bars.

        Uses non-zero tick count as a proxy for market activity.
        """
        try:
            rates = self._get_rates(symbol, count=60)
            if not rates:
                return 0.0

            tick_count = sum(
                1 for r in rates if r.get("tick_volume") is not None and r["tick_volume"] > 0
            )
            return min(tick_count / 60, 1.0)
        except Exception:
            logger.warning("_compute_liquidity failed for %s", symbol, exc_info=True)
            return 0.0

    def _compute_spread_quality(self, symbol: str) -> float:
        """Spread quality score based on current bid/ask spread.

        Higher scores mean tighter spreads.
        """
        try:
            tick = mt5.symbol_info_tick(symbol)
            info = mt5.symbol_info(symbol)
            if tick is None or info is None:
                return 0.0

            raw_spread = tick.ask - tick.bid
            point = info.point
            if point == 0:
                return 0.0

            spread_points = raw_spread / point
            numerator = spread_points - self.min_spread
            denominator = self.max_spread - self.min_spread
            if denominator <= 0:
                return 1.0

            score = max(0.0, 1.0 - numerator / denominator)
            return min(score, 1.0)
        except Exception:
            logger.warning(
                "_compute_spread_quality failed for %s", symbol, exc_info=True
            )
            return 0.0

    def _compute_volatility_usability(self, symbol: str) -> float:
        """Volatility usability score based on 14-period ATR of M1 data.

        Normalises ATR to [0, 1] — higher values indicate usable volatility.
        """
        try:
            rates = self._get_rates(symbol, count=14)
            if not rates or len(rates) < 2:
                return 0.0

            ranges = [abs(r["high"] - r["low"]) for r in rates if r.get("high") and r.get("low")]
            if not ranges:
                return 0.0

            atr = np.mean(ranges)
            score = atr / (atr + 0.001)
            return min(score, 1.0)
        except Exception:
            logger.warning(
                "_compute_volatility_usability failed for %s", symbol, exc_info=True
            )
            return 0.0

    def _compute_regime_activity(self, symbol: str) -> float:
        """Regime activity score based on 14-period RSI of M1 closes.

        RSI close to 50 indicates an active two-sided market; returns
        a score that peaks at 1.0 when RSI == 50.
        """
        try:
            rates = self._get_rates(symbol, count=14)
            if not rates or len(rates) < 2:
                return 0.0

            closes = np.array(
                [r["close"] for r in rates if r.get("close") is not None], dtype=float
            )
            if len(closes) < 2:
                return 0.0

            deltas = np.diff(closes)
            gains = np.where(deltas > 0, deltas, 0.0)
            losses = np.where(deltas < 0, -deltas, 0.0)

            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)

            if avg_loss == 0.0:
                rsi = 100.0
            elif avg_gain == 0.0:
                rsi = 0.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))

            score = 1.0 - abs(rsi - 50.0) / 50.0
            return max(0.0, min(score, 1.0))
        except Exception:
            logger.warning(
                "_compute_regime_activity failed for %s", symbol, exc_info=True
            )
            return 0.0

    # ------------------------------------------------------------------
    # Data fetching helpers
    # ------------------------------------------------------------------

    def _get_rates(self, symbol: str, count: int = 15) -> list:
        """Fetch *count* M1 rates for *symbol*.

        Returns a list of rate tuples converted to named-accessible dicts.
        """
        try:
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, count)
            if rates is None or len(rates) == 0:
                return []
            return [{
                "time": r[0], "open": r[1], "high": r[2],
                "low": r[3], "close": r[4],
                "tick_volume": r[5], "spread": r[6], "real_volume": r[7],
            } for r in rates]
        except Exception:
            logger.warning("_get_rates failed for %s", symbol, exc_info=True)
            return []

    def _collect_meta(self, symbol: str) -> dict:
        """Collect metadata snapshot for a symbol.

        Includes spread_points, ATR, RSI, and tick_coverage so callers
        have raw values alongside the normalised scores.
        """
        meta = {
            "spread_points": None,
            "atr": None,
            "rsi": None,
            "tick_coverage": None,
        }

        try:
            tick = mt5.symbol_info_tick(symbol)
            info = mt5.symbol_info(symbol)
            if tick is not None and info is not None and info.point != 0:
                raw_spread = tick.ask - tick.bid
                meta["spread_points"] = round(raw_spread / info.point, 2)
        except Exception:
            pass

        try:
            rates = self._get_rates(symbol, count=14)
            if rates and len(rates) >= 2:
                ranges = [abs(r["high"] - r["low"]) for r in rates if r.get("high") and r.get("low")]
                if ranges:
                    meta["atr"] = round(float(np.mean(ranges)), 6)

                closes = np.array(
                    [r["close"] for r in rates if r.get("close") is not None], dtype=float
                )
                if len(closes) >= 2:
                    deltas = np.diff(closes)
                    gains = np.where(deltas > 0, deltas, 0.0)
                    losses = np.where(deltas < 0, -deltas, 0.0)
                    avg_gain = np.mean(gains)
                    avg_loss = np.mean(losses)

                    if avg_loss == 0.0:
                        rsi = 100.0
                    elif avg_gain == 0.0:
                        rsi = 0.0
                    else:
                        rs = avg_gain / avg_loss
                        rsi = 100.0 - (100.0 / (1.0 + rs))
                    meta["rsi"] = round(rsi, 2)
        except Exception:
            pass

        try:
            rates60 = self._get_rates(symbol, count=60)
            if rates60:
                tick_count = sum(
                    1
                    for r in rates60
                    if r.get("tick_volume") is not None and r["tick_volume"] > 0
                )
                meta["tick_coverage"] = round(tick_count / 60, 4)
        except Exception:
            pass

        return meta

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(min_spread={self.min_spread}, max_spread={self.max_spread})"
        )


if __name__ == "__main__":
    mt5.initialize()
    sil = SymbolIntelligenceLayer()
    results = sil.score_symbols(
        ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "XAUUSD", "BTCUSD"]
    )
    for r in sorted(results, key=lambda x: x["total_score"], reverse=True):
        print(
            f"{r['symbol']:10s} score={r['total_score']:.3f}  "
            f"liq={r['scores'].get('liquidity', 0):.2f} "
            f"spread={r['scores'].get('spread_quality', 0):.2f} "
            f"vol={r['scores'].get('volatility_usability', 0):.2f} "
            f"act={r['scores'].get('regime_activity', 0):.2f}"
        )
    mt5.shutdown()
