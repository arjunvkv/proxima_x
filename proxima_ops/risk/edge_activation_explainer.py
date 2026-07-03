"""Edge Activation Explainer — per-edge causality attribution for signal scarcity.

For each edge per cycle, captures:
  - Feature state (RSI, ATR, EMA, price delta, volatility)
  - Activation failure reason (categorical: RSI_NOT_OVERSOLD, ATR_THRESHOLD_NOT_CROSSED, etc.)
  - Distance-to-trigger metric (0-1): how close to activation threshold
"""

from __future__ import annotations

import json, os, math, time
from typing import Optional, Any
import numpy as np

_MANIFEST_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "deployment_manifest.json")
)

_RSI_PERIOD = 14

# ---- helpers (mirror edge_signal_mapper) ----

def _compute_rsi(closes: np.ndarray, period: int = _RSI_PERIOD) -> np.ndarray:
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.full_like(closes, np.nan)
    avg_loss = np.full_like(closes, np.nan)
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    for i in range(period + 1, len(closes)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period
    rs = avg_gain / np.maximum(avg_loss, 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = 50.0
    return rsi

def _compute_atr(highs, lows, closes, period=14):
    n = min(len(highs), len(lows), len(closes))
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
    atr = np.full(n, np.nan)
    atr[period] = np.mean(tr[:period])
    for i in range(period + 1, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr

def _ema(values: np.ndarray, span: int) -> np.ndarray:
    out = np.full_like(values, np.nan)
    if len(values) == 0: return out
    out[0] = values[0]
    alpha = 2.0 / (span + 1)
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out

# ---- failure reason constants ----

FAILURE_REASONS = {
    "mean_reversion": [
        "RSI_NOT_OVERSOLD",
        "RSI_NOT_OVERBOUGHT",
        "RSI_EXITING_NEUTRAL",
        "RSI_MOMENTUM_NEUTRAL",
    ],
    "vol_expansion": [
        "ATR_THRESHOLD_NOT_CROSSED",
        "PRICE_DIRECTION_AMBIGUOUS",
    ],
    "pullback": [
        "TREND_AMBIGUOUS",
        "NO_UPTREND_PULLBACK",
        "NO_DOWNTREND_RALLY",
        "EMA_DISTANCE_TOO_LARGE",
    ],
}

# ---- explainer ----

class EdgeActivationExplainer:
    """Per-edge, per-cycle explanation of why no signal was produced."""

    def __init__(self, manifest_path: Optional[str] = None):
        self._manifest_path = manifest_path or _MANIFEST_PATH
        self._edges: list[dict] = []
        self._load_manifest()

    def _load_manifest(self):
        if not os.path.exists(self._manifest_path):
            self._edges = []
            return
        with open(self._manifest_path) as f:
            manifest = json.load(f)
        self._edges = manifest.get("edges", [])

    def explain_all(
        self,
        closes_by_symbol: dict[str, np.ndarray],
        highs_by_symbol: Optional[dict[str, np.ndarray]] = None,
        lows_by_symbol: Optional[dict[str, np.ndarray]] = None,
        mof_state: str = "INFORMATION_RICH",
    ) -> list[dict]:
        """Return per-edge explanation dicts."""
        explanations = []
        for edge in self._edges:
            symbol = edge["symbol"]
            closes = closes_by_symbol.get(symbol)
            if closes is None or len(closes) < 20:
                continue
            highs = (highs_by_symbol or {}).get(symbol)
            lows = (lows_by_symbol or {}).get(symbol)
            price = float(closes[-1])
            expl = self._explain_edge(edge, closes, highs, lows, price, mof_state)
            if expl:
                explanations.append(expl)
        return explanations

    def _explain_edge(
        self, edge: dict, closes: np.ndarray,
        highs: Optional[np.ndarray], lows: Optional[np.ndarray],
        price: float, mof_state: str,
    ) -> Optional[dict]:
        strategy = edge.get("strategy", "")
        params = edge.get("params", {})
        edge_id = edge.get("id", "unknown")
        symbol = edge.get("symbol", "UNKNOWN")

        if strategy == "mean_reversion":
            return self._explain_mean_reversion(edge_id, symbol, strategy, params, closes, mof_state)
        elif strategy == "vol_expansion":
            return self._explain_vol_expansion(edge_id, symbol, strategy, params, closes, highs, lows, mof_state)
        elif strategy == "pullback":
            return self._explain_pullback(edge_id, symbol, strategy, params, closes, mof_state)
        return None

    def _explain_mean_reversion(self, edge_id, symbol, strategy, params, closes, mof_state):
        rsi = _compute_rsi(closes)
        current_rsi = float(rsi[-1])
        prev_rsi = float(rsi[-2])
        rsi_oversold = params.get("rsi_oversold", 30)
        rsi_exit = params.get("rsi_exit", 50)
        oversold_thresh = rsi_oversold
        overbought_thresh = 100 - rsi_oversold if rsi_oversold < 40 else 70

        failure = "MOMENTUM_NEUTRAL"
        distance_to_trigger = 1.0

        if current_rsi <= oversold_thresh:
            failure = None
            distance_to_trigger = 0.0
        elif prev_rsi <= oversold_thresh and current_rsi > oversold_thresh:
            failure = None
            distance_to_trigger = 0.0
        elif current_rsi >= overbought_thresh and rsi_oversold < 40:
            failure = None
            distance_to_trigger = 0.0
        elif current_rsi >= rsi_exit and prev_rsi < rsi_exit:
            failure = "RSI_EXITING_NEUTRAL"
            distance_to_trigger = max(0, (current_rsi - rsi_exit) / (100 - rsi_exit))
        else:
            # Neutral zone: how far from oversold?
            if current_rsi < 50:
                gap = oversold_thresh - current_rsi
                if gap > 0:
                    distance_to_trigger = gap / oversold_thresh
                    failure = "RSI_NOT_OVERSOLD"
                else:
                    distance_to_trigger = abs(gap) / 50.0
                    failure = "RSI_MOMENTUM_NEUTRAL"
            else:
                gap = current_rsi - overbought_thresh
                if gap > 0:
                    distance_to_trigger = gap / (100 - overbought_thresh)
                    failure = "RSI_NOT_OVERBOUGHT"
                else:
                    distance_to_trigger = abs(gap) / 50.0
                    failure = "RSI_MOMENTUM_NEUTRAL"

        return {
            "edge_id": edge_id, "symbol": symbol, "strategy": strategy,
            "failure_reason": failure,
            "distance_to_trigger": round(min(distance_to_trigger, 1.0), 4),
            "direction": 0 if failure else (+1 if current_rsi <= oversold_thresh else -1),
            "features": {
                "current_rsi": round(current_rsi, 2),
                "prev_rsi": round(prev_rsi, 2),
                "rsi_oversold": rsi_oversold,
                "rsi_exit": rsi_exit,
                "rsi_delta": round(current_rsi - prev_rsi, 2),
            },
        }

    def _explain_vol_expansion(self, edge_id, symbol, strategy, params, closes, highs, lows, mof_state):
        if highs is None: highs = closes * 1.002
        if lows is None: lows = closes * 0.998
        atr = _compute_atr(highs, lows, closes)
        current_atr = float(atr[-1]) if not np.isnan(atr[-1]) else 0.0
        valid_atr = atr[~np.isnan(atr)]
        baseline_atr = float(np.nanmedian(valid_atr[-50:])) if len(valid_atr) >= 50 else float(np.nanmean(valid_atr)) if len(valid_atr) > 0 else 0.0
        atr_mult = params.get("atr_mult", 1.6)
        atr_pct_threshold = params.get("atr_pct_threshold", 60)
        price_change = float(closes[-1] - closes[-5]) if len(closes) >= 5 else 0.0

        if baseline_atr <= 0 or np.isnan(baseline_atr):
            return {
                "edge_id": edge_id, "symbol": symbol, "strategy": strategy,
                "failure_reason": "VOLATILITY_TOO_LOW",
                "distance_to_trigger": 1.0, "direction": 0,
                "features": {"current_atr": 0, "baseline_atr": 0, "atr_ratio": 0, "price_change": 0},
            }

        atr_ratio = current_atr / baseline_atr
        atr_threshold = float(np.percentile(valid_atr, atr_pct_threshold)) if len(valid_atr) > 10 else baseline_atr
        trigger_level = atr_threshold * atr_mult
        distance = max(0, trigger_level - current_atr) / max(trigger_level, 1e-12)

        failure = "ATR_THRESHOLD_NOT_CROSSED"
        if current_atr >= trigger_level:
            if abs(price_change) < 0.0001 * float(closes[-1]):
                failure = "PRICE_DIRECTION_AMBIGUOUS"
            else:
                failure = None

        return {
            "edge_id": edge_id, "symbol": symbol, "strategy": strategy,
            "failure_reason": failure,
            "distance_to_trigger": round(min(distance, 1.0), 4),
            "direction": 0,
            "features": {
                "current_atr": round(current_atr, 5),
                "baseline_atr": round(baseline_atr, 5),
                "atr_ratio": round(atr_ratio, 4),
                "trigger_level": round(trigger_level, 5),
                "price_change": round(price_change, 5),
            },
        }

    def _explain_pullback(self, edge_id, symbol, strategy, params, closes, mof_state):
        trend_span = params.get("trend_ema", 100)
        pullback_span = params.get("pullback_ema", 10)
        if len(closes) < trend_span + 5:
            return {
                "edge_id": edge_id, "symbol": symbol, "strategy": strategy,
                "failure_reason": "INSUFFICIENT_DATA",
                "distance_to_trigger": 1.0, "direction": 0,
                "features": {"reason": f"need {trend_span+5} bars, got {len(closes)}"},
            }

        trend = _ema(closes, trend_span)
        pullback = _ema(closes, pullback_span)
        price = float(closes[-1])
        trend_val = float(trend[-1])
        pull_val = float(pullback[-1])

        if np.isnan(trend_val) or np.isnan(pull_val):
            return {
                "edge_id": edge_id, "symbol": symbol, "strategy": strategy,
                "failure_reason": "EMA_NAN",
                "distance_to_trigger": 1.0, "direction": 0,
                "features": {},
            }

        trend_up = trend_val > float(trend[-5]) if len(trend) >= 5 else False
        trend_down = trend_val < float(trend[-5]) if len(trend) >= 5 else False

        dist_to_pull = abs(price - pull_val) / max(price, 1e-12)
        dist_pull_to_trend = abs(pull_val - trend_val) / max(price, 1e-12)
        max_dist = max(dist_pull_to_trend + 0.0001, 0.0001)

        failure = "TREND_AMBIGUOUS"
        direction = 0
        distance_to_trigger = 1.0

        if trend_up and price <= pull_val:
            depth = max(0, min(1.0, 1.0 - dist_to_pull / max_dist))
            if depth > 0.5:
                failure = None
                direction = +1
                distance_to_trigger = 1.0 - depth
            else:
                failure = "NO_UPTREND_PULLBACK"
                distance_to_trigger = 1.0 - depth
        elif trend_down and price >= pull_val:
            depth = max(0, min(1.0, 1.0 - dist_to_pull / max_dist))
            if depth > 0.5:
                failure = None
                direction = -1
                distance_to_trigger = 1.0 - depth
            else:
                failure = "NO_DOWNTREND_RALLY"
                distance_to_trigger = 1.0 - depth
        else:
            failure = "TREND_AMBIGUOUS"
            distance_to_trigger = 1.0

        return {
            "edge_id": edge_id, "symbol": symbol, "strategy": strategy,
            "failure_reason": failure,
            "distance_to_trigger": round(min(distance_to_trigger, 1.0), 4),
            "direction": direction,
            "features": {
                "trend_ema": round(trend_val, 5),
                "pullback_ema": round(pull_val, 5),
                "price": round(price, 5),
                "trend_up": trend_up,
                "trend_down": trend_down,
                "dist_to_pull": round(dist_to_pull, 6),
                "dist_pull_to_trend": round(dist_pull_to_trend, 6),
            },
        }

    def summary(self, explanations: list[dict]) -> dict:
        counts = {}
        for e in explanations:
            r = e.get("failure_reason")
            if r:
                counts[r] = counts.get(r, 0) + 1
        return {
            "total_edges": len(explanations),
            "total_active": sum(1 for e in explanations if e.get("direction") != 0),
            "failure_counts": counts,
            "avg_distance_to_trigger": round(
                sum(e.get("distance_to_trigger", 1.0) for e in explanations) / max(len(explanations), 1), 4
            ),
        }
