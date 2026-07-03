"""Edge Signal Mapper — Connects deployment manifest edges to the live pipeline.

Reads the 13 validated research edges from deployment_manifest.json and
converts them to OSS-compatible signal dicts that can flow through the
existing GovernancePipeline → MOF → RestrictedExecutionBridge → MT5 flow.

Each edge produces a signal dict with the standard OSS fields:
    symbol, direction, confidence, ecdf, drift, price

Strategy implementations:
    - mean_reversion (RSI-based)
    - vol_expansion (ATR-breakout-based)
    - pullback (EMA-pullback-based)
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("proxima_ops.risk.edge_signal_mapper")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_MANIFEST_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "deployment_manifest.json"
)
_MANIFEST_PATH = os.path.normpath(_MANIFEST_PATH)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RSI_PERIOD = 14


def _compute_rsi(closes: np.ndarray, period: int = _RSI_PERIOD) -> np.ndarray:
    """Compute RSI over *closes*."""
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


def _compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                 period: int = 14) -> np.ndarray:
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


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average."""
    out = np.full_like(values, np.nan)
    if len(values) == 0:
        return out
    out[0] = values[0]
    alpha = 2.0 / (span + 1)
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


# ---------------------------------------------------------------------------
# Strategy Functions
# ---------------------------------------------------------------------------

def _mean_reversion_signal(
    closes: np.ndarray,
    params: dict,
    rsi_values: Optional[np.ndarray] = None,
) -> Tuple[int, float, float, float]:
    """Generate mean-reversion signal from RSI.

    Parameters
    ----------
    closes : np.ndarray
        Close prices.
    params : dict
        Must contain 'rsi_oversold', 'rsi_exit', 'max_hold'.
    rsi_values : np.ndarray, optional
        Pre-computed RSI (computed if None).

    Returns
    -------
    (direction, confidence, rsi_norm, drift)
        direction: -1, 0, +1
        confidence: 0..1
        rsi_norm: 0..1 proxy for ecdf
        drift: -1, 0, +1
    """
    rsi_oversold = params.get("rsi_oversold", 30)
    rsi_exit = params.get("rsi_exit", 50)
    max_hold = params.get("max_hold", 18)

    if rsi_values is None:
        rsi_values = _compute_rsi(closes)

    if len(rsi_values) < 2:
        return 0, 0.0, 0.5, 0

    current_rsi = rsi_values[-1]
    prev_rsi = rsi_values[-2]

    # Oversold bounce: RSI was below oversold, now recovering upward
    if current_rsi <= rsi_oversold:
        # Strong oversold — direction +1 (buy)
        depth = max(0, (rsi_oversold - current_rsi) / rsi_oversold)
        confidence = min(1.0, 0.5 + depth * 0.5)
        direction = +1
    elif prev_rsi <= rsi_oversold and current_rsi > rsi_oversold:
        # Just exited oversold — momentum buy
        confidence = 0.4
        direction = +1
    elif current_rsi >= (100 - rsi_oversold) and rsi_oversold < 40:
        # Overbought (symmetric) — direction -1 (sell)
        depth = max(0, (current_rsi - (100 - rsi_oversold)) / rsi_oversold)
        confidence = min(1.0, 0.5 + depth * 0.5)
        direction = -1
    elif current_rsi >= rsi_exit and prev_rsi < rsi_exit:
        # Exiting — neutral signal
        direction = 0
        confidence = 0.3
    else:
        direction = 0
        confidence = 0.1

    # RSI normalized to [0, 1] as ecdf proxy
    rsi_norm = max(0.0, min(1.0, current_rsi / 100.0))

    # Drift: direction of RSI change
    rsi_drift = rsi_values[-1] - (rsi_values[-3] if len(rsi_values) >= 3 else rsi_values[-2])
    drift = 1 if rsi_drift > 2 else (-1 if rsi_drift < -2 else 0)

    return direction, round(confidence, 4), round(rsi_norm, 4), drift


def _vol_expansion_signal(
    closes: np.ndarray,
    params: dict,
    highs: Optional[np.ndarray] = None,
    lows: Optional[np.ndarray] = None,
    atr_values: Optional[np.ndarray] = None,
) -> Tuple[int, float, float, float]:
    """Generate vol-expansion signal from ATR.

    Parameters
    ----------
    closes : np.ndarray
    params : dict
        Must contain 'atr_mult', 'atr_pct_threshold', 'max_hold'.
    highs, lows : np.ndarray, optional
        High/low prices (simulated from closes if None).
    atr_values : np.ndarray, optional

    Returns
    -------
    (direction, confidence, atr_norm, drift)
    """
    atr_mult = params.get("atr_mult", 1.6)
    atr_pct_threshold = params.get("atr_pct_threshold", 60)
    max_hold = params.get("max_hold", 12)

    if highs is None:
        highs = closes * 1.002  # simulated high
    if lows is None:
        lows = closes * 0.998  # simulated low

    if atr_values is None:
        atr_values = _compute_atr(highs, lows, closes)

    if len(atr_values) < 20 or len(closes) < 20:
        return 0, 0.0, 0.5, 0

    current_atr = atr_values[-1]
    baseline_atr = np.nanmedian(atr_values[-50:]) if len(atr_values) >= 50 else np.nanmean(atr_values[~np.isnan(atr_values)])

    if baseline_atr <= 0 or np.isnan(baseline_atr):
        return 0, 0.0, 0.5, 0

    # ATR ratio
    atr_ratio = current_atr / baseline_atr

    # Check if ATR exceeds threshold (using percentile-based threshold)
    valid_atr = atr_values[~np.isnan(atr_values)]
    atr_threshold = np.percentile(valid_atr, atr_pct_threshold) if len(valid_atr) > 10 else baseline_atr

    # Recent price direction
    price_change = closes[-1] - closes[-5] if len(closes) >= 5 else 0

    if current_atr >= atr_threshold * atr_mult:
        # Volatility expansion detected
        expansion_strength = min(1.0, (atr_ratio - 1.0) / 2.0)
        confidence = min(1.0, 0.5 + expansion_strength * 0.5)
        # Direction follows price breakout
        direction = 1 if price_change > 0 else (-1 if price_change < 0 else 0)
    else:
        direction = 0
        confidence = 0.1

    # ATR normalized as ecdf proxy (using percentile rank)
    valid_sorted = np.sort(valid_atr)
    atr_pos = np.searchsorted(valid_sorted, current_atr) / max(len(valid_sorted), 1)
    atr_norm = max(0.0, min(1.0, float(atr_pos)))

    # Drift: ATR direction
    atr_drift = atr_values[-1] - (atr_values[-3] if len(atr_values) >= 3 else atr_values[-2])
    drift = 1 if atr_drift > baseline_atr * 0.1 else (-1 if atr_drift < -baseline_atr * 0.1 else 0)

    return direction, round(confidence, 4), round(atr_norm, 4), drift


def _pullback_signal(
    closes: np.ndarray,
    params: dict,
) -> Tuple[int, float, float, float]:
    """Generate pullback signal from EMA cross.

    Parameters
    ----------
    closes : np.ndarray
    params : dict
        Must contain 'trend_ema', 'pullback_ema', 'max_hold'.

    Returns
    -------
    (direction, confidence, norm_price, drift)
    """
    trend_span = params.get("trend_ema", 100)
    pullback_span = params.get("pullback_ema", 10)
    max_hold = params.get("max_hold", 18)

    if len(closes) < trend_span + 5:
        return 0, 0.0, 0.5, 0

    trend = _ema(closes, trend_span)
    pullback = _ema(closes, pullback_span)

    if np.isnan(trend[-1]) or np.isnan(pullback[-1]):
        return 0, 0.0, 0.5, 0

    # Trend direction
    trend_up = trend[-1] > trend[-5] if len(trend) >= 5 else False
    trend_down = trend[-1] < trend[-5] if len(trend) >= 5 else False

    # Pullback: price moves toward pullback_ema in opposite direction of trend
    price = closes[-1]
    trend_val = trend[-1]
    pull_val = pullback[-1]

    # Distance from price to pullback EMA (as fraction of price)
    dist_to_pull = abs(price - pull_val) / price

    # Distance from pullback EMA to trend EMA (as fraction of price)
    dist_pull_to_trend = abs(pull_val - trend_val) / max(price, 1e-12)

    direction = 0
    confidence = 0.1

    if trend_up and price <= pull_val:
        # Uptrend pullback to pullback EMA → buy
        pullback_depth = max(0, min(1.0, 1.0 - dist_to_pull / max(dist_pull_to_trend + 0.0001, 0.0001)))
        direction = +1
        confidence = min(1.0, 0.4 + pullback_depth * 0.4)
    elif trend_down and price >= pull_val:
        # Downtrend rally to pullback EMA → sell
        pullback_depth = max(0, min(1.0, 1.0 - dist_to_pull / max(dist_pull_to_trend + 0.0001, 0.0001)))
        direction = -1
        confidence = min(1.0, 0.4 + pullback_depth * 0.4)

    # Normalized price position as ecdf proxy (percentile rank)
    price_percentile = np.searchsorted(np.sort(closes), price) / max(len(closes), 1)
    norm_price = max(0.0, min(1.0, float(price_percentile)))

    # Drift: price direction
    price_drift = closes[-1] - (closes[-3] if len(closes) >= 3 else closes[-2])
    drift = 1 if price_drift > 0 else (-1 if price_drift < 0 else 0)

    return direction, round(confidence, 4), round(norm_price, 4), drift


# ---------------------------------------------------------------------------
# Strategy Registry
# ---------------------------------------------------------------------------

_STRATEGY_FUNCTIONS = {
    "mean_reversion": _mean_reversion_signal,
    "vol_expansion": _vol_expansion_signal,
    "pullback": _pullback_signal,
}


# ---------------------------------------------------------------------------
# Edge Signal Mapper
# ---------------------------------------------------------------------------

class EdgeSignalMapper:
    """Maps deployment manifest edges to live OSS-compatible signals.

    Loads the 13 edges from deployment_manifest.json and provides methods
    to generate signals from market data that are compatible with the
    existing GovernancePipeline → MOF → Bridge execution path.

    Usage::

        mapper = EdgeSignalMapper()
        signals = mapper.generate_all(closes_dict)

        # Or for a specific symbol:
        edges = mapper.get_edges_for_symbol("EURUSD")
        signals = mapper.generate_for_symbol("EURUSD", closes)
    """

    def __init__(self, manifest_path: Optional[str] = None) -> None:
        self._manifest_path = manifest_path or _MANIFEST_PATH
        self._edges: List[dict] = []
        self._by_symbol: Dict[str, List[dict]] = {}
        self._load_manifest()
        self._last_signals: List[dict] = []
        self._signal_timestamp: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def edges(self) -> List[dict]:
        """All 13 edge definitions from the manifest."""
        return list(self._edges)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def get_edges_for_symbol(self, symbol: str) -> List[dict]:
        """Return edge definitions for a given symbol."""
        return list(self._by_symbol.get(symbol, []))

    def generate_all(
        self,
        closes_by_symbol: Dict[str, np.ndarray],
        highs_by_symbol: Optional[Dict[str, np.ndarray]] = None,
        lows_by_symbol: Optional[Dict[str, np.ndarray]] = None,
        prices_by_symbol: Optional[Dict[str, float]] = None,
    ) -> List[dict]:
        """Generate signals for ALL edges using per-symbol OHLC data.

        Parameters
        ----------
        closes_by_symbol : dict
            Symbol → ndarray of close prices.
        highs_by_symbol : dict, optional
            Symbol → ndarray of high prices.
        lows_by_symbol : dict, optional
            Symbol → ndarray of low prices.
        prices_by_symbol : dict, optional
            Symbol → current price (uses closes[-1] if not provided).

        Returns
        -------
        list of dict
            Signal dicts in OSS format, one per edge that produced a
            non-zero direction.
        """
        signals: List[dict] = []

        for edge in self._edges:
            symbol = edge["symbol"]
            closes = closes_by_symbol.get(symbol)
            if closes is None or len(closes) < 20:
                continue

            highs = highs_by_symbol.get(symbol) if highs_by_symbol else None
            lows = lows_by_symbol.get(symbol) if lows_by_symbol else None
            price = prices_by_symbol.get(symbol, float(closes[-1])) if prices_by_symbol else float(closes[-1])

            sig = self._generate_edge_signal(edge, closes, highs, lows, price)
            if sig is not None:
                signals.append(sig)

        self._last_signals = signals
        self._signal_timestamp = datetime.now().isoformat()
        return signals

    def generate_for_symbol(
        self,
        symbol: str,
        closes: np.ndarray,
        highs: Optional[np.ndarray] = None,
        lows: Optional[np.ndarray] = None,
        price: Optional[float] = None,
    ) -> List[dict]:
        """Generate signals for all edges matching *symbol*."""
        edges = self.get_edges_for_symbol(symbol)
        signals: List[dict] = []
        for edge in edges:
            sig = self._generate_edge_signal(
                edge, closes, highs, lows,
                price or float(closes[-1] if len(closes) > 0 else 0),
            )
            if sig is not None:
                signals.append(sig)
        return signals

    def get_edge_status(self, edge_id: str) -> dict:
        """Get status info for a specific edge."""
        for edge in self._edges:
            if edge["id"] == edge_id:
                return {
                    "edge": edge,
                    "active": any(
                        s.get("edge_id") == edge_id
                        for s in self._last_signals
                    ),
                }
        return {"edge": None, "active": False}

    def get_last_signals(self) -> List[dict]:
        """Return the most recently generated signal batch."""
        return list(self._last_signals)

    def get_symbols_with_edges(self) -> List[str]:
        """Return list of symbols that have edge definitions."""
        return sorted(self._by_symbol.keys())

    def get_summary(self) -> dict:
        """Return a summary of mapper state."""
        active = len(self._last_signals)
        return {
            "total_edges": len(self._edges),
            "symbols": list(self._by_symbol.keys()),
            "active_signals": active,
            "signal_timestamp": self._signal_timestamp,
            "symbol_breakdown": {
                sym: len(edges)
                for sym, edges in self._by_symbol.items()
            },
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_manifest(self) -> None:
        """Load and parse deployment_manifest.json."""
        if not os.path.exists(self._manifest_path):
            logger.warning(
                "Manifest not found at %s — edge mapper will be empty",
                self._manifest_path,
            )
            self._edges = []
            self._by_symbol = {}
            return

        try:
            with open(self._manifest_path, "r") as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load manifest: %s", exc)
            self._edges = []
            self._by_symbol = {}
            return

        self._edges = manifest.get("edges", [])
        self._by_symbol.clear()
        for edge in self._edges:
            sym = edge.get("symbol", "UNKNOWN")
            self._by_symbol.setdefault(sym, []).append(edge)

        logger.info(
            "Loaded %d edges across %d symbols from %s",
            len(self._edges),
            len(self._by_symbol),
            self._manifest_path,
        )

    def _generate_edge_signal(
        self,
        edge: dict,
        closes: np.ndarray,
        highs: Optional[np.ndarray] = None,
        lows: Optional[np.ndarray] = None,
        price: float = 0.0,
    ) -> Optional[dict]:
        """Generate an OSS-format signal dict from one edge definition."""
        strategy = edge.get("strategy", "")
        params = edge.get("params", {})
        edge_id = edge.get("id", "unknown")
        symbol = edge.get("symbol", "UNKNOWN")

        if strategy not in _STRATEGY_FUNCTIONS:
            logger.debug("Unknown strategy %s for edge %s", strategy, edge_id)
            return None

        func = _STRATEGY_FUNCTIONS[strategy]

        try:
            if strategy == "vol_expansion":
                direction, confidence, ecdf_val, drift = func(
                    closes, params, highs, lows
                )
            elif strategy == "pullback":
                direction, confidence, ecdf_val, drift = func(closes, params)
            else:
                direction, confidence, ecdf_val, drift = func(closes, params)
        except Exception as exc:
            logger.error("Edge %s (%s) signal error: %s", edge_id, strategy, exc)
            return None

        if direction == 0 and confidence < 0.2:
            # Non-signal — still return it for diagnostics
            pass

        return {
            "symbol": symbol,
            "direction": direction,
            "confidence": confidence,
            "ecdf": ecdf_val,
            "drift": drift,
            "price": price,
            "edge_id": edge_id,
            "strategy": strategy,
            "params": params,
            "edge_pf": edge.get("pf_after_costs", 0.0),
            "source": f"edge_{edge_id}",
        }


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def create_edge_signals(
    closes_by_symbol: Dict[str, np.ndarray],
    manifest_path: Optional[str] = None,
) -> List[dict]:
    """One-shot: load manifest, generate signals for all edges, return them.

    Parameters
    ----------
    closes_by_symbol : dict
        Symbol → ndarray of close prices.
    manifest_path : str, optional

    Returns
    -------
    list of dict
        OSS-format signal dicts.
    """
    mapper = EdgeSignalMapper(manifest_path=manifest_path)
    return mapper.generate_all(closes_by_symbol)


def format_edge_signals(signals: List[dict]) -> str:
    """Pretty-print edge signals."""
    if not signals:
        return "  No edge signals"

    lines = []
    lines.append(f"\n  Edge Signals ({len(signals)} active):")
    lines.append(f"  {'Edge ID':<12s} {'Symbol':<8s} {'Strategy':<16s} "
                 f"{'Dir':<5s} {'Conf':<7s} {'ECDF':<7s} {'Drift':<6s} "
                 f"{'PF':<7s}")
    lines.append(f"  {'-' * 72}")
    for s in sorted(signals, key=lambda x: (x.get("symbol", ""), x.get("edge_id", ""))):
        lines.append(
            f"  {s.get('edge_id', '?'):<12s} {s.get('symbol', '?'):<8s} "
            f"{s.get('strategy', '?'):<16s} "
            f"{s.get('direction', 0):+d}    "
            f"{s.get('confidence', 0):.3f}  "
            f"{s.get('ecdf', 0):.3f}  "
            f"{s.get('drift', 0):+d}     "
            f"{s.get('edge_pf', 0):.3f}"
        )
    return "\n".join(lines)
