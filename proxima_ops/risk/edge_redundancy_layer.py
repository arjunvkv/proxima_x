"""Edge Redundancy Layer — orthogonal signal projections.

Produces two additional signal geometries per edge:
- PRESSURE: pre-event compression (RSI slope reversal, volatility compression, distance-from-mean)
- MOMENTUM: transition dynamics (RSI velocity, price acceleration, micro-trend break)

These are not "softer thresholds" — they detect different temporal positions
within the same latent regime event."""
import logging
from typing import Optional
import numpy as np

logger = logging.getLogger("proxima_ops.risk.erp")

SYMBOLS = ["EURUSD", "GBPUSD", "EURJPY", "USDJPY"]


def _ema(arr: np.ndarray, period: int = 20) -> np.ndarray:
    if len(arr) < 2:
        return arr.copy()
    result = arr.copy()
    alpha = 2.0 / (period + 1)
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result


def _compute_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    if len(closes) < period + 1:
        return np.full_like(closes, 50.0)
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.full_like(closes, np.nan)
    avg_loss = np.full_like(closes, np.nan)
    avg_gain[period] = gains[:period].mean()
    avg_loss[period] = losses[:period].mean()
    for i in range(period + 1, len(closes)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period
    rs = avg_gain / np.maximum(avg_loss, 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = 50.0
    return rsi


def _compute_atr(highs: np.ndarray, lows: np.ndarray,
                 closes: np.ndarray, period: int = 14) -> np.ndarray:
    if len(closes) < period + 1:
        return np.full_like(closes, 0.0)
    tr = np.zeros(len(closes))
    for i in range(1, len(closes)):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]))
    atr = np.full_like(closes, np.nan)
    atr[period] = tr[1:period + 1].mean()
    for i in range(period + 1, len(closes)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    atr[:period] = 0.0
    return atr


class EdgeRedundancyLayer:
    """Generate PRESSURE and MOMENTUM signals from market data.
    Edge IDs are stable across cycles: erp_pressure_{symbol}, erp_momentum_{symbol}."""

    def __init__(self):
        self._manifests: list[dict] = []

    def load_manifests(self, manifests: list[dict]):
        self._manifests = manifests

    def generate_pressure_signals(self, closes_by_symbol: dict,
                                   highs_by_symbol: dict,
                                   lows_by_symbol: dict) -> list[dict]:
        """PRESSURE: detect pre-event compression that precedes regime extremes.
        
        Signal fires when:
        - RSI is 30-40 or 60-70 (near but not at extreme)
        - RSI slope is pointing TOWARD extreme (increasing if >50, decreasing if <50)
        - ATR is compressing (short-term ATR / long-term ATR < 0.8)
        - Price is near EMA(20) (distance < 0.5×ATR)
        """
        signals = []
        for sym in SYMBOLS:
            cls = closes_by_symbol.get(sym)
            his = highs_by_symbol.get(sym)
            los = lows_by_symbol.get(sym)
            if cls is None or len(cls) < 30:
                continue

            rsi = _compute_rsi(cls)
            ema20 = _ema(cls, 20)
            atr14 = _compute_atr(his, los, cls, 14)
            current_rsi = rsi[-1]
            prev_rsi = rsi[-2]
            current_price = cls[-1]
            current_atr = atr14[-1]

            if current_atr <= 0 or np.isnan(current_atr):
                continue

            # RSI slope direction (positive = moving up, negative = moving down)
            rsi_slope = current_rsi - rsi[-3] if len(rsi) >= 3 else current_rsi - prev_rsi

            # Price distance from EMA, normalized by ATR
            dist_from_ema = (current_price - ema20[-1]) / max(current_atr, 1e-10)

            # ATR compression: ratio of last 3 ATR values to last 14 ATR values
            atr_short = np.mean(atr14[-3:]) if len(atr14) >= 3 else current_atr
            atr_compression = atr_short / max(current_atr, 1e-10)

            # Direction detection
            direction = 0
            confidence = 0.0

            # Bullish PRESSURE: RSI coming UP from 30-40 zone, slope positive
            if 30 <= prev_rsi <= 42 and prev_rsi <= 42 and rsi_slope > 0:
                direction = 1  # BUY
                slope_contribution = min(abs(rsi_slope) / 5.0, 1.0) * 0.3
                atr_contrib = max(0, 1.0 - atr_compression) * 0.3 if atr_compression < 0.85 else 0.1
                distance_contrib = max(0, 0.3 - abs(dist_from_ema) * 0.5) if abs(dist_from_ema) < 0.6 else 0.0
                confidence = 0.40 + slope_contribution + atr_contrib + distance_contrib

            # Bearish PRESSURE: RSI coming DOWN from 60-70 zone, slope negative
            elif 58 <= prev_rsi <= 70 and prev_rsi >= 58 and rsi_slope < 0:
                direction = -1  # SELL
                slope_contribution = min(abs(rsi_slope) / 5.0, 1.0) * 0.3
                atr_contrib = max(0, 1.0 - atr_compression) * 0.3 if atr_compression < 0.85 else 0.1
                distance_contrib = max(0, 0.3 - abs(dist_from_ema) * 0.5) if abs(dist_from_ema) < 0.6 else 0.0
                confidence = 0.40 + slope_contribution + atr_contrib + distance_contrib

            if direction != 0:
                eid = f"erp_pressure_{sym}"
                signals.append({
                    "edge_id": eid,
                    "parent_edge_id": f"pressure_{sym}",
                    "symbol": sym,
                    "direction": direction,
                    "confidence": min(confidence, 0.95),
                    "strategy": "pressure",
                    "has_active_signal": "True",
                    "price": float(current_price),
                    "side": "BUY" if direction > 0 else "SELL",
                })

        return signals

    def generate_momentum_signals(self, closes_by_symbol: dict,
                                   highs_by_symbol: dict,
                                   lows_by_symbol: dict) -> list[dict]:
        """MOMENTUM: detect transition dynamics as price enters regime space.
        
        Signal fires when:
        - RSI velocity (abs change over N bars) exceeds threshold
        - Price acceleration (change in displacement velocity) is high
        - Micro-trend: consecutive bars moving in same direction
        """
        signals = []
        for sym in SYMBOLS:
            cls = closes_by_symbol.get(sym)
            his = highs_by_symbol.get(sym)
            los = lows_by_symbol.get(sym)
            if cls is None or len(cls) < 30:
                continue

            rsi = _compute_rsi(cls)
            ema20 = _ema(cls, 20)
            atr14 = _compute_atr(his, los, cls, 14)
            current_rsi = rsi[-1]
            current_price = cls[-1]
            current_atr = atr14[-1]

            if current_atr <= 0 or np.isnan(current_atr):
                continue

            distance = (current_price - ema20[-1]) / max(current_atr, 1e-10)

            # RSI velocity: how fast RSI is changing over last 5 bars
            lookback = min(5, len(rsi) - 1)
            rsi_velocity = abs(current_rsi - rsi[-lookback]) / lookback if lookback > 0 else 0

            # Price acceleration: change in displacement
            prev_distance = (cls[-2] - ema20[-2]) / max(atr14[-2] if not np.isnan(atr14[-2]) else current_atr, 1e-10) if len(cls) >= 2 else 0
            displacement_velocity = abs(distance - prev_distance) if len(cls) >= 2 else 0

            # Micro-trend: consecutive directional closes
            if len(cls) >= 4:
                recent_deltas = np.diff(cls[-4:])
                micro_up = np.sum(recent_deltas > 0)
                micro_down = np.sum(recent_deltas < 0)
            else:
                micro_up = 0
                micro_down = 0

            direction = 0
            confidence = 0.0

            # Bullish MOMENTUM
            if current_rsi >= 45 and current_rsi <= 65:
                if rsi_velocity > 2.0 and displacement_velocity > 0.3 and micro_up >= micro_down:
                    direction = 1
                    contrib_v = min(rsi_velocity / 8.0, 1.0) * 0.3
                    contrib_d = min(displacement_velocity, 1.0) * 0.2
                    contrib_m = (micro_up - micro_down) * 0.1
                    confidence = 0.40 + contrib_v + contrib_d + contrib_m

            # Bearish MOMENTUM
            elif current_rsi >= 35 and current_rsi <= 55:
                if rsi_velocity > 2.0 and displacement_velocity > 0.3 and micro_down >= micro_up:
                    direction = -1
                    contrib_v = min(rsi_velocity / 8.0, 1.0) * 0.3
                    contrib_d = min(displacement_velocity, 1.0) * 0.2
                    contrib_m = (micro_down - micro_up) * 0.1
                    confidence = 0.40 + contrib_v + contrib_d + contrib_m

            if direction != 0:
                eid = f"erp_momentum_{sym}"
                signals.append({
                    "edge_id": eid,
                    "parent_edge_id": f"momentum_{sym}",
                    "symbol": sym,
                    "direction": direction,
                    "confidence": min(confidence, 0.95),
                    "strategy": "momentum",
                    "has_active_signal": "True",
                    "price": float(current_price),
                    "side": "BUY" if direction > 0 else "SELL",
                })

        return signals

    def generate_all(self, closes_by_symbol: dict,
                     highs_by_symbol: dict,
                     lows_by_symbol: dict) -> list[dict]:
        """Generate all ERL signals (PRESSURE + MOMENTUM)."""
        signals = []
        signals.extend(self.generate_pressure_signals(closes_by_symbol, highs_by_symbol, lows_by_symbol))
        signals.extend(self.generate_momentum_signals(closes_by_symbol, highs_by_symbol, lows_by_symbol))
        return signals
