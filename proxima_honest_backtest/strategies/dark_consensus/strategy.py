"""
Dark Consensus — Universal Cross-Broker Version (Spec v1.0)

Design per CROSS_BROKER_STRATEGY_SPEC.md:
  §2  Market structure: broad multi-pair directional agreement, not tick patterns
  §7  Time-defined windows: M5 UTC bars, not tick counts
  §9  Decision margin: trade only when |z| >> z_entry
  §10 Margin tracking: confidence = f(margin / z_entry)
  §11 Hysteresis: entry at |z| > z_entry, exit at |z| < z_exit
  §12 Cross-pair confirmation: 3 independent pairs must agree on direction
  §13 Currency strength: decompose pair returns into EUR/USD/JPY/GBP states
  §14 Consensus features: direction + magnitude + currency + timescale all agree
  §15-16 Persistence: conditions required for N consecutive bars before entry
  §17 Multi-timescale: short (1-bar) + medium (3-bar) returns must agree
  §19 Relative quantities: z-scores normalize returns by rolling volatility
  §22 Closed information: completed bars only, no look-ahead
  §23 Exact formulas: all math explicitly defined below
"""
import math
from collections import deque
from typing import Any, Dict, List, Optional

from proxima_honest_backtest.engine.types import SignalResult
from proxima_honest_backtest.strategies.multi_pair_base import MultiPairStrategy

def _sign(x: float) -> float:
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)

# ---------------------------------------------------------------------------
# §23: Exact mathematical specifications
# ---------------------------------------------------------------------------
# mid(t)  = close price of completed M5 bar at UTC time t
# r_i(T)  = ln(mid_i(t) / mid_i(t - T))    for pair i, lookback T
# z_i(T)  = (r_i(T) - mean(r_i over last W bars)) / std(r_i over last W bars)
#           where W = z_window
# EUR strength = z_EURUSD(T) + z_EURJPY(T)          (pairs where EUR is base)
# JPY strength = -z_EURJPY(T) - z_GBPJPY(T)         (pairs where JPY is quote)
# USD strength = z_EURUSD(T)                          (EURUSD has USD as quote)
# GBP strength = z_GBPJPY(T)                          (GBPJPY has GBP as base)
# consensus_dir = sign(mean(z_i for all pairs))
# margin = mean(|z_i|) - z_entry
# entry  iff: all z_i have same sign AND mean(|z_i|) > z_entry
#            AND short and long timescales agree
#            AND conditions hold for persistence_bars
# exit   iff: hold_bars elapsed OR mean(|z_i|) < z_exit
# ---------------------------------------------------------------------------


class _StreamingZScore:
    """O(1) per-bar rolling z-score via running sum/sum-sq over fixed window."""

    __slots__ = ("_window", "_deque", "_sum", "_sum_sq", "_n_obs")

    def __init__(self, window: int) -> None:
        self._window = window
        self._deque: deque = deque(maxlen=window)
        self._sum = 0.0
        self._sum_sq = 0.0
        self._n_obs = 0

    @property
    def ready(self) -> bool:
        return self._n_obs >= self._window

    def update(self, value: float) -> float:
        if self._n_obs >= self._window:
            old = self._deque[0]
            self._sum -= old
            self._sum_sq -= old * old
        self._deque.append(value)
        self._sum += value
        self._sum_sq += value * value
        self._n_obs += 1
        n = min(self._n_obs, self._window)
        if n < 2:
            return 0.0
        mean = self._sum / n
        var = max(self._sum_sq / n - mean * mean, 0.0)
        std = math.sqrt(var) if var > 1e-24 else 1e-12
        return (value - mean) / std


class DarkConsensusStrategy(MultiPairStrategy):
    """Cross-pair directional consensus with z-score normalization,
    hysteresis, multi-timescale confirmation, and persistence gating.

    Compliant with CROSS_BROKER_STRATEGY_SPEC.md v1.0.
    """

    DEFAULT_PARAMS: Dict[str, Any] = {
        "pairs": ["EURJPY", "EURUSD", "GBPJPY"],
        "z_entry": 2.0,
        "z_exit": 1.0,
        "lookback_short": 1,
        "lookback_long": 3,
        "z_window": 50,
        "persistence_bars": 1,
        "hold_bars": 1,
        "session_start": 0,
        "session_end": 24,
        "min_confidence": 0.30,
        "require_currency_agreement": True,
    }

    def __init__(self, parameters: Optional[Dict[str, Any]] = None) -> None:
        merged = dict(self.DEFAULT_PARAMS)
        if parameters:
            merged.update(parameters)
        super().__init__(merged)
        self._position: Optional[Dict[str, Any]] = None
        self._persistence_count: int = 0
        self._prev_direction: Optional[float] = None
        self._z_streams: Dict[str, _StreamingZScore] = {}

    def reset(self) -> None:
        self._position = None
        self._persistence_count = 0
        self._prev_direction = None
        self._z_streams = {}

    @property
    def in_position(self) -> bool:
        return self._position is not None

    # ------------------------------------------------------------------
    # §22: Entry logic — closed information only
    # ------------------------------------------------------------------
    def _compute_z_scores(self, history: Dict[str, List[float]], lookback: int) -> Optional[Dict[str, float]]:
        """§19: Compute z-scores of log returns using O(1) streaming."""
        pairs = list(self.parameters["pairs"])
        z_scores: Dict[str, float] = {}
        window = int(self.parameters["z_window"])

        for pair in pairs:
            closes = history.get(pair, [])
            if len(closes) < lookback + 1 + window:
                return None

            curr = closes[-1]
            prev = closes[-(lookback + 1)]
            if prev <= 0 or curr <= 0:
                return None
            ret = math.log(curr / prev)

            key = f"{pair}_{lookback}"
            if key not in self._z_streams:
                self._z_streams[key] = _StreamingZScore(window)
            z = self._z_streams[key].update(ret)

            if not self._z_streams[key].ready:
                return None
            z_scores[pair] = z

        return z_scores

    def _check_entry(self, bars, history) -> Optional[SignalResult]:
        """§12+§14+§17: Multi-feature consensus gated by persistence."""
        pairs = list(self.parameters["pairs"])
        lookback_short = int(self.parameters["lookback_short"])
        lookback_long = int(self.parameters["lookback_long"])

        short_z = self._compute_z_scores(history, lookback_short)
        long_z = self._compute_z_scores(history, lookback_long)
        if short_z is None or long_z is None:
            self._persistence_count = 0
            return None

        # §17: Multi-timescale agreement — short and long must agree
        short_signs = [_sign(v) for v in short_z.values()]
        long_signs = [_sign(v) for v in long_z.values()]
        if len(short_signs) < 3 or len(long_signs) < 3:
            self._persistence_count = 0
            return None
        if 0.0 in short_signs or 0.0 in long_signs:
            self._persistence_count = 0
            return None
        if len(set(short_signs)) != 1 or len(set(long_signs)) != 1:
            self._persistence_count = 0
            return None
        if short_signs[0] != long_signs[0]:
            self._persistence_count = 0
            return None

        # §12: Compute mean |z| across all pairs (magnitude agreement)
        z_values = [abs(short_z[p]) for p in pairs if p in short_z]
        mean_abs_z = sum(z_values) / len(z_values) if z_values else 0.0
        z_entry = float(self.parameters["z_entry"])
        if mean_abs_z <= z_entry:
            self._persistence_count = 0
            return None

        # §13: Currency strength decomposition (optional additional gate)
        if self.parameters.get("require_currency_agreement", True):
            eur_strength = short_z.get("EURUSD", 0) + short_z.get("EURJPY", 0)
            jpy_strength = -short_z.get("EURJPY", 0) - short_z.get("GBPJPY", 0)
            consensus_dir = short_signs[0]
            if consensus_dir > 0:
                if eur_strength <= 0 and jpy_strength >= 0:
                    self._persistence_count = 0
                    return None
            else:
                if eur_strength >= 0 and jpy_strength <= 0:
                    self._persistence_count = 0
                    return None

        # §15-16: Persistence — same direction for N consecutive bars
        if self._prev_direction is not None and self._prev_direction != short_signs[0]:
            self._persistence_count = 0

        self._prev_direction = short_signs[0]
        self._persistence_count += 1

        if self._persistence_count < int(self.parameters["persistence_bars"]):
            return None

        # §9-10: Decision margin — prefer signals far beyond threshold
        margin = mean_abs_z - z_entry
        best_pair = max(pairs, key=lambda p: abs(short_z[p]))
        direction = 1.0 if short_z[best_pair] > 0 else -1.0
        confidence = min(0.99, 0.30 + margin / z_entry * 0.69)

        return SignalResult(
            timestamp=list(bars.values())[0]["time"],
            signal=direction,
            confidence=round(confidence, 4),
            metadata={
                "strategy": self.name,
                "pair": best_pair,
                "action": "ENTER_LONG" if direction > 0 else "ENTER_SHORT",
                "mean_abs_z": round(float(mean_abs_z), 3),
                "margin": round(float(margin), 3),
                "persistence": self._persistence_count,
                "z_scores": {p: round(float(short_z.get(p, 0)), 2) for p in pairs},
                "timescale_agreement": True,
            },
        )

    # ------------------------------------------------------------------
    # §11: Hysteresis exit — exit on z mean reversion or time
    # ------------------------------------------------------------------
    def _check_exit(self, bars, history) -> Optional[SignalResult]:
        if self._position is None:
            return None

        ts = list(bars.values())[0]["time"]
        self._position["bars_held"] += 1

        # §11: Exit if z has reverted below z_exit
        lookback_short = int(self.parameters["lookback_short"])
        short_z = self._compute_z_scores(history, lookback_short)
        if short_z is not None:
            pairs = list(self.parameters["pairs"])
            z_vals = [abs(short_z.get(p, 0)) for p in pairs]
            mean_abs_z = sum(z_vals) / len(z_vals) if z_vals else 0.0
            z_exit = float(self.parameters["z_exit"])
            if mean_abs_z < z_exit:
                pos = self._position
                self._position = None
                self._persistence_count = 0
                self._prev_direction = None
                return SignalResult(
                    timestamp=ts,
                    signal=0.0,
                    confidence=1.0,
                    metadata={
                        "strategy": self.name,
                        "pair": pos.get("pair", ""),
                        "action": "EXIT",
                        "reason": "z_reversion",
                        "mean_abs_z": round(float(mean_abs_z), 3),
                    },
                )

        # §16: Time-based exit (hold duration)
        hold_bars = int(self.parameters["hold_bars"])
        if self._position["bars_held"] >= hold_bars:
            pos = self._position
            self._position = None
            self._persistence_count = 0
            self._prev_direction = None
            return SignalResult(
                timestamp=ts,
                signal=0.0,
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "pair": pos.get("pair", ""),
                    "action": "EXIT",
                    "reason": "hold_expired",
                    "bars_held": pos["bars_held"],
                },
            )

        return None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def on_bars(self, bars: Dict[str, Dict], history: Dict[str, Any]) -> List[SignalResult]:
        signals: List[SignalResult] = []

        ts = None
        for _, bar in bars.items():
            if bar:
                ts = bar.get("time")
                break
        if not ts:
            return signals

        hour = ts.hour if hasattr(ts, "hour") else 0
        session_start = int(self.parameters["session_start"])
        session_end = int(self.parameters["session_end"])

        is_session = session_start <= hour < session_end

        if self._position is not None:
            exit_sig = self._check_exit(bars, history)
            if exit_sig:
                signals.append(exit_sig)
            return signals

        if not is_session:
            self._persistence_count = 0
            self._prev_direction = None
            return signals

        entry_sig = self._check_entry(bars, history)
        if entry_sig:
            pair = entry_sig.metadata.get("pair", "")
            if pair in bars and bars[pair]:
                self._position = {
                    "pair": pair,
                    "direction": entry_sig.signal,
                    "entry_time": ts,
                    "bars_held": 0,
                    "entry_price": bars[pair]["close"],
                }
                signals.append(entry_sig)

        return signals

    def describe(self) -> str:
        p = self.parameters
        return (
            f"DarkConsensus(z_entry={p['z_entry']}, z_exit={p['z_exit']}, "
            f"persist={p['persistence_bars']}, hold={p['hold_bars']}, "
            f"multi_timescale={p['lookback_short']}/{p['lookback_long']})"
        )
