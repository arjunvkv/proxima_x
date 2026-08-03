"""
NY H21 — NY Closing Bell Dislocation Reversion

Target Anomaly:
  Institutional benchmark fixing dislocations at NY Market Close (21:00 UTC).
  WM/Refinitiv fix at 16:00 EST triggers temporary 30-min over-extensions
  that revert as liquidity normalizes at 22:00 UTC.

Empirical finding (Jul 2026 data):
  Cross-sectional mean reversion at 21 UTC is pair-specific, not universal.
  JPY crosses (EURJPY, GBPJPY, USDJPY) show 55-70% WR, while AUD/CAD pairs
  show 42-48% WR. Strategy uses a refined subset of high-WR pairs.
"""
import math
from typing import Any, Dict, List, Optional

from proxima_honest_backtest.engine.types import SignalResult
from proxima_honest_backtest.strategies.multi_pair_base import MultiPairStrategy

# Proven high-WR pairs at NY close (21 UTC) from empirical testing
NY_HIGH_WR_PAIRS = [
    "EURJPY", "GBPJPY", "EURNZD", "USDJPY", "USDCHF", "EURGBP",
]

# Full 18-pair universe for cross-sectional ranking
ALL_PAIRS = [
    "AUDJPY", "AUDNZD", "AUDUSD", "EURAUD", "EURCHF", "EURGBP",
    "EURJPY", "EURNZD", "EURUSD", "GBPAUD", "GBPCAD", "GBPJPY",
    "GBPNZD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY",
]

_POINT = {
    "AUDNZD": 1e-5, "EURAUD": 1e-5, "EURNZD": 1e-5,
    "GBPAUD": 1e-5, "GBPCAD": 1e-5, "GBPNZD": 1e-5,
    "EURUSD": 1e-5, "GBPUSD": 1e-5, "AUDUSD": 1e-5,
    "USDCAD": 1e-5, "NZDUSD": 1e-5, "EURGBP": 1e-5,
    "EURCHF": 1e-5, "USDCHF": 1e-5, "AUDJPY": 0.001,
    "EURJPY": 0.001, "GBPJPY": 0.001, "USDJPY": 0.001,
}


class NYH21Strategy(MultiPairStrategy):
    """NY H21 — Cross-pair mean reversion at NY close (21:00 UTC).

    Enter LONG on most-declined pairs from proven high-WR subset.
    Fires once daily at 21:00 UTC, holds 12 M5 bars (60 min).
    """

    DEFAULT_PARAMS: Dict[str, Any] = {
        "pairs": ALL_PAIRS,
        "trade_pairs": ["EURJPY", "GBPJPY", "USDJPY"],
        "top_n": 5,
        "lookback_bars": 6,
        "lookback_confirm_bars": 0,
        "hold_bars": 12,
        "hold_bars_map": {},  # per-pair override, e.g. {"GBPJPY": 9, "USDJPY": 9}
        "session_hour": 21,
        "direction": "LONG",
        "gap_threshold_pct": 10.0,
        "min_pairs": 0,
        "min_confidence": 0.0,
        "require_decline_persistence": False,
        "vol_window": 50,
    }

    def __init__(self, parameters: Optional[Dict[str, Any]] = None) -> None:
        merged = dict(self.DEFAULT_PARAMS)
        if parameters:
            merged.update(parameters)
        super().__init__(merged)
        self._positions: Dict[str, Dict] = {}
        self._last_entry_date: Optional[str] = None
        self._vol_streams: Dict[str, _RunningVol] = {}

    def reset(self) -> None:
        self._positions.clear()
        self._last_entry_date = None
        self._vol_streams.clear()

    def describe(self) -> str:
        p = self.parameters
        n_trade = len(p["trade_pairs"])
        return (f"NYH21(pairs={n_trade}, top_n={p['top_n']}, "
                f"lookback={p['lookback_bars']}, hold={p['hold_bars']})")

    def _update_vol(self, pair: str, ret: float) -> Optional[float]:
        key = f"vol_{pair}"
        if key not in self._vol_streams:
            self._vol_streams[key] = _RunningVol(int(self.parameters["vol_window"]))
        return self._vol_streams[key].update(ret)

    def _get_vol(self, pair: str) -> float:
        key = f"vol_{pair}"
        vs = self._vol_streams.get(key)
        return vs.std if vs and vs.count >= 5 else 0.001

    def on_bars(self, bars: Dict[str, Dict], history: Dict[str, Any]) -> List[SignalResult]:
        signals: List[SignalResult] = []
        p = self.parameters
        session_hour = int(p["session_hour"])
        hold_bars = int(p["hold_bars"])

        ts = next((b["time"] for _, b in bars.items() if b), None)
        if not ts:
            return signals
        hour = ts.hour if hasattr(ts, "hour") else 0
        today = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)

        self._check_exits(ts, bars, signals, hold_bars, p)

        if hour != session_hour:
            return signals
        if self._last_entry_date == today:
            return signals
        self._last_entry_date = today

        lb = int(p["lookback_bars"])
        trade_pairs = p["trade_pairs"]

        candidates = []
        for pair in p["pairs"]:
            closes = history.get(pair, [])
            if len(closes) < lb + 2:
                continue
            curr = closes[-1]
            prev_lb = closes[-(lb + 1)]
            if prev_lb <= 0 or curr <= 0:
                continue
            ret = math.log(curr / prev_lb)

            bar = bars.get(pair)
            if bar is None:
                continue
            vol = self._update_vol(pair, ret) or 0.001
            margin = abs(ret) / max(vol, 1e-10)

            candidates.append((pair, ret, margin))

        if not candidates:
            return signals

        candidates.sort(key=lambda x: x[1])

        top_n = int(p["top_n"])
        entered = 0
        for pair, ret, margin in candidates[:top_n]:
            if ret >= 0:
                break
            if pair not in trade_pairs:
                continue
            bar = bars.get(pair)
            if bar is None:
                continue

            entry_price = float(bar["open"])
            confidence = min(0.95, margin * 0.15)

            self._positions[pair] = {
                "direction": 1.0,
                "entry_time": ts,
                "bars_held": 0,
                "entry_price": entry_price,
            }
            signals.append(SignalResult(
                timestamp=ts,
                signal=1.0,
                confidence=round(confidence, 4),
                metadata={
                    "strategy": self.name,
                    "pair": pair,
                    "action": "ENTER_LONG",
                    "entry_price": entry_price,
                    "ret_log": round(float(ret), 6),
                    "margin": round(float(margin), 3),
                },
            ))
            entered += 1

        return signals

    def _check_exits(self, ts, bars, signals, hold_bars, p):
        hold_map = p.get("hold_bars_map", {})
        to_remove = []
        for pair, pos in self._positions.items():
            pos["bars_held"] = pos.get("bars_held", 0) + 1
            effective_hold = hold_map.get(pair, hold_bars)
            if pos["bars_held"] >= effective_hold:
                to_remove.append(pair)
        for pair in to_remove:
            self._positions.pop(pair, None)
            signals.append(SignalResult(
                timestamp=ts,
                signal=0.0,
                confidence=1.0,
                metadata={
                    "strategy": self.name,
                    "pair": pair,
                    "action": "EXIT",
                    "reason": "hold_expired",
                },
            ))


class _RunningVol:
    """O(1) streaming standard deviation via Welford."""

    __slots__ = ("_window", "_deque", "_sum", "_sum_sq", "count")

    def __init__(self, window: int) -> None:
        self._window = window
        self._deque: Any = __import__("collections").deque(maxlen=window)
        self._sum = 0.0
        self._sum_sq = 0.0
        self.count = 0

    @property
    def std(self) -> float:
        n = min(self.count, self._window)
        if n < 2:
            return 0.0
        mean = self._sum / n
        var = max(self._sum_sq / n - mean * mean, 0.0)
        return math.sqrt(var) if var > 1e-24 else 0.0

    def update(self, value: float) -> float:
        if self.count >= self._window:
            old = self._deque[0]
            self._sum -= old
            self._sum_sq -= old * old
        self._deque.append(value)
        self._sum += value
        self._sum_sq += value * value
        self.count += 1
        return self.std
