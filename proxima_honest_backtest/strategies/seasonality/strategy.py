import math
from typing import Any, Dict, List, Optional
from proxima_honest_backtest.engine.types import SignalResult
from proxima_honest_backtest.strategies.multi_pair_base import MultiPairStrategy

ALL_PAIRS = [
    "AUDJPY", "AUDNZD", "AUDUSD", "EURAUD", "EURCHF", "EURGBP",
    "EURJPY", "EURNZD", "EURUSD", "GBPAUD", "GBPCAD", "GBPJPY",
    "GBPNZD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY",
]

class SeasonalityStrategy(MultiPairStrategy):
    """Intraday Seasonality — LONG all pairs at 01:00 UTC, hold 60 min."""

    DEFAULT_PARAMS: Dict[str, Any] = {
        "pairs": ALL_PAIRS,
        "entry_hour": 1,
        "entry_minute": 0,
        "hold_bars": 12,
        "top_n": 18,
    }

    def __init__(self, parameters: Optional[Dict[str, Any]] = None) -> None:
        merged = dict(self.DEFAULT_PARAMS)
        if parameters:
            merged.update(parameters)
        super().__init__(merged)
        self._positions: Dict[str, Dict] = {}
        self._current_date: Optional[str] = None

    def reset(self) -> None:
        self._positions.clear()
        self._current_date = None

    def describe(self) -> str:
        p = self.parameters
        return f"Seasonality(h={p['entry_hour']}, hold={p['hold_bars']}, n={p['top_n']})"

    def on_bars(self, bars: Dict[str, Dict], history: Dict[str, Any]) -> List[SignalResult]:
        signals: List[SignalResult] = []
        p = self.parameters
        entry_h = int(p["entry_hour"])
        entry_m = int(p["entry_minute"])
        hold_bars = int(p["hold_bars"])
        top_n = int(p["top_n"])

        ts = next((b["time"] for _, b in bars.items() if b), None)
        if not ts:
            return signals
        hour = ts.hour if hasattr(ts, "hour") else 0
        minute = ts.hour * 60 + (ts.minute if hasattr(ts, "minute") else 0)
        t_min = hour * 60 + (ts.minute if hasattr(ts, "minute") else 0)
        today = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)

        # Exit checks
        to_remove = []
        for pair, pos in self._positions.items():
            pos["bars_held"] = pos.get("bars_held", 0) + 1
            if pos["bars_held"] >= hold_bars:
                to_remove.append(pair)
        for pair in to_remove:
            self._positions.pop(pair, None)
            signals.append(SignalResult(
                timestamp=ts, signal=0.0, confidence=1.0,
                metadata={"strategy": self.name, "pair": pair, "action": "EXIT", "reason": "hold_expired"},
            ))

        # Entry at specified hour
        target_t = entry_h * 60 + entry_m
        if t_min != target_t:
            return signals
        if self._current_date == today:
            return signals
        self._current_date = today

        # LONG all pairs (or top N - we trade all since all showed upward bias)
        count = 0
        for pair, bar in bars.items():
            if pair.startswith("_"):
                continue
            if pair in self._positions:
                continue
            close = bar.get("close")
            if close is None or (isinstance(close, float) and math.isnan(close)):
                continue
            entry_price = float(bar.get("open", close))

            self._positions[pair] = {
                "direction": 1.0,  # always LONG
                "entry_time": ts,
                "bars_held": 0,
                "entry_price": entry_price,
            }
            signals.append(SignalResult(
                timestamp=ts, signal=1.0, confidence=0.6,
                metadata={
                    "strategy": self.name, "pair": pair,
                    "action": "ENTER_LONG", "entry_price": entry_price,
                },
            ))
            count += 1
            if count >= top_n:
                break

        return signals
