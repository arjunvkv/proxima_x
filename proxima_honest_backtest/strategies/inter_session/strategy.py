import math
from typing import Any, Dict, List, Optional
from proxima_honest_backtest.engine.types import SignalResult
from proxima_honest_backtest.strategies.multi_pair_base import MultiPairStrategy

ALL_PAIRS = [
    "AUDJPY", "AUDNZD", "AUDUSD", "EURAUD", "EURCHF", "EURGBP",
    "EURJPY", "EURNZD", "EURUSD", "GBPAUD", "GBPCAD", "GBPJPY",
    "GBPNZD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY",
]


class InterSessionStrategy(MultiPairStrategy):

    DEFAULT_PARAMS: Dict[str, Any] = {
        "pairs": ALL_PAIRS,
        "entry_hours": [8, 16],
        "entry_weekdays": [],  # empty = Mon-Fri, e.g. [4] = Friday only
        "top_n": 3,
        "lookback_bars": 12,
        "hold_bars": 12,
        "direction": "LONG",
        "sort_descending": False,
        "min_pairs": 8,
    }

    def __init__(self, parameters: Optional[Dict[str, Any]] = None) -> None:
        merged = dict(self.DEFAULT_PARAMS)
        if parameters:
            merged.update(parameters)
        super().__init__(merged)
        self._positions: Dict[str, Dict] = {}
        self._entry_dates: set = set()

    def reset(self) -> None:
        self._positions.clear()
        self._entry_dates.clear()

    def describe(self) -> str:
        p = self.parameters
        return (f"InterSession(hours={p['entry_hours']}, top_n={p['top_n']}, "
                f"lookback={p['lookback_bars']}, hold={p['hold_bars']})")

    def on_bars(self, bars: Dict[str, Dict], history: Dict[str, Any]) -> List[SignalResult]:
        signals: List[SignalResult] = []
        p = self.parameters
        entry_hours = [int(h) for h in p["entry_hours"]]
        hold_bars = int(p["hold_bars"])

        ts = next((b["time"] for _, b in bars.items() if b), None)
        if not ts:
            return signals
        hour = ts.hour if hasattr(ts, "hour") else 0
        weekday = ts.weekday() if hasattr(ts, "weekday") else -1
        today = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)

        self._check_exits(ts, bars, signals, hold_bars)

        if hour not in entry_hours:
            return signals
        wd = p.get("entry_weekdays", [])
        if wd and weekday not in wd:
            return signals

        entry_key = f"{today}_{hour}"
        if entry_key in self._entry_dates:
            return signals
        self._entry_dates.add(entry_key)

        lb = int(p["lookback_bars"])
        min_pairs = int(p["min_pairs"])
        top_n = int(p["top_n"])

        candidates = []
        for pair in p["pairs"]:
            closes = history.get(pair, [])
            if len(closes) < lb + 2:
                continue
            curr = closes[-1]
            prev = closes[-(lb + 1)]
            if prev <= 0 or curr <= 0:
                continue
            ret = (curr - prev) / prev
            candidates.append((pair, ret))

        if len(candidates) < min_pairs:
            return signals

        if p.get("sort_descending", False):
            candidates.sort(key=lambda x: x[1], reverse=True)
        else:
            candidates.sort(key=lambda x: x[1])

        for pair, ret in candidates[:top_n]:
            if p.get("sort_descending", False):
                if ret <= 0:
                    break
            else:
                if ret >= 0:
                    break
            bar = bars.get(pair)
            if bar is None:
                continue
            entry_price = float(bar["open"])
            self._positions[pair] = {
                "direction": 1.0,
                "entry_time": ts,
                "bars_held": 0,
                "entry_price": entry_price,
            }
            signals.append(SignalResult(
                timestamp=ts,
                signal=1.0,
                confidence=round(min(0.95, abs(ret) * 50), 4),
                metadata={
                    "strategy": self.name,
                    "pair": pair,
                    "action": "ENTER_LONG",
                    "entry_price": entry_price,
                    "ret": round(float(ret), 6),
                },
            ))

        return signals

    def _check_exits(self, ts, bars, signals, hold_bars):
        to_remove = []
        for pair, pos in self._positions.items():
            pos["bars_held"] = pos.get("bars_held", 0) + 1
            if pos["bars_held"] >= hold_bars:
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
