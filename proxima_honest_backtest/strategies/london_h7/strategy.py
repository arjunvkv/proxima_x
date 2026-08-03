import math
from typing import Any, Dict, List, Optional, Tuple

from proxima_honest_backtest.engine.types import SignalResult
from proxima_honest_backtest.strategies.multi_pair_base import MultiPairStrategy

ALL_PAIRS = [
    "AUDJPY", "AUDNZD", "AUDUSD", "EURAUD", "EURCHF", "EURGBP",
    "EURJPY", "EURNZD", "EURUSD", "GBPAUD", "GBPCAD", "GBPJPY",
    "GBPNZD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY",
]

MAJOR_PAIRS = {"EURUSD", "GBPUSD", "USDJPY", "USDCAD", "USDCHF"}
JPY_CROSS_PAIRS = {"EURJPY", "GBPJPY", "AUDJPY"}
VOLATILE_CROSS_PAIRS = {"EURAUD", "GBPAUD", "EURNZD", "GBPNZD", "AUDNZD", "GBPCAD", "AUDUSD", "NZDUSD", "EURGBP", "EURCHF"}


def _pip_value(pair: str) -> float:
    return 0.0001 if not pair.endswith("JPY") else 0.01


def _pair_group(pair: str) -> str:
    if pair in MAJOR_PAIRS:
        return "major"
    if pair in JPY_CROSS_PAIRS:
        return "jpy"
    return "volatile"


class LondonH7Strategy(MultiPairStrategy):

    DEFAULT_PARAMS: Dict[str, Any] = {
        "pairs": ALL_PAIRS,
        "trade_pairs": ALL_PAIRS,
        "entry_hour": 7,
        "entry_minute": 10,
        "sweep_min_pips": 3,
        "sweep_max_pips": 30,
        "min_range_pips": 8,
        "max_range_pips": 50,
        "top_n": 3,
        "hold_bars_major": 8,
        "hold_bars_jpy": 12,
        "hold_bars_volatile": 15,
        "sweep_lookback_bars": 84,
    }

    def __init__(self, parameters: Optional[Dict[str, Any]] = None) -> None:
        merged = dict(self.DEFAULT_PARAMS)
        if parameters:
            merged.update(parameters)
        super().__init__(merged)
        self._positions: Dict[str, Dict] = {}
        self._cur_date: Optional[str] = None
        self._asian_ranges: Dict[str, Dict] = {}
        self._sweep_candidates: Dict[str, Dict] = {}
        self._close_0705: Dict[str, float] = {}
        self._last_entry_date: Optional[str] = None

    def reset(self) -> None:
        self._positions.clear()
        self._cur_date = None
        self._asian_ranges.clear()
        self._sweep_candidates.clear()
        self._close_0705.clear()
        self._last_entry_date = None

    def describe(self) -> str:
        p = self.parameters
        return (f"LondonH7(sweep={p['sweep_min_pips']}-{p['sweep_max_pips']}p, "
                f"range={p['min_range_pips']}-{p['max_range_pips']}p, "
                f"top_n={p['top_n']}, hold_major={p['hold_bars_major']}, "
                f"hold_jpy={p['hold_bars_jpy']}, hold_vol={p['hold_bars_volatile']})")

    def _hold_bars(self, pair: str) -> int:
        p = self.parameters
        g = _pair_group(pair)
        if g == "major":
            return int(p["hold_bars_major"])
        if g == "jpy":
            return int(p["hold_bars_jpy"])
        return int(p["hold_bars_volatile"])

    def on_bars(self, bars: Dict[str, Dict], history: Dict[str, Any]) -> List[SignalResult]:
        signals: List[SignalResult] = []
        p = self.parameters
        entry_hour = int(p["entry_hour"])
        entry_minute = int(p["entry_minute"])

        ts = next((b["time"] for _, b in bars.items() if b), None)
        if not ts:
            return signals
        hour = ts.hour if hasattr(ts, "hour") else 0
        minute = ts.minute if hasattr(ts, "minute") else 0
        today = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)

        # Reset on new day
        if self._cur_date != today:
            self._cur_date = today
            self._asian_ranges = {}
            self._sweep_candidates = {}
            self._close_0705 = {}
            self._last_entry_date = None

        # Track Asian range (00:00-06:55)
        if hour < 7:
            for pair, bar in bars.items():
                if bar is None:
                    continue
                ar = self._asian_ranges.setdefault(pair, {
                    "high": -1e9, "low": 1e9, "count": 0,
                })
                ar["high"] = max(ar["high"], bar["high"])
                ar["low"] = min(ar["low"], bar["low"])
                ar["count"] += 1

        # At 07:00 bar: detect sweep
        if hour == 7 and minute == 0:
            for pair, bar in bars.items():
                if bar is None:
                    continue
                ar = self._asian_ranges.get(pair)
                if not ar:
                    continue
                pv = _pip_value(pair)
                range_pips = (ar["high"] - ar["low"]) / pv
                if range_pips < float(p["min_range_pips"]) or range_pips > float(p["max_range_pips"]):
                    continue

                high_sweep = bar["high"] - ar["high"]
                low_sweep = ar["low"] - bar["low"]
                max_sweep = max(high_sweep, low_sweep)
                sweep_pips = max_sweep / pv
                if sweep_pips < float(p["sweep_min_pips"]) or sweep_pips > float(p["sweep_max_pips"]):
                    continue

                is_high_sweep = high_sweep > low_sweep
                self._sweep_candidates[pair] = {
                    "high_sweep": high_sweep,
                    "low_sweep": low_sweep,
                    "sweep_pips": sweep_pips,
                    "is_high_sweep": is_high_sweep,
                    "asian_high": ar["high"],
                    "asian_low": ar["low"],
                    "range_pips": range_pips,
                    "direction_taken": False,
                }

        # At 07:05 bar: store close for confirmation
        if hour == 7 and minute == 5:
            for pair, bar in bars.items():
                if bar is None or pair not in self._sweep_candidates:
                    continue
                self._close_0705[pair] = bar["close"]

        # At 07:10 bar: entry
        if hour == entry_hour and minute == entry_minute:
            if self._last_entry_date == today:
                return signals
            self._last_entry_date = today

            candidates: List[Tuple[str, float, float, bool]] = []
            for pair in p["trade_pairs"]:
                sw = self._sweep_candidates.get(pair)
                if not sw or sw.get("direction_taken", False):
                    continue

                close_0705 = self._close_0705.get(pair)
                if close_0705 is None:
                    continue

                # Judas Swing: 07:05 close back inside Asian range
                if not (sw["asian_low"] <= close_0705 <= sw["asian_high"]):
                    continue

                bar = bars.get(pair)
                if bar is None:
                    continue

                if sw["is_high_sweep"]:
                    candidates.append((pair, sw["high_sweep"], sw["sweep_pips"], True))
                else:
                    candidates.append((pair, sw["low_sweep"], sw["sweep_pips"], False))

            if not candidates:
                return signals

            candidates.sort(key=lambda x: x[1], reverse=True)
            top_n = min(int(p["top_n"]), len(candidates))
            selected = candidates[:top_n]

            for pair, sweep_dist, sweep_pips, is_high in selected:
                bar = bars[pair]
                entry_price = float(bar["open"])
                direction = -1.0 if is_high else 1.0

                self._sweep_candidates[pair]["direction_taken"] = True
                self._positions[pair] = {
                    "direction": direction,
                    "entry_time": ts,
                    "bars_held": 0,
                    "entry_price": entry_price,
                    "hold_bars": self._hold_bars(pair),
                }
                signals.append(SignalResult(
                    timestamp=ts,
                    signal=direction,
                    confidence=round(min(0.95, 0.5 + sweep_dist / (_pip_value(pair) * 15)), 4),
                    metadata={
                        "strategy": self.name,
                        "pair": pair,
                        "action": "ENTER_LONG" if direction > 0 else "ENTER_SHORT",
                        "entry_price": entry_price,
                        "sweep_pips": round(sw["sweep_pips"], 1),
                        "range_pips": round(sw["range_pips"], 1),
                    },
                ))

        # Exit checks
        to_remove = []
        for pair, pos in self._positions.items():
            pos["bars_held"] = pos.get("bars_held", 0) + 1
            hold = pos.get("hold_bars", 12)
            if pos["bars_held"] >= hold:
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

        return signals
