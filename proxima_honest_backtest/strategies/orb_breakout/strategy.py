import math
from typing import Any, Dict, List, Optional
from proxima_honest_backtest.engine.types import SignalResult
from proxima_honest_backtest.strategies.multi_pair_base import MultiPairStrategy

ALL_PAIRS = [
    "AUDJPY", "AUDNZD", "AUDUSD", "EURAUD", "EURCHF", "EURGBP",
    "EURJPY", "EURNZD", "EURUSD", "GBPAUD", "GBPCAD", "GBPJPY",
    "GBPNZD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY",
]


class ORBBreakoutStrategy(MultiPairStrategy):

    DEFAULT_PARAMS: Dict[str, Any] = {
        "pairs": ALL_PAIRS,
        "orb_start_hour": 8,
        "orb_start_min": 0,
        "orb_duration_min": 30,
        "entry_window_end": 55,
        "hold_bars": 12,
        "min_breakout_pips": 3,
    }

    def __init__(self, parameters: Optional[Dict[str, Any]] = None) -> None:
        merged = dict(self.DEFAULT_PARAMS)
        if parameters:
            merged.update(parameters)
        super().__init__(merged)
        self._positions: Dict[str, Dict] = {}
        self._orbs: Dict[str, Dict] = {}
        self._current_date: Optional[str] = None

    def reset(self) -> None:
        self._positions.clear()
        self._orbs.clear()
        self._current_date = None

    def describe(self) -> str:
        p = self.parameters
        return f"ORBBreakout(hold={p['hold_bars']}, orb_start={p['orb_start_hour']}:{p['orb_start_min']:02d})"

    def on_bars(self, bars: Dict[str, Dict], history: Dict[str, Any]) -> List[SignalResult]:
        signals: List[SignalResult] = []
        p = self.parameters
        hold_bars = int(p["hold_bars"])
        orb_sh = int(p["orb_start_hour"])
        orb_sm = int(p["orb_start_min"])
        orb_dm = int(p["orb_duration_min"])
        ew_end = int(p["entry_window_end"])
        min_pips = float(p.get("min_breakout_pips", 3))

        ts = next((b["time"] for _, b in bars.items() if b), None)
        if not ts:
            return signals
        hour = ts.hour if hasattr(ts, "hour") else 0
        minute = ts.hour * 60 + (ts.minute if hasattr(ts, "minute") else 0)
        t_min = hour * 60 + (ts.minute if hasattr(ts, "minute") else 0)
        today = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)

        # Date reset
        if self._current_date != today:
            self._current_date = today
            self._orbs = {}

        # Exit checks
        self._check_exits(ts, signals, hold_bars)

        # Build ORB during the opening range
        orb_start = orb_sh * 60 + orb_sm
        orb_end = orb_start + orb_dm

        if orb_start <= t_min <= orb_end:
            for pair, bar in bars.items():
                high = bar.get("high")
                low = bar.get("low")
                if high is None or low is None:
                    continue
                orb = self._orbs.setdefault(pair, {"hi": -1e9, "lo": 1e9})
                orb["hi"] = max(orb["hi"], high)
                orb["lo"] = min(orb["lo"], low)
                orb["start_min"] = orb_start
                orb["end_min"] = orb_end

        # Entry window: first bar AFTER ORB closes, look for breakouts
        entry_window_start = orb_end + 5  # first bar after ORB period
        entry_window_end = orb_sh * 60 + ew_end  # e.g., 08:55

        if entry_window_start <= t_min <= entry_window_end:
            for pair, bar in bars.items():
                if pair in self._positions:
                    continue
                close = bar.get("close")
                if close is None:
                    continue
                orb = self._orbs.get(pair)
                if orb is None:
                    continue

                pip_size = 1e-5 if pair not in ("AUDJPY","EURJPY","GBPJPY","USDJPY") else 0.001
                if close > orb["hi"]:
                    breakout_pips = (close - orb["hi"]) / pip_size
                    if breakout_pips < min_pips:
                        continue
                    direction = "LONG"
                elif close < orb["lo"]:
                    breakout_pips = (orb["lo"] - close) / pip_size
                    if breakout_pips < min_pips:
                        continue
                    direction = "SHORT"
                else:
                    continue

                entry_price = float(bar["open"])
                self._positions[pair] = {
                    "direction": 1.0 if direction == "LONG" else -1.0,
                    "entry_time": ts,
                    "bars_held": 0,
                    "entry_price": entry_price,
                }
                signals.append(SignalResult(
                    timestamp=ts,
                    signal=1.0 if direction == "LONG" else -1.0,
                    confidence=min(0.95, breakout_pips / 10),
                    metadata={
                        "strategy": self.name,
                        "pair": pair,
                        "action": f"ENTER_{direction}",
                        "entry_price": entry_price,
                        "breakout_pips": round(breakout_pips, 1),
                    },
                ))

        return signals

    def _check_exits(self, ts, signals, hold_bars):
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
