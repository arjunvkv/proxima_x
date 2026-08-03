import math
from typing import Any, Dict, List, Optional
from proxima_honest_backtest.engine.types import SignalResult
from proxima_honest_backtest.strategies.multi_pair_base import MultiPairStrategy

ALL_PAIRS = [
    "AUDJPY", "AUDNZD", "AUDUSD", "EURAUD", "EURCHF", "EURGBP",
    "EURJPY", "EURNZD", "EURUSD", "GBPAUD", "GBPCAD", "GBPJPY",
    "GBPNZD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY",
]

class RollingORBStrategy(MultiPairStrategy):
    """Rolling Hourly ORB — breakout from first 2 bars of each hour, ride 30 min."""

    DEFAULT_PARAMS: Dict[str, Any] = {
        "pairs": ALL_PAIRS,
        "hold_bars": 6,
        "min_breakout_pips": 2,
    }

    def __init__(self, parameters: Optional[Dict[str, Any]] = None) -> None:
        merged = dict(self.DEFAULT_PARAMS)
        if parameters:
            merged.update(parameters)
        super().__init__(merged)
        self._positions: Dict[str, Dict] = {}
        self._orb_hi: Dict[str, float] = {}
        self._orb_lo: Dict[str, float] = {}
        self._current_hour: Optional[int] = None
        self._orb_active = False
        self._check_bars = 0

    def reset(self) -> None:
        self._positions.clear()
        self._orb_hi.clear()
        self._orb_lo.clear()
        self._current_hour = None
        self._orb_active = False

    def describe(self) -> str:
        p = self.parameters
        return f"RollingORB(hold={p['hold_bars']}, min_pips={p['min_breakout_pips']})"

    def on_bars(self, bars: Dict[str, Dict], history: Dict[str, Any]) -> List[SignalResult]:
        signals: List[SignalResult] = []
        p = self.parameters
        hold_bars = int(p["hold_bars"])
        min_pips = float(p.get("min_breakout_pips", 2))

        ts = next((b["time"] for _, b in bars.items() if b), None)
        if not ts:
            return signals
        hour = ts.hour if hasattr(ts, "hour") else 0
        minute = ts.hour * 60 + (ts.minute if hasattr(ts, "minute") else 0)
        t_min = hour * 60 + (ts.minute if hasattr(ts, "minute") else 0)

        # Hour rollover
        if self._current_hour != hour:
            self._current_hour = hour
            self._orb_hi = {}
            self._orb_lo = {}
            self._orb_active = True
            self._check_bars = 0

        # Track ORB from first 2 bars of hour
        if self._orb_active:
            self._check_bars += 1
            for pair, bar in bars.items():
                if pair.startswith("_"):
                    continue
                high = bar.get("high")
                low = bar.get("low")
                if high is None or low is None:
                    continue
                self._orb_hi[pair] = max(self._orb_hi.get(pair, -1e9), high)
                self._orb_lo[pair] = min(self._orb_lo.get(pair, 1e9), low)
            if self._check_bars >= 2:
                self._orb_active = False

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

        # Entry: check if previous bar's close broke the ORB (use history)
        if not self._orb_active and self._check_bars >= 2:
            for pair, bar in bars.items():
                if pair.startswith("_"):
                    continue
                if pair in self._positions:
                    continue
                hist = history.get(pair, [])
                if len(hist) < 1:
                    continue
                prev_close = hist[-1]
                if prev_close is None or (isinstance(prev_close, float) and math.isnan(prev_close)):
                    continue
                hi = self._orb_hi.get(pair)
                lo = self._orb_lo.get(pair)
                if hi is None or lo is None:
                    continue

                pip_size = 1e-5 if pair not in ("AUDJPY","EURJPY","GBPJPY","USDJPY") else 0.001
                entry_price = float(bar.get("open", bar.get("close", prev_close)))
                if prev_close > hi:
                    breakout_pips = (prev_close - hi) / pip_size
                    if breakout_pips < min_pips:
                        continue
                    direction = "LONG"
                elif prev_close < lo:
                    breakout_pips = (lo - prev_close) / pip_size
                    if breakout_pips < min_pips:
                        continue
                    direction = "SHORT"
                else:
                    continue

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
