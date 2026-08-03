import math
from typing import Any, Dict, List, Optional
from proxima_honest_backtest.engine.types import SignalResult
from proxima_honest_backtest.strategies.multi_pair_base import MultiPairStrategy

ALL_PAIRS = [
    "AUDJPY", "AUDNZD", "AUDUSD", "EURAUD", "EURCHF", "EURGBP",
    "EURJPY", "EURNZD", "EURUSD", "GBPAUD", "GBPCAD", "GBPJPY",
    "GBPNZD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY",
]

class RangeContractionStrategy(MultiPairStrategy):
    """Range Contraction Breakout — rolling ATR compression + close range breakout on M5."""

    DEFAULT_PARAMS: Dict[str, Any] = {
        "pairs": ALL_PAIRS,
        "hold_bars": 6,
        "lookback": 20,
        "compression_ratio": 0.5,
    }

    def __init__(self, parameters: Optional[Dict[str, Any]] = None) -> None:
        merged = dict(self.DEFAULT_PARAMS)
        if parameters:
            merged.update(parameters)
        super().__init__(merged)
        self._positions: Dict[str, Dict] = {}
        self._histories: Dict[str, List[float]] = {}

    def reset(self) -> None:
        self._positions.clear()
        self._histories.clear()

    def describe(self) -> str:
        p = self.parameters
        return f"RangeContr(hold={p['hold_bars']}, lb={p['lookback']})"

    def on_bars(self, bars: Dict[str, Dict], history: Dict[str, Any]) -> List[SignalResult]:
        signals: List[SignalResult] = []
        p = self.parameters
        hold_bars = int(p["hold_bars"])
        lookback = int(p["lookback"])
        comp_ratio = float(p["compression_ratio"])

        ts = next((b["time"] for _, b in bars.items() if b), None)
        if not ts:
            return signals

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

        # Entry
        for pair, bar in bars.items():
            if pair.startswith("_"):
                continue
            if pair in self._positions:
                continue
            close = bar.get("close")
            high = bar.get("high")
            low = bar.get("low")
            if close is None or high is None or low is None:
                continue
            if any(isinstance(x, float) and math.isnan(x) for x in (close, high, low)):
                continue

            # Track bar range (high-low) as compression metric
            rng_hist = self._histories.setdefault(pair, [])
            bar_range = high - low
            rng_hist.append(bar_range)
            if len(rng_hist) > lookback * 3:
                rng_hist[:] = rng_hist[-(lookback * 3):]

            if len(rng_hist) < lookback * 2:
                continue

            recent_avg = sum(rng_hist[-lookback:]) / lookback
            prior_avg = sum(rng_hist[-(lookback*2):-lookback]) / lookback

            if recent_avg > comp_ratio * prior_avg:
                continue

            # Breakout detection: previous bar close vs range of earlier bars
            close_hist = history.get(pair, [])
            if len(close_hist) < lookback + 2:
                continue
            range_closes = close_hist[-(lookback+2):-2]
            hi20 = max(range_closes)
            lo20 = min(range_closes)
            prev_close = close_hist[-2]
            entry_price = float(bar.get("open", close))

            pip_size = 1e-5 if pair not in ("AUDJPY","EURJPY","GBPJPY","USDJPY") else 0.001

            if prev_close > hi20:
                breakout_pips = (prev_close - hi20) / pip_size
                if breakout_pips < 2:
                    continue
                direction = "LONG"
            elif prev_close < lo20:
                breakout_pips = (lo20 - prev_close) / pip_size
                if breakout_pips < 2:
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
