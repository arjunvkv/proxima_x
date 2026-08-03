import math
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

from proxima_honest_backtest.engine.types import SignalResult
from proxima_honest_backtest.strategies.multi_pair_base import MultiPairStrategy

ALL_PAIRS = [
    "AUDJPY", "AUDNZD", "AUDUSD", "EURAUD", "EURCHF", "EURGBP",
    "EURJPY", "EURNZD", "EURUSD", "GBPAUD", "GBPCAD", "GBPJPY",
    "GBPNZD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY",
]


def _pip_value(pair: str) -> float:
    return 0.0001 if not pair.endswith("JPY") else 0.01


class VWAPReversionStrategy(MultiPairStrategy):

    DEFAULT_PARAMS: Dict[str, Any] = {
        "pairs": ALL_PAIRS,
        "trade_pairs": ALL_PAIRS,
        "sigma_entry": 2.0,
        "sigma_exit": 0.5,
        "vwap_min_bars": 24,
        "dev_window": 20,
        "hold_bars": 10,
        "top_n": 3,
        "max_positions": 5,
        "min_deviation_pips": 3,
    }

    def __init__(self, parameters: Optional[Dict[str, Any]] = None) -> None:
        merged = dict(self.DEFAULT_PARAMS)
        if parameters:
            merged.update(parameters)
        super().__init__(merged)
        self._cur_date: Optional[str] = None
        self._vol_tp: Dict[str, float] = {}
        self._vol_sum: Dict[str, float] = {}
        self._deviations: Dict[str, deque] = {}
        self._positions: Dict[str, Dict] = {}

    def reset(self) -> None:
        self._cur_date = None
        self._vol_tp.clear()
        self._vol_sum.clear()
        self._deviations.clear()
        self._positions.clear()

    def describe(self) -> str:
        p = self.parameters
        return (f"VWAPRev(sigma={p['sigma_entry']}, hold={p['hold_bars']}, "
                f"top_n={p['top_n']})")

    def _vwap(self, pair: str) -> Optional[float]:
        vs = self._vol_sum.get(pair, 0.0)
        if vs > 0:
            return self._vol_tp.get(pair, 0.0) / vs
        return None

    def _dev_std(self, pair: str, window: int) -> float:
        dq = self._deviations.get(pair)
        if not dq or len(dq) < 2:
            return 0.0
        arr = list(dq)[-window:]
        if len(arr) < 2:
            return 0.0
        mean = sum(arr) / len(arr)
        var = sum((d - mean) ** 2 for d in arr) / len(arr)
        return math.sqrt(var) if var > 1e-24 else 0.0

    def on_bars(self, bars: Dict[str, Dict], history: Dict[str, Any]) -> List[SignalResult]:
        signals: List[SignalResult] = []
        p = self.parameters

        ts = next((b["time"] for _, b in bars.items() if b), None)
        if not ts:
            return signals
        today = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)

        if self._cur_date != today:
            self._cur_date = today
            self._vol_tp.clear()
            self._vol_sum.clear()
            self._deviations.clear()
            self._positions.clear()

        sigma_entry = float(p["sigma_entry"])
        sigma_exit = float(p["sigma_exit"])
        min_bars = int(p["vwap_min_bars"])
        dev_win = int(p["dev_window"])
        hold_bars = int(p["hold_bars"])
        top_n = int(p["top_n"])
        max_pos = int(p["max_positions"])
        min_dev_pips = float(p["min_deviation_pips"])

        # Update VWAP + track deviations
        for pair, bar in bars.items():
            if bar is None:
                continue
            tp = (bar["high"] + bar["low"] + bar["close"]) / 3.0
            vol = max(bar["volume"], 0.0)
            self._vol_tp[pair] = self._vol_tp.get(pair, 0.0) + tp * vol
            self._vol_sum[pair] = self._vol_sum.get(pair, 0.0) + vol

            vwap = self._vwap(pair)
            if vwap is not None:
                dev = bar["close"] - vwap
                self._deviations.setdefault(pair, deque(maxlen=50)).append(dev)

        # Exit checks
        to_remove = []
        for pair, pos in list(self._positions.items()):
            pos["bars_held"] = pos.get("bars_held", 0) + 1
            if pos["bars_held"] >= pos.get("hold_bars", hold_bars):
                to_remove.append(pair)
                continue
            vwap = self._vwap(pair)
            if vwap is not None:
                dq = self._deviations.get(pair)
                if dq and len(dq) >= 3:
                    current_dev = bars.get(pair, {}).get("close", 0) - vwap
                    arr = list(dq)
                    rm = sum(arr) / len(arr)
                    # Exit when current deviation is back near zero
                    if abs(current_dev - rm) < sigma_exit * self._dev_std(pair, dev_win):
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
                    "reason": "reversion_or_hold",
                },
            ))

        if len(self._positions) >= max_pos:
            return signals

        # Entry logic
        candidates_long: List[Tuple[str, float, float]] = []
        candidates_short: List[Tuple[str, float, float]] = []

        for pair in p["trade_pairs"]:
            bar = bars.get(pair)
            if bar is None or pair in self._positions:
                continue
            vwap = self._vwap(pair)
            if vwap is None:
                continue
            dq = self._deviations.get(pair)
            if not dq or len(dq) < min_bars:
                continue
            std = self._dev_std(pair, dev_win)
            if std <= 0:
                continue

            current_dev = bar["close"] - vwap
            z = current_dev / std
            pip = _pip_value(pair)
            dev_pips = abs(current_dev) / pip

            if dev_pips < min_dev_pips:
                continue

            if z > sigma_entry:
                candidates_short.append((pair, dev_pips, z))
            elif z < -sigma_entry:
                candidates_long.append((pair, dev_pips, abs(z)))

        candidates_long.sort(key=lambda x: x[1], reverse=True)
        candidates_short.sort(key=lambda x: x[1], reverse=True)

        n_long = min(top_n, len(candidates_long))
        n_short = min(top_n, len(candidates_short))
        available = max_pos - len(self._positions)
        total_wanted = n_long + n_short
        if total_wanted > available:
            ratio = available / max(total_wanted, 1)
            n_long = min(int(n_long * ratio), n_long)
            n_short = min(available - n_long, n_short)

        for pair, dev_pips, z_abs in candidates_long[:n_long]:
            bar = bars[pair]
            entry_price = float(bar["open"])
            self._positions[pair] = {
                "direction": 1.0, "entry_time": ts,
                "bars_held": 0, "entry_price": entry_price,
                "hold_bars": hold_bars,
            }
            signals.append(SignalResult(
                timestamp=ts, signal=1.0,
                confidence=round(min(0.95, 0.5 + dev_pips / 40), 4),
                metadata={
                    "strategy": self.name, "pair": pair,
                    "action": "ENTER_LONG", "entry_price": entry_price,
                    "dev_pips": round(dev_pips, 1), "z_score": round(z_abs, 2),
                },
            ))

        for pair, dev_pips, z_abs in candidates_short[:n_short]:
            bar = bars[pair]
            entry_price = float(bar["open"])
            self._positions[pair] = {
                "direction": -1.0, "entry_time": ts,
                "bars_held": 0, "entry_price": entry_price,
                "hold_bars": hold_bars,
            }
            signals.append(SignalResult(
                timestamp=ts, signal=-1.0,
                confidence=round(min(0.95, 0.5 + dev_pips / 40), 4),
                metadata={
                    "strategy": self.name, "pair": pair,
                    "action": "ENTER_SHORT", "entry_price": entry_price,
                    "dev_pips": round(dev_pips, 1), "z_score": round(z_abs, 2),
                },
            ))

        return signals
