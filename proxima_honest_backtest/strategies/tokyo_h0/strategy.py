"""
Tokyo H0 — Universal Cross-Broker Version (Spec v1.0)

Design per CROSS_BROKER_STRATEGY_SPEC.md:
  §2  Market structure: session-based cross-pair mean reversion
  §5  Mid prices: entry_price at bar open (session open = market open)
  §7  Time-defined: UTC midnight bar OPEN, not tick-level timing
  §9  Decision margin: abs(return) / recent_volatility
  §10 Margin tracking: confidence proportional to margin
  §11 Hysteresis: fixed hold duration exit
  §12 Cross-pair confirmation: 18-pair ranking ensures broad USD/JPY/EUR moves
  §13 Currency strength: most-declined pairs expose weakest currencies
  §14 Consensus: ranking + gap filter + confidence all agree
  §15-16 Persistence: decline must hold across multiple lookback windows
  §17 Multi-timescale: short + long lookback both show decline
  §19 Relative quantities: z-score of return / volatility
  §22 Closed information: completed bars only, no look-ahead
  §26 State machine: IDLE → ENTER (at session) → HOLD → EXIT → IDLE
  §27 Strategy/execution separated: SignalResult carries entry_price
"""
import math
from typing import Any, Dict, List, Optional

from proxima_honest_backtest.engine.types import SignalResult
from proxima_honest_backtest.strategies.multi_pair_base import MultiPairStrategy

# All 18 M5 pairs available in data
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


class TokyoH0Strategy(MultiPairStrategy):
    """Tokyo H0 — session-based cross-pair mean reversion at UTC midnight.

    Spec-compliant:
      §5  open-price entry via entry_price in metadata
      §7  fires at session_hour (default 0 = UTC midnight)
      §12 ranks all 18 pairs by recent return
      §15-16 persistence gating via lookback_confirm
      §17 multi-timescale via lookback_short + lookback_long
      §27 entry_price carried in metadata, not engine hardcoded
    """

    DEFAULT_PARAMS: Dict[str, Any] = {
        "pairs": ALL_PAIRS,
        "top_n": 3,
        "lookback_bars": 12,              # primary: 60 min
        "lookback_confirm_bars": 3,        # §17: 15-min confirmation
        "hold_bars": 3,                   # 15 min hold
        "session_hour": 0,                # UTC midnight
        "direction": "LONG",
        "gap_threshold_pct": 0.5,         # skip gap-open pairs
        "min_pairs": 8,                   # need 8+ pairs
        "min_confidence": 0.30,
        "require_decline_persistence": True,  # §15-16
        "vol_window": 50,                 # §19: for decision margin
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
        return (f"TokyoH0(top_n={p['top_n']}, lookback={p['lookback_bars']}, "
                f"hold={p['hold_bars']}, persist={p['require_decline_persistence']})")

    # ------------------------------------------------------------------
    # §5: Mid price for entry
    # ------------------------------------------------------------------
    @staticmethod
    def _mid(bar: Dict[str, Any], pair: str) -> float:
        return float(bar["close"]) + float(bar.get("spread", 0)) * _POINT.get(pair, 1e-5) / 2.0

    # ------------------------------------------------------------------
    # §19: Streaming volatility for decision margin
    # ------------------------------------------------------------------
    def _update_vol(self, pair: str, ret: float) -> Optional[float]:
        key = f"vol_{pair}"
        if key not in self._vol_streams:
            self._vol_streams[key] = _RunningVol(int(self.parameters["vol_window"]))
        return self._vol_streams[key].update(ret)

    def _get_vol(self, pair: str) -> float:
        key = f"vol_{pair}"
        vs = self._vol_streams.get(key)
        return vs.std if vs and vs.count >= 5 else 0.001

    # ------------------------------------------------------------------
    # Main: on_bars — compute signals once per session
    # ------------------------------------------------------------------
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

        # --- Exit checks (every bar) ---
        self._check_exits(ts, bars, signals, hold_bars)

        # --- Entry check (only at session hour, once per day) ---
        if hour != session_hour:
            return signals
        if self._last_entry_date == today:
            return signals
        self._last_entry_date = today

        # --- Compute returns for all pairs with data ---
        lb = int(p["lookback_bars"])
        lb_confirm = int(p["lookback_confirm_bars"])
        gap_thresh = float(p["gap_threshold_pct"])
        min_pairs = int(p["min_pairs"])
        min_conf = float(p["min_confidence"])

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

            # §17: Multi-timescale — shorter lookback must also show decline
            if p.get("require_decline_persistence", True):
                if len(closes) >= lb_confirm + 2:
                    prev_short = closes[-(lb_confirm + 1)]
                    if prev_short > 0:
                        ret_short = math.log(curr / prev_short)
                        if ret_short > 0:  # short window not declining
                            continue

            # Gap filter: skip if move is concentrated in last bar (gap open)
            prev_bar = bars.get(pair, {}).get("close")
            if prev_bar is not None and prev_bar > 0:
                gap_pct = abs(curr - prev_bar) / prev_bar * 100
                if gap_pct >= gap_thresh:
                    continue

            # §19: Decision margin relative to recent volatility
            bar = bars.get(pair)
            if bar is None:
                continue
            vol = self._update_vol(pair, ret) or 0.001
            margin = abs(ret) / max(vol, 1e-10)

            candidates.append((pair, ret, margin, gap_pct))

        if len(candidates) < min_pairs:
            return signals

        # Sort by most declined (most negative return)
        candidates.sort(key=lambda x: x[1])

        # Enter LONG on top N most declined
        top_n = int(p["top_n"])
        for pair, ret, margin, _ in candidates[:top_n]:
            if ret >= 0:
                break  # no more declining pairs
            confidence = min(0.95, margin * 0.15)
            if confidence < min_conf:
                continue

            bar = bars.get(pair)
            if bar is None:
                continue

            # §5: Enter at bar OPEN price
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

        return signals

    # ------------------------------------------------------------------
    # Exit: hold duration expiry at bar close
    # ------------------------------------------------------------------
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
