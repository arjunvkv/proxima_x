"""
V2+z Cross-Broker — Universal Specification v1.0 Compliant

Design per CROSS_BROKER_STRATEGY_SPEC.md:
  §5  Mid prices: mid = close + spread * point / 2
  §7  Time-defined windows: M5 UTC bars, not tick-based
  §9  Decision margin: margin = |z| - z_entry
  §10 Margin tracking: confidence = f(margin / z_entry)
  §11 Hysteresis: z_entry for entry, z_exit for exit
  §12 Cross-pair confirmation: independent pairs confirm currency move
  §13 Currency strength: decompose pair Z into base-currency state
  §14 Consensus features: Z + persistence + cross-confirmation + timescale
  §15-16 Persistence: N consecutive bars meeting threshold
  §17 Multi-timescale: short (1-bar) + long (3-bar) returns agree
  §19 Relative quantities: z-scores normalize returns by rolling volatility
  §22 Closed information: completed M5 bars only, no look-ahead
  §23 Exact formulas: all math defined below
  §26 State machine: IDLE -> ENTER -> POSITION -> EXIT -> IDLE per pair
  §27 Strategy/execution separated: SignalResult, not broker API calls
"""
from collections import deque
import math
from typing import Any, Dict, List, Optional, Set

from proxima_honest_backtest.engine.types import SignalResult
from proxima_honest_backtest.strategies.multi_pair_base import MultiPairStrategy

# ---------------------------------------------------------------------------
# §23: Exact mathematical specifications
# ---------------------------------------------------------------------------
# mid(p, t) = close(p, t) + spread(p, t) * point(p) / 2        [§5]
# r1(p, t)  = ln(mid(p, t) / mid(p, t - 1))                    [short return]
# r3(p, t)  = ln(mid(p, t) / mid(p, t - 3))                    [long return]
# z(p, t)   = (r(p, t) - mean(W)) / std(W)                     [§19]
#            over last W = z_window returns via streaming
# margin    = abs(z) - z_entry                                  [§9-10]
# entry iff: abs(z_short) > z_entry
#          AND sign(z_short) == sign(z_long)                    [§17]
#          AND direction filter allows                          [empirical]
#          AND cross-confirmation pairs agree                   [§12-13]
#          AND same direction for persistence_bars              [§15-16]
# exit  iff: trailing stop triggered                            [V2+z]
#          OR abs(z) < z_exit                                   [§11]
# ---------------------------------------------------------------------------

_POINT: Dict[str, float] = {
    "AUDNZD": 0.00001, "EURAUD": 0.00001, "EURNZD": 0.00001,
    "GBPAUD": 0.00001, "GBPCAD": 0.00001, "GBPNZD": 0.00001,
    "EURUSD": 0.00001, "GBPUSD": 0.00001, "AUDUSD": 0.00001,
    "USDCAD": 0.00001, "NZDUSD": 0.00001, "EURGBP": 0.00001,
    "EURCHF": 0.00001, "USDCHF": 0.00001,
    "EURJPY": 0.001, "GBPJPY": 0.001, "USDJPY": 0.001, "AUDJPY": 0.001,
}

_CONFIRM: Dict[str, Dict[str, List[str]]] = {
    "EURAUD": {"direct": ["EURUSD", "EURJPY"], "inverse": []},
    "EURNZD": {"direct": ["EURUSD", "EURJPY"], "inverse": []},
    "GBPAUD": {"direct": ["GBPJPY"], "inverse": ["EURGBP"]},
    "GBPCAD": {"direct": ["GBPJPY"], "inverse": ["EURGBP"]},
    "GBPNZD": {"direct": ["GBPJPY"], "inverse": ["EURGBP"]},
    "AUDNZD": {"direct": ["AUDUSD", "AUDJPY"], "inverse": []},
}

_EMPIRICAL_DIR: Dict[str, str] = {
    "EURAUD": "LONG", "GBPAUD": "LONG", "GBPCAD": "LONG",
    "EURNZD": "SHORT", "AUDNZD": "BOTH", "GBPNZD": "BOTH",
}


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


def _sign(x: float) -> float:
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)


class V2zCrossBrokerStrategy(MultiPairStrategy):
    """V2+z: z-score mean-reversion with trailing stop.
    Cross-broker compliant per CROSS_BROKER_STRATEGY_SPEC.md.
    Trades each pair independently.
    """

    DEFAULT_PARAMS: Dict[str, Any] = {
        "pairs": ["EURAUD", "GBPAUD", "GBPCAD", "EURNZD", "AUDNZD", "GBPNZD"],
        "z_entry": 3.5, "z_exit": 1.0, "z_window": 50,
        "lookback_short": 1, "lookback_long": 3,
        "persistence_bars": 1, "trailing_stop_a": 3.0,
        "trailing_trig_a": 1.0, "trailing_gap_a": 0.05,
        "direction": "BOTH", "require_cross_confirmation": True,
        "session_start": 0, "session_end": 24,
    }

    def __init__(self, parameters: Optional[Dict[str, Any]] = None) -> None:
        merged = dict(self.DEFAULT_PARAMS)
        if parameters:
            merged.update(parameters)
        super().__init__(merged)
        self._positions: Dict[str, Dict] = {}
        self._persistence: Dict[str, int] = {}
        self._prev_dir: Dict[str, Optional[float]] = {}
        self._z_streams: Dict[str, _StreamingZScore] = {}
        self._all_pairs: List[str] = self._resolve_all_pairs()
        self._pair_dir: Dict[str, str] = {}
        gd = self.parameters.get("direction", "BOTH")
        for p in self.parameters["pairs"]:
            self._pair_dir[p] = _EMPIRICAL_DIR.get(p, "BOTH") if gd == "EMPIRICAL" else gd

    def _resolve_all_pairs(self) -> List[str]:
        seen: Set[str] = set(self.parameters["pairs"])
        for p in self.parameters["pairs"]:
            c = _CONFIRM.get(p)
            if c:
                seen.update(c["direct"])
                seen.update(c["inverse"])
        return list(seen)

    def reset(self) -> None:
        self._positions.clear()
        self._persistence.clear()
        self._prev_dir.clear()
        self._z_streams.clear()

    def describe(self) -> str:
        p = self.parameters
        return (f"V2zCrossBroker(z_entry={p['z_entry']}, dir={p['direction']}, "
                f"persist={p['persistence_bars']}, cross={p['require_cross_confirmation']})")

    # --- §5: Mid price ---
    @staticmethod
    def _mid(bar: Dict[str, Any], pair: str) -> float:
        return float(bar["close"]) + float(bar.get("spread", 0)) * _POINT.get(pair, 1e-5) / 2.0

    # --- §19: Streaming z-scores (O(1) per bar) — UPDATES streams ---
    def _compute_z_scores(self, history: Dict, pairs: List[str],
                          lookback: int) -> Optional[Dict[str, float]]:
        z_scores: Dict[str, float] = {}
        window = int(self.parameters["z_window"])
        for pair in pairs:
            closes = history.get(pair, [])
            if len(closes) < lookback + 1 + window:
                return None
            curr, prev = closes[-1], closes[-(lookback + 1)]
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

    # --- §12-13: Cross-pair currency confirmation ---
    @staticmethod
    def _check_confirmation(pair: str, z_scores: Dict[str, float],
                            signal_dir: float) -> bool:
        conf = _CONFIRM.get(pair)
        if not conf:
            return True
        for cp in conf["direct"]:
            if _sign(z_scores.get(cp, 0.0)) != _sign(signal_dir):
                return False
        for cp in conf["inverse"]:
            if _sign(z_scores.get(cp, 0.0)) == _sign(signal_dir):
                return False
        return True

    # --- §11 entry + §15-16 persistence + §17 multi-timescale ---
    def _check_entry(self, pair: str, z_short: Dict, z_long: Dict,
                     ts) -> Optional[SignalResult]:
        p = self.parameters
        z_entry = float(p["z_entry"])
        zs, zl = z_short.get(pair, 0.0), z_long.get(pair, 0.0)
        if _sign(zs) != _sign(zl):
            return None
        abs_z = abs(zs)
        if abs_z <= z_entry:
            return None
        pair_dir = self._pair_dir.get(pair, "BOTH")
        sig_dir = _sign(zs)
        if (pair_dir == "LONG" and sig_dir < 0) or (pair_dir == "SHORT" and sig_dir > 0):
            return None
        if p.get("require_cross_confirmation", True):
            if not self._check_confirmation(pair, z_short, sig_dir):
                return None
        prv = self._prev_dir.get(pair)
        if prv is not None and _sign(prv) != sig_dir:
            self._persistence[pair] = 0
        self._prev_dir[pair] = sig_dir
        self._persistence[pair] = self._persistence.get(pair, 0) + 1
        if self._persistence[pair] < int(p["persistence_bars"]):
            return None
        margin = abs_z - z_entry
        confidence = min(0.99, 0.30 + margin / max(z_entry, 1e-12) * 0.69)
        return SignalResult(
            timestamp=ts, signal=sig_dir, confidence=round(confidence, 4),
            metadata={"strategy": self.name, "pair": pair,
                      "action": "ENTER_LONG" if sig_dir > 0 else "ENTER_SHORT",
                      "z": round(zs, 3), "z_long": round(zl, 3),
                      "margin": round(margin, 3),
                      "persistence": self._persistence[pair]},
        )

    # --- §11 exit: trailing stop + z reversion ---
    def _check_exit(self, pair: str, bars: Dict, ts,
                    z_short: Dict) -> Optional[SignalResult]:
        pos = self._positions.get(pair)
        if pos is None:
            return None
        bar = bars.get(pair)
        if bar is None:
            return None
        price = self._mid(bar, pair)
        p = self.parameters
        pos["bars_held"] = pos.get("bars_held", 0) + 1
        stop_dist = float(p["trailing_stop_a"]) * (price * float(p["trailing_gap_a"]))
        if pos["direction"] > 0:
            pos["trailing_lo"] = min(pos.get("trailing_lo", price), price)
            if price >= pos["trailing_lo"] + stop_dist:
                self._positions.pop(pair, None)
                return SignalResult(timestamp=ts, signal=-1.0, confidence=0.95,
                    metadata={"strategy": self.name, "pair": pair,
                              "action": "EXIT_LONG", "reason": "trailing_stop"})
        else:
            pos["trailing_hi"] = max(pos.get("trailing_hi", price), price)
            if price <= pos["trailing_hi"] - stop_dist:
                self._positions.pop(pair, None)
                return SignalResult(timestamp=ts, signal=1.0, confidence=0.95,
                    metadata={"strategy": self.name, "pair": pair,
                              "action": "EXIT_SHORT", "reason": "trailing_stop"})
        # §11: Exit on z reversion
        if z_short and pair in z_short and abs(z_short[pair]) < float(p["z_exit"]):
            self._positions.pop(pair, None)
            ex_sig = -1.0 if pos["direction"] > 0 else 1.0
            return SignalResult(timestamp=ts, signal=ex_sig, confidence=1.0,
                metadata={"strategy": self.name, "pair": pair,
                          "action": "EXIT_LONG" if pos["direction"] > 0 else "EXIT_SHORT",
                          "reason": "z_reversion", "z": round(z_short[pair], 3)})
        return None

    # --- Main entry point: compute z-scores ONCE per bar ---
    def on_bars(self, bars: Dict[str, Dict], history: Dict[str, Any]) -> List[SignalResult]:
        signals: List[SignalResult] = []
        p = self.parameters
        s_start, s_end = int(p["session_start"]), int(p["session_end"])
        ts = next((b["time"] for _, b in bars.items() if b), None)
        if not ts:
            return signals
        hour = ts.hour if hasattr(ts, "hour") else 0
        in_session = s_start <= hour < s_end

        # --- §19+§17: Compute ALL z-scores ONCE per bar (critical perf fix) ---
        lb_s = int(p["lookback_short"])
        lb_l = int(p["lookback_long"])
        z_short = self._compute_z_scores(history, self._all_pairs, lb_s)
        z_long = self._compute_z_scores(history, self._all_pairs, lb_l)
        if z_short is None or z_long is None:
            return signals

        for pair in self.parameters["pairs"]:
            if pair in self._positions:
                ex = self._check_exit(pair, bars, ts, z_short)
                if ex:
                    signals.append(ex)
                continue
            if not in_session:
                self._persistence[pair] = 0
                self._prev_dir[pair] = None
                continue
            if pair not in bars:
                continue
            en = self._check_entry(pair, z_short, z_long, ts)
            if en:
                price = self._mid(bars[pair], pair)
                self._positions[pair] = {"pair": pair, "direction": en.signal,
                    "bars_held": 0, "trailing_lo": price, "trailing_hi": price}
                signals.append(en)
        return signals
