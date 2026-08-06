from typing import Any, Dict, List, Optional

import numpy as np

from proxima_honest_backtest.engine.types import SignalResult
from proxima_honest_backtest.strategies.multi_pair_base import MultiPairStrategy

ALL_CURRENCIES = ['USD', 'EUR', 'JPY', 'GBP', 'AUD', 'NZD', 'CAD', 'CHF']

ALL_PAIRS_28 = [
    "EURUSD","GBPUSD","USDJPY","AUDUSD","NZDUSD","USDCAD","USDCHF",
    "EURJPY","GBPJPY","EURGBP","EURAUD","EURCHF","EURCAD","EURNZD",
    "GBPAUD","GBPCAD","GBPCHF","GBPNZD",
    "AUDJPY","AUDCAD","AUDCHF","AUDNZD",
    "NZDJPY","NZDCAD","NZDCHF",
    "CADJPY","CADCHF",
    "CHFJPY",
]

BEST_PAIR = {
    "USD": "AUDUSD", "EUR": "EURUSD", "JPY": "NZDJPY",
    "GBP": "GBPUSD", "AUD": "AUDUSD", "NZD": "NZDUSD",
    "CAD": "NZDCAD", "CHF": "USDCHF",
}


def _base_quote(pair):
    for c in ALL_CURRENCIES:
        if pair.startswith(c):
            return c, pair[len(c):]
    return None, None


def _currency_pairs():
    result = {c: [] for c in ALL_CURRENCIES}
    for pair in ALL_PAIRS_28:
        base, quote = _base_quote(pair)
        if base and quote:
            if base in result:
                result[base].append((1.0, pair))
            if quote in result:
                result[quote].append((-1.0, pair))
    return result

CURR_PAIRS = _currency_pairs()


def _pair_map():
    m = {}
    for pj, p in enumerate(ALL_PAIRS_28):
        b, q = _base_quote(p)
        ci1 = ALL_CURRENCIES.index(b)
        ci2 = ALL_CURRENCIES.index(q)
        m[(ci1, ci2)] = pj
        m[(ci2, ci1)] = pj
    return m

CURR_PAIR_MAP = _pair_map()


def _curr_sign():
    m = {}
    for ci, c in enumerate(ALL_CURRENCIES):
        for sg, pn in CURR_PAIRS.get(c, []):
            pj = ALL_PAIRS_28.index(pn)
            m[(ci, pj)] = sg
    return m

CURR_SIGN = _curr_sign()


class BlindSpotAlphaStrategy(MultiPairStrategy):
    """Blind Spot Alpha — Currency Divergence Momentum.

    Find strongest (max z) and weakest (min z) currencies.
    Both must exceed |z| >= z_threshold in opposite directions.
    Trade the pair between them where both currencies confirm direction.
    """

    DEFAULT_PARAMS: Dict[str, Any] = {
        "z_threshold": 2.0,
        "z_window": 400,
        "vol_window": 200,
        "hold_bars": 5,
        "pairs": ALL_PAIRS_28,
    }

    def __init__(self, parameters: Optional[Dict[str, Any]] = None) -> None:
        merged = dict(self.DEFAULT_PARAMS)
        if parameters:
            merged.update(parameters)
        super().__init__(merged)
        self._curr_history: Dict[str, List[float]] = {c: [] for c in ALL_CURRENCIES}
        self._pair_returns: Dict[str, List[float]] = {}
        self._positions: Dict[str, Dict] = {}
        self._pair_vol_fixed: Dict[str, float] = {}
        self._vol_warmup_done = False

    def reset(self) -> None:
        self._curr_history = {c: [] for c in ALL_CURRENCIES}
        self._pair_returns = {}
        self._positions = {}
        self._pair_vol_fixed = {}
        self._vol_warmup_done = False

    def on_bars(self, bars: Dict[str, Dict], history: Dict[str, Any]) -> List[SignalResult]:
        signals: List[SignalResult] = []

        ts = None
        for _, bar in bars.items():
            if bar:
                ts = bar.get("time")
                break
        if not ts:
            return signals

        self._check_exits(ts, signals)

        if self._positions:
            return signals

        # Strict contract: close-to-close returns from COMPLETED closes only.
        minute_returns = {}
        for pair in self.parameters["pairs"]:
            closes = history.get(pair, [])
            if len(closes) < 2:
                continue
            curr = closes[-1]
            prev = closes[-2]
            if prev > 0 and curr > 0:
                ret = np.log(curr / prev)
                minute_returns[pair] = ret
                if pair not in self._pair_returns:
                    self._pair_returns[pair] = []
                self._pair_returns[pair].append(ret)

        if len(minute_returns) < 5:
            return signals

        if not self._vol_warmup_done:
            self._compute_fixed_vols()
            return signals

        curr_returns = self._compute_currency_returns(minute_returns)
        if len(curr_returns) < 3:
            return signals

        for c, ret in curr_returns.items():
            self._curr_history[c].append(ret)

        z_threshold = float(self.parameters["z_threshold"])
        z_window = int(self.parameters["z_window"])

        z_scores = np.zeros(len(ALL_CURRENCIES))
        for ci, c in enumerate(ALL_CURRENCIES):
            hist = self._curr_history.get(c, [])
            if len(hist) < max(z_window, 10):
                continue
            recent = hist[-z_window:]
            mean = float(np.mean(recent))
            std = float(np.std(recent))
            if std < 1e-12:
                continue
            z_scores[ci] = (curr_returns.get(c, 0) - mean) / std

        sorted_idx = np.argsort(z_scores)
        strongest_ci = int(sorted_idx[-1])
        weakest_ci = int(sorted_idx[0])
        sz = z_scores[strongest_ci]
        wz = z_scores[weakest_ci]

        if sz < z_threshold or wz > -z_threshold:
            return signals

        if abs(sz) >= abs(wz):
            trade_ci = strongest_ci
            other_ci = weakest_ci
        else:
            trade_ci = weakest_ci
            other_ci = strongest_ci

        pair = BEST_PAIR.get(ALL_CURRENCIES[trade_ci])
        if pair is None or pair not in minute_returns:
            return signals

        pj = ALL_PAIRS_28.index(pair)
        sg = CURR_SIGN.get((trade_ci, pj), 0)
        if sg == 0:
            return signals
        direction = 1 if z_scores[trade_ci] > 0 else -1
        d_star = int(direction * sg)

        other_sg = CURR_SIGN.get((other_ci, pj), 0)
        if other_sg == 0:
            return signals
        other_dir = 1 if z_scores[other_ci] > 0 else -1
        other_dstar = int(other_dir * other_sg)

        if d_star != other_dstar:
            return signals

        confidence = min(0.99, max(abs(sz), abs(wz)) / 5.0)

        self._positions[pair] = {
            "direction": d_star,
            "entry_time": ts,
            "bars_held": 0,
            "entry_price": bars[pair]["open"],
        }

        signals.append(SignalResult(
            timestamp=ts,
            signal=float(d_star),
            confidence=round(confidence, 4),
            metadata={
                "strategy": self.name,
                "pair": pair,
                "action": "ENTER_LONG" if d_star > 0 else "ENTER_SHORT",
                "strongest_currency": ALL_CURRENCIES[strongest_ci],
                "strongest_z": round(float(sz), 2),
                "weakest_currency": ALL_CURRENCIES[weakest_ci],
                "weakest_z": round(float(wz), 2),
                "entry_price": float(bars[pair]["open"]),
            },
        ))

        return signals

    def _compute_fixed_vols(self):
        vol_window = int(self.parameters["vol_window"])
        for pair in self.parameters["pairs"]:
            buf = self._pair_returns.get(pair, [])
            if len(buf) >= vol_window:
                self._pair_vol_fixed[pair] = float(np.std(buf[:vol_window])) + 1e-10
        if len(self._pair_vol_fixed) >= 5:
            self._vol_warmup_done = True

    def _compute_currency_returns(self, minute_returns):
        curr_rets: Dict[str, List] = {c: [[], []] for c in ALL_CURRENCIES}
        for pair, ret in minute_returns.items():
            vol = self._pair_vol_fixed.get(pair, 1e-10)
            base, quote = _base_quote(pair)
            if base and base in curr_rets:
                curr_rets[base][0].append(ret)
                curr_rets[base][1].append(vol)
            if quote and quote in curr_rets:
                curr_rets[quote][0].append(-ret)
                curr_rets[quote][1].append(vol)

        result = {}
        for c in ALL_CURRENCIES:
            rets, vols = curr_rets[c]
            if len(rets) < 2:
                continue
            w = np.array([1.0 / v for v in vols])
            w = w / np.sum(w)
            result[c] = float(np.dot(rets, w))
        return result

    def _check_exits(self, ts, signals):
        hold = int(self.parameters["hold_bars"])
        to_remove = []
        for pair, pos in self._positions.items():
            pos["bars_held"] += 1
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

    def describe(self) -> str:
        p = self.parameters
        return f"BlindSpotAlpha(z={p['z_threshold']}, hold={p['hold_bars']}, win={p['z_window']})"
