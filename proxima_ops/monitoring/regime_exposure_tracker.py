import json
import logging
import os
from collections import defaultdict
from typing import Any

import numpy as np


logger = logging.getLogger("proxima_ops.monitoring.regime_exposure_tracker")

STATE_PATH = "state/regime_exposure_state.json"
SYMBOLS = ["EURUSD", "GBPUSD", "EURJPY", "USDJPY"]

RSI_BIN_KEYS = ["<20", "20-25", "25-30", "30-35", "35-40", "40-60", "60-65", "65-70", "70-75", "75-80", ">80"]
ATR_PCTILE_BIN_KEYS = ["0-20", "20-40", "40-60", "60-80", "80-100"]
TREND_BIN_KEYS = ["bearish", "neutral", "bullish"]


def _empty_bins(bin_keys: list[str]) -> dict[str, int]:
    return {k: 0 for k in bin_keys}


def _compute_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.full_like(closes, np.nan)
    avg_loss = np.full_like(closes, np.nan)
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    for i in range(period + 1, len(closes)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period
    rs = avg_gain / np.maximum(avg_loss, 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = 50.0
    return rsi


def _compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    n = min(len(highs), len(lows), len(closes))
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
    atr = np.full(n, np.nan)
    atr[period] = np.mean(tr[:period])
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _atr_current_percentile(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14, lookback: int = 100) -> float:
    n = min(len(highs), len(lows), len(closes))
    if n < period + 1:
        return 50.0
    atr_series = _compute_atr(highs, lows, closes, period)
    valid_atr = atr_series[~np.isnan(atr_series)]
    if len(valid_atr) < 2:
        return 50.0
    current_atr = valid_atr[-1]
    window = valid_atr[-min(lookback, len(valid_atr)):]
    count_below = np.sum(window <= current_atr)
    percentile = (count_below / len(window)) * 100.0
    return percentile


def _trend_from_m5_ma_cross(close_prices: np.ndarray) -> int:
    if len(close_prices) < 10:
        return 0
    if len(close_prices) >= 20:
        m5_closes = close_prices[4::5]
        m5_closes = m5_closes[~np.isnan(m5_closes)]
        if len(m5_closes) >= 8:
            fast_period = max(2, len(m5_closes) // 4)
            slow_period = max(5, len(m5_closes) // 2)
            fast_ma = np.mean(m5_closes[-fast_period:])
            slow_ma = np.mean(m5_closes[-slow_period:])
        else:
            fast_ma = np.mean(close_prices[-5:])
            slow_ma = np.mean(close_prices[-20:])
    else:
        fast_ma = np.mean(close_prices[-5:])
        slow_ma = np.mean(close_prices[-max(10, len(close_prices)):])
    if fast_ma > slow_ma * 1.001:
        return 1
    elif fast_ma < slow_ma * 0.999:
        return -1
    return 0


def _classify_rsi(rsi: float) -> str:
    if rsi < 20:
        return "<20"
    elif rsi < 25:
        return "20-25"
    elif rsi < 30:
        return "25-30"
    elif rsi < 35:
        return "30-35"
    elif rsi < 40:
        return "35-40"
    elif rsi < 60:
        return "40-60"
    elif rsi < 65:
        return "60-65"
    elif rsi < 70:
        return "65-70"
    elif rsi < 75:
        return "70-75"
    elif rsi < 80:
        return "75-80"
    else:
        return ">80"


def _classify_atr_percentile(pctile: float) -> str:
    if pctile < 20:
        return "0-20"
    elif pctile < 40:
        return "20-40"
    elif pctile < 60:
        return "40-60"
    elif pctile < 80:
        return "60-80"
    else:
        return "80-100"


def _classify_trend(trend_val: int) -> str:
    if trend_val < 0:
        return "bearish"
    elif trend_val > 0:
        return "bullish"
    return "neutral"


class RegimeExposureTracker:
    def __init__(self, state_path: str = STATE_PATH):
        self._state_path = state_path
        self.total_cycles_tracked: int = 0

        self.rsi_bins: dict[str, int] = _empty_bins(RSI_BIN_KEYS)
        self.atr_percentile_bins: dict[str, int] = _empty_bins(ATR_PCTILE_BIN_KEYS)
        self.trend_bins: dict[str, int] = _empty_bins(TREND_BIN_KEYS)

        self.per_symbol: dict[str, dict[str, Any]] = {}
        for sym in SYMBOLS:
            self.per_symbol[sym] = {
                "rsi_bins": _empty_bins(RSI_BIN_KEYS),
                "atr_percentile_bins": _empty_bins(ATR_PCTILE_BIN_KEYS),
                "trend_bins": _empty_bins(TREND_BIN_KEYS),
                "rsi_extreme_events": [],
                "atr_spike_events": [],
                "total_cycles": 0,
            }

        self.rsi_extreme_events: list[tuple[int, str, float]] = []
        self.atr_spike_events: list[tuple[int, str, float]] = []

        self._cycles_with_rsi_extreme: int = 0
        self._cycles_with_atr_spike: int = 0

        self._latest_rsi: dict[str, float] = {}
        self._latest_atr_pctile: dict[str, float] = {}
        self._latest_trend: dict[str, int] = {}
        self._prev_regime_state: dict[str, dict] = {}

        self._load_state()
        logger.debug("RegimeExposureTracker initialized from %s (%d cycles tracked)", self._state_path, self.total_cycles_tracked)

    def record_cycle(self, cycle_data: dict, md: dict) -> None:
        self.total_cycles_tracked += 1
        closes_by_sym = md.get("closes", {})
        highs_by_sym = md.get("highs", {})
        lows_by_sym = md.get("lows", {})

        cycle_has_rsi_extreme = False
        cycle_has_atr_spike = False

        for sym in SYMBOLS:
            closes = closes_by_sym.get(sym)
            highs = highs_by_sym.get(sym)
            lows = lows_by_sym.get(sym)

            if closes is None or highs is None or lows is None:
                continue
            if len(closes) < 15 or len(highs) < 15 or len(lows) < 15:
                continue

            closes_arr = np.asarray(closes, dtype=np.float64)
            highs_arr = np.asarray(highs, dtype=np.float64)
            lows_arr = np.asarray(lows, dtype=np.float64)

            rsi_series = _compute_rsi(closes_arr)
            rsi_val = float(rsi_series[~np.isnan(rsi_series)][-1]) if np.sum(~np.isnan(rsi_series)) > 0 else 50.0
            self._latest_rsi[sym] = rsi_val

            atr_pctile = _atr_current_percentile(highs_arr, lows_arr, closes_arr)
            self._latest_atr_pctile[sym] = atr_pctile

            trend_val = _trend_from_m5_ma_cross(closes_arr)
            self._latest_trend[sym] = trend_val

            rsi_bin = _classify_rsi(rsi_val)
            atr_bin = _classify_atr_percentile(atr_pctile)
            trend_bin = _classify_trend(trend_val)

            self.rsi_bins[rsi_bin] += 1
            self.atr_percentile_bins[atr_bin] += 1
            self.trend_bins[trend_bin] += 1

            ps = self.per_symbol[sym]
            ps["rsi_bins"][rsi_bin] += 1
            ps["atr_percentile_bins"][atr_bin] += 1
            ps["trend_bins"][trend_bin] += 1
            ps["total_cycles"] += 1

            if rsi_val < 30 or rsi_val > 70:
                cycle_has_rsi_extreme = True
                event = (self.total_cycles_tracked, sym, round(rsi_val, 2))
                self.rsi_extreme_events.append(event)
                ps["rsi_extreme_events"].append(event)

            if atr_pctile > 60:
                cycle_has_atr_spike = True
                event = (self.total_cycles_tracked, sym, round(atr_pctile, 2))
                self.atr_spike_events.append(event)
                ps["atr_spike_events"].append(event)

        if cycle_has_rsi_extreme:
            self._cycles_with_rsi_extreme += 1
        if cycle_has_atr_spike:
            self._cycles_with_atr_spike += 1

        self._log_state_changes(closes_by_sym)
        self._save_state()

    def regime_exposure_report(self) -> dict:
        total_cycles = self.total_cycles_tracked
        total_observations = sum(self.rsi_bins.values())

        def pct(count: int, divisor: int) -> float:
            return round((count / divisor) * 100.0, 2) if divisor > 0 else 0.0

        rsi_distribution = {k: pct(v, total_observations) for k, v in self.rsi_bins.items()}
        atr_distribution = {k: pct(v, total_observations) for k, v in self.atr_percentile_bins.items()}
        trend_distribution = {k: pct(v, total_observations) for k, v in self.trend_bins.items()}

        extreme_rate = pct(self._cycles_with_rsi_extreme, total_cycles)
        atr_spike_rate = pct(self._cycles_with_atr_spike, total_cycles)

        per_symbol_report = {}
        for sym in SYMBOLS:
            ps = self.per_symbol[sym]
            sym_obs = sum(ps["rsi_bins"].values())
            per_symbol_report[sym] = {
                "total_cycles": ps["total_cycles"],
                "rsi_distribution": {k: pct(v, sym_obs) for k, v in ps["rsi_bins"].items()},
                "atr_distribution": {k: pct(v, sym_obs) for k, v in ps["atr_percentile_bins"].items()},
                "trend_distribution": {k: pct(v, sym_obs) for k, v in ps["trend_bins"].items()},
                "rsi_extreme_rate": pct(len(ps["rsi_extreme_events"]), ps["total_cycles"]),
                "atr_spike_rate": pct(len(ps["atr_spike_events"]), ps["total_cycles"]),
                "rsi_extreme_count": len(ps["rsi_extreme_events"]),
                "atr_spike_count": len(ps["atr_spike_events"]),
            }

        return {
            "total_cycles_tracked": total_cycles,
            "rsi_distribution": rsi_distribution,
            "atr_distribution": atr_distribution,
            "trend_distribution": trend_distribution,
            "rsi_extreme_rate": extreme_rate,
            "rsi_extreme_count": len(self.rsi_extreme_events),
            "atr_spike_rate": atr_spike_rate,
            "atr_spike_count": len(self.atr_spike_events),
            "per_symbol": per_symbol_report,
        }

    def is_regime_active(self) -> bool:
        if not self._latest_rsi:
            return False
        return any(rsi < 30 or rsi > 70 for rsi in self._latest_rsi.values())

    def active_symbols_summary(self) -> dict:
        summary = {}
        for sym in SYMBOLS:
            rsi = self._latest_rsi.get(sym)
            atr = self._latest_atr_pctile.get(sym)
            trend = self._latest_trend.get(sym)
            if rsi is None:
                continue
            summary[sym] = {
                "rsi": round(rsi, 2),
                "rsi_extreme": rsi < 30 or rsi > 70,
                "atr_percentile": round(atr, 2) if atr is not None else None,
                "atr_spike": (atr or 0) > 60,
                "trend": trend if trend is not None else 0,
                "trend_label": _classify_trend(trend) if trend is not None else "neutral",
            }
        return summary

    def _log_state_changes(self, closes_by_sym: dict) -> None:
        changes = []
        for sym in SYMBOLS:
            if sym not in closes_by_sym:
                continue
            current = {
                "rsi": round(self._latest_rsi.get(sym, 50.0), 2),
                "atr_pctile": round(self._latest_atr_pctile.get(sym, 50.0), 2),
                "trend": _classify_trend(self._latest_trend.get(sym, 0)),
            }
            prev = self._prev_regime_state.get(sym)
            if prev is None or current != prev:
                changes.append(f"{sym}: rsi={current['rsi']} atr_pctile={current['atr_pctile']} trend={current['trend']}")
                self._prev_regime_state[sym] = current
        if changes:
            logger.info("Regime state change [cycle %d]: %s", self.total_cycles_tracked, " | ".join(changes))

    def _save_state(self) -> None:
        state = {
            "total_cycles_tracked": self.total_cycles_tracked,
            "rsi_bins": dict(self.rsi_bins),
            "atr_percentile_bins": dict(self.atr_percentile_bins),
            "trend_bins": dict(self.trend_bins),
            "per_symbol": {},
            "rsi_extreme_events": [(c, s, r) for (c, s, r) in self.rsi_extreme_events],
            "atr_spike_events": [(c, s, p) for (c, s, p) in self.atr_spike_events],
            "_cycles_with_rsi_extreme": self._cycles_with_rsi_extreme,
            "_cycles_with_atr_spike": self._cycles_with_atr_spike,
            "_latest_rsi": dict(self._latest_rsi),
            "_latest_atr_pctile": dict(self._latest_atr_pctile),
            "_latest_trend": {k: int(v) for k, v in self._latest_trend.items()},
        }
        for sym in SYMBOLS:
            ps = self.per_symbol[sym]
            state["per_symbol"][sym] = {
                "rsi_bins": dict(ps["rsi_bins"]),
                "atr_percentile_bins": dict(ps["atr_percentile_bins"]),
                "trend_bins": dict(ps["trend_bins"]),
                "rsi_extreme_events": [(c, s, r) for (c, s, r) in ps["rsi_extreme_events"]],
                "atr_spike_events": [(c, s, p) for (c, s, p) in ps["atr_spike_events"]],
                "total_cycles": ps["total_cycles"],
            }
        if os.path.dirname(self._state_path):
            os.makedirs(os.path.dirname(self._state_path), exist_ok=True)
        with open(self._state_path, "w") as f:
            json.dump(state, f, indent=2, default=str)
        logger.debug("RegimeExposureTracker state saved to %s", self._state_path)

    def _load_state(self) -> bool:
        if not os.path.exists(self._state_path):
            return False
        try:
            with open(self._state_path) as f:
                state = json.load(f)
            self.total_cycles_tracked = state.get("total_cycles_tracked", 0)
            self.rsi_bins = defaultdict(int, state.get("rsi_bins", {}))
            self.atr_percentile_bins = defaultdict(int, state.get("atr_percentile_bins", {}))
            self.trend_bins = defaultdict(int, state.get("trend_bins", {}))

            raw_ps = state.get("per_symbol", {})
            for sym in SYMBOLS:
                if sym in raw_ps:
                    d = raw_ps[sym]
                    self.per_symbol[sym] = {
                        "rsi_bins": defaultdict(int, d.get("rsi_bins", {})),
                        "atr_percentile_bins": defaultdict(int, d.get("atr_percentile_bins", {})),
                        "trend_bins": defaultdict(int, d.get("trend_bins", {})),
                        "rsi_extreme_events": [tuple(x) for x in d.get("rsi_extreme_events", [])],
                        "atr_spike_events": [tuple(x) for x in d.get("atr_spike_events", [])],
                        "total_cycles": d.get("total_cycles", 0),
                    }
                else:
                    self.per_symbol[sym] = {
                        "rsi_bins": _empty_bins(RSI_BIN_KEYS),
                        "atr_percentile_bins": _empty_bins(ATR_PCTILE_BIN_KEYS),
                        "trend_bins": _empty_bins(TREND_BIN_KEYS),
                        "rsi_extreme_events": [],
                        "atr_spike_events": [],
                        "total_cycles": 0,
                    }

            self.rsi_extreme_events = [tuple(x) for x in state.get("rsi_extreme_events", [])]
            self.atr_spike_events = [tuple(x) for x in state.get("atr_spike_events", [])]
            self._cycles_with_rsi_extreme = state.get("_cycles_with_rsi_extreme", 0)
            self._cycles_with_atr_spike = state.get("_cycles_with_atr_spike", 0)
            self._latest_rsi = defaultdict(float, state.get("_latest_rsi", {}))
            self._latest_atr_pctile = defaultdict(float, state.get("_latest_atr_pctile", {}))
            self._latest_trend = defaultdict(int, {k: int(v) for k, v in state.get("_latest_trend", {}).items()})

            logger.debug("RegimeExposureTracker state loaded from %s (%d cycles)", self._state_path, self.total_cycles_tracked)
            return True
        except Exception as e:
            logger.warning("Failed to load RegimeExposureTracker state: %s", e)
            return False
