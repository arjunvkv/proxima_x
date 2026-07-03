from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numba
from numpy.typing import NDArray

from research.mechanism_discovery.temporal_topology import TemporalTopology
from research.mechanism_discovery.energy_dynamics import EnergyDynamics
from research.mechanism_discovery.memory_landscape import MemoryLandscape
from research.information_discovery.mi_estimator import _fast_conditional_mutual_info


@numba.jit(nopython=True, cache=True)
def _numba_rolling_state_mutation(states: NDArray[np.int64], window: int) -> NDArray[np.float64]:
    n = len(states)
    result = np.zeros(n, dtype=np.float64)
    for i in range(window, n):
        changes = 0
        for j in range(i - window + 1, i):
            if states[j] != states[j - 1]:
                changes += 1
        result[i] = float(changes) / float(window)
    return result


@numba.jit(nopython=True, cache=True)
def _numba_rolling_regime_prob(states: NDArray[np.int64], window: int) -> NDArray[np.float64]:
    n = len(states)
    result = np.zeros(n, dtype=np.float64)
    for i in range(window, n):
        current = states[i]
        diff_count = 0
        for j in range(i - window, i):
            if states[j] != current:
                diff_count += 1
        result[i] = float(diff_count) / float(window)
    return result


@numba.jit(nopython=True, cache=True)
def _future_returns(price: NDArray[np.float64], horizons: NDArray[np.int32]) -> NDArray[np.float64]:
    n = len(price)
    h = len(horizons)
    result = np.full((n, h), np.nan, dtype=np.float64)
    for i in range(n):
        for j in range(h):
            future_idx = i + horizons[j]
            if future_idx < n:
                result[i, j] = np.log(price[future_idx] / price[i])
    return result


@numba.jit(nopython=True, cache=True)
def _future_drawdown(price: NDArray[np.float64], horizon: int) -> NDArray[np.float64]:
    n = len(price)
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(n - horizon):
        entry = price[i]
        peak = entry
        max_dd = 0.0
        for j in range(1, horizon + 1):
            p = price[i + j]
            if p > peak:
                peak = p
            dd = (p - peak) / peak
            if dd < max_dd:
                max_dd = dd
        result[i] = max_dd
    return result


@numba.jit(nopython=True, cache=True)
def _future_runup(price: NDArray[np.float64], horizon: int) -> NDArray[np.float64]:
    n = len(price)
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(n - horizon):
        entry = price[i]
        trough = entry
        max_ru = 0.0
        for j in range(1, horizon + 1):
            p = price[i + j]
            if p < trough:
                trough = p
            ru = (p - trough) / trough
            if ru > max_ru:
                max_ru = ru
        result[i] = max_ru
    return result


@numba.jit(nopython=True, cache=True)
def _numba_skew(arr: NDArray[np.float64]) -> float:
    n = len(arr)
    if n < 3:
        return 0.0
    m = 0.0
    for i in range(n):
        m += arr[i]
    m /= n
    v = 0.0
    for i in range(n):
        d = arr[i] - m
        v += d * d
    std = np.sqrt(v / n)
    if std < 1e-12:
        return 0.0
    s = 0.0
    for i in range(n):
        d = (arr[i] - m) / std
        s += d * d * d
    return s / n


@numba.jit(nopython=True, cache=True)
def _numba_kurtosis(arr: NDArray[np.float64]) -> float:
    n = len(arr)
    if n < 4:
        return 0.0
    m = 0.0
    for i in range(n):
        m += arr[i]
    m /= n
    v = 0.0
    for i in range(n):
        d = arr[i] - m
        v += d * d
    std = np.sqrt(v / n)
    if std < 1e-12:
        return 0.0
    k = 0.0
    for i in range(n):
        d = (arr[i] - m) / std
        k += d * d * d * d
    return k / n - 3.0


@numba.jit(nopython=True, cache=True)
def _cross_correlate_nb(x: NDArray[np.float64], y: NDArray[np.float64], max_lag: int) -> NDArray[np.float64]:
    n = len(x)
    result = np.empty(2 * max_lag + 1, dtype=np.float64)
    for k in range(-max_lag, max_lag + 1):
        if k >= 0:
            sx, ex = k, n
            sy, ey = 0, n - k
        else:
            sx, ex = 0, n + k
            sy, ey = -k, n
        length = ex - sx
        if length < 3:
            result[k + max_lag] = 0.0
            continue
        xs = x[sx:ex]
        ys = y[sy:ey]
        xm = 0.0
        ym = 0.0
        for i in range(length):
            xm += xs[i]
            ym += ys[i]
        xm /= length
        ym /= length
        xv, yv, cov = 0.0, 0.0, 0.0
        for i in range(length):
            dx = xs[i] - xm
            dy = ys[i] - ym
            cov += dx * dy
            xv += dx * dx
            yv += dy * dy
        den = np.sqrt(max(xv, 1e-12)) * np.sqrt(max(yv, 1e-12))
        result[k + max_lag] = cov / den if den > 0 else 0.0
    return result


@numba.jit(nopython=True, cache=True)
def _find_peak_lag(x: NDArray[np.float64], y: NDArray[np.float64], max_lag: int) -> tuple[int, float]:
    corr = _cross_correlate_nb(x, y, max_lag)
    peak_idx = 0
    for i in range(1, len(corr)):
        if abs(corr[i]) > abs(corr[peak_idx]):
            peak_idx = i
    lag = peak_idx - max_lag
    return lag, corr[peak_idx]


VALIDATED_VARIABLES = ["memory_density", "energy_storage", "adaptive_time",
                       "state_mutation_rate", "regime_change_probability"]
PRIMARY_VARIABLES = ["memory_density", "energy_storage", "adaptive_time"]
SECONDARY_VARIABLES = ["state_mutation_rate", "regime_change_probability"]
HORIZONS = [1, 5, 20, 50, 100, 500]
TARGET_ASSETS = ["EURJPY", "USDJPY", "GBPJPY", "XAUUSD"]
TIME_WINDOWS = [
    ("2018-01-01", "2020-01-01", "2018-2020"),
    ("2020-01-01", "2022-01-01", "2020-2022"),
    ("2022-01-01", "2024-01-01", "2022-2024"),
    ("2024-01-01", "2027-01-01", "2024-2026"),
]


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, tuple):
        return list(_clean(v) for v in obj)
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist() if hasattr(obj, 'tolist') else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    return obj


@dataclass
class AELResult:
    rq_name: str
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"rq_name": self.rq_name, "status": self.status,
                "metrics": _clean(self.metrics)}


class AlphaValidator:
    def __init__(self, data_dir: str = "data/market"):
        self.data_dir = Path(data_dir)
        self.tt = TemporalTopology()
        self.ed = EnergyDynamics()
        self.ml = MemoryLandscape()

    def load_asset_data(self, asset: str) -> dict[str, NDArray]:
        path = self.data_dir / f"{asset}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Data not found: {path}")
        import polars as pl
        df = pl.read_parquet(str(path))
        price = df["close"].to_numpy().astype(np.float64)
        returns = (df["log_return"].to_numpy().astype(np.float64)
                   if "log_return" in df.columns
                   else np.diff(np.log(price), prepend=np.log(price[0])))
        volume = (df["volume"].to_numpy().astype(np.float64)
                  if "volume" in df.columns
                  else np.ones(len(price), dtype=np.float64))
        high = df["high"].to_numpy().astype(np.float64) if "high" in df.columns else price.copy()
        low = df["low"].to_numpy().astype(np.float64) if "low" in df.columns else price.copy()
        return {"price": price, "returns": returns, "volume": volume, "high": high, "low": low}

    def load_data_window(self, asset: str, start: str, end: str) -> dict[str, NDArray]:
        path = self.data_dir / f"{asset}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Data not found: {path}")
        import polars as pl
        import datetime
        df = pl.read_parquet(str(path))
        dt_start = datetime.datetime.strptime(start, "%Y-%m-%d")
        dt_end = datetime.datetime.strptime(end, "%Y-%m-%d")
        df = df.filter((pl.col("timestamp") >= dt_start) & (pl.col("timestamp") < dt_end))
        price = df["close"].to_numpy().astype(np.float64)
        returns = (df["log_return"].to_numpy().astype(np.float64)
                   if "log_return" in df.columns
                   else np.diff(np.log(price), prepend=np.log(price[0])))
        volume = (df["volume"].to_numpy().astype(np.float64)
                  if "volume" in df.columns
                  else np.ones(len(price), dtype=np.float64))
        high = df["high"].to_numpy().astype(np.float64) if "high" in df.columns else price.copy()
        low = df["low"].to_numpy().astype(np.float64) if "low" in df.columns else price.copy()
        return {"price": price, "returns": returns, "volume": volume, "high": high, "low": low}

    def compute_signals(self, data: dict) -> dict[str, Any]:
        price = data["price"]
        returns = data["returns"]
        n = len(price)

        result_tt = self.tt.compute(data)
        result_ed = self.ed.compute(data)
        result_ml = self.ml.compute(data)

        states = result_tt.get("time_regime", np.zeros(n, dtype=np.int64)).astype(np.int64)

        analysis: dict[str, Any] = {
            "price": price,
            "returns": returns,
            "volume": data.get("volume", np.ones(n, dtype=np.float64)),
            "adaptive_time": np.asarray(result_tt.get("adaptive_time_coordinate", np.zeros(n)), dtype=np.float64),
            "time_density": result_tt.get("time_density", np.zeros(n)),
            "event_density": result_tt.get("event_density", np.zeros(n)),
            "information_density": result_tt.get("information_density", np.zeros(n)),
            "behavior_density": result_tt.get("behavior_density", np.zeros(n)),
            "energy_creation": result_ed.get("energy_creation", np.zeros(n)),
            "energy_storage": result_ed.get("energy_storage", np.zeros(n)),
            "energy_release": result_ed.get("energy_release", np.zeros(n)),
            "energy_dissipation": result_ed.get("energy_dissipation", np.zeros(n)),
            "energy_balance": result_ed.get("energy_balance", np.zeros(n)),
            "memory_density": np.asarray(result_ml.get("memory_density", np.zeros(n)), dtype=np.float64),
            "memory_gradient": np.asarray(result_ml.get("memory_gradient", np.zeros(n)), dtype=np.float64),
            "memory_conflict": np.abs(
                np.asarray(result_ml.get("memory_density", np.zeros(n)), dtype=np.float64)
                - np.asarray(result_ml.get("memory_gradient", np.zeros(n)), dtype=np.float64)),
            "memory_interference": result_ml.get("memory_interference", np.zeros(n)),
            "memory_landscape": result_ml.get("memory_landscape", np.zeros(n)),
            "states": states,
        }
        analysis["state_mutation_rate"] = _numba_rolling_state_mutation(states, 20)
        analysis["regime_change_probability"] = _numba_rolling_regime_prob(states, 20)

        return analysis

    def decile_bins(self, signal: NDArray[np.float64]) -> tuple[NDArray[np.float64], list[NDArray[np.bool_]]]:
        percentiles = np.linspace(0, 100, 11)
        boundaries = np.nanpercentile(signal, percentiles)
        masks = []
        for i in range(10):
            if i == 0:
                mask = signal <= boundaries[i + 1]
            elif i == 9:
                mask = signal > boundaries[i]
            else:
                mask = (signal > boundaries[i]) & (signal <= boundaries[i + 1])
            masks.append(mask)
        return boundaries, masks

    def bucket_statistics(self, future_returns: NDArray[np.float64], mask: NDArray[np.bool_]) -> dict[str, float]:
        vals = future_returns[mask]
        return {
            "mean": float(np.nanmean(vals)),
            "median": float(np.nanmedian(vals)),
            "std": float(np.nanstd(vals)),
            "skew": _numba_skew(vals),
            "kurtosis": _numba_kurtosis(vals),
            "n": int(np.sum(mask)),
        }

    def peak_lag_analysis(self, source: str, target: str, signals: dict, max_lag: int = 200) -> dict[str, Any]:
        s = np.asarray(signals.get(source, np.zeros(1)), dtype=np.float64)
        t = np.asarray(signals.get(target, np.zeros(1)), dtype=np.float64)
        n = min(len(s), len(t))
        if n < max_lag * 2 + 1:
            return {"peak_lag": 0, "peak_corr": 0.0, "n": n}
        lag, corr = _find_peak_lag(s[:n], t[:n], max_lag)
        return {"peak_lag": lag, "peak_corr": corr, "n": n}

    def information_flow(self, source: str, target: str, signals: dict, n_bins: int = 20) -> float:
        s = np.asarray(signals.get(source, np.zeros(1)), dtype=np.float64)
        t = np.asarray(signals.get(target, np.zeros(1)), dtype=np.float64)
        n = min(len(s), len(t))
        if n < 3:
            return 0.0
        return float(_fast_conditional_mutual_info(s[:n - 1], t[1:n], t[:n - 1], n_bins))
