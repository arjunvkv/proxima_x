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
def _zscore(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    n = len(arr)
    m = 0.0
    for i in range(n):
        m += arr[i]
    m /= n
    v = 0.0
    for i in range(n):
        d = arr[i] - m
        v += d * d
    s = np.sqrt(v / max(n - 1, 1))
    s = max(s, 1e-12)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = (arr[i] - m) / s
    return out


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
class ARLResult:
    rq_name: str
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"rq_name": self.rq_name, "status": self.status,
                "metrics": _clean(self.metrics)}


PRIMARY_VARIABLES = ["memory_density", "energy_storage", "adaptive_time"]
HORIZONS = [1, 5, 20, 50, 100, 500]
TARGET_ASSETS = ["EURJPY", "USDJPY", "GBPJPY", "XAUUSD"]
TIME_WINDOWS = [
    ("2018-01-01", "2020-01-01", "2018-2020"),
    ("2020-01-01", "2022-01-01", "2020-2022"),
    ("2022-01-01", "2024-01-01", "2022-2024"),
    ("2024-01-01", "2027-01-01", "2024-2026"),
]


class ARLValidator:
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
        return {"price": price, "returns": returns, "volume": volume, "high": high, "low": low,
                "raw": df}

    def load_data_window(self, asset: str, start: str, end: str) -> dict[str, NDArray]:
        path = self.data_dir / f"{asset}.parquet"
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

        memory_density = np.asarray(result_ml.get("memory_density", np.zeros(n)), dtype=np.float64)
        energy_storage = np.asarray(result_ed.get("energy_storage", np.zeros(n)), dtype=np.float64)
        adaptive_time = np.asarray(result_tt.get("adaptive_time_coordinate", np.zeros(n)), dtype=np.float64)

        analysis: dict[str, Any] = {
            "price": price, "returns": returns,
            "volume": data.get("volume", np.ones(n, dtype=np.float64)),
            "adaptive_time": adaptive_time,
            "energy_storage": energy_storage,
            "memory_density": memory_density,
            "states": states,
        }
        analysis["state_mutation_rate"] = _numba_rolling_state_mutation(states, 20)
        analysis["regime_change_probability"] = _numba_rolling_regime_prob(states, 20)
        return analysis

    def compute_detrended_signals(self, data: dict, trend_period: int = 200) -> dict[str, Any]:
        price = data["price"]
        n = len(price)
        trend = np.zeros(n, dtype=np.float64)
        for i in range(min(trend_period, n), n):
            trend[i] = np.mean(price[i - trend_period:i])
        detrended_price = price - trend + trend[min(trend_period, n - 1)]
        detrended_returns = np.diff(np.log(np.maximum(detrended_price, 1e-12)),
                                    prepend=np.log(detrended_price[0]))

        detrended_data = {
            "price": detrended_price,
            "returns": detrended_returns,
            "volume": data.get("volume", np.ones(n, dtype=np.float64)),
            "high": data.get("high", detrended_price.copy()),
            "low": data.get("low", detrended_price.copy()),
        }
        return self.compute_signals(detrended_data)

    def alpha_signal(self, signals: dict, normalize: bool = True) -> NDArray[np.float64]:
        es = np.asarray(signals["energy_storage"], dtype=np.float64)
        md = np.asarray(signals["memory_density"], dtype=np.float64)
        at = np.asarray(signals["adaptive_time"], dtype=np.float64)
        n = min(len(es), len(md), len(at))
        if normalize:
            es, md, at = es[:n], md[:n], at[:n]
            es_z = _zscore(es)
            md_z = _zscore(md)
            at_z = _zscore(at)
            return es_z * md_z * at_z
        return es[:n] * md[:n] * at[:n]

    def alpha_by_threshold(self, signals: dict, pct: float = 10) -> NDArray[np.bool_]:
        es = np.asarray(signals["energy_storage"], dtype=np.float64)
        md = np.asarray(signals["memory_density"], dtype=np.float64)
        at = np.asarray(signals["adaptive_time"], dtype=np.float64)
        n = min(len(es), len(md), len(at))
        es_t = np.nanpercentile(es, 100 - pct)
        md_t = np.nanpercentile(md, 100 - pct)
        at_t = np.nanpercentile(at, 100 - pct)
        return (es[:n] > es_t) & (md[:n] > md_t) & (at[:n] > at_t)

    def eval_alpha(self, alpha_sig: NDArray[np.float64],
                   future_ret: NDArray[np.float64], horizon_idx: int = 2) -> dict:
        fwd = future_ret[:, horizon_idx]
        n = min(len(alpha_sig), len(fwd))
        as_, fw = alpha_sig[:n], fwd[:n]

        _, masks = self._decile_bins(as_)
        sig_mask = masks[-1]

        if np.sum(sig_mask) < 5:
            return {"mean": 0.0, "pp": 0.5, "std": 0.0, "sharpe": 0.0, "n": 0}

        vals = fw[sig_mask]
        m = float(np.nanmean(vals))
        s = float(np.nanstd(vals))
        return {
            "mean": m,
            "pp": float(np.mean(vals > 0)),
            "std": s,
            "sharpe": m / max(s, 1e-12),
            "skew": float(_numba_skew(vals.astype(np.float64))) if len(vals) > 3 else 0.0,
            "kurtosis": float(_numba_kurtosis(vals.astype(np.float64))) if len(vals) > 4 else 0.0,
            "n": int(np.sum(sig_mask)),
        }

    def _decile_bins(self, signal: NDArray[np.float64]) -> tuple[NDArray[np.float64], list[NDArray[np.bool_]]]:
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
