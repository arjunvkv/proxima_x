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


@numba.jit(nopython=True, cache=True)
def _numba_rolling_percentile(arr: NDArray[np.float64], window: int, pct: float) -> NDArray[np.float64]:
    n = len(arr)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return result
    # Seed with first window
    first = np.sort(arr[:window])
    idx = int(window * pct / 100.0)
    idx = max(0, min(idx, window - 1))
    result[window - 1] = first[idx]
    for i in range(window, n):
        # Simple O(n log n) per window - ok for moderate n
        chunk = np.sort(arr[i - window:i])
        result[i] = chunk[idx]
    return result


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
class AAEResult:
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


class AAEValidator:
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

    def alpha_signal(self, signals: dict) -> NDArray[np.float64]:
        es = np.asarray(signals["energy_storage"], dtype=np.float64)
        n = len(es)
        es_z = _zscore(es)
        return es_z

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
        return {"mean": m, "pp": float(np.mean(vals > 0)), "std": s,
                "sharpe": m / max(s, 1e-12),
                "skew": float(_numba_skew(vals.astype(np.float64))) if len(vals) > 3 else 0.0,
                "kurtosis": float(_numba_kurtosis(vals.astype(np.float64))) if len(vals) > 4 else 0.0,
                "n": int(np.sum(sig_mask))}

    def rolling_percentile(self, signal: NDArray[np.float64], window: int, pct: float) -> NDArray[np.float64]:
        return _numba_rolling_percentile(signal, window, pct)

    def compute_at_quantile(self, at_signal: NDArray[np.float64], n_buckets: int = 5) -> list[NDArray[np.bool_]]:
        pcts = np.linspace(0, 100, n_buckets + 1)
        boundaries = np.nanpercentile(at_signal, pcts)
        masks = []
        for i in range(n_buckets):
            if i == 0:
                masks.append(at_signal <= boundaries[i + 1])
            elif i == n_buckets - 1:
                masks.append(at_signal > boundaries[i])
            else:
                masks.append((at_signal > boundaries[i]) & (at_signal <= boundaries[i + 1]))
        return masks

    def portfolio_stats(self, returns_list: list[NDArray[np.float64]]) -> dict:
        n = min(len(r) for r in returns_list)
        aligned = np.array([r[:n] for r in returns_list])
        ew = np.mean(aligned, axis=0)
        vols = np.std(aligned, axis=1) + 1e-12
        vw = np.sum(aligned * (1.0 / vols)[:, np.newaxis], axis=0) / np.sum(1.0 / vols)
        sw = np.mean(aligned, axis=0)
        return {
            "equal_weight": {"mean": float(np.mean(ew)), "std": float(np.std(ew)),
                             "sharpe": float(np.mean(ew)) / max(float(np.std(ew)), 1e-12),
                             "pp": float(np.mean(ew > 0))},
            "vol_weight": {"mean": float(np.mean(vw)), "std": float(np.std(vw)),
                           "sharpe": float(np.mean(vw)) / max(float(np.std(vw)), 1e-12),
                           "pp": float(np.mean(vw > 0))},
            "signal_weight": {"mean": float(np.mean(sw)), "std": float(np.std(sw)),
                              "sharpe": float(np.mean(sw)) / max(float(np.std(sw)), 1e-12),
                              "pp": float(np.mean(sw > 0))},
        }
