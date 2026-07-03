"""RQ6: Is compression's role stable across time?"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from pathlib import Path

from research.compression_physics.compression_validator import CompressionValidator, CPIResult, _find_peak_lag


class TimeStability:
    def __init__(self, validator: CompressionValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset
        self._max_lag = 200
        self._windows = 4

    def run(self) -> CPIResult:
        path = Path("data/market") / f"{self.asset}.parquet"
        if not path.exists():
            return CPIResult("time_stability", "FAILED", metrics={"error": f"data not found: {path}"})
        df = pl.read_parquet(str(path))
        n = len(df)
        chunk = n // self._windows
        results = []

        for w in range(self._windows):
            start = w * chunk
            end = n if w == self._windows - 1 else (w + 1) * chunk
            window_df = df.slice(start, end - start)

            data = {
                "price": window_df["close"].to_numpy().astype(np.float64),
                "returns": (window_df["log_return"].to_numpy().astype(np.float64)
                            if "log_return" in window_df.columns
                            else np.diff(np.log(window_df["close"].to_numpy().astype(np.float64)),
                                        prepend=np.log(window_df["close"].to_numpy().astype(np.float64)[0]))),
                "volume": (window_df["volume"].to_numpy().astype(np.float64)
                           if "volume" in window_df.columns
                           else np.ones(len(window_df), dtype=np.float64)),
                "high": window_df["high"].to_numpy().astype(np.float64) if "high" in window_df.columns else window_df["close"].to_numpy().astype(np.float64),
                "low": window_df["low"].to_numpy().astype(np.float64) if "low" in window_df.columns else window_df["close"].to_numpy().astype(np.float64),
            }

            signals = self.validator.compute_signals(data)
            compression = np.asarray(signals["compression"], dtype=np.float64)
            es = np.asarray(signals["energy_storage"], dtype=np.float64)

            lag, corr = _find_peak_lag(compression, es, self._max_lag)
            flow = self.validator.information_flow("compression", "energy_storage", signals)

            results.append({
                "window": w,
                "start": start,
                "end": end,
                "peak_lag": lag,
                "peak_corr": corr,
                "information_flow": flow,
            })

        lags = [r["peak_lag"] for r in results]
        corrs = [r["peak_corr"] for r in results]

        lag_drift = float(np.max(lags) - np.min(lags)) if lags else 0
        corr_stability = float(np.std(corrs)) if corrs else 0

        metrics = {
            "results": results,
            "n_windows": self._windows,
            "lag_mean": float(np.mean(lags)) if lags else 0,
            "lag_min": int(np.min(lags)) if lags else 0,
            "lag_max": int(np.max(lags)) if lags else 0,
            "lag_drift": lag_drift,
            "lag_std": float(np.std(lags)) if lags else 0,
            "corr_mean": float(np.mean(corrs)) if corrs else 0,
            "corr_std": corr_stability,
            "flow_mean": float(np.mean([r["information_flow"] for r in results])),
        }

        print(f"  Time stability (compression->energy):")
        for r in results:
            print(f"    Window {r['window']}: rows {r['start']}-{r['end']}: lag={r['peak_lag']:+4d}, corr={r['peak_corr']:.4f}")

        print(f"  Lag drift: {lag_drift} (std={metrics['lag_std']:.1f}), corr std: {corr_stability:.4f}")

        if lag_drift <= 20 and corr_stability < 0.2:
            status = "PASSED"
            print("  Compression->energy relationship is STABLE across time")
        elif lag_drift <= 50:
            status = "INCONCLUSIVE"
        else:
            status = "FAILED"

        return CPIResult("time_stability", status, metrics=metrics)
