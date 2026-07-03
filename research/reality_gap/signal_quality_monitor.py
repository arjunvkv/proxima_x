import json
import numpy as np
import polars as pl
from research.adaptive_alpha_engine.aae_validator import AAEValidator


class SignalQualityMonitor:

    @staticmethod
    def compute_rolling_pp(signals: np.ndarray, future_returns: np.ndarray, window: int = 100, threshold: float = 0.7) -> np.ndarray:
        n = min(len(signals), len(future_returns))
        result = np.full(n, np.nan)
        if n < window:
            return result
        for i in range(window, n):
            sig_slice = signals[i - window:i]
            ret_slice = future_returns[i - window:i]
            mask = sig_slice > threshold
            if np.sum(mask) > 0:
                result[i] = float(np.mean(ret_slice[mask] > 0.0))
            else:
                result[i] = 0.5
        return result

    @staticmethod
    def compute_rolling_sharpe(pnls: np.ndarray, window: int = 100, periods_per_year: float = 12.6) -> np.ndarray:
        n = len(pnls)
        result = np.full(n, np.nan)
        if n < window:
            return result
        for i in range(window, n):
            seg = pnls[i - window:i]
            m = float(np.mean(seg))
            s = float(np.std(seg))
            result[i] = m / max(s, 1e-12) * np.sqrt(periods_per_year)
        return result

    @staticmethod
    def compute_threshold_drift(signal: np.ndarray, window: int = 504, percentile: float = 90.0) -> np.ndarray:
        n = len(signal)
        result = np.full(n, np.nan)
        if n < window:
            return result
        for i in range(window, n):
            result[i] = float(np.nanpercentile(signal[i - window:i], percentile))
        return result

    @staticmethod
    def compute_signal_frequency(signal: np.ndarray, window: int = 100, threshold: float = 0.7) -> np.ndarray:
        n = len(signal)
        result = np.full(n, 0.0)
        if n < window:
            return result
        for i in range(window, n):
            result[i] = float(np.mean(signal[i - window:i] > threshold))
        return result

    @staticmethod
    def compute_signal_strength(signal: np.ndarray, window: int = 100, threshold: float = 0.7) -> np.ndarray:
        n = len(signal)
        result = np.full(n, 0.0)
        if n < window:
            return result
        for i in range(window, n):
            seg = signal[i - window:i]
            mask = seg > threshold
            if np.sum(mask) > 0:
                result[i] = float(np.mean(seg[mask]))
        return result

    @staticmethod
    def date_to_index_mapping(data: dict, periods: list[tuple[str, str, str]]) -> dict[str, tuple[int, int]]:
        raw = data.get("raw")
        if raw is not None and hasattr(raw, "filter"):
            import datetime
            mapping = {}
            for start, end, label in periods:
                dt_start = datetime.datetime.strptime(start, "%Y-%m-%d")
                dt_end = datetime.datetime.strptime(end, "%Y-%m-%d")
                sub = raw.filter(
                    (pl.col("timestamp") >= dt_start) & (pl.col("timestamp") < dt_end)
                )
                if len(sub) == 0:
                    mapping[label] = (0, 0)
                else:
                    all_ts = raw["timestamp"].to_numpy().astype("datetime64[ns]")
                    start_pos = int(np.searchsorted(all_ts, np.datetime64(dt_start, "ns")))
                    end_pos = int(np.searchsorted(all_ts, np.datetime64(dt_end, "ns")) - 1)
                    mapping[label] = (max(0, start_pos), min(len(raw) - 1, end_pos))
            return mapping
        n = len(data.get("price", np.array([])))
        if n == 0:
            return {label: (0, 0) for _, _, label in periods}
        total_bars = n
        mapping = {}
        for start, end, label in periods:
            import datetime
            dt_start = datetime.datetime.strptime(start, "%Y-%m-%d")
            dt_end = datetime.datetime.strptime(end, "%Y-%m-%d")
            ref_start = datetime.datetime(2018, 1, 1)
            ref_end = datetime.datetime(2027, 1, 1)
            ref_span = (ref_end - ref_start).total_seconds()
            start_frac = (dt_start - ref_start).total_seconds() / ref_span
            end_frac = (dt_end - ref_start).total_seconds() / ref_span
            s = max(0, int(start_frac * total_bars))
            e = min(total_bars - 1, int(end_frac * total_bars))
            mapping[label] = (s, e)
        return mapping

    @staticmethod
    def _numba_skew(arr: np.ndarray) -> float:
        n = len(arr)
        if n < 3:
            return 0.0
        m = float(np.mean(arr))
        std = float(np.std(arr))
        if std < 1e-12:
            return 0.0
        return float(np.mean(((arr - m) / std) ** 3))

    @staticmethod
    def _period_metrics(composite: np.ndarray, residual: np.ndarray, es: np.ndarray, at: np.ndarray, price: np.ndarray, start_idx: int, end_idx: int) -> dict:
        seg_c = composite[start_idx:end_idx + 1]
        seg_r = residual[start_idx:end_idx + 1]
        seg_e = es[start_idx:end_idx + 1]
        seg_a = at[start_idx:end_idx + 1]
        seg_p = price[start_idx:end_idx + 1]
        if len(seg_c) < 100:
            return {"signal_frequency": 0.0, "signal_strength": 0.0, "residual_energy_distribution": {"mean": 0.0, "std": 0.0, "skew": 0.0}, "energy_storage_distribution": {"mean": 0.0, "std": 0.0, "skew": 0.0}, "adaptive_time_distribution": {"mean": 0.0, "std": 0.0}, "threshold_90th": 0.0, "pp_h20": 0.5}
        signal_freq = float(np.mean(seg_c > 0.7))
        gt07 = seg_c[seg_c > 0.7]
        signal_strength = float(np.mean(gt07)) if len(gt07) > 0 else 0.0
        res_dist = {"mean": float(np.mean(seg_r)), "std": float(np.std(seg_r)), "skew": SignalQualityMonitor._numba_skew(seg_r)}
        es_dist = {"mean": float(np.mean(seg_e)), "std": float(np.std(seg_e)), "skew": SignalQualityMonitor._numba_skew(seg_e)}
        at_dist = {"mean": float(np.mean(seg_a)), "std": float(np.std(seg_a))}
        threshold_90th = float(np.nanpercentile(seg_c, 90))
        h20_returns = np.full(len(seg_c), np.nan)
        for j in range(len(seg_p) - 20):
            h20_returns[j] = float(np.log(seg_p[j + 20] / seg_p[j]))
        valid = (seg_c > 0.7) & ~np.isnan(h20_returns)
        pp_h20 = float(np.mean(h20_returns[valid] > 0.0)) if np.sum(valid) > 0 else 0.5
        return {"signal_frequency": signal_freq, "signal_strength": signal_strength, "residual_energy_distribution": res_dist, "energy_storage_distribution": es_dist, "adaptive_time_distribution": at_dist, "threshold_90th": threshold_90th, "pp_h20": pp_h20}
