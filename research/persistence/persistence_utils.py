import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, entropy
from sklearn.feature_selection import mutual_info_regression
from proxima_v1.core.signal_engine import SignalEngine


def extract_persistence_events(composite: np.ndarray, threshold: float = 0.7, min_gap: int = 0):
    events = []
    i = 0
    n = len(composite)
    while i < n:
        if composite[i] > threshold:
            start = i
            while i < n and composite[i] > threshold:
                i += 1
            end = i - 1
            duration = end - start + 1
            events.append({
                "start_idx": int(start), "end_idx": int(end),
                "duration": int(duration),
                "peak_score": float(np.max(composite[start:end+1])),
                "mean_score": float(np.mean(composite[start:end+1])),
            })
        else:
            i += 1
    return events


class PersistenceDataLoader:
    """Loads signal engine data and extracts persistence-layer measurements."""

    LAYER_KEYS = [
        "residual_energy", "energy_storage", "adaptive_time",
        "memory_density", "state_mutation_rate", "regime_change_probability",
        "memory_conflict", "memory_gradient",
        "energy_creation", "energy_release", "energy_dissipation",
        "compression", "information_pressure",
    ]

    def __init__(self, asset: str):
        self.asset = asset
        self.engine = SignalEngine(asset)
        self.engine.precompute_full()
        self.n = min(len(self.engine._full_residual), len(self.engine._full_es), len(self.engine._full_at))
        self._build_layer_matrix()

    def _build_layer_matrix(self):
        n = self.n
        sig = self.engine._signals
        self.layers = {}

        self.layers["residual_energy"] = self.engine._full_residual[:n]
        self.layers["energy_storage"] = self.engine._full_es[:n]
        self.layers["adaptive_time"] = self.engine._full_at[:n]

        for k in self.LAYER_KEYS:
            if k in self.layers:
                continue
            arr = sig.get(k, np.zeros(n))
            if len(arr) > n:
                arr = arr[:n]
            elif len(arr) < n:
                arr = np.pad(arr, (0, n - len(arr)), constant_values=0.0)
            self.layers[k] = np.nan_to_num(arr, nan=0.0)

        res = self.layers["residual_energy"]
        es = self.layers["energy_storage"]
        at = self.layers["adaptive_time"]
        res_r = np.full(n, 0.5)
        es_r = np.full(n, 0.5)
        at_r = np.full(n, 0.5)
        w = 504
        for i in range(w, n):
            res_r[i] = float(np.sum(res[max(0, i - w):i + 1] <= res[i])) / float(min(i + 1, w + 1))
            es_r[i] = float(np.sum(es[max(0, i - w):i + 1] <= es[i])) / float(min(i + 1, w + 1))
            at_r[i] = float(np.sum(at[max(0, i - w):i + 1] <= at[i])) / float(min(i + 1, w + 1))
        self.composite = np.clip(0.60 * res_r + 0.30 * es_r + 0.10 * at_r, 0.0, 1.0)

        self.events = extract_persistence_events(self.composite, threshold=0.7)

    def get_events_df(self) -> pd.DataFrame:
        rows = []
        for ev in self.events:
            s, e = ev["start_idx"], ev["end_idx"]
            row = {
                "start_idx": s, "end_idx": e,
                "duration": ev["duration"],
                "peak_score": ev["peak_score"],
                "mean_score": ev["mean_score"],
            }
            for lk in self.LAYER_KEYS:
                arr = self.layers[lk][s:e + 1]
                row[f"{lk}_entry"] = float(self.layers[lk][s]) if s < self.n else 0.0
                row[f"{lk}_exit"] = float(self.layers[lk][e]) if e < self.n else 0.0
                row[f"{lk}_mean"] = float(np.mean(arr))
                row[f"{lk}_delta"] = float(self.layers[lk][e] - self.layers[lk][s]) if e < self.n and s < self.n else 0.0
                row[f"{lk}_peak"] = float(np.max(arr))
            rows.append(row)
        return pd.DataFrame(rows)

    def get_signal_durations(self) -> np.ndarray:
        return np.array([e["duration"] for e in self.events], dtype=float)

    def get_rolling_duration(self, window: int = 252) -> np.ndarray:
        n = self.n
        out = np.zeros(n)
        if not self.events:
            return out
        ev_arr = np.zeros(n)
        for ev in self.events:
            ev_arr[ev["start_idx"]:ev["end_idx"] + 1] = ev["duration"]
        for i in range(window, n):
            out[i] = float(np.mean(ev_arr[max(0, i - window):i + 1]))
        return out

    def get_persistence_half_life(self) -> float:
        durations = self.get_signal_durations()
        if len(durations) < 5:
            return 0.0
        sorted_d = np.sort(durations)
        total = np.sum(sorted_d)
        cumulative = 0.0
        for d in sorted_d:
            cumulative += d
            if cumulative >= total / 2:
                return float(d)
        return 0.0


class PersistenceMeasure:
    """Compute persistence-related metrics from layer data."""

    @staticmethod
    def pearson_spearman_mi(x: np.ndarray, y: np.ndarray) -> dict:
        mask = ~(np.isnan(x) | np.isnan(y))
        xc, yc = x[mask], y[mask]
        if len(xc) < 5:
            return {"pearson": 0.0, "spearman": 0.0, "mutual_info": 0.0}
        p, _ = pearsonr(xc, yc)
        s, _ = spearmanr(xc, yc)
        mi = mutual_info_regression(xc.reshape(-1, 1), yc, random_state=42)[0]
        return {"pearson": float(p), "spearman": float(s), "mutual_info": float(mi)}


def align_by_min(*arrays: np.ndarray) -> list[np.ndarray]:
    n = min(len(a) for a in arrays)
    return [a[:n] for a in arrays]
