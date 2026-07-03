"""DSR shared infrastructure: state vector builder, directional metrics, walk-forward splits."""
import sys, json, warnings
from pathlib import Path
import numpy as np
import polars as pl

warnings.filterwarnings("ignore")

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

from research.directional_physics.dpl_core import DPLData, SYMBOLS, HORIZON_LABELS
from sklearn.metrics import mutual_info_score

HORIZONS = [1, 5, 20, 50, 100, 500]
HORIZON_KEYS = ["H1", "H5", "H20", "H50", "H100", "H500"]
STATE_HORIZONS = [5, 20, 50]
STATE_HORIZON_KEYS = ["H5", "H20", "H50"]

class DSRState:
    """Complete directional state vector for a symbol at a given time."""

    def __init__(self):
        self.regime_state = np.nan
        self.regime_transition_from = np.nan
        self.regime_transition_to = np.nan
        self.residual_sign = np.nan
        self.residual_pressure = np.nan
        self.residual_acceleration = np.nan
        self.memory_imbalance = np.nan
        self.memory_saturation = np.nan
        self.memory_cluster = np.nan
        self.macro_regime = np.nan
        self.macro_es_rank = np.nan
        self.propagation_state = np.nan
        self.propagation_bias = np.nan
        self._as_tuple = None

    def to_tuple(self):
        if self._as_tuple is None:
            self._as_tuple = (
                self._clean(self.regime_state),
                self._clean(self.regime_transition_from),
                self._clean(self.regime_transition_to),
                self._clean(self.residual_sign),
                self._clean(self.residual_pressure),
                self._clean(self.residual_acceleration),
                self._clean(self.memory_imbalance),
                self._clean(self.memory_saturation),
                self._clean(self.memory_cluster),
                self._clean(self.macro_regime),
                self._clean(self.macro_es_rank),
                self._clean(self.propagation_state),
                self._clean(self.propagation_bias),
            )
        return self._as_tuple

    @staticmethod
    def _clean(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return -999
        if isinstance(v, (np.floating,)) and np.isnan(v):
            return -999
        return int(round(float(v))) if isinstance(v, (float, np.floating)) else int(v)

    @property
    def valid(self):
        return not (np.isnan(self.regime_state) if isinstance(self.regime_state, float) else False)

    def __repr__(self):
        return (f"DSRState(regime={self.regime_state}, residual_sign={self.residual_sign}, "
                f"residual_pressure={self.residual_pressure:.2f}, mb_imb={self.memory_imbalance:.2f})")

class DSRCore:
    """Builds complete state vectors for all symbols and computes directional metrics."""

    def __init__(self, cache_dir: str = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path(__file__).parent / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._data = {}

    def load_symbol(self, symbol: str, force_reload: bool = False):
        cache_path = self.cache_dir / f"{symbol}_state.npz"
        if cache_path.exists() and not force_reload:
            arr = np.load(cache_path)
            self._data[symbol] = {k: arr[k] for k in arr.files}
            return self._data[symbol]

        d = DPLData(symbol)
        n = len(d.es)

        residual = d.residuals.get("linear", np.full(n, np.nan))
        residual_mae = d.residuals.get("mae_7", np.full(n, np.nan))
        residual_xgb = d.residuals.get("xgb", np.full(n, np.nan))
        residual_rf = d.residuals.get("rf", np.full(n, np.nan))

        es = d.es
        memory_density = d.memory_density
        fut_ret = d.fut_ret

        vol = d.vol_metrics.get("realized_vol", np.full(n, np.nan)) if d.vol_metrics else np.full(n, np.nan)

        combined_density = self._combine_density(memory_density, vol)
        regimes = self._tertile_quantize(combined_density)

        residual_sign = np.sign(residual)
        residual_sign[(residual == 0) | np.isnan(residual)] = 0

        residual_pressure = self._cumulative_pressure(residual)

        residual_acc = np.full(n, np.nan)
        residual_acc[5:] = residual_pressure[5:] - residual_pressure[:-5]

        memory_imbalance = self._compute_imbalance(memory_density)

        saturation_pct = 90
        saturation_thr = np.nanpercentile(memory_density, saturation_pct)
        memory_saturation = (memory_density > saturation_thr).astype(np.int64)
        memory_saturation[np.isnan(memory_density)] = 0

        memory_clusters = self._cluster_memory(memory_density)

        macro_regime = np.full(n, np.nan, dtype=np.int64)
        macro_es_rank = np.full(n, np.nan)
        window_100 = 100
        for i in range(window_100, n):
            window = es[i - window_100:i]
            macro_es_rank[i] = np.sum(window <= es[i]) / window_100
            density_slice = combined_density[i - window_100:i]
            macro_regime[i] = int(np.nanmedian(self._tertile_quantize(density_slice)[-10:]))

        reg_transition_from = np.full(n, -1, dtype=np.int64)
        reg_transition_to = np.full(n, -1, dtype=np.int64)
        for i in range(1, n):
            if regimes[i] != regimes[i-1]:
                reg_transition_from[i] = int(regimes[i-1])
                reg_transition_to[i] = int(regimes[i])

        result = {
            "es": es.astype(np.float32),
            "fut_ret": fut_ret.astype(np.float32),
            "regime": regimes.astype(np.int64),
            "reg_transition_from": reg_transition_from.astype(np.int64),
            "reg_transition_to": reg_transition_to.astype(np.int64),
            "residual": residual.astype(np.float32),
            "residual_sign": residual_sign.astype(np.int64),
            "residual_pressure": residual_pressure.astype(np.float32),
            "residual_acceleration": residual_acc.astype(np.float32),
            "memory_density": memory_density.astype(np.float32),
            "memory_imbalance": memory_imbalance.astype(np.float32),
            "memory_saturation": memory_saturation.astype(np.int64),
            "memory_cluster": memory_clusters.astype(np.int64),
            "macro_regime": macro_regime.astype(np.int64),
            "macro_es_rank": macro_es_rank.astype(np.float32),
            "combined_density": combined_density.astype(np.float32),
            "vol": vol.astype(np.float32),
        }
        np.savez_compressed(cache_path, **result)
        self._data[symbol] = result
        return self._data[symbol]

    @staticmethod
    def _combine_density(memory_density, vol):
        m = (memory_density - np.nanmean(memory_density)) / max(np.nanstd(memory_density), 1e-12)
        v = (vol - np.nanmean(vol)) / max(np.nanstd(vol), 1e-12)
        combined = np.full_like(m, np.nan)
        valid = ~np.isnan(m) & ~np.isnan(v)
        combined[valid] = m[valid] + v[valid]
        combined[~valid & ~np.isnan(m)] = m[~valid & ~np.isnan(m)]
        return combined

    @staticmethod
    def _tertile_quantize(x):
        result = np.full(len(x), -1, dtype=np.int64)
        valid = ~np.isnan(x)
        if np.sum(valid) < 10:
            return result
        t1 = np.nanpercentile(x[valid], 33.33)
        t2 = np.nanpercentile(x[valid], 66.67)
        result[valid & (x <= t1)] = 0
        result[valid & (x > t1) & (x <= t2)] = 1
        result[valid & (x > t2)] = 2
        return result

    @staticmethod
    def _cumulative_pressure(residual):
        pressure = np.full_like(residual, np.nan)
        pressure[0] = 0
        for i in range(1, len(residual)):
            if np.isnan(residual[i]):
                pressure[i] = pressure[i-1]
            else:
                pressure[i] = pressure[i-1] + residual[i]
        return pressure

    @staticmethod
    def _compute_imbalance(memory_density):
        n = len(memory_density)
        imb = np.full(n, np.nan)
        window = 50
        for i in range(window, n):
            chunk = memory_density[i - window:i]
            above = np.sum(chunk > memory_density[i])
            below = np.sum(chunk < memory_density[i])
            total = above + below
            imb[i] = (above - below) / total if total > 0 else 0.0
        return imb

    @staticmethod
    def _cluster_memory(memory_density, n_clusters=3):
        n = len(memory_density)
        labels = np.full(n, -1, dtype=np.int64)
        valid = ~np.isnan(memory_density)
        if np.sum(valid) < n_clusters * 10:
            return labels
        x = memory_density[valid].reshape(-1, 1)
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
        clusters = km.fit_predict(x)
        labels[valid] = clusters.astype(np.int64)
        return labels

    def get_state(self, symbol: str, idx: int) -> DSRState:
        d = self._data.get(symbol)
        if d is None:
            d = self.load_symbol(symbol)
        s = DSRState()
        s.regime_state = int(d["regime"][idx]) if d["regime"][idx] >= 0 else np.nan
        s.regime_transition_from = int(d["reg_transition_from"][idx]) if d["reg_transition_from"][idx] >= 0 else np.nan
        s.regime_transition_to = int(d["reg_transition_to"][idx]) if d["reg_transition_to"][idx] >= 0 else np.nan
        s.residual_sign = int(d["residual_sign"][idx]) if not np.isnan(d["residual_sign"][idx]) else np.nan
        s.residual_pressure = d["residual_pressure"][idx]
        s.residual_acceleration = d["residual_acceleration"][idx]
        s.memory_imbalance = d["memory_imbalance"][idx]
        s.memory_saturation = int(d["memory_saturation"][idx])
        s.memory_cluster = int(d["memory_cluster"][idx]) if d["memory_cluster"][idx] >= 0 else np.nan
        s.macro_regime = int(d["macro_regime"][idx]) if not np.isnan(d["macro_regime"][idx]) else np.nan
        s.macro_es_rank = d["macro_es_rank"][idx]
        return s

    def state_array(self, symbol: str) -> np.ndarray:
        d = self._data.get(symbol)
        if d is None:
            d = self.load_symbol(symbol)
        n = len(d["es"])
        states = np.full(n, -1, dtype=np.int64)
        unique_map = {}
        counter = 0
        for i in range(n):
            s = self.get_state(symbol, i)
            if not s.valid:
                states[i] = -1
                continue
            t = s.to_tuple()
            if t not in unique_map:
                unique_map[t] = counter
                counter += 1
            states[i] = unique_map[t]
        return states, unique_map

    def directional_metrics(self, symbol: str, state_ids: np.ndarray, fut_ret: np.ndarray,
                            horizon_idx: int = 2, min_samples: int = 5):
        up = (fut_ret > 0).astype(float)
        down = (fut_ret <= 0).astype(float)
        unique_sids = np.unique(state_ids[state_ids >= 0])
        results = {}
        for sid in unique_sids:
            mask = state_ids == sid
            cnt = int(np.sum(mask))
            if cnt < min_samples:
                continue
            n_up = int(np.sum(up[mask]))
            n_down = cnt - n_up
            p_up = n_up / cnt
            p_down = n_down / cnt
            entropy = 0.0
            if p_up > 0 and p_down > 0:
                entropy = -(p_up * np.log2(p_up) + p_down * np.log2(p_down))
            base_p_up = np.mean(up)
            ig = 0.0
            if p_up > 0 and base_p_up > 0:
                ig = p_up * np.log2(p_up / base_p_up)
            if p_down > 0 and (1 - base_p_up) > 0:
                ig += p_down * np.log2(p_down / (1 - base_p_up))
            p_up_sem = np.sqrt(p_up * (1 - p_up) / cnt) if cnt > 0 else 0
            results[int(sid)] = {
                "count": cnt,
                "n_up": n_up,
                "n_down": n_down,
                "p_up": round(p_up, 4),
                "p_down": round(p_down, 4),
                "entropy": round(entropy, 4),
                "information_gain": round(ig, 4),
                "p_up_sem": round(p_up_sem, 4),
                "z_score": round((p_up - base_p_up) / max(p_up_sem, 1e-12), 4) if p_up_sem > 0 else 0,
            }
        return results

    def run_all_symbols(self, force_reload: bool = False):
        for sym in SYMBOLS:
            self.load_symbol(sym, force_reload=force_reload)

class WalkForwardValidator:
    """Walk-forward validation splits for DSR."""

    SPLITS = [
        ("2018-2022", "2023"),
        ("2019-2023", "2024"),
        ("2020-2024", "2025"),
    ]

    def __init__(self, dsr: DSRCore, tick_data_path: str = None):
        self.dsr = dsr
        self._year_ranges = {}

    def prepare(self, symbol: str):
        d = self.dsr._data.get(symbol)
        if d is None:
            d = self.dsr.load_symbol(symbol)
        data_len = len(d["es"])
        years = np.full(data_len, 2018, dtype=np.int32)
        from research.energy_reality.energy_validator import EnergyValidator
        ev = EnergyValidator(symbol)
        ev.load(symbol)
        data = ev.data
        if data is not None and isinstance(data, dict):
            raw = data.get("raw")
            if raw is not None and hasattr(raw, "columns"):
                col = "timestamp" if "timestamp" in raw.columns else "time"
                times = raw[col].to_list() if hasattr(raw[col], "to_list") else list(raw[col])
                years_arr = np.array([t.year for t in times], dtype=np.int32)
                min_len = min(len(years_arr), data_len)
                years[:min_len] = years_arr[:min_len]
        self._year_ranges[symbol] = years
        return years

    def split(self, symbol: str, train_name: str, test_name: str):
        train_year_end = int(test_name) - 1
        test_year = int(test_name)
        years = self._year_ranges.get(symbol)
        if years is None:
            years = self.prepare(symbol)
        train_mask = (years >= int(train_name[:4])) & (years <= train_year_end)
        test_mask = years == test_year
        return train_mask, test_mask


def save_report(report: dict, name: str):
    path = Path(__file__).parent / "reports" / f"{name}.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved {path}")
    return path


if __name__ == "__main__":
    dsr = DSRCore()
    for sym in SYMBOLS:
        print(f"Loading {sym}...")
        dsr.load_symbol(sym)
    print("DSR core ready.")
