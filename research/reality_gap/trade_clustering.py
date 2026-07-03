import json
import numpy as np
from proxima_v1.core.signal_engine import SignalEngine


class TradeClustering:
    CLUSTER_THRESHOLD = 5

    def __init__(self, assets: list[str] | None = None):
        self.assets = assets or ["EURJPY", "USDJPY", "GBPJPY"]

    def run(self) -> dict:
        composites = {}
        prices = {}
        for asset in self.assets:
            eng = SignalEngine(asset)
            eng.precompute_full()
            full_res = np.nan_to_num(eng._full_residual, nan=0.0)
            full_es = np.nan_to_num(eng._full_es, nan=0.0)
            full_at = np.nan_to_num(eng._full_at, nan=0.0)
            n = len(full_res)
            rolling_res = np.full(n, 0.5)
            rolling_es = np.full(n, 0.5)
            rolling_at = np.full(n, 0.5)
            for i in range(503, n):
                r_slice = full_res[max(0, i - 503):i + 1]
                e_slice = full_es[max(0, i - 503):i + 1]
                a_slice = full_at[max(0, i - 503):i + 1]
                rolling_res[i] = float(np.sum(r_slice <= full_res[i])) / len(r_slice)
                rolling_es[i] = float(np.sum(e_slice <= full_es[i])) / len(e_slice)
                rolling_at[i] = float(np.sum(a_slice <= full_at[i])) / len(a_slice)
            composites[asset] = np.clip(0.60 * rolling_res + 0.30 * rolling_es + 0.10 * rolling_at, 0.0, 1.0)
            price_arr = eng._data.get("price", np.zeros(n))
            prices[asset] = price_arr[:n]

        n = min(len(composites[a]) for a in self.assets)
        signal_bars = []
        for i in range(504, n):
            for asset in self.assets:
                if float(composites[asset][i]) > 0.7:
                    signal_bars.append(i)
                    break

        signal_indices = np.array(signal_bars, dtype=np.int64)
        if len(signal_indices) < 2:
            return {"total_signals": int(len(signal_indices)), "isolated_signals": 0, "clustered_signals": 0, "cluster_ratio": 0.0, "mean_inter_signal_gap": 0.0, "median_inter_signal_gap": 0.0, "mean_cluster_drawdown": 0.0, "mean_isolated_drawdown": 0.0, "cluster_drawdown_worse": False}

        gaps = np.diff(signal_indices).astype(np.float64)

        is_clustered = np.zeros(len(signal_indices), dtype=bool)
        for i in range(len(signal_indices)):
            if i > 0 and signal_indices[i] - signal_indices[i - 1] <= self.CLUSTER_THRESHOLD:
                is_clustered[i] = True
                is_clustered[i - 1] = True

        def _drawdown_at(bar: int, price_arr: np.ndarray, lookahead: int = 50) -> float:
            end = min(bar + lookahead, len(price_arr))
            if end <= bar:
                return 0.0
            seg = price_arr[bar:end]
            entry = float(seg[0])
            if entry <= 1e-12:
                return 0.0
            eq = seg / entry
            peak = float(eq[0])
            max_dd = 0.0
            for v in eq:
                v = float(v)
                if v > peak:
                    peak = v
                dd = (peak - v) / peak
                if dd > max_dd:
                    max_dd = dd
            return max_dd

        price_ref = prices[self.assets[0]]
        cluster_drawdowns = []
        isolated_drawdowns = []
        i = 0
        while i < len(signal_indices):
            if is_clustered[i]:
                cluster_start = i
                while i < len(signal_indices) and is_clustered[i]:
                    i += 1
                dd = _drawdown_at(int(signal_indices[cluster_start]), price_ref)
                cluster_drawdowns.append(dd)
            else:
                dd = _drawdown_at(int(signal_indices[i]), price_ref)
                isolated_drawdowns.append(dd)
                i += 1

        mean_cluster_dd = float(np.mean(cluster_drawdowns)) if cluster_drawdowns else 0.0
        mean_isolated_dd = float(np.mean(isolated_drawdowns)) if isolated_drawdowns else 0.0

        return {
            "total_signals": int(len(signal_indices)),
            "isolated_signals": int(np.sum(~is_clustered)),
            "clustered_signals": int(np.sum(is_clustered)),
            "cluster_ratio": float(float(np.sum(is_clustered)) / max(len(signal_indices), 1)),
            "mean_inter_signal_gap": float(np.mean(gaps)) if len(gaps) > 0 else 0.0,
            "median_inter_signal_gap": float(np.median(gaps)) if len(gaps) > 0 else 0.0,
            "mean_cluster_drawdown": mean_cluster_dd,
            "mean_isolated_drawdown": mean_isolated_dd,
            "cluster_drawdown_worse": bool(mean_cluster_dd > mean_isolated_dd),
        }

    def save(self, path: str):
        results = self.run()
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
