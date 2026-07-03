import json
import numpy as np
from proxima_v1.core.signal_engine import SignalEngine


class SignalDecay:
    DELAYS = [0, 1, 2, 5, 10, 20]

    def __init__(self, asset: str = "EURJPY"):
        self.asset = asset
        self.engine = SignalEngine(asset)

    def run(self) -> dict:
        self.engine.precompute_full()
        n = len(self.engine._full_es)
        price = np.array(self.engine._data["price"][:n])
        full_res = np.nan_to_num(self.engine._full_residual, nan=0.0)
        full_es = np.nan_to_num(self.engine._full_es, nan=0.0)
        full_at = np.nan_to_num(self.engine._full_at, nan=0.0)
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
        composite = np.clip(0.60 * rolling_res + 0.30 * rolling_es + 0.10 * rolling_at, 0.0, 1.0)
        delay_returns = {d: [] for d in self.DELAYS}
        for i in range(504, n, 20):
            if composite[i] <= 0.7:
                continue
            for d in self.DELAYS:
                entry_idx = i + d
                exit_idx = entry_idx + 20
                if exit_idx < len(price):
                    ret = float(np.log(price[exit_idx] / price[entry_idx]))
                    delay_returns[d].append(ret)
        results = {}
        for d in self.DELAYS:
            arr = np.array(delay_returns[d], dtype=np.float64)
            if len(arr) < 3:
                results[d] = {"pp": 0.5, "sharpe": 0.0, "mean_return": 0.0, "n_signals": len(arr)}
            else:
                mean_ret = float(np.mean(arr))
                std_ret = float(np.std(arr))
                sharpe = mean_ret / max(std_ret, 1e-12)
                pp = float(np.mean(arr > 0.0))
                results[d] = {"pp": pp, "sharpe": sharpe, "mean_return": mean_ret, "n_signals": len(arr)}
        half_life = self.DELAYS[-1]
        for d in self.DELAYS:
            if results[d]["pp"] < 0.55 or results[d]["sharpe"] < 0.3:
                half_life = d
                break
        results["half_life"] = half_life
        return results

    def save(self, path: str):
        results = self.run()
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
