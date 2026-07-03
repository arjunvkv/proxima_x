"""RQ2: Does alpha survive after controlling for volatility regime?"""

from __future__ import annotations

import numpy as np
import numba
from numpy.typing import NDArray

from research.alpha_reality.arl_validator import ARLValidator, ARLResult, HORIZONS, _future_returns


@numba.jit(nopython=True, cache=True)
def _rolling_vol(returns: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    n = len(returns)
    result = np.zeros(n, dtype=np.float64)
    for i in range(window, n):
        s = 0.0
        for j in range(i - window, i):
            s += returns[j]
        m = s / window
        v = 0.0
        for j in range(i - window, i):
            d = returns[j] - m
            v += d * d
        result[i] = np.sqrt(v / window)
    return result


class VolatilityIndependence:
    def __init__(self, validator: ARLValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset

    def run(self) -> ARLResult:
        data = self.validator.load_asset_data(self.asset)
        price = data["price"]
        returns = data["returns"]

        signals = self.validator.compute_signals(data)
        alpha = self.validator.alpha_signal(signals)
        fr_all = _future_returns(price, np.array(HORIZONS, dtype=np.int32))

        vol = _rolling_vol(returns, 50)
        n = min(len(vol), len(alpha), fr_all.shape[0])
        vol, alpha, fr = vol[:n], alpha[:n], fr_all[:n]

        v_thirds = [float(np.nanpercentile(vol, p)) for p in [33.3, 66.6]]
        vol_labels = ["Low", "Med", "High"]
        vol_masks = [
            (vol > 0) & (vol <= v_thirds[0]),
            (vol > v_thirds[0]) & (vol <= v_thirds[1]),
            vol > v_thirds[1],
        ]

        def _eval_subset(vmask: np.ndarray, hi: int) -> dict:
            aidx = np.where(vmask)[0]
            if len(aidx) < 20:
                return {"mean": 0.0, "pp": 0.5, "std": 0.0, "n": 0}
            sub_alpha = alpha[aidx]
            sub_fwd = fr[aidx, hi]
            _, masks = self.validator._decile_bins(sub_alpha)
            top = masks[-1]
            if np.sum(top) < 5:
                return {"mean": 0.0, "pp": 0.5, "std": 0.0, "n": 0}
            vals = sub_fwd[top]
            m = float(np.nanmean(vals))
            s = float(np.nanstd(vals))
            return {
                "mean": m,
                "pp": float(np.mean(vals > 0)),
                "std": s,
                "sharpe": m / max(s, 1e-12),
                "n": int(np.sum(top)),
            }

        results = {}
        for vi, vmask in enumerate(vol_masks):
            label = vol_labels[vi]
            results[label] = {}
            for hi, h in enumerate(HORIZONS):
                results[label][f"H{h}"] = _eval_subset(vmask, hi)

        print("  Volatility Independence @ H20:")
        for label in vol_labels:
            r = results.get(label, {}).get("H20", {})
            print(f"    {label:5s} vol: mean={r.get('mean', 0):.6f}, pp={r.get('pp', 0):.3f}, n={r.get('n', 0)}")

        low_pp = results.get("Low", {}).get("H20", {}).get("pp", 0.5)
        med_pp = results.get("Med", {}).get("H20", {}).get("pp", 0.5)
        high_pp = results.get("High", {}).get("H20", {}).get("pp", 0.5)
        survives = (low_pp > 0.53) and (med_pp > 0.53) and (high_pp > 0.5)

        print(f"    Alpha survives all vol regimes: {'YES' if survives else 'NO'}")

        status = "PASSED" if survives else "FAILED"
        return ARLResult("volatility_independence", status, metrics=results)
