from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Any

from research.adaptive_alpha_engine.aae_validator import AAEValidator, HORIZONS, TARGET_ASSETS, _zscore, _future_returns


@dataclass
class ERLResult:
    rq_name: str
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)


VOL_METRICS = ["realized_vol", "parkinson_vol", "garman_klass_vol", "rogers_satchell_vol",
               "atr", "vol_of_vol", "entropy", "entropy_change"]


class EnergyValidator:
    def __init__(self, asset: str = "EURJPY"):
        self.asset = asset
        self.aae = AAEValidator()
        self.data: dict | None = None
        self.signals: dict | None = None
        self.price: NDArray[np.float64] | None = None
        self.vol_metrics: dict[str, NDArray[np.float64]] = {}
        self.fut_ret: NDArray[np.float64] | None = None

    def load(self, asset: str | None = None) -> dict:
        a = asset or self.asset
        self.data = self.aae.load_asset_data(a)
        self.signals = self.aae.compute_signals(self.data)
        self.price = self.signals["price"]
        horizons_arr = np.array(HORIZONS, dtype=np.int32)
        self.fut_ret = _future_returns(self.price, horizons_arr)
        self._compute_vol_metrics()
        return self.signals

    def _compute_vol_metrics(self, window: int = 20):
        p = self.price
        h = self.data["high"]
        lo = self.data["low"]
        c = self.price
        r = self.data["returns"]
        n = len(p)

        rv = np.zeros(n, dtype=np.float64)
        for i in range(window, n):
            rv[i] = float(np.nanstd(r[i - window:i])) * np.sqrt(252)
        self.vol_metrics["realized_vol"] = rv

        pk = np.zeros(n, dtype=np.float64)
        for i in range(window, n):
            logs = np.log(h[i - window:i] / lo[i - window:i])
            pk[i] = float(np.sqrt(np.mean(logs ** 2) / (4.0 * np.log(2.0)))) * np.sqrt(252)
        self.vol_metrics["parkinson_vol"] = pk

        gk = np.zeros(n, dtype=np.float64)
        for i in range(window, n):
            hl = np.log(h[i - window:i] / lo[i - window:i])
            co = np.log(c[i - window:i] / p[i - window - 1:i - 1]) if i - window > 0 else np.zeros(window)
            term1 = 0.5 * np.mean(hl ** 2)
            term2 = (2.0 * np.log(2.0) - 1.0) * np.mean(co ** 2)
            gk[i] = float(np.sqrt(max(term1 - term2, 1e-12))) * np.sqrt(252)
        self.vol_metrics["garman_klass_vol"] = gk

        rs = np.zeros(n, dtype=np.float64)
        for i in range(window, n):
            hl = np.log(h[i - window:i] / c[i - window:i])
            lc = np.log(lo[i - window:i] / c[i - window:i])
            hc = np.log(h[i - window:i] / p[i - window - 1:i - 1]) if i - window > 0 else np.zeros(window)
            lc_2 = np.log(lo[i - window:i] / p[i - window - 1:i - 1]) if i - window > 0 else np.zeros(window)
            term = np.mean(hl * (hl - lc) + hc * (hc - lc_2))
            rs[i] = float(np.sqrt(max(term, 1e-12))) * np.sqrt(252)
        self.vol_metrics["rogers_satchell_vol"] = rs

        atr = np.zeros(n, dtype=np.float64)
        tr = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            tr[i] = max(h[i] - lo[i], abs(h[i] - p[i - 1]), abs(lo[i] - p[i - 1]))
        for i in range(window, n):
            atr[i] = float(np.mean(tr[i - window:i]))
        self.vol_metrics["atr"] = atr

        vv = np.zeros(n, dtype=np.float64)
        for i in range(window * 2, n):
            vv[i] = float(np.std(rv[i - window:i]))
        self.vol_metrics["vol_of_vol"] = vv

        ent = np.zeros(n, dtype=np.float64)
        bins = 20
        for i in range(window, n):
            hist, _ = np.histogram(r[i - window:i], bins=bins, density=True)
            hist = hist[hist > 0]
            ent[i] = float(-np.sum(hist * np.log(hist + 1e-12)))
        self.vol_metrics["entropy"] = ent

        ec = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            ec[i] = ent[i] - ent[i - 1]
        self.vol_metrics["entropy_change"] = ec

    def es_signal(self) -> NDArray[np.float64]:
        return np.asarray(self.signals["energy_storage"], dtype=np.float64)

    def eval_alpha(self, signal: NDArray[np.float64], horizon_idx: int = 2) -> dict:
        return self.aae.eval_alpha(signal, self.fut_ret, horizon_idx)

    def decile_alpha(self, signal: NDArray[np.float64], horizon_idx: int = 2) -> dict:
        return self.aae.eval_alpha(signal, self.fut_ret, horizon_idx)

    def _clean(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): self._clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._clean(v) for v in obj]
        if isinstance(obj, tuple):
            return list(self._clean(v) for v in obj)
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist() if hasattr(obj, 'tolist') else float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        return obj
