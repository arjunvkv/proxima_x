"""RQ4: Is energy_storage a required mediator between compression and memory_density?"""

from __future__ import annotations

from typing import Any

import numpy as np
import numba
from numpy.typing import NDArray

from research.compression_physics.compression_validator import CompressionValidator, CPIResult


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
def _pearson(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    n = len(x)
    mx, my = 0.0, 0.0
    for i in range(n):
        mx += x[i]
        my += y[i]
    m = mx / n
    my /= n
    num, dx2, dy2 = 0.0, 0.0, 0.0
    for i in range(n):
        dx = x[i] - m
        dy = y[i] - my
        num += dx * dy
        dx2 += dx * dx
        dy2 += dy * dy
    return num / (np.sqrt(dx2 * dy2) + 1e-12)


@numba.jit(nopython=True, cache=True)
def _partial_corr(x: NDArray[np.float64], y: NDArray[np.float64], z: NDArray[np.float64]) -> float:
    n = min(len(x), len(y), len(z))
    rx, ry, rz = _zscore(x[:n]), _zscore(y[:n]), _zscore(z[:n])
    r_xy = _pearson(rx, ry)
    r_xz = _pearson(rx, rz)
    r_yz = _pearson(ry, rz)
    return (r_xy - r_xz * r_yz) / (np.sqrt(1 - r_xz * r_xz) * np.sqrt(1 - r_yz * r_yz) + 1e-12)


class CompressionMediation:
    def __init__(self, validator: CompressionValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset
        self._max_lag = 200

    def run(self) -> CPIResult:
        data = self.validator.load_asset_data(self.asset)
        signals = self.validator.compute_signals(data)

        compression = np.asarray(signals["compression"], dtype=np.float64)
        es = np.asarray(signals["energy_storage"], dtype=np.float64)
        md = np.asarray(signals["memory_density"], dtype=np.float64)
        n = min(len(compression), len(es), len(md))

        if n < self._max_lag * 2 + 1:
            return CPIResult("compression_mediation", "FAILED", metrics={"error": "insufficient data"})

        from research.compression_physics.compression_validator import _find_peak_lag
        lag_ce, r_ce = _find_peak_lag(compression[:n], es[:n], self._max_lag)
        lag_cm, r_cm = _find_peak_lag(compression[:n], md[:n], self._max_lag)
        lag_em, r_em = _find_peak_lag(es[:n], md[:n], self._max_lag)

        # Align by discovered lags, then standardize
        max_neg = max(0, -lag_ce, -lag_em, -lag_cm)
        seg_len = n - max_neg - 10
        if seg_len < 10:
            return CPIResult("compression_mediation", "INCONCLUSIVE",
                             metrics={"error": f"insufficient aligned: {seg_len}"})

        c_arr = compression[max_neg:max_neg + seg_len]
        e_arr = es[max_neg + max(0, lag_ce):max_neg + seg_len + max(0, lag_ce)] if lag_ce >= 0 else es[max_neg - lag_ce:max_neg + seg_len - lag_ce]
        m_arr = md[max_neg + max(0, lag_cm):max_neg + seg_len + max(0, lag_cm)] if lag_cm >= 0 else md[max_neg - lag_cm:max_neg + seg_len - lag_cm]

        vn = min(len(c_arr), len(e_arr), len(m_arr))
        cf, ef, mf = c_arr[:vn], e_arr[:vn], m_arr[:vn]

        # Standardized path coefficients (beta = correlation for single-variable regression)
        zc = _zscore(cf)
        ze = _zscore(ef)
        zm = _zscore(mf)

        path_a = _pearson(zc, ze)           # compression -> energy_storage
        path_b = _partial_corr(ze, zm, zc)   # energy -> memory | compression
        path_c_total = _pearson(zc, zm)      # compression -> memory (total)
        path_c_direct = _partial_corr(zc, zm, ze)  # compression -> memory | energy

        indirect_effect = path_a * path_b
        mediated_proportion = indirect_effect / max(abs(path_c_total), 1e-12)
        sobel_num = indirect_effect
        sobel_den = np.sqrt(max(path_a * path_a * (1 - path_b * path_b) / max(vn - 3, 1) +
                                path_b * path_b * (1 - path_a * path_a) / max(vn - 3, 1), 1e-12))
        sobel_z = sobel_num / sobel_den

        metrics = {
            "lag_compression_to_energy": lag_ce,
            "lag_compression_to_memory": lag_cm,
            "lag_energy_to_memory": lag_em,
            "r_compression_energy": r_ce,
            "r_compression_memory": r_cm,
            "r_energy_memory": r_em,
            "path_a_std": path_a,
            "path_b_std": path_b,
            "path_c_total_std": path_c_total,
            "path_c_direct_std": path_c_direct,
            "indirect_effect_ab": indirect_effect,
            "mediated_proportion": mediated_proportion,
            "sobel_z": sobel_z,
        }

        if mediated_proportion > 0.5 and sobel_z > 1.96:
            status = "PASSED"
            print(f"  Energy IS required mediator (prop={mediated_proportion:.3f}, sobel_z={sobel_z:.2f})")
        elif mediated_proportion > 0.3:
            status = "INCONCLUSIVE"
            print(f"  Partial mediation (prop={mediated_proportion:.3f}, sobel_z={sobel_z:.2f})")
        else:
            status = "FAILED"
            print(f"  Compression->memory DIRECT (prop={mediated_proportion:.3f}, sobel_z={sobel_z:.2f})")

        return CPIResult("compression_mediation", status, metrics=metrics)
