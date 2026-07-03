from __future__ import annotations

from typing import Any

import numpy as np
import numba
from numpy.typing import NDArray

from research.memory_physics.memory_validator import MemoryValidator, MPRResult
from research.temporal_reality.causality_analysis import AdaptiveTimeCausality


@numba.jit(nopython=True, cache=True)
def _univariate_slope(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    mx, my = 0.0, 0.0
    for i in range(n):
        mx += x[i]
        my += y[i]
    mx /= n
    my /= n
    num, den = 0.0, 0.0
    for i in range(n):
        dx = x[i] - mx
        dy = y[i] - my
        num += dx * dy
        den += dx * dx
    return num / den if abs(den) > 1e-12 else 0.0


@numba.jit(nopython=True, cache=True)
def _partial_slope(x: NDArray[np.float64], y: NDArray[np.float64], z: NDArray[np.float64]) -> float:
    n = min(len(x), len(y), len(z))
    if n < 3:
        return 0.0
    xf, yf, zf = x[:n], y[:n], z[:n]
    b_xz = _univariate_slope(zf, xf)
    b_yz = _univariate_slope(zf, yf)
    rx = np.empty(n, dtype=np.float64)
    ry = np.empty(n, dtype=np.float64)
    for i in range(n):
        rx[i] = xf[i] - b_xz * zf[i]
        ry[i] = yf[i] - b_yz * zf[i]
    return _univariate_slope(rx, ry)


class MediatorResolution:
    """RQ4: Is memory_density a required mediator between memory_conflict and adaptive_time?

    Tests two paths:
    - conflict -> density -> AT (mediated)
    - conflict -> AT (direct)
    """

    def __init__(self, validator: MemoryValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset
        self._max_lag = 200

    def run(self) -> MPRResult:
        data = self.validator.load_asset_data(self.asset)
        signals = self.validator.compute_signals(data)

        conflict = np.asarray(signals["memory_conflict"], dtype=np.float64)
        density = np.asarray(signals["memory_density"], dtype=np.float64)
        at = np.asarray(signals["adaptive_time"], dtype=np.float64)
        n = min(len(conflict), len(density), len(at))

        if n < self._max_lag * 2 + 1:
            return MPRResult("mediator_resolution", "FAILED", metrics={"error": "insufficient data"})

        from research.memory_physics.memory_validator import _find_peak_lag
        lag_cd, r_cd = _find_peak_lag(conflict[:n], density[:n], self._max_lag)
        lag_ca, r_ca = _find_peak_lag(conflict[:n], at[:n], self._max_lag)
        lag_da, r_da = _find_peak_lag(density[:n], at[:n], self._max_lag)

        total_shift = max(0, -lag_cd, -lag_da)
        seg_len = n - total_shift - abs(lag_ca) - 10
        if seg_len < 10:
            return MPRResult("mediator_resolution", "INCONCLUSIVE",
                             metrics={"error": f"insufficient aligned samples: {seg_len}"})

        shift = max(0, -lag_cd)
        c_aligned = conflict[shift:n - 10]
        d_aligned = density[:n - shift - 10] if lag_cd < 0 else density[lag_cd:n - 10]
        a_aligned = at[shift:n - 10] if lag_ca < 0 else at[lag_ca:n - 10]

        vn = min(len(c_aligned), len(d_aligned), len(a_aligned))
        if vn < 10:
            return MPRResult("mediator_resolution", "INCONCLUSIVE",
                             metrics={"error": f"insufficient vn={vn}"})

        cf, df_, af = c_aligned[:vn], d_aligned[:vn], a_aligned[:vn]

        # Path a: conflict -> density
        a_slope = _univariate_slope(cf, df_)
        # Path b: density -> AT controlling for conflict
        b_slope = _partial_slope(df_, af, cf)
        # Path c total: conflict -> AT
        c_total = _univariate_slope(cf, af)
        # Path c' direct: conflict -> AT controlling for density
        c_direct = _partial_slope(cf, af, df_)

        indirect_effect = a_slope * b_slope
        total_effect = c_total
        mediated_proportion = indirect_effect / max(abs(total_effect), 1e-12)

        non_mediated = abs(c_direct)
        mediated = abs(indirect_effect)

        # Correlations for interpretability
        r_cd_raw = float(np.corrcoef(cf, df_)[0, 1])
        r_ca_raw = float(np.corrcoef(cf, af)[0, 1])
        r_da_raw = float(np.corrcoef(df_, af)[0, 1])
        from research.temporal_reality.causality_analysis import AdaptiveTimeCausality
        r_ca_given_d = float(AdaptiveTimeCausality._cross_correlate(cf, af - b_slope * df_, 50)[50])

        metrics = {
            "lag_conflict_to_density": lag_cd,
            "lag_conflict_to_at": lag_ca,
            "lag_density_to_at": lag_da,
            "r_conflict_density": r_cd_raw,
            "r_conflict_at": r_ca_raw,
            "r_density_at": r_da_raw,
            "a_slope_conflict_to_density": a_slope,
            "b_slope_density_to_at_given_conflict": b_slope,
            "c_total_conflict_to_at": c_total,
            "c_direct_conflict_to_at_given_density": c_direct,
            "indirect_effect_ab": indirect_effect,
            "mediated_proportion": mediated_proportion,
            "density_is_required": abs(non_mediated) < abs(mediated) * 0.5,
        }

        if abs(mediated) > abs(non_mediated):
            status = "PASSED"
            print(f"  Density IS required mediator (mediated={mediated:.6f} vs direct={non_mediated:.6f})")
        elif abs(non_mediated) > abs(mediated) * 2:
            status = "FAILED"
            print(f"  Conflict->AT is primarily direct (direct={non_mediated:.6f} vs mediated={mediated:.6f})")
        else:
            status = "INCONCLUSIVE"
            print(f"  Inconclusive (mediated={mediated:.6f} vs direct={non_mediated:.6f})")

        return MPRResult("mediator_resolution", status, metrics=metrics)
