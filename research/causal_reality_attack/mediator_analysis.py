from __future__ import annotations

from typing import Any

import numpy as np
import numba
from numpy.typing import NDArray

from research.causal_reality_attack.attack_validator import (
    AttackValidator, AttackResult, TARGET_VARIABLES, _clean_serializable,
)
from research.temporal_reality.causality_analysis import AdaptiveTimeCausality


@numba.jit(nopython=True, cache=True)
def _univariate_slope(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    mx = 0.0
    my = 0.0
    for i in range(n):
        mx += x[i]
        my += y[i]
    mx /= n
    my /= n
    num = 0.0
    den = 0.0
    for i in range(n):
        dx = x[i] - mx
        dy = y[i] - my
        num += dx * dy
        den += dx * dx
    if abs(den) < 1e-12:
        return 0.0
    return num / den


@numba.jit(nopython=True, cache=True)
def _partial_slope(x: NDArray[np.float64], y: NDArray[np.float64], z: NDArray[np.float64]) -> float:
    """Slope of y on x controlling for z (regression residual method)."""
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


@numba.jit(nopython=True, cache=True)
def _find_best_lag(x: NDArray[np.float64], y: NDArray[np.float64], max_lag: int) -> tuple[int, float]:
    n = min(len(x), len(y))
    if n < max_lag * 2 + 1:
        return 0, 0.0
    best_lag = 0
    best_corr = -1.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            sx, ex = lag, n
            sy, ey = 0, n - lag
        else:
            sx, ex = 0, n + lag
            sy, ey = -lag, n
        length = ex - sx
        if length < 3:
            continue
        x_seg = x[sx:ex]
        y_seg = y[sy:ey]
        xm = 0.0
        ym = 0.0
        for i in range(length):
            xm += x_seg[i]
            ym += y_seg[i]
        xm /= length
        ym /= length
        xv = 0.0
        yv = 0.0
        cov = 0.0
        for i in range(length):
            dx = x_seg[i] - xm
            dy = y_seg[i] - ym
            cov += dx * dy
            xv += dx * dx
            yv += dy * dy
        denom = np.sqrt(max(xv, 1e-12)) * np.sqrt(max(yv, 1e-12))
        r = cov / denom if denom > 0 else 0.0
        if abs(r) > abs(best_corr):
            best_corr = r
            best_lag = lag
    return best_lag, best_corr


class MediatorAnalysis:
    """Attack 7: Mediator Analysis (Baron & Kenny product-of-coefficients).

    Tests whether adaptive_time is a true mediator between memory_density
    and state_mutation_rate using empirically-discovered lags and OLS slopes.
    """

    def __init__(self, validator: AttackValidator, asset: str = "EURJPY"):
        self.validator = validator
        self.asset = asset
        self._max_lag = 50

    def run(self) -> AttackResult:
        data = self.validator.load_asset_data(self.asset)
        signals = self.validator.compute_signals(data)

        md = np.asarray(signals.get("memory_density", np.zeros(1)), dtype=np.float64)
        at = np.asarray(signals.get("adaptive_time", np.zeros(1)), dtype=np.float64)
        smr = np.asarray(signals.get("state_mutation_rate", np.zeros(1)), dtype=np.float64)
        n = min(len(md), len(at), len(smr))

        if n < self._max_lag * 2 + 1:
            return AttackResult("mediator_analysis", "FAILED",
                                metrics={"error": "insufficient data"})

        # Step 1: Discover empirical lags
        lag_md_at, r_md_at = _find_best_lag(md, at, self._max_lag)
        lag_at_smr, r_at_smr = _find_best_lag(at, smr, self._max_lag)
        lag_md_smr, r_md_smr = _find_best_lag(md, smr, self._max_lag)

        metrics = {
            "lag_md_to_at": lag_md_at,
            "lag_at_to_smr": lag_at_smr,
            "lag_md_to_smr": lag_md_smr,
            "r_md_at": float(r_md_at),
            "r_at_smr": float(r_at_smr),
            "r_md_smr": float(r_md_smr),
        }

        # Step 2: Align series based on discovered lags
        # If md leads at (lag_md_at < 0), md is past of at
        # Shift so that: md(t+lag_md_at) predicts at(t)
        effective_lag = lag_md_at + lag_at_smr
        total_shift = max(0, -effective_lag, -lag_md_at, -lag_at_smr)

        # Align all three to common window
        seg_len = n - abs(lag_md_at) - abs(lag_at_smr) - total_shift
        if seg_len < 10:
            metrics["note"] = "insufficient overlap after lag alignment"
            return AttackResult("mediator_analysis", "INCONCLUSIVE", metrics=metrics)

        # Align: md leads at by lag_md_at, at leads smr by lag_at_smr
        if lag_md_at < 0:
            start_md = -lag_md_at
            start_at = 0
        else:
            start_md = 0
            start_at = lag_md_at

        if lag_at_smr < 0:
            at_end = n + lag_at_smr
            smr_end = n
        else:
            at_end = n
            smr_end = n - lag_at_smr

        common_start = start_md
        common_end = min(at_end, smr_end, n)
        seg_len = common_end - common_start
        if seg_len < 10:
            metrics["note"] = f"insufficient seg_len={seg_len}"
            return AttackResult("mediator_analysis", "INCONCLUSIVE", metrics=metrics)

        md_seg = md[common_start:common_end]
        at_seg = at[common_start + lag_md_at - lag_md_at:common_end + lag_md_at - lag_md_at]

        # Simpler: just use the lag structure
        # After lag alignment, we want: md predicts at predicts smr
        # So we use contemporaneous values after shift
        shift1 = max(0, -lag_md_at)
        shift2 = max(0, -lag_at_smr)
        total = shift1 + shift2
        if n - total < 10:
            metrics["note"] = f"insufficient after shift total={total}"
            return AttackResult("mediator_analysis", "INCONCLUSIVE", metrics=metrics)

        md_aligned = md[shift1:n - total + shift1]
        at_aligned = at[shift1 + shift2:n]
        smr_aligned = smr[total:]

        vn = min(len(md_aligned), len(at_aligned), len(smr_aligned))
        md_a = md_aligned[:vn].copy()
        at_a = at_aligned[:vn].copy()
        smr_a = smr_aligned[:vn].copy()

        if vn < 10:
            metrics["note"] = f"insufficient vn={vn}"
            return AttackResult("mediator_analysis", "INCONCLUSIVE", metrics=metrics)

        # Step 3: Product-of-coefficients mediation
        # Path a: MD → AT
        a_slope = _univariate_slope(md_a, at_a)
        # Path b: AT → SMR (controlling for MD)
        b_slope = _partial_slope(at_a, smr_a, md_a)
        # Path c (total): MD → SMR
        c_slope = _univariate_slope(md_a, smr_a)
        # Path c' (direct): MD → SMR (controlling for AT)
        c_prime_slope = _partial_slope(md_a, smr_a, at_a)

        indirect_effect = a_slope * b_slope
        total_effect = c_slope
        direct_effect = c_prime_slope
        mediated_proportion = indirect_effect / max(abs(total_effect), 1e-12)

        # Sobel-like test using standardized coefficients for interpretability
        r_direct = float(np.corrcoef(md_a, smr_a)[0, 1])
        r_md_at_val = float(np.corrcoef(md_a, at_a)[0, 1])
        r_at_smr_val = float(np.corrcoef(at_a, smr_a)[0, 1])

        reduction_ratio = (abs(r_direct) - abs(c_prime_slope / max(abs(c_slope), 1e-12) * abs(r_direct) if abs(c_slope) > 1e-12 else 0.0)) / max(abs(r_direct), 1e-12)

        # Simpler: compare direct path with and without mediator
        indirect_r = abs(r_md_at_val * r_at_smr_val)
        direct_r = abs(r_direct)
        reduction_r = (direct_r - abs(float(np.corrcoef(md_a, smr_a - b_slope * at_a)[0, 1] if abs(b_slope) > 1e-12 else r_direct))) / max(direct_r, 1e-12)

        # Use partial correlation for cleaner reduction measure
        from research.temporal_reality.causality_analysis import AdaptiveTimeCausality
        residual = smr_a - (b_slope if abs(b_slope) > 1e-12 else 0.0) * at_a
        corr_md_smr_given_at = float(np.corrcoef(md_a, residual)[0, 1]) if len(md_a) > 1 else 0.0

        metrics.update({
            "a_slope_md_to_at": float(a_slope),
            "b_slope_at_to_smr_given_md": float(b_slope),
            "c_slope_total_md_to_smr": float(c_slope),
            "c_prime_slope_direct_md_to_smr": float(c_prime_slope),
            "indirect_effect_ab": float(indirect_effect),
            "total_effect_c": float(total_effect),
            "direct_effect_c_prime": float(direct_effect),
            "mediated_proportion": float(mediated_proportion),
            "r_direct_contemporaneous": float(r_direct),
        })

        is_mediator = abs(indirect_effect) > 1e-6 and abs(reduction_r) > 0.15
        is_decorative = abs(indirect_effect) < 1e-8 and abs(reduction_r) < 0.05

        if is_mediator:
            status = "PASSED"
            print(f"  adaptive_time IS a mediator (ab={indirect_effect:.6f}, c={total_effect:.6f}, proportion={mediated_proportion:.3f})")
        elif is_decorative:
            status = "FAILED"
            print(f"  adaptive_time is DECORATIVE (ab={indirect_effect:.6f})")
        else:
            status = "INCONCLUSIVE"
            print(f"  adaptive_time: inconclusive (ab={indirect_effect:.6f}, c={total_effect:.6f})")

        return AttackResult("mediator_analysis", status, metrics=metrics)
