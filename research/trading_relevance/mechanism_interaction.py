from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numba
import numpy as np
from numpy.typing import NDArray

from research.information_discovery.mi_estimator import (
    _fast_digitize,
    _fast_entropy_digitized,
    _fast_joint_entropy_digitized,
    _fast_mutual_info,
    _fast_percentile,
    _fast_triple_entropy_digitized,
)


@dataclass
class MechanismInteractionReport:
    asset: str
    adaptive_time_only_ig: float
    energy_only_ig: float
    memory_only_ig: float
    combined_ig: float
    adaptive_time_improvement: float  # combined_ig - max(energy_only_ig, memory_only_ig)

    low_adaptive_time: dict  # {"ig": float, "sid": float, "sir": float}
    high_adaptive_time: dict  # {"ig": float, "sid": float, "sir": float}
    ig_difference: float  # high_ig - low_ig
    sid_difference: float
    sir_difference: float

    verdict: str
    # "adaptive_time_adds_information"
    # "adaptive_time_is_redundant"
    # "adaptive_time_is_regime_filter"


@numba.jit(nopython=True, cache=True)
def _composite_encode(
    dig_arrays: list[NDArray[np.int32]], n_bins: int
) -> NDArray[np.int32]:
    n = len(dig_arrays[0])
    n_arr = len(dig_arrays)
    out = np.zeros(n, dtype=np.int32)
    for i in range(n):
        code = 0
        for j in range(n_arr):
            code = code * n_bins + dig_arrays[j][i]
        out[i] = code
    return out


@numba.jit(nopython=True, cache=True)
def _conditional_mi_multiple(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    condition_arrays: list[NDArray[np.float64]],
    n_bins: int,
    cond_bins: int,
) -> float:
    valid = ~(np.isnan(x) | np.isnan(y))
    for c in condition_arrays:
        valid = valid & ~np.isnan(c)
    xc = x[valid]
    yc = y[valid]
    conds = [c[valid] for c in condition_arrays]
    if len(xc) < 2:
        return 0.0

    q = np.linspace(0.0, 1.0, n_bins + 1)
    x_bins = _fast_percentile(xc, q)
    y_bins = _fast_percentile(yc, q)

    ux = np.unique(x_bins)
    uy = np.unique(y_bins)
    if len(ux) < 2 or len(uy) < 2:
        return 0.0

    dx = _fast_digitize(xc, x_bins)
    dy = _fast_digitize(yc, y_bins)

    cq = np.linspace(0.0, 1.0, cond_bins + 1)
    d_conds = []
    for c in conds:
        cb = _fast_percentile(c, cq)
        dc = _fast_digitize(c, cb)
        d_conds.append(dc)

    comp = _composite_encode(d_conds, cond_bins)
    n_comp = cond_bins ** len(condition_arrays)

    h_xz = _fast_joint_entropy_digitized(dx, comp, n_comp)
    h_yz = _fast_joint_entropy_digitized(dy, comp, n_comp)
    h_z = _fast_entropy_digitized(comp, n_comp)
    h_xyz = _fast_triple_entropy_digitized(dx, dy, comp, n_comp)

    return max(0.0, h_xz + h_yz - h_z - h_xyz)


@numba.jit(nopython=True, cache=True)
def _mi_composite_vs_float(
    dig_composite: NDArray[np.int32],
    n_composite: int,
    y: NDArray[np.float64],
    n_bins: int,
) -> float:
    q = np.linspace(0.0, 1.0, n_bins + 1)
    y_bins = _fast_percentile(y, q)
    uy = np.unique(y_bins)
    if len(uy) < 2:
        return 0.0
    dy = _fast_digitize(y, y_bins)

    max_bins = max(n_composite, n_bins)
    hx = _fast_entropy_digitized(dig_composite, n_composite)
    hy = _fast_entropy_digitized(dy, n_bins)
    hxy = _fast_joint_entropy_digitized(dig_composite, dy, max_bins)
    return max(0.0, hx + hy - hxy)


@numba.jit(nopython=True, cache=True)
def _sid_score(
    states: NDArray[np.int32], returns: NDArray[np.float64], n_bins: int
) -> float:
    valid = ~np.isnan(returns)
    s = states[valid]
    r = returns[valid]
    if len(r) < 2:
        return 0.0

    q = np.linspace(0.0, 1.0, n_bins + 1)
    r_bins = _fast_percentile(r, q)
    if len(np.unique(r_bins)) < 2:
        return 0.0
    dr = _fast_digitize(r, r_bins)
    hy = _fast_entropy_digitized(dr, n_bins)

    unique_s = np.unique(s)
    h_y_given_s = 0.0
    total = len(r)
    for sid_val in unique_s:
        if sid_val < 0:
            continue
        mask = s == sid_val
        sub_r = r[mask]
        sub_n = len(sub_r)
        if sub_n < 2:
            continue
        sub_bins = _fast_percentile(sub_r, q)
        if len(np.unique(sub_bins)) < 2:
            continue
        dsub = _fast_digitize(sub_r, sub_bins)
        hsub = _fast_entropy_digitized(dsub, n_bins)
        h_y_given_s += (sub_n / total) * hsub

    return max(0.0, hy - h_y_given_s)


@numba.jit(nopython=True, cache=True)
def _state_complexity(states: NDArray[np.int32]) -> float:
    valid = states >= 0
    s = states[valid]
    n = len(s)
    if n < 2:
        return 1.0

    transitions = 0
    for i in range(1, n):
        if s[i] != s[i - 1]:
            transitions += 1
    transition_rate = transitions / n

    unique_s = np.unique(s)
    n_unique = len(unique_s)
    entropy = 0.0
    for sid_val in unique_s:
        cnt = 0
        for i in range(n):
            if s[i] == sid_val:
                cnt += 1
        if cnt > 0:
            p = cnt / n
            entropy -= p * np.log(p)

    return n_unique + transition_rate * 100.0 + entropy


@numba.jit(nopython=True, cache=True)
def _sir_score(
    states: NDArray[np.int32], returns: NDArray[np.float64], n_bins: int
) -> float:
    sid = _sid_score(states, returns, n_bins)
    if sid <= 0.0:
        return 0.0
    cplx = _state_complexity(states)
    if cplx < 1e-10:
        return 0.0
    return sid / cplx


@numba.jit(nopython=True, cache=True)
def _compute_ig_for_regime(
    adaptive_time: NDArray[np.float64],
    energy_storage: NDArray[np.float64],
    memory_density: NDArray[np.float64],
    memory_gradient: NDArray[np.float64],
    future_returns: NDArray[np.float64],
    states: NDArray[np.int32],
    mask: NDArray[np.bool_],
    n_bins: int,
    cond_bins: int,
) -> tuple[float, float, float]:
    at_r = adaptive_time[mask]
    en_r = energy_storage[mask]
    md_r = memory_density[mask]
    mg_r = memory_gradient[mask]
    fr_r = future_returns[mask]
    st_r = states[mask]

    if len(at_r) < 10:
        return 0.0, 0.0, 0.0

    ig = _fast_mutual_info(at_r, fr_r, n_bins)

    sid_val = _sid_score(st_r, fr_r, n_bins) if np.any(st_r >= 0) else 0.0
    sir_val = _sir_score(st_r, fr_r, n_bins) if np.any(st_r >= 0) else 0.0

    return ig, sid_val, sir_val


class MechanismInteractionAnalyzer:
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.cond_bins = 5

    def compute(
        self,
        adaptive_time: NDArray,
        returns: NDArray,
        energy_storage: NDArray,
        memory_density: NDArray,
        memory_gradient: NDArray,
        states: NDArray = None,
    ) -> MechanismInteractionReport:
        at = np.asarray(adaptive_time, dtype=np.float64)
        es = np.asarray(energy_storage, dtype=np.float64)
        md = np.asarray(memory_density, dtype=np.float64)
        mg = np.asarray(memory_gradient, dtype=np.float64)
        st = np.asarray(states, dtype=np.int32) if states is not None else np.zeros(len(at), dtype=np.int32)

        n = len(at)
        future_ret = np.empty(n, dtype=np.float64)
        future_ret[:-1] = returns[1:]
        future_ret[-1] = 0.0

        valid = ~(np.isnan(at) | np.isnan(es) | np.isnan(md) | np.isnan(mg) | np.isnan(future_ret))
        at_v = at[valid]
        es_v = es[valid]
        md_v = md[valid]
        mg_v = mg[valid]
        fr_v = future_ret[valid]
        st_v = st[valid]

        # ---- RQ5: Information Gain comparison ----
        at_ig = _fast_mutual_info(at_v, fr_v, self.n_bins)
        en_ig = _fast_mutual_info(es_v, fr_v, self.n_bins)
        md_ig = _fast_mutual_info(md_v, fr_v, self.n_bins)
        mg_ig = _fast_mutual_info(mg_v, fr_v, self.n_bins)
        memory_ig = max(md_ig, mg_ig)

        conds = [es_v, md_v, mg_v]
        combined_ig = _conditional_mi_multiple(at_v, fr_v, conds, self.n_bins, self.cond_bins)

        improvement = combined_ig - max(en_ig, memory_ig)

        # ---- RQ6: Regime Filter Evaluation ----
        median_at = np.median(at_v)
        low_mask = at_v <= median_at
        high_mask = at_v > median_at

        low_ig, low_sid, low_sir = _compute_ig_for_regime(
            at_v, es_v, md_v, mg_v, fr_v, st_v, low_mask,
            self.n_bins, self.cond_bins,
        )
        high_ig, high_sid, high_sir = _compute_ig_for_regime(
            at_v, es_v, md_v, mg_v, fr_v, st_v, high_mask,
            self.n_bins, self.cond_bins,
        )

        ig_diff = high_ig - low_ig
        sid_diff = high_sid - low_sid
        sir_diff = high_sir - low_sir

        low_dict = {"ig": low_ig, "sid": low_sid, "sir": low_sir}
        high_dict = {"ig": high_ig, "sid": high_sid, "sir": high_sir}

        report = MechanismInteractionReport(
            asset="unknown",
            adaptive_time_only_ig=float(at_ig),
            energy_only_ig=float(en_ig),
            memory_only_ig=float(memory_ig),
            combined_ig=float(combined_ig),
            adaptive_time_improvement=float(improvement),
            low_adaptive_time=low_dict,
            high_adaptive_time=high_dict,
            ig_difference=float(ig_diff),
            sid_difference=float(sid_diff),
            sir_difference=float(sir_diff),
            verdict="unknown",
        )
        report.verdict = self._determine_verdict(report)
        return report

    def _determine_verdict(self, report: MechanismInteractionReport) -> str:
        improv = report.adaptive_time_improvement
        best_individual = max(report.energy_only_ig, report.memory_only_ig)
        relative_improv = improv / best_individual if best_individual > 1e-12 else 0.0

        has_rq6_effect = (
            abs(report.ig_difference) > 0.05
            or abs(report.sid_difference) > 0.05
            or abs(report.sir_difference) > 0.01
        )

        if relative_improv > 0.1 or improv > 0.05:
            return "adaptive_time_adds_information"
        if has_rq6_effect:
            return "adaptive_time_is_regime_filter"
        return "adaptive_time_is_redundant"

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _discrete_mi(
        x: NDArray[np.float64], y: NDArray[np.float64], n_bins: int
    ) -> float:
        return _fast_mutual_info(x, y, n_bins)

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _sid_score(
        states: NDArray[np.int32], returns: NDArray[np.float64], n_bins: int
    ) -> float:
        return _sid_score(states, returns, n_bins)
