from __future__ import annotations

from typing import Any

import numpy as np
import numba
from numpy.typing import NDArray

from research.information_discovery.mi_estimator import (
    _fast_percentile,
    _fast_digitize,
    _fast_entropy_digitized,
    _fast_joint_entropy_digitized,
    _fast_triple_entropy_digitized,
    _fast_mutual_info,
    _fast_conditional_mutual_info,
)


@numba.jit(nopython=True, cache=True)
def _fast_composite_encode(
    digited_conditions: list[NDArray[np.int32]],
    n_bins: int,
) -> NDArray[np.int32]:
    n = len(digited_conditions[0])
    n_conds = len(digited_conditions)
    composite = np.zeros(n, dtype=np.int32)
    for i in range(n):
        code = 0
        for j in range(n_conds):
            code = code * n_bins + digited_conditions[j][i]
        composite[i] = code
    return composite


@numba.jit(nopython=True, cache=True)
def _fast_conditional_mi_multiple(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    condition_arrays: list[NDArray[np.float64]],
    n_bins: int,
    cond_bins: int = 5,
) -> float:
    valid = ~np.isnan(x) & ~np.isnan(y)
    for c in condition_arrays:
        valid = valid & ~np.isnan(c)
    x_clean = x[valid]
    y_clean = y[valid]
    conds_clean = [c[valid] for c in condition_arrays]
    if len(x_clean) < 2:
        return 0.0

    q = np.linspace(0.0, 1.0, n_bins + 1)
    x_bins = _fast_percentile(x_clean, q)
    y_bins = _fast_percentile(y_clean, q)

    dig_x = _fast_digitize(x_clean, x_bins)
    dig_y = _fast_digitize(y_clean, y_bins)

    cq = np.linspace(0.0, 1.0, cond_bins + 1)
    dig_conds = []
    for c in conds_clean:
        c_bins = _fast_percentile(c, cq)
        dig = _fast_digitize(c, c_bins)
        dig_conds.append(dig)

    composite = _fast_composite_encode(dig_conds, cond_bins)
    n_composite_bins = cond_bins ** len(condition_arrays)

    h_xz = _fast_joint_entropy_digitized(dig_x, composite, n_composite_bins)
    h_yz = _fast_joint_entropy_digitized(dig_y, composite, n_composite_bins)
    h_z = _fast_entropy_digitized(composite, n_composite_bins)
    h_xyz = _fast_triple_entropy_digitized(dig_x, dig_y, composite, n_composite_bins)

    return max(0.0, h_xz + h_yz - h_z - h_xyz)


class ConditionalInformationAnalyzer:
    """Analyzes conditional mutual information between time and state mutations.

    Computes how much predictive information is preserved when conditioning
    on volatility, entropy, event density, and state transition rate.
    """

    def __init__(self, n_bins: int = 20, cond_bins: int = 5) -> None:
        self.n_bins = n_bins
        self.cond_bins = cond_bins

    def conditional_mi(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        condition: NDArray[np.float64],
    ) -> float:
        """Conditional mutual information I(x;y|condition).

        Uses the identity: I(x;y|z) = H(x,z) + H(y,z) - H(z) - H(x,y,z)
        with percentile-based discretization.

        Parameters
        ----------
        x : NDArray[np.float64]
            First variable (e.g. adaptive_time_coordinate).
        y : NDArray[np.float64]
            Second variable (e.g. future_state_mutation).
        condition : NDArray[np.float64]
            Conditioning variable.

        Returns
        -------
        float
            Conditional mutual information in nats.
        """
        return _fast_conditional_mutual_info(
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            np.asarray(condition, dtype=np.float64),
            self.n_bins,
        )

    def conditional_mi_multiple(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        conditions: list[NDArray[np.float64]],
    ) -> float:
        """Conditional mutual information I(x;y|z1,z2,...,zk).

        Conditions are jointly encoded via composite discretization.
        Each condition is discretized into ``cond_bins`` buckets, then
        combined into a single composite index using mixed-radix encoding.

        Parameters
        ----------
        x : NDArray[np.float64]
            First variable.
        y : NDArray[np.float64]
            Second variable.
        conditions : list[NDArray[np.float64]]
            Sequence of conditioning variables.

        Returns
        -------
        float
            Conditional mutual information in nats.
        """
        if not conditions:
            return _fast_mutual_info(
                np.asarray(x, dtype=np.float64),
                np.asarray(y, dtype=np.float64),
                self.n_bins,
            )
        return _fast_conditional_mi_multiple(
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            [np.asarray(c, dtype=np.float64) for c in conditions],
            self.n_bins,
            self.cond_bins,
        )

    def analyze(
        self,
        adaptive_time_coordinate: NDArray[np.float64],
        future_state_mutation: NDArray[np.float64],
        volatility: NDArray[np.float64],
        entropy: NDArray[np.float64],
        event_density: NDArray[np.float64],
        state_transition_rate: NDArray[np.float64],
    ) -> dict[str, Any]:
        """Run full conditional information analysis.

        Computes raw mutual information, per-condition CMI, and joint-CMI
        across all four conditioning variables.

        Parameters
        ----------
        adaptive_time_coordinate : NDArray[np.float64]
            Adaptive time coordinate array.
        future_state_mutation : NDArray[np.float64]
            Future state mutation array.
        volatility : NDArray[np.float64]
            Volatility conditioning signal.
        entropy : NDArray[np.float64]
            Entropy conditioning signal.
        event_density : NDArray[np.float64]
            Event density conditioning signal.
        state_transition_rate : NDArray[np.float64]
            State transition rate conditioning signal.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:
                - raw_mi : float — unconditional mutual information
                - per_condition_cmi : dict[str, float] — CMI per condition
                - conditioned_mi : float — CMI conditioned on all variables jointly
                - information_survival_ratio : float — conditioned_mi / raw_mi
        """
        x = np.asarray(adaptive_time_coordinate, dtype=np.float64)
        y = np.asarray(future_state_mutation, dtype=np.float64)

        raw_mi = _fast_mutual_info(x, y, self.n_bins)

        conditions: dict[str, NDArray[np.float64]] = {
            "volatility": np.asarray(volatility, dtype=np.float64),
            "entropy": np.asarray(entropy, dtype=np.float64),
            "event_density": np.asarray(event_density, dtype=np.float64),
            "state_transition_rate": np.asarray(state_transition_rate, dtype=np.float64),
        }

        per_condition_cmi: dict[str, float] = {}
        for name, cond_arr in conditions.items():
            per_condition_cmi[name] = self.conditional_mi(x, y, cond_arr)

        conditioned_mi = self.conditional_mi_multiple(x, y, list(conditions.values()))

        information_survival_ratio = (
            conditioned_mi / raw_mi if raw_mi > 1e-12 else 0.0
        )

        return {
            "raw_mi": raw_mi,
            "per_condition_cmi": per_condition_cmi,
            "conditioned_mi": conditioned_mi,
            "information_survival_ratio": information_survival_ratio,
        }
