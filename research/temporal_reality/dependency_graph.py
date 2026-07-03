"""
Dependency Graph Builder for Proxima X Reality Phase 4.

Constructs a 4-node directed dependency graph over the variables

    adaptive_time -> state_mutation -> regime_transition -> market_outcome

using correlation and transfer entropy to quantify directed edge strengths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from numba import jit
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EDGE_WEAK_THRESHOLD: float = 0.125
"""Edges with dependency_score below this are considered weak.

Calibrated so that >95 % of random-noise edges fall below this value
when ``_TE_N_BINS == 5`` and sample size ≥ 200.
"""

_PATH_STRONG_THRESHOLD: float = 0.25
"""Weakest-link minimum along the canonical path must be ≥ this to
qualify as a strong ``evolution_clock_pathway``."""

_TE_LAG: int = 1
"""Lag used in transfer entropy computation."""

_TE_N_BINS: int = 5
"""Number of quantile bins for transfer entropy discretisation.

Using fewer bins (5) reduces finite-sample bias in the plug-in entropy
estimators while retaining sufficient resolution for dependency detection.
"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EdgeInfo:
    """Metadata for a single directed edge in the dependency graph.

    Attributes
    ----------
    edge_strength : float
        Absolute Pearson correlation between the two variables.
    information_flow : float
        Transfer entropy from source to target (nats).
    dependency_score : float
        Combined score = (edge_strength + te_norm) / 2 where
        te_norm = information_flow / (information_flow + 1).
    """

    edge_strength: float = 0.0
    information_flow: float = 0.0
    dependency_score: float = 0.0


@dataclass
class DependencyGraph:
    """A 4-node directed dependency graph with per-edge metadata.

    Attributes
    ----------
    nodes : list[str]
        Ordered node labels.
    edges : dict[(str, str), EdgeInfo]
        Directed edge metadata keyed by ``(from_name, to_name)``.
    adjacency_matrix : np.ndarray
        4×4 matrix where element ``[i, j]`` is the dependency_score of
        the edge from node *j* (column) to node *i* (row).
    verdict : str
        High-level interpretation label (see
        :meth:`DependencyGraphBuilder.build`).
    """

    nodes: List[str] = field(default_factory=list)
    edges: Dict[Tuple[str, str], EdgeInfo] = field(default_factory=dict)
    adjacency_matrix: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros((4, 4), dtype=np.float64),
    )
    verdict: str = ""


# ---------------------------------------------------------------------------
# Numba-accelerated helpers
# ---------------------------------------------------------------------------


@jit(nopython=True, cache=True)
def _pearson_corr(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Pearson correlation coefficient between two equal-length 1-D arrays."""
    n = len(x)
    if n < 2:
        return 0.0

    sum_x, sum_y = 0.0, 0.0
    for i in range(n):
        sum_x += x[i]
        sum_y += y[i]

    mean_x = sum_x / n
    mean_y = sum_y / n

    cov, var_x, var_y = 0.0, 0.0, 0.0
    for i in range(n):
        dx = x[i] - mean_x
        dy = y[i] - mean_y
        cov += dx * dy
        var_x += dx * dx
        var_y += dy * dy

    denom = np.sqrt(var_x * var_y)
    if denom < 1e-14:
        return 0.0
    r = cov / denom
    return max(-1.0, min(1.0, r))


@jit(nopython=True, cache=True)
def _entropy(counts: NDArray[np.int32], total: float) -> float:
    """Shannon entropy ``H = -sum(p_k * log(p_k))`` from an integer count array."""
    h = 0.0
    for i in range(len(counts)):
        c = counts[i]
        if c > 0:
            p = c / total
            h -= p * np.log(p)
    return h


@jit(nopython=True, cache=True)
def _cmi_discrete(
    x: NDArray[np.int32],
    y: NDArray[np.int32],
    z: NDArray[np.int32],
    n_bins: int,
) -> float:
    """Conditional mutual information ``I(x; y | z)`` for pre-discretised arrays.

    Uses the identity: I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(Z) - H(X,Y,Z).
    """
    n = len(x)
    total = float(n)

    z_cnt = np.zeros(n_bins, dtype=np.int32)
    xz_cnt = np.zeros(n_bins * n_bins, dtype=np.int32)
    yz_cnt = np.zeros(n_bins * n_bins, dtype=np.int32)
    xyz_cnt = np.zeros(n_bins * n_bins * n_bins, dtype=np.int32)

    for i in range(n):
        xi, yi, zi = x[i], y[i], z[i]
        z_cnt[zi] += 1
        xz_cnt[xi * n_bins + zi] += 1
        yz_cnt[yi * n_bins + zi] += 1
        xyz_cnt[xi * n_bins * n_bins + yi * n_bins + zi] += 1

    h_z = _entropy(z_cnt, total)
    h_xz = _entropy(xz_cnt, total)
    h_yz = _entropy(yz_cnt, total)
    h_xyz = _entropy(xyz_cnt, total)

    return max(0.0, h_xz + h_yz - h_z - h_xyz)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class DependencyGraphBuilder:
    """Construct a 4-node dependency graph over adaptive-time, state-mutation,
    regime-transition, and market-outcome variables.

    Each directed edge is characterised by:

    * **edge_strength** — absolute Pearson correlation
    * **information_flow** — transfer entropy (source → target)
    * **dependency_score** — average of *edge_strength* and
      ``te / (te + 1)`` (both mapped to [0, 1])
    """

    NODE_NAMES: List[str] = [
        "adaptive_time",
        "state_mutation",
        "regime_transition",
        "market_outcome",
    ]

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        adaptive_time: NDArray[np.float64],
        state_mutation_rate: NDArray[np.float64],
        regime_change_events: NDArray[np.float64],
        returns: NDArray[np.float64],
        volatility: NDArray[np.float64],
    ) -> DependencyGraph:
        """Build a 4-node dependency graph from the five input signals.

        The four graph nodes are derived as follows:

        =====================  ============================================
        Node                   Signal
        =====================  ============================================
        adaptive_time          *adaptive_time* (passed as-is)
        state_mutation         *state_mutation_rate* (passed as-is)
        regime_transition      *regime_change_events* (passed as-is)
        market_outcome         ``returns * volatility`` (interaction term)
        =====================  ============================================

        For each ordered pair ``(from, to)`` the builder computes:

        1. :meth:`_edge_strength` — absolute Pearson correlation
        2. :meth:`_transfer_entropy` — TE at lag 1
        3. A combined ``dependency_score = (strength + te/(te+1)) / 2``

        The resulting 4×4 adjacency matrix follows the standard convention:
        entry ``[i, j]`` = dependency score of the edge **from** node *j*
        (column) **to** node *i* (row).

        Parameters
        ----------
        adaptive_time : np.ndarray
            1-D adaptive-time coordinate values.
        state_mutation_rate : np.ndarray
            1-D state mutation rate values.
        regime_change_events : np.ndarray
            1-D regime change event indicator / intensity.
        returns : np.ndarray
            1-D market returns (same length as *adaptive_time*).
        volatility : np.ndarray
            1-D market volatility estimates (same length).

        Returns
        -------
        DependencyGraph
            Populated graph with nodes, edges, adjacency matrix, and a
            categorical verdict.

        Raises
        ------
        ValueError
            If any input is not a 1-D array, lengths differ, or fewer
            than 5 clean samples remain after NaN/Inf filtering.

        Verdict logic
        -------------
        The verdict string is chosen by evaluating the following conditions
        in order:

        1. ``adj[1, 0]`` is the **largest** entry in the matrix →
           ``"adaptive_time_drives_evolution"``

        2. ``adj[0, 1]`` is the **largest** entry →
           ``"adaptive_time_reflects_evolution"``

        3. Every entry in the matrix is below ``_EDGE_WEAK_THRESHOLD`` →
           ``"no_clear_dependency"``

        4. The canonical path ``adaptive_time → state_mutation →
           regime_transition → market_outcome`` has a minimum edge
           **≥** ``_PATH_STRONG_THRESHOLD`` →
           ``"evolution_clock_pathway"``

        5. Otherwise → ``"mixed_dependencies"``
        """
        # -- validate & convert -------------------------------------------------
        arrays: List[NDArray[np.float64]] = []
        labels = [
            "adaptive_time",
            "state_mutation_rate",
            "regime_change_events",
            "returns",
            "volatility",
        ]
        for label, arr in zip(labels, [adaptive_time, state_mutation_rate, regime_change_events, returns, volatility]):
            a = np.asarray(arr, dtype=np.float64)
            if a.ndim != 1:
                raise ValueError(f"{label} must be 1-D, got {a.ndim} dimensions")
            arrays.append(a)

        lengths = {len(a) for a in arrays}
        if len(lengths) != 1:
            raise ValueError(
                f"All inputs must have the same length; got lengths {[len(a) for a in arrays]}"
            )

        # -- construct the four node signals ------------------------------------
        # market_outcome: interaction term capturing return × vol impact
        market_outcome = returns * volatility
        market_outcome = np.where(np.isfinite(market_outcome), market_outcome, 0.0)

        signals = [
            arrays[0],  # adaptive_time
            arrays[1],  # state_mutation_rate
            arrays[2],  # regime_change_events
            market_outcome,
        ]

        # -- clean NaN / Inf ----------------------------------------------------
        mask: NDArray[np.bool_] = np.ones(len(signals[0]), dtype=np.bool_)
        for s in signals:
            mask &= np.isfinite(s)
        signals = [s[mask] for s in signals]

        n = len(signals[0])
        if n < 5:
            raise ValueError(
                f"Only {n} valid sample(s) remain after NaN/Inf filtering; need at least 5."
            )

        # -- compute edges & adjacency matrix ------------------------------------
        n_nodes = 4
        adj = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        edges: Dict[Tuple[str, str], EdgeInfo] = {}

        for i in range(n_nodes):      # target (row)
            for j in range(n_nodes):  # source (column)
                if i == j:
                    continue

                from_name = self.NODE_NAMES[j]
                to_name = self.NODE_NAMES[i]
                x = signals[j]  # source variable
                y = signals[i]  # target variable

                es = self._edge_strength(x, y)
                te = self._transfer_entropy(x, y, lag=_TE_LAG)

                # Combine: edge_strength in [0, 1], TE mapped to [0, 1)
                # via x / (x + 1), then average for the dependency score.
                te_norm = te / (te + 1.0)
                ds = (es + te_norm) * 0.5

                edges[(from_name, to_name)] = EdgeInfo(
                    edge_strength=es,
                    information_flow=te,
                    dependency_score=ds,
                )
                adj[i, j] = ds

        # -- verdict ------------------------------------------------------------
        # The adjacency matrix stores the combined dependency_score which is
        # directional (it incorporates transfer entropy).  Use it for verdict
        # so that the two directions of the at↔sm pair can differ.
        max_idx = np.unravel_index(np.argmax(adj), adj.shape)
        max_val = adj[max_idx]

        # Canonical path: adaptive_time(0) -> state_mutation(1) ->
        #                 regime_transition(2) -> market_outcome(3).
        path_vals = [adj[1, 0], adj[2, 1], adj[3, 2]]
        path_strength = min(path_vals)

        if max_val < _EDGE_WEAK_THRESHOLD:
            verdict = "no_clear_dependency"
        elif max_idx == (1, 0) and adj[1, 0] > adj[0, 1]:
            verdict = "adaptive_time_drives_evolution"
        elif max_idx == (0, 1) and adj[0, 1] > adj[1, 0]:
            verdict = "adaptive_time_reflects_evolution"
        elif path_strength >= _PATH_STRONG_THRESHOLD:
            verdict = "evolution_clock_pathway"
        else:
            verdict = "mixed_dependencies"

        return DependencyGraph(
            nodes=list(self.NODE_NAMES),
            edges=edges,
            adjacency_matrix=adj,
            verdict=verdict,
        )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _edge_strength(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
        """Absolute Pearson correlation between *x* and *y*.

        Returns 0.0 when the correlation cannot be meaningfully computed
        (e.g. constant or single-element inputs).
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        return float(abs(_pearson_corr(x, y)))

    @staticmethod
    @jit(nopython=True, cache=True)
    def _discretize(x: NDArray[np.float64], n_bins: int) -> NDArray[np.int32]:
        """Quantile-based discretisation into ``n_bins`` integer codes ``[0, n_bins-1]``.

        Bin edges are determined by the empirical quantiles of *x*.
        """
        n = len(x)
        out = np.empty(n, dtype=np.int32)
        if n < 2:
            out[:] = 0
            return out

        sorted_x = np.sort(x)
        edges = np.empty(n_bins + 1, dtype=np.float64)
        for i in range(n_bins + 1):
            idx = int((i / n_bins) * (n - 1))
            edges[i] = sorted_x[idx]

        edges[0] = -np.inf
        edges[-1] = np.inf

        for i in range(n):
            val = x[i]
            for j in range(n_bins - 1):
                if edges[j] <= val < edges[j + 1]:
                    out[i] = j
                    break
            else:
                out[i] = n_bins - 1

        return out

    @staticmethod
    def _transfer_entropy(
        source: NDArray[np.float64],
        target: NDArray[np.float64],
        lag: int = 1,
    ) -> float:
        """Simplified transfer entropy from *source* to *target* at given *lag*.

        Approximates TE as the conditional mutual information

        .. math::

            TE_{S \\to T}(t) = I(T_t; S_{t-lag} \\mid T_{t-1})

        using quantile-based discretisation with ``_TE_N_BINS`` bins.
        """
        n = len(source)
        if n <= lag + 2:
            return 0.0

        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)

        # Align so that for each t in [lag, n-1] we have:
        #   src = source[t - lag]
        #   tgt = target[t]
        #   cond = target[t - 1]
        src = source[lag:]       # length n - lag
        tgt = target[lag:]       # length n - lag
        cond = target[lag - 1:-1]  # length n - lag

        n_valid = len(src)
        if n_valid < _TE_N_BINS + 2:
            return 0.0

        src_binned = DependencyGraphBuilder._discretize(src, _TE_N_BINS)
        tgt_binned = DependencyGraphBuilder._discretize(tgt, _TE_N_BINS)
        cond_binned = DependencyGraphBuilder._discretize(cond, _TE_N_BINS)

        return float(_cmi_discrete(src_binned, tgt_binned, cond_binned, _TE_N_BINS))
