"""
causal_graph_builder.py --- Causal dependency graph from telemetry streams.

Builds a directed weighted graph between engine dimensions, scalar metrics,
and regime state by approximating causal relationships (not just correlation)
from the 432-byte shared-memory telemetry frames.

Two complementary methods are used per metric pair:

1. **Cross-correlation with lag detection** --- Pearson correlation at lags
   -10 to +10 frames. The lag with max |correlation| indicates the most
   likely causal direction and delay.

2. **Granger causality approximation** --- Linear regression F-test: does
   adding the past of metric A significantly improve prediction of metric B
   beyond B's own history?

Edges are weighted by |correlation| * (1 - p_value_approx) * lag_decay.
"""

from __future__ import annotations

import math
import struct
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

__all__ = [
    "CausalNode",
    "CausalEdge",
    "CausalGraph",
    "CausalGraphBuilder",
]

# ---------------------------------------------------------------------------
# Constants --- frame layout (matches shared_memory_telemetry.py)
# ---------------------------------------------------------------------------

_HEADER_FORMAT = struct.Struct("<QdQQQQ2Q")  # 64 bytes
_FRAME_FORMAT = struct.Struct("<32f13f4x")   # 184 bytes

HEADER_SIZE = 64
FRAME_SIZE = 184
BUF0_OFFSET = 64
BUF1_OFFSET = 248

# Number of engine dimensions in the 32f portion
N_ENGINE_DIMS = 32

# Scalar metric names --- indices 0-11 of the 13 floats (index 12 is padding).
# The order must match TelemetryCore.write_scalars.
SCALAR_NAMES: List[str] = [
    "alignment",
    "stability",
    "entropy",
    "regime_state",
    "tpi_confidence",
    "shadow_alignment",
    "sof_score",
    "kill_switch_pressure",
    "rollout_progress",
    "execution_intensity",
    "risk_exposure",
    "system_integrity",
]

# ---------------------------------------------------------------------------
# Analysis tuning
# ---------------------------------------------------------------------------

# Maximum lag (in frames) tested for cross-correlation
_MAX_CORR_LAG = 10

# Number of lag terms in the Granger regression models
_GRANGER_ORDER = 3

# Minimum frames required before any analysis is attempted
_MIN_FRAMES = 30

# Minimum |correlation| to consider an edge
_CORR_THRESHOLD = 0.3

# Maximum p-value for Granger causality to contribute
_GRANGER_P_THRESHOLD = 0.10

# Decay factor applied per unit lag to prefer shorter causal delays
_LAG_DECAY_BASE = 0.9


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CausalNode:
    """A node in the causal graph.

    Attributes
    ----------
    id : str
        Unique identifier (e.g. ``"engine_7"``, ``"metric_stability"``).
    node_type : str
        One of ``"engine"``, ``"metric"``, ``"regime"``.
    label : str
        Human-readable label.
    """
    id: str
    node_type: str
    label: str


@dataclass
class CausalEdge:
    """A directed edge representing a causal influence.

    Attributes
    ----------
    source : str
        Node ID of the cause.
    target : str
        Node ID of the effect.
    weight : float
        Causal strength in [0.0, 1.0].
    lag : int
        Detected lag in frames (positive means *source* leads *target*).
    method : str
        Detection method: ``"granger"``, ``"cross_corr"``, or
        ``"transfer_entropy"`` (reserved for future use).
    """
    source: str
    target: str
    weight: float
    lag: int
    method: str


@dataclass
class CausalGraph:
    """A complete causal dependency graph at a point in time.

    Attributes
    ----------
    nodes : list[CausalNode]
        All nodes in the graph.
    edges : list[CausalEdge]
        All detected causal edges.
    timestamp : float
        Unix timestamp of the latest frame that contributed to the graph.
    """
    nodes: list[CausalNode]
    edges: list[CausalEdge]
    timestamp: float


# ---------------------------------------------------------------------------
# CausalGraphBuilder
# ---------------------------------------------------------------------------


class CausalGraphBuilder:
    """Builds a causal dependency graph from a rolling telemetry window.

    Maintains deques of the 32 engine dimensions and 12 scalar metrics
    extracted from raw 432-byte SHM frames.  On demand, computes pairwise
    cross-correlation and Granger-causality approximations to produce a
    weighted directed graph.

    Parameters
    ----------
    window_size : int
        Maximum number of frames retained for rolling-window analysis.
        Larger windows improve statistical power but slow adaptation.
    """

    def __init__(self, window_size: int = 500) -> None:
        self.window_size = window_size

        # Rolling windows: one deque per engine dimension (0 .. 31)
        self._engine_series: List[deque] = [
            deque(maxlen=window_size) for _ in range(N_ENGINE_DIMS)
        ]

        # Rolling windows: one deque per scalar metric (12 entries)
        self._scalar_series: List[deque] = [
            deque(maxlen=window_size) for _ in range(len(SCALAR_NAMES))
        ]

        # Timestamps corresponding to each ingested frame
        self._timestamps: deque = deque(maxlen=window_size)

        self._frames_seen: int = 0
        self._latest_timestamp: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, frame: bytes) -> None:
        """Parse a raw 432-byte SHM frame and update internal rolling windows.

        Parameters
        ----------
        frame : bytes
            Exactly 432 bytes as written by ``TelemetryCore`` (see
            ``shared_memory_telemetry.py``).
        """
        if len(frame) < HEADER_SIZE + FRAME_SIZE:
            return  # malformed --- silently ignore

        # -- Parse header
        hdr = _HEADER_FORMAT.unpack_from(frame, 0)
        active_idx: int = hdr[2]       # 0 or 1
        timestamp: float = hdr[1]

        # -- Read the active frame buffer
        offset = BUF0_OFFSET if active_idx == 0 else BUF1_OFFSET
        raw = _FRAME_FORMAT.unpack_from(frame, offset)

        # 32 floats --- engine vector
        engine_vector = raw[:32]

        # 13 floats --- scalars (indices 0-11 are meaningful, index 12 padding)
        scalars = raw[32:44]

        # -- Update rolling windows
        for i in range(N_ENGINE_DIMS):
            self._engine_series[i].append(engine_vector[i])

        for i in range(len(SCALAR_NAMES)):
            self._scalar_series[i].append(scalars[i])

        self._timestamps.append(timestamp)
        self._latest_timestamp = timestamp
        self._frames_seen += 1

    def build_graph(self) -> CausalGraph:
        """Analyse all metric pairs and return a weighted causal graph.

        Returns
        -------
        CausalGraph
            A graph containing nodes for each engine dimension, each scalar
            metric, and the regime state, with causal edges whose weights
            reflect combined evidence from cross-correlation and Granger
            causality.  If insufficient data are available, returns a graph
            with nodes but no edges.
        """
        nodes = self._build_nodes()
        edges: List[CausalEdge] = []

        if self._frames_seen < _MIN_FRAMES:
            return CausalGraph(
                nodes=nodes,
                edges=edges,
                timestamp=self._latest_timestamp or time.time(),
            )

        # Collect all time series as (id, values) pairs
        series: List[Tuple[str, List[float]]] = []

        # Engine dimensions
        for i in range(N_ENGINE_DIMS):
            series.append((f"engine_{i}", list(self._engine_series[i])))

        # Scalar metrics
        for i, name in enumerate(SCALAR_NAMES):
            series.append((f"metric_{name}", list(self._scalar_series[i])))

        n_series = len(series)

        # Pairwise analysis
        for i in range(n_series):
            for j in range(n_series):
                if i == j:
                    continue

                id_i, vals_i = series[i]
                id_j, vals_j = series[j]

                edge = self._analyse_pair(id_i, vals_i, id_j, vals_j)
                if edge is not None:
                    edges.append(edge)

        return CausalGraph(
            nodes=nodes,
            edges=edges,
            timestamp=self._latest_timestamp or time.time(),
        )

    def get_top_drivers(
        self, target: str, top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """Return the top *N* causal drivers of a given node.

        Builds a fresh graph and extracts incoming edges to *target*,
        sorted by descending weight.

        Parameters
        ----------
        target : str
            Node ID (e.g. ``"metric_stability"``, ``"engine_7"``).
        top_n : int
            Maximum number of drivers to return.

        Returns
        -------
        list[tuple[str, float]]
            ``(driver_id, causal_weight)`` pairs, highest weight first.
        """
        graph = self.build_graph()
        drivers: List[Tuple[str, float]] = [
            (e.source, e.weight) for e in graph.edges if e.target == target
        ]
        drivers.sort(key=lambda x: -x[1])
        return drivers[:top_n]

    def get_engine_coupling(self, engine_i: int, engine_j: int) -> float:
        """Return the bidirectional causal coupling strength between engines.

        The coupling is the *maximum* of the two directed edge weights
        between the pair, or 0.0 if no significant edge was detected.

        Parameters
        ----------
        engine_i : int
            First engine dimension index (0-31).
        engine_j : int
            Second engine dimension index (0-31).

        Returns
        -------
        float
            Causal coupling strength in [0.0, 1.0].
        """
        graph = self.build_graph()
        id_i = f"engine_{engine_i}"
        id_j = f"engine_{engine_j}"

        w_ij = 0.0
        w_ji = 0.0
        for e in graph.edges:
            if e.source == id_i and e.target == id_j:
                w_ij = e.weight
            elif e.source == id_j and e.target == id_i:
                w_ji = e.weight

        return max(w_ij, w_ji)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_nodes(self) -> List[CausalNode]:
        """Create the full set of graph nodes."""
        nodes: List[CausalNode] = []

        for i in range(N_ENGINE_DIMS):
            nodes.append(CausalNode(
                id=f"engine_{i}",
                node_type="engine",
                label=f"Engine Dim {i}",
            ))

        for name in SCALAR_NAMES:
            label = name.replace("_", " ").title()
            nodes.append(CausalNode(
                id=f"metric_{name}",
                node_type="metric",
                label=label,
            ))

        nodes.append(CausalNode(
            id="regime",
            node_type="regime",
            label="Regime State",
        ))

        return nodes

    def _analyse_pair(
        self,
        src_id: str,
        src_vals: List[float],
        tgt_id: str,
        tgt_vals: List[float],
    ) -> Optional[CausalEdge]:
        """Analyse a single directed pair using both methods.

        Returns a ``CausalEdge`` if the combined evidence exceeds the
        correlation threshold, otherwise ``None``.
        """
        n = min(len(src_vals), len(tgt_vals))
        if n < _MIN_FRAMES:
            return None

        x = src_vals[:n]
        y = tgt_vals[:n]

        best_weight = 0.0
        best_lag = 0
        best_method = "cross_corr"

        # -- Method 1: cross-correlation with lag detection
        corr_result = self._cross_correlation(x, y)
        if corr_result is not None:
            lag, corr_val = corr_result
            decay = _LAG_DECAY_BASE ** abs(lag)
            corr_weight = abs(corr_val) * decay

            if corr_weight > best_weight:
                best_weight = min(corr_weight, 1.0)
                best_lag = lag
                best_method = "cross_corr"

        # -- Method 2: Granger causality approximation (src -> tgt)
        granger_result = self._granger_causality(x, y)
        if granger_result is not None:
            f_stat, p_value = granger_result
            if p_value < _GRANGER_P_THRESHOLD and f_stat > 0:
                # Normalise F-stat contribution and multiply by (1 - p)
                f_norm = min(f_stat / 10.0, 1.0)
                granger_weight = (1.0 - p_value) * f_norm

                if granger_weight > best_weight:
                    best_weight = min(granger_weight, 1.0)
                    best_lag = _GRANGER_ORDER
                    best_method = "granger"

        if best_weight >= _CORR_THRESHOLD:
            return CausalEdge(
                source=src_id,
                target=tgt_id,
                weight=best_weight,
                lag=best_lag,
                method=best_method,
            )

        return None

    # ------------------------------------------------------------------
    # Cross-correlation with lag detection
    # ------------------------------------------------------------------

    @staticmethod
    def _cross_correlation(
        x: List[float],
        y: List[float],
        max_lag: int = _MAX_CORR_LAG,
    ) -> Optional[Tuple[int, float]]:
        """Compute cross-correlation at lags [-max_lag .. +max_lag].

        Returns the lag (positive means *x* leads *y*) at which the absolute
        Pearson correlation is maximised, together with the correlation value.
        Returns ``None`` if there are too few data points.

        Parameters
        ----------
        x : list[float]
            First time series (potential cause).
        y : list[float]
            Second time series (potential effect).
        max_lag : int
            Maximum number of frames to shift in either direction.

        Returns
        -------
        tuple[int, float] or None
            ``(best_lag, best_correlation)`` in [-1, 1].
        """
        n = min(len(x), len(y))
        if n < _MIN_FRAMES:
            return None

        x_arr = list(x[:n])
        y_arr = list(y[:n])

        mean_x = sum(x_arr) / n
        mean_y = sum(y_arr) / n

        dx = [v - mean_x for v in x_arr]
        dy = [v - mean_y for v in y_arr]

        var_x = sum(d * d for d in dx) / n
        var_y = sum(d * d for d in dy) / n

        std_x = math.sqrt(var_x) if var_x > 0 else 1e-8
        std_y = math.sqrt(var_y) if var_y > 0 else 1e-8

        best_lag = 0
        best_corr = 0.0

        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                # y leads x (negative lag): shift y forward relative to x
                shift = -lag
                p = n - shift
                if p < 5:
                    continue
                numer = sum(dx[i + shift] * dy[i] for i in range(p))
            else:
                # x leads y (positive lag): shift x forward relative to y
                shift = lag
                p = n - shift
                if p < 5:
                    continue
                numer = sum(dx[i] * dy[i + shift] for i in range(p))

            corr = numer / (p * std_x * std_y)
            corr = max(-1.0, min(1.0, corr))  # clamp fp noise

            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag

        return best_lag, best_corr

    # ------------------------------------------------------------------
    # Granger causality approximation (F-test)
    # ------------------------------------------------------------------

    @staticmethod
    def _granger_causality(
        x: List[float],
        y: List[float],
        order: int = _GRANGER_ORDER,
    ) -> Optional[Tuple[float, float]]:
        """Approximate Granger causality test of X -> Y.

        Compares two autoregressive models:

        * Restricted  --- ``y[t] ~ y[t-1 .. t-order]``
        * Unrestricted --- ``y[t] ~ y[t-1 .. t-order] + x[t-1 .. t-order]``

        Returns the F-statistic and an approximate p-value using the
        incomplete-beta / F-distribution survival function.

        Parameters
        ----------
        x : list[float]
            Potential cause time series.
        y : list[float]
            Potential effect time series.
        order : int
            Number of lag terms in each regression.

        Returns
        -------
        tuple[float, float] or None
            ``(f_statistic, p_value)`` or ``None`` if data are insufficient
            or the regression matrix is singular.
        """
        n = min(len(x), len(y))
        if n < 2 * order + 5:
            return None

        x_arr = list(x[:n])
        y_arr = list(y[:n])

        num_samples = n - order
        y_target: List[float] = []
        y_lags: List[List[float]] = []   # restricted features
        x_lags: List[List[float]] = []   # extra unrestricted features

        for t in range(order, n):
            y_target.append(y_arr[t])
            y_lags.append([y_arr[t - k - 1] for k in range(order)])
            x_lags.append([x_arr[t - k - 1] for k in range(order)])

        # -- Restricted model: Y ~ Y_lagged
        beta_r = _ols(y_lags, y_target)
        if beta_r is None:
            return None

        rss_r = 0.0
        for i in range(num_samples):
            pred = sum(beta_r[k] * y_lags[i][k] for k in range(order))
            rss_r += (y_target[i] - pred) ** 2

        # -- Unrestricted model: Y ~ Y_lagged + X_lagged
        ur_features = [y_lags[i] + x_lags[i] for i in range(num_samples)]
        beta_ur = _ols(ur_features, y_target)
        if beta_ur is None:
            return None

        rss_ur = 0.0
        for i in range(num_samples):
            pred = sum(beta_ur[k] * ur_features[i][k]
                       for k in range(2 * order))
            rss_ur += (y_target[i] - pred) ** 2

        # -- F-statistic
        #   F = ((RSS_r - RSS_ur) / p)  /  (RSS_ur / (n - 2p - 1))
        dof = num_samples - 2 * order - 1
        if dof < 1 or rss_ur < 1e-15:
            return None

        f_stat = ((rss_r - rss_ur) / order) / (rss_ur / dof)
        if f_stat < 0:
            # Unrestricted model should never have larger RSS;
            # numerical noise yields a conservative zero.
            f_stat = 0.0

        p_value = _f_sf(f_stat, order, dof)
        return f_stat, p_value


# ---------------------------------------------------------------------------
# Internal math helpers  (module-level for testability)
# ---------------------------------------------------------------------------


def _ols(
    features: List[List[float]], target: List[float]
) -> Optional[List[float]]:
    """Ordinary least squares via normal equations (closed form).

    Solves ``beta = (X^T X)^-1 X^T y`` where ``X`` includes a bias (intercept)
    column of ones.  Returns ``None`` when the Gram matrix is singular.

    Parameters
    ----------
    features : list[list[float]]
        Shape ``(n_samples, n_features)``.
    target : list[float]
        Shape ``(n_samples,)``.

    Returns
    -------
    list[float] or None
        Coefficient vector length ``n_features + 1`` (bias first).
    """
    n = len(features)
    if n == 0:
        return None

    m = len(features[0])
    size = m + 1  # +1 for bias

    # Build Gram matrix X^T X  and  X^T y
    XtX = [[0.0] * size for _ in range(size)]
    Xty = [0.0] * size

    for i in range(n):
        row = features[i]
        y_val = target[i]

        # bias column (index 0)
        XtX[0][0] += 1.0
        Xty[0] += y_val

        for j in range(m):
            vj = row[j]

            XtX[0][j + 1] += vj
            XtX[j + 1][0] += vj
            Xty[j + 1] += vj * y_val

            for k in range(j, m):
                vk = row[k]
                XtX[j + 1][k + 1] += vj * vk
                if k != j:
                    XtX[k + 1][j + 1] = XtX[j + 1][k + 1]

    return _solve_linear(XtX, Xty)


def _solve_linear(
    A: List[List[float]], b: List[float]
) -> Optional[List[float]]:
    """Solve ``Ax = b`` via Gaussian elimination with partial pivoting.

    Returns ``None`` if *A* is (near-)singular.
    """
    n = len(A)
    # Augmented matrix
    aug = [A[i][:] + [b[i]] for i in range(n)]

    for col in range(n):
        # Partial pivoting
        pivot_row = col
        pivot_val = abs(aug[col][col])
        for row in range(col + 1, n):
            if abs(aug[row][col]) > pivot_val:
                pivot_val = abs(aug[row][col])
                pivot_row = row

        if pivot_val < 1e-12:
            return None

        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        # Eliminate rows below
        pivot = aug[col][col]
        for row in range(col + 1, n):
            factor = aug[row][col] / pivot
            for k in range(col, n + 1):
                aug[row][k] -= factor * aug[col][k]

    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = aug[i][n]
        for j in range(i + 1, n):
            s -= aug[i][j] * x[j]
        if abs(aug[i][i]) < 1e-12:
            return None
        x[i] = s / aug[i][i]

    return x


# ---------------------------------------------------------------------------
# F-distribution survival function  (1 - CDF)
# ---------------------------------------------------------------------------


def _f_sf(f_stat: float, df1: int, df2: int) -> float:
    """Survival function of the F-distribution: ``P(F > f_stat)``.

    Uses the identity relating the F-distribution to the regularised
    incomplete beta function::

        P(F > f) = I_x(df2/2, df1/2)

    where ``x = df2 / (df2 + df1 * f)``.

    Parameters
    ----------
    f_stat : float
        Observed F-statistic (>= 0).
    df1 : int
        Numerator degrees of freedom.
    df2 : int
        Denominator degrees of freedom.

    Returns
    -------
    float
        Approximate p-value in [0, 1].
    """
    if f_stat <= 0.0:
        return 1.0

    x = df2 / (df2 + df1 * f_stat)
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    a = df2 / 2.0
    b = df1 / 2.0

    return _betainc(x, a, b)


def _betainc(x: float, a: float, b: float) -> float:
    """Regularised incomplete beta function ``I_x(a, b)``.

    Implemented via Lentz's continued fraction method (modified Lentz).
    This is the same function used by SciPy's ``scipy.special.betainc``.

    Parameters
    ----------
    x : float
        Upper limit (0 <= x <= 1).
    a : float
        First shape parameter (> 0).
    b : float
        Second shape parameter (> 0).

    Returns
    -------
    float
        ``I_x(a, b)`` in [0, 1].
    """
    # Handle edge cases
    if x < 0.0 or x > 1.0:
        return 0.0
    if x == 0.0 or x == 1.0:
        return x

    # Use symmetry:  I_x(a, b) = 1 - I_{1-x}(b, a)
    # This improves numerical stability when x is large.
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc(1.0 - x, b, a)

    # Pre-compute the normalisation factor via log-gamma
    ln_beta = (
        math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    )
    front = math.exp(a * math.log(x) + b * math.log1p(-x) - ln_beta) / a

    # Lentz continued fraction (modified)
    MAX_ITER = 300
    TINY = 1e-60
    EPS = 1e-14

    # Initialise with first term (m = 0)
    # Using the continued fraction representation from
    #   Numerical Recipes, sec 6.4 "Incomplete Beta Function"
    f = 1.0
    C = 1.0
    D = 1.0 - (a + b) * x / (a + 1.0)
    if abs(D) < TINY:
        D = TINY
    D = 1.0 / D
    f = D

    for m in range(1, MAX_ITER + 1):
        # Even step (m)
        numer = m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m))
        D = 1.0 + numer * D
        if abs(D) < TINY:
            D = TINY
        C = 1.0 + numer / C
        if abs(C) < TINY:
            C = TINY
        D = 1.0 / D
        delta = C * D
        f *= delta

        # Odd step (m)
        numer = -(a + m) * (a + b + m) * x / (
            (a + 2.0 * m) * (a + 2.0 * m + 1.0)
        )
        D = 1.0 + numer * D
        if abs(D) < TINY:
            D = TINY
        C = 1.0 + numer / C
        if abs(C) < TINY:
            C = TINY
        D = 1.0 / D
        delta = C * D
        f *= delta

        if abs(delta - 1.0) < EPS:
            break

    return front * f
