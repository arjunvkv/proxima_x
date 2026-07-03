"""
Decision Surface Visualizer — maps input→p_cont mapping to answer
"is OSS a plateau surface?" Computes sensitivity gradients, flat region
detection, and plateau classification.

Input space dimensions
---------------------
- ecdf
- drift
- spread
- volatility
- entropy

Each dimension is binned into N equal-width buckets to create a
discretised grid.  For each cell (input combination) the mean p_cont
is tracked, enabling plateau detection and sensitivity analysis.
"""

import math
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_BINS = 10                       # Buckets per input dimension
PLATEAU_ZONE_LOW = 0.45           # Lower bound of plateau p_cont range
PLATEAU_ZONE_HIGH = 0.55          # Upper bound of plateau p_cont range
PLATEAU_THRESHOLD = 0.80          # >80 % cells in plateau zone → PLATEAU
HEALTHY_THRESHOLD = 0.30          # <30 % cells in plateau zone → HEALTHY
FLAT_ZONE_LOW = 0.49              # Narrow "don't know" zone lower bound
FLAT_ZONE_HIGH = 0.51             # Narrow "don't know" zone upper bound
DEAD_SENSITIVITY_THRESHOLD = 0.01  # Dimensions below this are "dead"

INPUT_DIMENSIONS = (
    "ecdf",
    "drift",
    "spread",
    "volatility",
    "entropy",
)

# ---------------------------------------------------------------------------
# Singleton registry
# ---------------------------------------------------------------------------

_instances = {}


def DecisionSurfaceVisualizer(instance_id="default"):
    """Singleton accessor — returns the same _DecisionSurfaceVisualizer
    for a given *instance_id*.

    Parameters
    ----------
    instance_id : str
        Logical identifier for the visualizer instance (default ``"default"``).

    Returns
    -------
    _DecisionSurfaceVisualizer
    """
    if instance_id not in _instances:
        _instances[instance_id] = _DecisionSurfaceVisualizer(instance_id)
    return _instances[instance_id]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _bin_index(value, min_val, max_val, n_bins):
    """Return 0-based bin index for *value* given observed *min_val* and
    *max_val*.

    When *min_val* == *max_val* all values fall into bin 0.
    The result is clamped to ``[0, n_bins - 1]``.
    """
    if min_val == max_val:
        return 0
    span = max_val - min_val
    idx = int((value - min_val) / span * n_bins)
    return max(0, min(n_bins - 1, idx))


def _in_closed_interval(value, low, high):
    """Return True if *value* is in ``[low, high]``."""
    return low <= value <= high


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------


class _DecisionSurfaceVisualizer:
    """Maps input→p_cont mapping for OSS plateau detection.

    Parameters
    ----------
    instance_id : str
        Logical instance identifier (used for logging).
    """

    def __init__(self, instance_id="default"):
        self._instance_id = instance_id
        # symbol -> list of (input_state_dict, p_cont) tuples
        self._observations = defaultdict(list)
        # symbol -> {dim_name: (min_observed, max_observed)}
        self._min_max = defaultdict(
            lambda: {d: (float("inf"), float("-inf")) for d in INPUT_DIMENSIONS}
        )
        logger.debug("DecisionSurfaceVisualizer(%r) initialised", instance_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed_observation(self, symbol, input_state, p_cont):
        """Record one input→p_cont mapping.

        Parameters
        ----------
        symbol : str
            Instrument / ticker identifier.
        input_state : dict
            Feature vector mapping dimension name (ecdf, drift, spread,
            volatility, entropy) to a float value.
        p_cont : float
            Continuation probability in ``[0, 1]``.
        """
        # Ensure all dimensions are present (fill missing with 0.0)
        sanitised = {}
        for d in INPUT_DIMENSIONS:
            val = input_state.get(d)
            if val is None:
                logger.warning(
                    "Missing input dimension '%s' in feed_observation for %s — "
                    "defaulting to 0.0",
                    d,
                    symbol,
                )
                sanitised[d] = 0.0
            else:
                sanitised[d] = float(val)

        p_val = float(p_cont)
        self._observations[symbol].append((sanitised, p_val))

        # Update per-dimension min / max
        mm = self._min_max[symbol]
        for d in INPUT_DIMENSIONS:
            lo, hi = mm[d]
            v = sanitised[d]
            mm[d] = (min(lo, v), max(hi, v))

        logger.debug("feed_observation %s p_cont=%.4f state=%s", symbol, p_val, sanitised)

    def analyze_surface(self, symbol):
        """Return a dict of decision-surface metrics for *symbol*.

        Parameters
        ----------
        symbol : str
            Instrument / ticker identifier.

        Returns
        -------
        dict with keys:
            symbol              — the symbol string
            total_observations  — number of observations recorded
            coverage            — fraction of discretised grid cells populated
            is_plateau          — True when *verdict* is ``"PLATEAU"``
            plateau_pct         — fraction of cells whose mean p_cont is in
                                  ``[0.45, 0.55]``
            sensitivity         — dict ``{dim_sensitivity: float}`` per dimension
            dead_dimensions     — list of dimension names with sensitivity < 0.01
            alive_dimensions    — list of dimension names with sensitivity >= 0.01
            verdict             — ``"PLATEAU"`` | ``"PARTIAL"`` | ``"HEALTHY"`` |
                                  ``"INSUFFICIENT_DATA"``
            flat_region_count   — number of cells with mean p_cont in
                                  ``[0.49, 0.51]``
            flat_region_pct     — fraction of cells in ``[0.49, 0.51]``
        """
        obs_list = self._observations.get(symbol, [])
        n = len(obs_list)
        empty_result = {
            "symbol": symbol,
            "total_observations": 0,
            "coverage": 0.0,
            "is_plateau": False,
            "plateau_pct": 0.0,
            "sensitivity": {f"{d}_sensitivity": 0.0 for d in INPUT_DIMENSIONS},
            "dead_dimensions": list(INPUT_DIMENSIONS),
            "alive_dimensions": [],
            "verdict": "INSUFFICIENT_DATA",
            "flat_region_count": 0,
            "flat_region_pct": 0.0,
        }

        if n == 0:
            return empty_result

        # ---- Build the discretised grid ----
        min_max = self._min_max[symbol]
        grid = defaultdict(list)  # bin_tuple -> [p_cont, ...]

        for input_state, p_cont in obs_list:
            bins = []
            for d in INPUT_DIMENSIONS:
                lo, hi = min_max[d]
                idx = _bin_index(input_state[d], lo, hi, N_BINS)
                bins.append(idx)
            grid[tuple(bins)].append(p_cont)

        total_cells = len(grid)
        if total_cells == 0:
            return empty_result

        # Mean p_cont per cell
        cell_means = {bins: sum(vals) / len(vals) for bins, vals in grid.items()}

        # ---- Plateau detection ----
        plateau_cells = sum(
            1 for m in cell_means.values()
            if _in_closed_interval(m, PLATEAU_ZONE_LOW, PLATEAU_ZONE_HIGH)
        )
        plateau_pct = plateau_cells / total_cells

        # ---- Flat (tight "don't know") regions ----
        flat_cells = sum(
            1 for m in cell_means.values()
            if _in_closed_interval(m, FLAT_ZONE_LOW, FLAT_ZONE_HIGH)
        )
        flat_region_pct = flat_cells / total_cells

        # ---- Coverage ----
        possible_cells = N_BINS ** len(INPUT_DIMENSIONS)
        coverage = total_cells / possible_cells

        # ---- Sensitivity gradients ----
        sensitivity = self._compute_sensitivity(cell_means, min_max)

        # ---- Dead / alive dimensions ----
        dead = []
        alive = []
        for d in INPUT_DIMENSIONS:
            sens_key = f"{d}_sensitivity"
            if sensitivity[sens_key] < DEAD_SENSITIVITY_THRESHOLD:
                dead.append(d)
            else:
                alive.append(d)

        # ---- Verdict ----
        if plateau_pct > PLATEAU_THRESHOLD:
            verdict = "PLATEAU"
        elif plateau_pct < HEALTHY_THRESHOLD:
            verdict = "HEALTHY"
        else:
            verdict = "PARTIAL"

        return {
            "symbol": symbol,
            "total_observations": n,
            "coverage": round(coverage, 6),
            "is_plateau": verdict == "PLATEAU",
            "plateau_pct": round(plateau_pct, 6),
            "sensitivity": {k: round(v, 6) for k, v in sensitivity.items()},
            "dead_dimensions": dead,
            "alive_dimensions": alive,
            "verdict": verdict,
            "flat_region_count": flat_cells,
            "flat_region_pct": round(flat_region_pct, 6),
        }

    def _compute_sensitivity(self, cell_means, min_max):
        """Return a dict ``{dim_sensitivity: float}`` — numerical partial
        derivative of p_cont w.r.t. each input dimension.

        For each dimension *d*, observations are grouped by the other four
        dimensions (held constant).  Within each group, the gradient
        ``|Δp_cont| / |Δinput|`` is computed between consecutive bin values
        and averaged.
        """
        sensitivity = {f"{d}_sensitivity": 0.0 for d in INPUT_DIMENSIONS}

        for d_idx, d in enumerate(INPUT_DIMENSIONS):
            lo, hi = min_max.get(d, (0.0, 1.0))
            if hi <= lo:  # no variation → gradient is zero
                continue
            bin_width = (hi - lo) / N_BINS

            # Group cells by the tuple of all other dimensions
            groups = defaultdict(list)  # other_bins -> [(d_bin, mean_p_cont)]
            for bins, mean_p in cell_means.items():
                other = tuple(bins[k] for k in range(len(INPUT_DIMENSIONS)) if k != d_idx)
                groups[other].append((bins[d_idx], mean_p))

            gradients = []
            for entries in groups.values():
                if len(entries) < 2:
                    continue
                # Sort by d-bin index so we compute slopes along the axis
                entries.sort(key=lambda x: x[0])
                for i in range(len(entries) - 1):
                    bin_i, mean_i = entries[i]
                    bin_j, mean_j = entries[i + 1]
                    d_diff = abs(bin_j - bin_i)
                    if d_diff == 0:
                        continue
                    input_diff = d_diff * bin_width
                    grad = abs(mean_i - mean_j) / input_diff
                    gradients.append(grad)

            if gradients:
                sensitivity[f"{d}_sensitivity"] = sum(gradients) / len(gradients)

        return sensitivity

    # ------------------------------------------------------------------
    # Batch & reset
    # ------------------------------------------------------------------

    def get_all_analyses(self):
        """Return dict mapping each symbol to its surface analysis."""
        return {sym: self.analyze_surface(sym) for sym in self._observations}

    def reset(self):
        """Clear all observations and min/max ranges across all symbols."""
        self._observations.clear()
        self._min_max.clear()
        logger.info("DecisionSurfaceVisualizer(%r) reset", self._instance_id)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _seed_state(rng, ecdf_range=(0.0, 1.0), drift_range=(-1.0, 1.0),
                spread_range=(0.0, 0.1), vol_range=(0.0, 0.5),
                ent_range=(0.0, 1.0)):
    """Generate a random input-state dict."""
    return {
        "ecdf": rng.uniform(*ecdf_range),
        "drift": rng.uniform(*drift_range),
        "spread": rng.uniform(*spread_range),
        "volatility": rng.uniform(*vol_range),
        "entropy": rng.uniform(*ent_range),
    }


if __name__ == "__main__":
    import random

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 70)
    print("Decision Surface Visualizer — Self Test")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Scenario 1 — PLATEAU: all p_cont ≈ 0.5 regardless of input
    # ------------------------------------------------------------------
    print("\n>>> Scenario 1: PLATEAU (p_cont ≈ 0.5 for all inputs)")
    viz1 = DecisionSurfaceVisualizer("plateau_test")
    rng = random.Random(42)
    for _ in range(500):
        state = _seed_state(rng)
        p_cont = 0.50 + rng.gauss(0, 0.02)  # tight cluster around 0.5
        p_cont = max(0.0, min(1.0, p_cont))
        viz1.feed_observation("PLATEAU_SYM", state, p_cont)

    r1 = viz1.analyze_surface("PLATEAU_SYM")
    print(f"  Verdict:            {r1['verdict']}")
    print(f"  is_plateau:         {r1['is_plateau']}")
    print(f"  plateau_pct:        {r1['plateau_pct']:.4f}")
    print(f"  total_observations: {r1['total_observations']}")
    print(f"  coverage:           {r1['coverage']:.6f}")
    print(f"  dead_dimensions:    {r1['dead_dimensions']}")
    print(f"  alive_dimensions:   {r1['alive_dimensions']}")
    for dim in INPUT_DIMENSIONS:
        print(f"    {dim}_sensitivity:  {r1['sensitivity'][f'{dim}_sensitivity']:.6f}")
    assert r1["verdict"] == "PLATEAU", f"Expected PLATEAU, got {r1['verdict']}"
    assert r1["is_plateau"] is True
    assert r1["plateau_pct"] > 0.80, f"Expected plateau_pct > 0.80, got {r1['plateau_pct']}"
    print("  ✓ PLATEAU assertions passed")

    # ------------------------------------------------------------------
    # Scenario 2 — HEALTHY: p_cont varies linearly with ecdf
    # ------------------------------------------------------------------
    print("\n>>> Scenario 2: HEALTHY (p_cont varies with ecdf)")
    viz2 = DecisionSurfaceVisualizer("healthy_test")
    rng = random.Random(123)
    for _ in range(500):
        state = _seed_state(rng)
        # p_cont strongly driven by ecdf, small noise from other dims
        p_cont = 0.2 + 0.6 * state["ecdf"] + rng.gauss(0, 0.03)
        p_cont = max(0.0, min(1.0, p_cont))
        viz2.feed_observation("HEALTHY_SYM", state, p_cont)

    r2 = viz2.analyze_surface("HEALTHY_SYM")
    print(f"  Verdict:            {r2['verdict']}")
    print(f"  is_plateau:         {r2['is_plateau']}")
    print(f"  plateau_pct:        {r2['plateau_pct']:.4f}")
    print(f"  total_observations: {r2['total_observations']}")
    print(f"  coverage:           {r2['coverage']:.6f}")
    print(f"  dead_dimensions:    {r2['dead_dimensions']}")
    print(f"  alive_dimensions:   {r2['alive_dimensions']}")
    for dim in INPUT_DIMENSIONS:
        print(f"    {dim}_sensitivity:  {r2['sensitivity'][f'{dim}_sensitivity']:.6f}")
    assert r2["verdict"] == "HEALTHY", f"Expected HEALTHY, got {r2['verdict']}"
    assert r2["is_plateau"] is False
    assert r2["plateau_pct"] < 0.30, f"Expected plateau_pct < 0.30, got {r2['plateau_pct']}"
    print("  ✓ HEALTHY assertions passed")

    # ------------------------------------------------------------------
    # Scenario 3 — PARTIAL: moderate p_cont spread, some flat-ish areas
    # ------------------------------------------------------------------
    print("\n>>> Scenario 3: PARTIAL (mixed behaviour)")
    viz3 = DecisionSurfaceVisualizer("partial_test")
    rng = random.Random(7)
    for _ in range(500):
        state = _seed_state(rng)
        # Half the time p_cont ≈ 0.5, half driven by drift
        if rng.random() < 0.5:
            p_cont = 0.50 + rng.gauss(0, 0.02)
        else:
            p_cont = 0.5 + 0.3 * state["drift"] + rng.gauss(0, 0.03)
        p_cont = max(0.0, min(1.0, p_cont))
        viz3.feed_observation("PARTIAL_SYM", state, p_cont)

    r3 = viz3.analyze_surface("PARTIAL_SYM")
    print(f"  Verdict:            {r3['verdict']}")
    print(f"  is_plateau:         {r3['is_plateau']}")
    print(f"  plateau_pct:        {r3['plateau_pct']:.4f}")
    print(f"  total_observations: {r3['total_observations']}")
    print(f"  coverage:           {r3['coverage']:.6f}")
    print(f"  dead_dimensions:    {r3['dead_dimensions']}")
    print(f"  alive_dimensions:   {r3['alive_dimensions']}")
    for dim in INPUT_DIMENSIONS:
        print(f"    {dim}_sensitivity:  {r3['sensitivity'][f'{dim}_sensitivity']:.6f}")
    assert r3["verdict"] == "PARTIAL", f"Expected PARTIAL, got {r3['verdict']}"
    assert r3["is_plateau"] is False
    assert 0.30 <= r3["plateau_pct"] <= 0.80, \
        f"Expected plateau_pct in [0.30, 0.80], got {r3['plateau_pct']}"
    print("  ✓ PARTIAL assertions passed")

    # ------------------------------------------------------------------
    # Scenario 4 — Empty / insufficient data
    # ------------------------------------------------------------------
    print("\n>>> Scenario 4: INSUFFICIENT_DATA (no observations)")
    viz4 = DecisionSurfaceVisualizer("empty_test")
    r4 = viz4.analyze_surface("GHOST_SYM")
    print(f"  Verdict:            {r4['verdict']}")
    print(f"  total_observations: {r4['total_observations']}")
    assert r4["verdict"] == "INSUFFICIENT_DATA"
    assert r4["total_observations"] == 0
    print("  ✓ INSUFFICIENT_DATA assertion passed")

    # ------------------------------------------------------------------
    # Scenario 5 — Reset & verify emptiness
    # ------------------------------------------------------------------
    print("\n>>> Scenario 5: Reset test")
    viz1.reset()
    r5 = viz1.analyze_surface("PLATEAU_SYM")
    print(f"  After reset, verdict: {r5['verdict']}")
    print(f"  total_observations:   {r5['total_observations']}")
    assert r5["verdict"] == "INSUFFICIENT_DATA"
    assert r5["total_observations"] == 0
    print("  ✓ Reset assertion passed")

    # ------------------------------------------------------------------
    # Scenario 6 — Singleton identity
    # ------------------------------------------------------------------
    print("\n>>> Scenario 6: Singleton identity")
    viz1_again = DecisionSurfaceVisualizer("plateau_test")
    assert viz1_again is viz1, "Singleton should return same object"
    print("  ✓ Singleton identity passed")
    viz_default = DecisionSurfaceVisualizer()
    viz_default2 = DecisionSurfaceVisualizer("default")
    assert viz_default is viz_default2
    print("  ✓ Default singleton identity passed")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ALL SELF-TEST ASSERTIONS PASSED ✓")
    print("=" * 70)
