"""
OSS Surface Diagnostic — measures entropy of p_cont distribution, variance of EV,
and sensitivity to input perturbations. Confirms whether OSS is a collapsed manifold.

Metrics
-------
p_cont entropy   : Shannon entropy of p_cont (binned into 20 buckets).
                   Max ≈ log2(20) ≈ 4.32 (totally uncertain).
                   Near 0 → deterministic.
EV variance      : Variance of the expected value across observations.
                   Near zero → model outputs the same thing regardless of input.
Signal entropy   : Entropy of the {-1, 0, +1} signal distribution.
                   Max = log2(3) ≈ 1.58.
p_cont stability : Standard deviation of p_cont over a sliding window of 20 obs.
"""

import math
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

P_CONT_BINS = 20
MAX_P_CONT_ENTROPY = math.log2(P_CONT_BINS)      # ≈ 4.3219
MAX_SIGNAL_ENTROPY = math.log2(3)                 # ≈ 1.585
STABILITY_WINDOW = 20
COLLAPSED_ENTROPY_THRESHOLD = 3.5                 # p_cont entropy near max
COLLAPSED_VARIANCE_THRESHOLD = 0.01               # EV variance near zero
HEALTHY_ENTROPY_LOWER = 1.0
HEALTHY_ENTROPY_UPPER = 3.5

# ---------------------------------------------------------------------------
# Singleton registry
# ---------------------------------------------------------------------------

_instances = {}


def OSSSurfaceDiagnostic(instance_id="default"):
    """Singleton accessor — returns the same _OSSSurfaceDiagnostic for a given id."""
    if instance_id not in _instances:
        _instances[instance_id] = _OSSSurfaceDiagnostic(instance_id)
    return _instances[instance_id]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _shannon_entropy(counts, total):
    """Compute Shannon entropy in bits from a sequence of counts.

    Parameters
    ----------
    counts : list of int
        Number of observations in each bin / category.
    total : int
        Sum of *counts*.

    Returns
    -------
    float
        Entropy in bits. Returns 0.0 when *total* is 0.
    """
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts:
        if c == 0:
            continue
        p = c / total
        entropy -= p * math.log2(p)
    return entropy


def _variance(values):
    """Compute sample variance of a sequence of floats."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / (n - 1)


def _std(values):
    """Compute sample standard deviation of a sequence of floats."""
    n = len(values)
    if n < 2:
        return 0.0
    return math.sqrt(_variance(values))


def _make_histogram_bins(n_bins=20):
    """Return (n_bins + 1) evenly spaced bin edges in [0, 1]."""
    return [i / n_bins for i in range(n_bins + 1)]


def _bin_index(value, edges):
    """Return 0-based bin index for *value* given sorted *edges*.

    Clamps to [0, len(edges)-2] so the last bin includes 1.0.
    """
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return i
    # value == 1.0 lands in the last bin
    return len(edges) - 2


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------

class _OSSSurfaceDiagnostic:
    """Tracks OSS surface outputs per symbol and computes diagnostic metrics."""

    def __init__(self, instance_id="default"):
        self._instance_id = instance_id
        # symbol -> list of dicts with recorded fields
        self._observations = defaultdict(list)
        # symbol -> list of p_cont values for sliding-window stability
        self._p_cont_window = defaultdict(list)
        logger.debug("OSSSurfaceDiagnostic(%r) initialised", instance_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed_observation(self, symbol, p_cont, ev, signal,
                         exec_drift=None, live_drift=None,
                         regime=None, ecdf=None):
        """Record one OSS surface output for *symbol*.

        Parameters
        ----------
        symbol : str
            Instrument / ticker identifier.
        p_cont : float
            Continuation probability in [0, 1].
        ev : float
            Expected value of the trade (could be signed).
        signal : int
            Signal direction: -1, 0, or +1.
        exec_drift : float or None
            Execution drift metric at time of observation.
        live_drift : float or None
            Live drift metric at time of observation.
        regime : str or None
            Current market regime label.
        ecdf : float or None
            ECDF value (optional).
        """
        obs = {
            "p_cont": float(p_cont),
            "ev": float(ev),
            "signal": int(signal),
            "exec_drift": float(exec_drift) if exec_drift is not None else None,
            "live_drift": float(live_drift) if live_drift is not None else None,
            "regime": regime,
            "ecdf": float(ecdf) if ecdf is not None else None,
        }
        self._observations[symbol].append(obs)
        self._p_cont_window[symbol].append(float(p_cont))
        # Trim sliding window
        if len(self._p_cont_window[symbol]) > STABILITY_WINDOW:
            self._p_cont_window[symbol].pop(0)

        logger.debug("feed_observation %s p_cont=%.4f ev=%.4f signal=%d",
                     symbol, p_cont, ev, signal)

    # ------------------------------------------------------------------
    # Per-symbol diagnostic
    # ------------------------------------------------------------------

    def get_diagnostic(self, symbol):
        """Return a dict of diagnostic metrics for *symbol*.

        Returns
        -------
        dict with keys:
            symbol, observation_count,
            p_cont_entropy, ev_variance, signal_entropy,
            p_cont_stability, verdict
        """
        obs_list = self._observations.get(symbol, [])
        n = len(obs_list)

        if n == 0:
            return {
                "symbol": symbol,
                "observation_count": 0,
                "p_cont_entropy": 0.0,
                "ev_variance": 0.0,
                "signal_entropy": 0.0,
                "p_cont_stability": 0.0,
                "verdict": "INSUFFICIENT_DATA",
            }

        # ---- p_cont entropy (binned into P_CONT_BINS buckets) ----
        edges = _make_histogram_bins(P_CONT_BINS)
        bin_counts = [0] * P_CONT_BINS
        for obs in obs_list:
            idx = _bin_index(obs["p_cont"], edges)
            bin_counts[idx] += 1
        p_cont_entropy = _shannon_entropy(bin_counts, n)

        # ---- EV variance ----
        ev_values = [obs["ev"] for obs in obs_list]
        ev_var = _variance(ev_values)

        # ---- Signal entropy ----
        signal_counts = [0, 0, 0]  # -1, 0, +1
        for obs in obs_list:
            s = obs["signal"]
            if s == -1:
                signal_counts[0] += 1
            elif s == 0:
                signal_counts[1] += 1
            elif s == 1:
                signal_counts[2] += 1
        signal_entropy = _shannon_entropy(signal_counts, n)

        # ---- p_cont stability (std over sliding window) ----
        window = self._p_cont_window.get(symbol, [])
        p_cont_stability = _std(window) if len(window) >= 2 else 0.0

        # ---- Per-symbol verdict ----
        verdict = self._classify_symbol(p_cont_entropy, ev_var)

        return {
            "symbol": symbol,
            "observation_count": n,
            "p_cont_entropy": round(p_cont_entropy, 4),
            "ev_variance": round(ev_var, 6),
            "signal_entropy": round(signal_entropy, 4),
            "p_cont_stability": round(p_cont_stability, 6),
            "verdict": verdict,
        }

    # ------------------------------------------------------------------
    # Batch diagnostics
    # ------------------------------------------------------------------

    def get_all_diagnostics(self):
        """Return dict mapping each symbol to its diagnostic report."""
        return {sym: self.get_diagnostic(sym) for sym in self._observations}

    # ------------------------------------------------------------------
    # Global verdict
    # ------------------------------------------------------------------

    def get_global_verdict(self):
        """Return an aggregate verdict across all tracked symbols.

        Returns
        -------
        str
            ``"COLLAPSED"`` — every symbol with sufficient data is collapsed.
            ``"PARTIALLY_COLLAPSED"`` — at least one collapsed, others healthy.
            ``"HEALTHY"`` — no symbols are collapsed.
            ``"INSUFFICIENT_DATA"`` — no symbols have enough observations.
        """
        diagnostics = self.get_all_diagnostics()
        if not diagnostics:
            return "INSUFFICIENT_DATA"

        collapsed = 0
        healthy = 0
        insufficient = 0

        for diag in diagnostics.values():
            v = diag["verdict"]
            if v == "COLLAPSED":
                collapsed += 1
            elif v == "HEALTHY":
                healthy += 1
            else:
                insufficient += 1

        if collapsed > 0 and healthy == 0:
            return "COLLAPSED"
        elif collapsed > 0 and healthy > 0:
            return "PARTIALLY_COLLAPSED"
        elif healthy > 0:
            return "HEALTHY"
        else:
            return "INSUFFICIENT_DATA"

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self):
        """Clear all observations across all symbols."""
        self._observations.clear()
        self._p_cont_window.clear()
        logger.info("OSSSurfaceDiagnostic(%r) reset", self._instance_id)

    # ------------------------------------------------------------------
    # Internal classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_symbol(p_cont_entropy, ev_variance):
        """Return verdict string for a single symbol given its metrics.

        Parameters
        ----------
        p_cont_entropy : float
            Shannon entropy of the binned p_cont distribution.
        ev_variance : float
            Variance of the expected value.

        Returns
        -------
        str
            ``"COLLAPSED"`` if p_cont entropy is near max AND EV variance is
            near zero (manifold collapse). ``"HEALTHY"`` if entropy is in a
            mid-range and variance is non-zero.
        """
        # Collapse detection: high entropy + near-zero variance
        if (p_cont_entropy >= COLLAPSED_ENTROPY_THRESHOLD
                and ev_variance <= COLLAPSED_VARIANCE_THRESHOLD):
            return "COLLAPSED"

        # Healthy detection: mid-range entropy + non-zero variance
        if (HEALTHY_ENTROPY_LOWER <= p_cont_entropy <= HEALTHY_ENTROPY_UPPER
                and ev_variance > COLLAPSED_VARIANCE_THRESHOLD):
            return "HEALTHY"

        # Borderline: some diagnostic signal but not clearly one or the other
        if p_cont_entropy >= COLLAPSED_ENTROPY_THRESHOLD:
            return "HIGH_ENTROPY"
        if ev_variance <= COLLAPSED_VARIANCE_THRESHOLD:
            return "LOW_VARIANCE"
        return "BORDERLINE"


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    diag = OSSSurfaceDiagnostic()

    print("=" * 60)
    print("OSS Surface Diagnostic — Self Test")
    print("=" * 60)

    # ---- Scenario 1: COLLAPSED ----
    # p_cont uniformly random → high entropy; EV nearly constant → low variance
    random.seed(1)
    for _ in range(200):
        p_cont = random.random()            # uniform [0,1] → max entropy
        ev = 0.01 + random.gauss(0, 0.001)  # tiny variance
        signal = random.choice([-1, 0, 1])
        diag.feed_observation(
            "COLLAPSED_SCE", p_cont, ev, signal,
            exec_drift=0.02, live_drift=random.random(),
            regime="regime_a", ecdf=0.5,
        )

    # ---- Scenario 2: HEALTHY ----
    # p_cont bimodal (peaks around 0.2 and 0.8) → mid entropy;
    # EV varies significantly
    random.seed(2)
    for _ in range(200):
        # Bimodal p_cont
        if random.random() < 0.5:
            p_cont = 0.2 + random.gauss(0, 0.05)
        else:
            p_cont = 0.8 + random.gauss(0, 0.05)
        p_cont = max(0.0, min(1.0, p_cont))
        # EV with meaningful variance
        ev = random.gauss(0.5, 0.3)
        signal = 1 if p_cont > 0.5 else -1
        diag.feed_observation(
            "HEALTHY_SCE", p_cont, ev, signal,
            exec_drift=0.1, live_drift=0.05,
            regime="regime_b", ecdf=0.7,
        )

    # ---- Scenario 3: BORDERLINE / partially collapsed ----
    # p_cont clustered around 0.5 → low-ish entropy; EV has some variance
    random.seed(3)
    for _ in range(200):
        p_cont = 0.5 + random.gauss(0, 0.02)  # tight cluster
        p_cont = max(0.0, min(1.0, p_cont))
        ev = random.gauss(0.1, 0.05)           # moderate variance
        signal = 0
        diag.feed_observation(
            "BORDERLINE_SCE", p_cont, ev, signal,
        )

    # ---- Scenario 4: deterministic (entropy = 0) ----
    for _ in range(50):
        diag.feed_observation(
            "DETERMINISTIC_SCE", 0.0, 0.0, 0,
        )

    # Print per-symbol diagnostics
    for sym in ["COLLAPSED_SCE", "HEALTHY_SCE", "BORDERLINE_SCE", "DETERMINISTIC_SCE"]:
        report = diag.get_diagnostic(sym)
        print(f"\n--- {sym} ---")
        for k, v in report.items():
            print(f"  {k:25s} = {v}")

    # Print global verdict
    print("\n" + "=" * 60)
    print("GLOBAL VERDICT")
    print("=" * 60)
    print(f"  {diag.get_global_verdict()}")

    # Print all diagnostics
    print("\n" + "=" * 60)
    print("ALL DIAGNOSTICS")
    print("=" * 60)
    all_d = diag.get_all_diagnostics()
    for sym, d in all_d.items():
        print(f"  {sym:20s} → {d['verdict']:20s}  (obs={d['observation_count']})")

    # Test reset
    print("\n" + "=" * 60)
    print("RESET TEST")
    print("=" * 60)
    diag.reset()
    print(f"  After reset, global verdict: {diag.get_global_verdict()}")
    print(f"  Observations for HEALTHY_SCE: {diag.get_diagnostic('HEALTHY_SCE')['observation_count']}")
