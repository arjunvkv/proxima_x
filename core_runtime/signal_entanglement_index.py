"""
Signal Entanglement Index — measures how strongly OSS and ALT signals
collapse into a single decision manifold via normalized mutual information.

Entanglement quantifies how much knowing one signal tells you about the
other, computed from Shannon entropy of the discrete {-1, 0, +1} signal
distributions.
"""

import logging
import math
from collections import Counter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton registry
# ---------------------------------------------------------------------------

_instances = {}


def SignalEntanglementIndex(instance_id="default"):
    """Accessor / singleton factory for ``_SignalEntanglementIndex``."""
    if instance_id not in _instances:
        _instances[instance_id] = _SignalEntanglementIndex(instance_id)
    return _instances[instance_id]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_SIGNALS = {-1, 0, 1}


def _validate_signal(value, name=""):
    if value not in _VALID_SIGNALS:
        raise ValueError(
            f"{name} signal must be one of {{-1, 0, 1}}, got {value!r}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entropy(counter):
    """Shannon entropy H(X) = -Σ p(x) log₂ p(x)  (bits).

    Parameters
    ----------
    counter : collections.Counter
        Counts of discrete outcomes.

    Returns
    -------
    float
        Entropy in bits.  Returns 0.0 when the counter is empty.
    """
    total = sum(counter.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _verdict(ei):
    """Classify entanglement index into a human-readable label.

    Thresholds
    ----------
    EI < 0.1   → INDEPENDENT
    EI < 0.3   → LOW
    EI < 0.6   → MODERATE
    EI < 0.9   → HIGH
    EI ≥ 0.9   → FULLY_ENTANGLED
    """
    if ei < 0.1:
        return "INDEPENDENT"
    if ei < 0.3:
        return "LOW"
    if ei < 0.6:
        return "MODERATE"
    if ei < 0.9:
        return "HIGH"
    return "FULLY_ENTANGLED"


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------

class _SignalEntanglementIndex:
    """Tracks (OSS, ALT) signal pairs per symbol and computes entanglement.

    Parameters
    ----------
    instance_id : str
        Arbitrary label for this instance (used in logging).
    """

    def __init__(self, instance_id="default"):
        self._instance_id = instance_id

        # Per-symbol counters — chosen over storing raw lists for memory
        # efficiency when dealing with many observations.
        self._oss_counts = {}       # symbol -> Counter{-1, 0, +1}
        self._alt_counts = {}       # symbol -> Counter{-1, 0, +1}
        self._joint_counts = {}     # symbol -> Counter{(oss, alt)}
        self._observation_count = {}  # symbol -> int

        logger.info("SignalEntanglementIndex(%s) initialised", instance_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed_observation(self, symbol, oss_signal, alt_signal):
        """Record one (OSS, ALT) signal pair for *symbol*.

        Parameters
        ----------
        symbol : hashable
            Ticker or identifier for the instrument.
        oss_signal : int
            OSS signal value: -1, 0, or +1.
        alt_signal : int
            ALT signal value: -1, 0, or +1.

        Raises
        ------
        ValueError
            If either signal is not in {-1, 0, 1}.
        """
        _validate_signal(oss_signal, "OSS")
        _validate_signal(alt_signal, "ALT")

        # Lazily initialise per-symbol containers
        if symbol not in self._oss_counts:
            self._oss_counts[symbol] = Counter()
            self._alt_counts[symbol] = Counter()
            self._joint_counts[symbol] = Counter()
            self._observation_count[symbol] = 0

        self._oss_counts[symbol][oss_signal] += 1
        self._alt_counts[symbol][alt_signal] += 1
        self._joint_counts[symbol][(oss_signal, alt_signal)] += 1
        self._observation_count[symbol] += 1

    def compute_entanglement(self, symbol):
        """Return a detailed entanglement report for *symbol*.

        Parameters
        ----------
        symbol : hashable
            Ticker or identifier for the instrument.

        Returns
        -------
        dict
            Keys::

                symbol                         str
                observations                   int
                oss_entropy                    float  (bits)
                alt_entropy                    float  (bits)
                joint_entropy                  float  (bits)
                conditional_entropy_oss_given_alt  float  (bits)
                conditional_entropy_alt_given_oss  float  (bits)
                mutual_information             float  (bits)
                entanglement_index             float  (0.0 – 1.0)
                verdict                        str

        Raises
        ------
        KeyError
            If *symbol* has no observations.
        """
        if symbol not in self._oss_counts:
            raise KeyError(
                f"Symbol {symbol!r} has no observations in "
                f"SignalEntanglementIndex({self._instance_id!r})"
            )

        n = self._observation_count[symbol]
        h_oss = _entropy(self._oss_counts[symbol])
        h_alt = _entropy(self._alt_counts[symbol])
        h_joint = _entropy(self._joint_counts[symbol])

        # Conditional entropies
        h_oss_given_alt = h_joint - h_alt
        h_alt_given_oss = h_joint - h_oss

        # Mutual information
        mi = h_oss + h_alt - h_joint

        # Normalised entanglement index
        min_h = min(h_oss, h_alt)
        if min_h == 0.0:
            # If either marginal has zero entropy the signal is perfectly
            # deterministic → fully entangled.
            ei = 1.0
        else:
            ei = mi / min_h

        # Clamp to [0, 1] to suppress floating-point noise
        ei = max(0.0, min(1.0, ei))

        return {
            "symbol": symbol,
            "observations": n,
            "oss_entropy": round(h_oss, 6),
            "alt_entropy": round(h_alt, 6),
            "joint_entropy": round(h_joint, 6),
            "conditional_entropy_oss_given_alt": round(h_oss_given_alt, 6),
            "conditional_entropy_alt_given_oss": round(h_alt_given_oss, 6),
            "mutual_information": round(mi, 6),
            "entanglement_index": round(ei, 6),
            "verdict": _verdict(ei),
        }

    def get_global_entanglement(self):
        """Aggregate entanglement across all tracked symbols.

        Returns
        -------
        dict
            Keys::

                symbols                    int  (number of symbols)
                total_observations         int
                average_entanglement       float
                average_mutual_information  float
                verdict                    str
                symbol_results             list[dict]
        """
        symbols = list(self._oss_counts.keys())
        if not symbols:
            return {
                "symbols": 0,
                "total_observations": 0,
                "average_entanglement": 0.0,
                "average_mutual_information": 0.0,
                "verdict": "NONE",
                "symbol_results": [],
            }

        results = []
        for sym in symbols:
            try:
                results.append(self.compute_entanglement(sym))
            except KeyError:
                continue

        total_obs = sum(r["observations"] for r in results)
        n_sym = len(results)
        avg_ei = sum(r["entanglement_index"] for r in results) / n_sym if n_sym else 0.0
        avg_mi = sum(r["mutual_information"] for r in results) / n_sym if n_sym else 0.0

        return {
            "symbols": n_sym,
            "total_observations": total_obs,
            "average_entanglement": round(avg_ei, 6),
            "average_mutual_information": round(avg_mi, 6),
            "verdict": _verdict(avg_ei),
            "symbol_results": results,
        }

    def reset(self):
        """Clear all stored signal data for every symbol."""
        self._oss_counts.clear()
        self._alt_counts.clear()
        self._joint_counts.clear()
        self._observation_count.clear()
        logger.info("SignalEntanglementIndex(%s) reset", self._instance_id)


# ===================================================================
# Self-test
# ===================================================================

def _selftest():
    """Run a quick sanity check to verify the module works correctly."""
    import random

    idx = SignalEntanglementIndex("selftest")

    # ---- 1. Independent signals → EI ≈ 0 ----
    for _ in range(2000):
        oss = random.choice([-1, 0, 1])
        alt = random.choice([-1, 0, 1])
        idx.feed_observation("INDEP", oss, alt)

    r_indep = idx.compute_entanglement("INDEP")
    assert r_indep["entanglement_index"] < 0.15, (
        f"Expected EI near 0 for independent signals, "
        f"got {r_indep['entanglement_index']}"
    )
    print(f"[SELFTEST] Independent signals:  EI={r_indep['entanglement_index']:.4f}  "
          f"{r_indep['verdict']}  \u2713")

    # ---- 2. Identical signals → EI ≈ 1 ----
    for _ in range(2000):
        v = random.choice([-1, 0, 1])
        idx.feed_observation("IDEN", v, v)

    r_iden = idx.compute_entanglement("IDEN")
    assert r_iden["entanglement_index"] > 0.85, (
        f"Expected EI near 1 for identical signals, "
        f"got {r_iden['entanglement_index']}"
    )
    print(f"[SELFTEST] Identical signals:    EI={r_iden['entanglement_index']:.4f}  "
          f"{r_iden['verdict']}  \u2713")

    # ---- 3. Partially correlated → EI ≈ 0.5 ----
    # alt = oss 50 % of the time, random otherwise
    for _ in range(2000):
        oss = random.choice([-1, 0, 1])
        if random.random() < 0.5:
            alt = oss
        else:
            alt = random.choice([-1, 0, 1])
        idx.feed_observation("CORR", oss, alt)

    r_corr = idx.compute_entanglement("CORR")
    assert 0.2 < r_corr["entanglement_index"] < 0.8, (
        f"Expected EI ~0.5 for partially correlated signals, "
        f"got {r_corr['entanglement_index']}"
    )
    print(f"[SELFTEST] Partial correlation:  EI={r_corr['entanglement_index']:.4f}  "
          f"{r_corr['verdict']}  \u2713")

    # ---- 4. Global entanglement ----
    global_r = idx.get_global_entanglement()
    assert global_r["symbols"] == 3, f"Expected 3 symbols, got {global_r['symbols']}"
    print(f"[SELFTEST] Global entanglement:  avg EI={global_r['average_entanglement']:.4f}  "
          f"over {global_r['symbols']} symbols  \u2713")

    # ---- 5. Reset ----
    idx.reset()
    assert not idx._observation_count, "Expected empty state after reset"
    print("[SELFTEST] Reset OK  \u2713")

    print("\nAll self-tests passed.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _selftest()
