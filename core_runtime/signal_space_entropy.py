"""
Signal Space Entropy — maps the distribution of ALL signal sources in the system
and computes entropy per source. Detects signal space collapse — the condition
where no signal source produces meaningful directional information.
"""

import math
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton registry
# ---------------------------------------------------------------------------

_instances = {}


def SignalSpaceEntropy(instance_id="default"):
    """Singleton accessor — returns the same _SignalSpaceEntropy for a given id."""
    if instance_id not in _instances:
        _instances[instance_id] = _SignalSpaceEntropy(instance_id)
    return _instances[instance_id]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _signal_class(value):
    """Map an arbitrary signal value to one of {-1, 0, +1}.

    Floats whose absolute value is below 0.05 are treated as flat (0).
    None and non-numeric inputs also resolve to 0.
    """
    if value is None:
        return 0
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0
    if abs(v) < 0.05:
        return 0
    return 1 if v > 0 else -1


def _entropy(counts):
    """Compute Shannon entropy (base 2) from a dict of label -> count.

    Parameters
    ----------
    counts : dict
        Mapping from category label (int) to count.

    Returns
    -------
    float
        Entropy in bits.  0.0 when total count is 0.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            ent -= p * math.log2(p)
    return ent


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------


class _SignalSpaceEntropy:
    """Tracks signal distributions from all known sources and computes entropy.

    Known source categories recognised by the system:

    ===========  ============================================================
    Source       Description
    ===========  ============================================================
    OSS          Outcome Surface Signal — p_cont, signal, ev
    ALT          Alternative / control — signal, confidence
    SHADOW       Shadow model signals — signal, confidence
    ECDF         ECDF-based signals — signal, ecdf_value
    INJECTED     Injected signals — signal, source_mode
    RESEARCH     Research model signals — r_pc, r_ph, r_pt
    ===========  ============================================================
    """

    KNOWN_SOURCES = {
        "OSS",
        "ALT",
        "SHADOW",
        "ECDF",
        "INJECTED",
        "RESEARCH",
    }

    def __init__(self, instance_id="default"):
        self._instance_id = instance_id
        # source_name -> list of signal classes (-1, 0, +1)
        self._observations = defaultdict(list)
        logger.debug("SignalSpaceEntropy(%r) initialised", instance_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed_observation(self, source_name, signal_value, confidence=None, metadata=None):
        """Record a signal observation from *source_name*.

        Parameters
        ----------
        source_name : str
            One of the known sources (OSS, ALT, SHADOW, ECDF, INJECTED,
            RESEARCH) or any custom identifier.
        signal_value : int or float
            The raw signal value.  Will be normalised to {-1, 0, +1}
            via :func:`_signal_class`.
        confidence : float or None
            Optional confidence score associated with the signal.
        metadata : dict or None
            Optional additional context (e.g. ``ecdf_value``, ``source_mode``).
        """
        cls = _signal_class(signal_value)
        self._observations[source_name].append(cls)
        logger.debug(
            "feed_observation source=%s raw=%s class=%s conf=%s meta=%s",
            source_name, signal_value, cls, confidence,
            {} if metadata is None else metadata,
        )

    def get_source_entropy(self, source_name):
        """Return entropy metrics for a single source.

        Parameters
        ----------
        source_name : str
            Source identifier.

        Returns
        -------
        dict with keys:
            source      — the source name
            entropy     — Shannon entropy of the {-1, 0, +1} distribution (bits)
            flat_rate   — fraction of observations where signal == 0
            bias        — net direction: buy_pct - sell_pct
            count       — total number of observations recorded
        """
        obs = self._observations.get(source_name, [])
        n = len(obs)
        if n == 0:
            return {
                "source": source_name,
                "entropy": 0.0,
                "flat_rate": 1.0,
                "bias": 0.0,
                "count": 0,
            }

        counts = {-1: 0, 0: 0, 1: 0}
        for cls in obs:
            counts[cls] += 1

        ent = _entropy(counts)
        flat_rate = counts[0] / n
        buy_pct = counts[1] / n
        sell_pct = counts[-1] / n

        return {
            "source": source_name,
            "entropy": round(ent, 4),
            "flat_rate": round(flat_rate, 4),
            "bias": round(buy_pct - sell_pct, 4),
            "count": n,
        }

    def get_all_entropies(self):
        """Return dict mapping each observed source name to its entropy metrics."""
        return {src: self.get_source_entropy(src) for src in self._observations}

    def get_global_assessment(self):
        """Return a global assessment of signal-space health.

        Returns
        -------
        dict with keys:
            total_sources        — number of distinct sources with observations
            active_sources       — sources whose entropy > 0.5
            collapsed_sources    — sources whose entropy < 0.3
            global_signal_entropy — mean entropy across all sources
            verdict              — one of:
                ``"SIGNAL_SPACE_HEALTHY"``
                ``"PARTIAL_COLLAPSE"``
                ``"FULL_COLLAPSE"``
            collapse_detail      — one of:
                ``"OSS_ONLY"``, ``"ALL_SOURCES"``, ``"MIXED"``

        Verdict logic
        -------------
        - **FULL_COLLAPSE**:  All sources have a flat rate > 80%.
        - **PARTIAL_COLLAPSE** with *OSS_ONLY*:  OSS flat rate > 80% while all
          other sources have flat rate < 50%.
        - **SIGNAL_SPACE_HEALTHY**:  All sources have flat rate < 50%.
        - **PARTIAL_COLLAPSE** with *MIXED*:  Everything else.
        """
        reports = self.get_all_entropies()
        total_sources = len(reports)

        # Count active / collapsed
        active_sources = sum(1 for r in reports.values() if r["entropy"] > 0.5)
        collapsed_sources = sum(1 for r in reports.values() if r["entropy"] < 0.3)

        # Global entropy = mean of all source entropies
        if total_sources > 0:
            global_signal_entropy = round(
                sum(r["entropy"] for r in reports.values()) / total_sources, 4
            )
        else:
            global_signal_entropy = 0.0

        # ---- Verdict logic ----
        flat_rates = {name: r["flat_rate"] for name, r in reports.items()}
        oss_flat = flat_rates.get("OSS", 0.0)

        all_flat_80 = all(fr > 0.80 for fr in flat_rates.values())

        oss_flat_80 = oss_flat > 0.80
        others_below_50 = all(
            fr < 0.50 for name, fr in flat_rates.items() if name != "OSS"
        )

        all_below_50 = all(fr < 0.50 for fr in flat_rates.values())

        if total_sources == 0:
            verdict = "FULL_COLLAPSE"
            collapse_detail = "ALL_SOURCES"
        elif all_flat_80:
            verdict = "FULL_COLLAPSE"
            collapse_detail = "ALL_SOURCES"
        elif oss_flat_80 and others_below_50:
            verdict = "PARTIAL_COLLAPSE"
            collapse_detail = "OSS_ONLY"
        elif all_below_50:
            verdict = "SIGNAL_SPACE_HEALTHY"
            collapse_detail = "MIXED"
        else:
            verdict = "PARTIAL_COLLAPSE"
            collapse_detail = "MIXED"

        return {
            "total_sources": total_sources,
            "active_sources": active_sources,
            "collapsed_sources": collapsed_sources,
            "global_signal_entropy": global_signal_entropy,
            "verdict": verdict,
            "collapse_detail": collapse_detail,
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ---- Scenario 1: Healthy signal space ----
    # All sources have flat rate < 50%, plenty of directional signal.
    sse_healthy = SignalSpaceEntropy("healthy")
    for _ in range(80):
        sse_healthy.feed_observation("OSS", 1)
    for _ in range(80):
        sse_healthy.feed_observation("OSS", -1)
    for _ in range(40):
        sse_healthy.feed_observation("OSS", 0)
    for _ in range(60):
        sse_healthy.feed_observation("ALT", 1)
    for _ in range(60):
        sse_healthy.feed_observation("ALT", -1)
    for _ in range(50):
        sse_healthy.feed_observation("ALT", 0)
    for _ in range(70):
        sse_healthy.feed_observation("SHADOW", 1)
    for _ in range(70):
        sse_healthy.feed_observation("SHADOW", -1)
    for _ in range(50):
        sse_healthy.feed_observation("SHADOW", 0)
    for _ in range(70):
        sse_healthy.feed_observation("ECDF", 1)
    for _ in range(70):
        sse_healthy.feed_observation("ECDF", -1)
    for _ in range(40):
        sse_healthy.feed_observation("ECDF", 0)
    for _ in range(60):
        sse_healthy.feed_observation("INJECTED", 1)
    for _ in range(60):
        sse_healthy.feed_observation("INJECTED", -1)
    for _ in range(50):
        sse_healthy.feed_observation("INJECTED", 0)
    for _ in range(90):
        sse_healthy.feed_observation("RESEARCH", 1)
    for _ in range(90):
        sse_healthy.feed_observation("RESEARCH", -1)
    for _ in range(20):
        sse_healthy.feed_observation("RESEARCH", 0)

    # ---- Scenario 2: OSS-only collapse ----
    # OSS entirely flat; other sources remain active (flat < 50%).
    sse_oss_collapse = SignalSpaceEntropy("oss_collapse")
    for _ in range(200):
        sse_oss_collapse.feed_observation("OSS", 0)
    for _ in range(60):
        sse_oss_collapse.feed_observation("ALT", 1)
    for _ in range(60):
        sse_oss_collapse.feed_observation("ALT", -1)
    for _ in range(80):
        sse_oss_collapse.feed_observation("ALT", 0)
    for _ in range(50):
        sse_oss_collapse.feed_observation("SHADOW", 1)
    for _ in range(50):
        sse_oss_collapse.feed_observation("SHADOW", -1)
    for _ in range(80):
        sse_oss_collapse.feed_observation("SHADOW", 0)

    # ---- Scenario 3: Full collapse ----
    # Every source is flat > 80%.
    sse_full_collapse = SignalSpaceEntropy("full_collapse")
    for _ in range(300):
        sse_full_collapse.feed_observation("OSS", 0)
    for _ in range(300):
        sse_full_collapse.feed_observation("ALT", 0)
    for _ in range(300):
        sse_full_collapse.feed_observation("SHADOW", 0)

    # ---- Scenario 4: Mixed partial collapse ----
    # OSS > 80% flat, but some other sources also > 50% flat → MIXED.
    sse_mixed = SignalSpaceEntropy("mixed_collapse")
    for _ in range(200):
        sse_mixed.feed_observation("OSS", 0)
    for _ in range(80):
        sse_mixed.feed_observation("ALT", 0)
    for _ in range(40):
        sse_mixed.feed_observation("ALT", 1)
    for _ in range(40):
        sse_mixed.feed_observation("ALT", -1)
    for _ in range(100):
        sse_mixed.feed_observation("SHADOW", 1)
    for _ in range(100):
        sse_mixed.feed_observation("SHADOW", -1)

    # ---- Print results ----
    print("=" * 60)
    print("Signal Space Entropy — Self Test")
    print("=" * 60)

    for label, instance_id in [
        ("HEALTHY", "healthy"),
        ("OSS_ONLY_COLLAPSE", "oss_collapse"),
        ("FULL_COLLAPSE", "full_collapse"),
        ("MIXED_COLLAPSE", "mixed_collapse"),
    ]:
        engine = SignalSpaceEntropy(instance_id)
        print(f"\n{'─' * 60}")
        print(f"  SCENARIO: {label}")
        print(f"{'─' * 60}")

        print("\n  Per-source entropies:")
        for src_name, report in sorted(engine.get_all_entropies().items()):
            print(
                f"    {src_name:15s}  entropy={report['entropy']:.4f}  "
                f"flat={report['flat_rate']:.4f}  "
                f"bias={report['bias']:+.4f}  "
                f"count={report['count']}"
            )

        print("\n  Global assessment:")
        assessment = engine.get_global_assessment()
        for k, v in assessment.items():
            print(f"    {k:25s} = {v}")
        print()

    # ---- Quick assertions ----
    healthy = SignalSpaceEntropy("healthy")
    ass = healthy.get_global_assessment()
    assert ass["verdict"] == "SIGNAL_SPACE_HEALTHY", f"Expected HEALTHY, got {ass['verdict']}"
    assert ass["total_sources"] == 6, f"Expected 6 sources, got {ass['total_sources']}"
    assert ass["active_sources"] >= 4, f"Expected >=4 active, got {ass['active_sources']}"

    oss_col = SignalSpaceEntropy("oss_collapse")
    ass2 = oss_col.get_global_assessment()
    assert ass2["verdict"] == "PARTIAL_COLLAPSE", f"Expected PARTIAL_COLLAPSE, got {ass2['verdict']}"
    assert ass2["collapse_detail"] == "OSS_ONLY", f"Expected OSS_ONLY, got {ass2['collapse_detail']}"

    full_col = SignalSpaceEntropy("full_collapse")
    ass3 = full_col.get_global_assessment()
    assert ass3["verdict"] == "FULL_COLLAPSE", f"Expected FULL_COLLAPSE, got {ass3['verdict']}"
    assert ass3["collapse_detail"] == "ALL_SOURCES", f"Expected ALL_SOURCES, got {ass3['collapse_detail']}"

    mixed = SignalSpaceEntropy("mixed_collapse")
    ass4 = mixed.get_global_assessment()
    assert ass4["verdict"] == "PARTIAL_COLLAPSE", f"Expected PARTIAL_COLLAPSE, got {ass4['verdict']}"
    assert ass4["collapse_detail"] == "MIXED", f"Expected MIXED, got {ass4['collapse_detail']}"

    logger.info("All self-test assertions passed ✓")
    print("All self-test assertions passed ✓")
