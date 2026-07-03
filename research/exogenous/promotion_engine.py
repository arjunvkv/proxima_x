"""
Program VI — Exogenous Amplitude Promotion Engine
===================================================
Evaluates exogenous surface entries against Tier 1 / Tier 2 promotion
criteria and decides whether each (exogenous_key, horizon) cell qualifies
for promotion to a higher-confidence regime.

Tiers
-----
**Tier 1** (ALL must pass):
    - AER >= 3.5
    - spread_multiple >= 5.0
    - P(|move| > 3 × spread) >= 0.50  (k=3 exceed prob)
    - n >= 150
    - stability >= 0.85

**Tier 2** (ALL must pass):
    - AER >= 2.5
    - spread_multiple >= 3.0
    - P(|move| > 2 × spread) >= 0.40  (k=2 exceed prob)
    - n >= 80
    - stability >= 0.75

**Reject**: anything that does not satisfy all criteria of either tier.
"""

import sys

sys.path.insert(0, ".")
sys.path.insert(0, "proxima_x")

from typing import List

import polars as pl

from research.exogenous.schemas import ExogenousSurfaceEntry

# Stability threshold for individual exceed-prob checks.
_STABILITY_EXCEED_THRESHOLD: float = 0.3


def _compute_stability(exceed_prob: dict) -> float:
    """Fraction of k values where exceed_prob[k] > 0.3.

    Parameters
    ----------
    exceed_prob : dict
        Mapping from k (float) to probability (float).

    Returns
    -------
    float
        Fraction of keys whose exceed_prob exceeds 0.3.
    """
    if not exceed_prob:
        return 0.0
    passing = sum(
        1 for v in exceed_prob.values() if v > _STABILITY_EXCEED_THRESHOLD
    )
    return passing / len(exceed_prob)


class ExogenousPromotionEngine:
    """Evaluates exogenous surface entries for tier-based promotion.

    Use :meth:`evaluate` for a single entry or :meth:`evaluate_batch` for
    a list of entries (returns a ``pl.DataFrame``).
    """

    # ------------------------------------------------------------------
    # Single-entry evaluation
    # ------------------------------------------------------------------

    def evaluate(self, entry: ExogenousSurfaceEntry) -> dict:
        """Evaluate a single surface entry and return the promotion verdict.

        Parameters
        ----------
        entry : ExogenousSurfaceEntry
            A surface entry to evaluate.

        Returns
        -------
        dict
            With keys:

            - ``exogenous_key``    — built from the entry's components
            - ``horizon``          — the forward horizon (seconds)
            - ``aer``              — amplitude excess ratio
            - ``spread_multiple``  — mean_abs / mean_spread
            - ``n``                — observation count
            - ``tier``             — 1, 2, or 0 (rejected)
            - ``promoted``         — True if tier > 0
            - ``reasons``          — list of human-readable strings
              explaining why each criterion passed
            - ``failures``         — list of human-readable strings
              explaining which criteria failed
        """
        reasons: List[str] = []
        failures: List[str] = []

        # ------------------- Tier 1 checks -------------------
        tier1_aer = entry.aer >= 3.5
        tier1_sm = entry.spread_multiple >= 5.0
        tier1_exceed = entry.exceed_prob.get(3.0, 0.0) >= 0.50
        tier1_n = entry.n >= 150
        stability = _compute_stability(entry.exceed_prob)
        tier1_stability = stability >= 0.85

        tier1_criteria = {
            "aer >= 3.5": tier1_aer,
            "spread_multiple >= 5.0": tier1_sm,
            "P(|move| > 3*spread) >= 0.50": tier1_exceed,
            "n >= 150": tier1_n,
            f"stability >= 0.85 (got {stability:.3f})": tier1_stability,
        }

        tier1_pass = all(tier1_criteria.values())

        # ------------------- Tier 2 checks -------------------
        tier2_aer = entry.aer >= 2.5
        tier2_sm = entry.spread_multiple >= 3.0
        tier2_exceed = entry.exceed_prob.get(2.0, 0.0) >= 0.40
        tier2_n = entry.n >= 80
        tier2_stability = stability >= 0.75

        tier2_criteria = {
            "aer >= 2.5": tier2_aer,
            "spread_multiple >= 3.0": tier2_sm,
            "P(|move| > 2*spread) >= 0.40": tier2_exceed,
            "n >= 80": tier2_n,
            f"stability >= 0.75 (got {stability:.3f})": tier2_stability,
        }

        tier2_pass = all(tier2_criteria.values())

        # ------------------- Assign tier ---------------------
        if tier1_pass:
            tier = 1
            for criterion, passed in tier1_criteria.items():
                if passed:
                    reasons.append(f"Tier1: {criterion}")
                else:
                    failures.append(f"Tier1: {criterion}")
        elif tier2_pass:
            tier = 2
            for criterion, passed in tier2_criteria.items():
                if passed:
                    reasons.append(f"Tier2: {criterion}")
                else:
                    failures.append(f"Tier2: {criterion}")
        else:
            tier = 0
            for criterion, passed in tier1_criteria.items():
                if not passed:
                    failures.append(f"Tier1: {criterion}")
            for criterion, passed in tier2_criteria.items():
                if not passed:
                    failures.append(f"Tier2: {criterion}")
            reasons.append("No tier criteria satisfied")

        # Build exogenous_key from entry components
        fixing = (
            "None" if entry.fixing_window is None else entry.fixing_window
        )
        exogenous_key = "|".join(
            [
                entry.session,
                fixing,
                str(entry.rollover),
                str(entry.liquidity_void),
                str(entry.news_proxy),
            ]
        )

        return {
            "exogenous_key": exogenous_key,
            "horizon": entry.horizon,
            "aer": entry.aer,
            "spread_multiple": entry.spread_multiple,
            "n": entry.n,
            "tier": tier,
            "promoted": tier > 0,
            "reasons": reasons,
            "failures": failures,
        }

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    def evaluate_batch(
        self, entries: List[ExogenousSurfaceEntry]
    ) -> pl.DataFrame:
        """Evaluate a list of surface entries and return results as a table.

        Parameters
        ----------
        entries : list[ExogenousSurfaceEntry]
            Sequence of surface entries to evaluate.

        Returns
        -------
        pl.DataFrame
            Columns:
                exogenous_key   str
                horizon         i64
                aer             f64
                spread_multiple f64
                n               i64
                tier            i64  (0, 1, or 2)
                promoted        bool
                reasons         str  (semicolon-joined list)
                failures        str  (semicolon-joined list)
        """
        rows = []
        for entry in entries:
            result = self.evaluate(entry)
            rows.append(
                {
                    "exogenous_key": result["exogenous_key"],
                    "horizon": result["horizon"],
                    "aer": result["aer"],
                    "spread_multiple": result["spread_multiple"],
                    "n": result["n"],
                    "tier": result["tier"],
                    "promoted": result["promoted"],
                    "reasons": "; ".join(result["reasons"]),
                    "failures": "; ".join(result["failures"]),
                }
            )

        if not rows:
            return pl.DataFrame(
                {
                    "exogenous_key": pl.Series([], dtype=pl.Utf8),
                    "horizon": pl.Series([], dtype=pl.Int64),
                    "aer": pl.Series([], dtype=pl.Float64),
                    "spread_multiple": pl.Series([], dtype=pl.Float64),
                    "n": pl.Series([], dtype=pl.Int64),
                    "tier": pl.Series([], dtype=pl.Int64),
                    "promoted": pl.Series([], dtype=pl.Boolean),
                    "reasons": pl.Series([], dtype=pl.Utf8),
                    "failures": pl.Series([], dtype=pl.Utf8),
                }
            )

        return pl.DataFrame(rows)
