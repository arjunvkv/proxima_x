"""
Program VI.5 — Event Promotion Engine
========================================
Evaluates event amplitude surface entries against Tier 1 / Tier 2 promotion
criteria and decides whether each (event_key, horizon) cell qualifies for
promotion to a higher-confidence regime.

Tiers
-----
**Tier 1** (ALL must pass):
    - AER >= 2.5
    - P(|move| > 3 × spread) >= 0.45  (k=3 exceed prob)
    - n >= 50
    - stability >= 0.75

**Tier 2** (ALL must pass):
    - AER >= 1.8
    - P(|move| > 2 × spread) >= 0.35  (k=2 exceed prob)
    - n >= 30
    - stability >= 0.65

**Reject**: anything that does not satisfy all criteria of either tier.
"""

import sys; sys.path.insert(0, "."); sys.path.insert(0, "proxima_x")

from typing import Any, List

import polars as pl

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


class EventPromotionEngine:
    """Evaluates event surface entries for tier-based promotion.

    Use :meth:`evaluate` for a single entry or :meth:`evaluate_all` for
    a full surface dictionary (returns a ``pl.DataFrame``).
    """

    # ------------------------------------------------------------------
    # Single-entry evaluation
    # ------------------------------------------------------------------

    def evaluate(self, key: str, horizon: int, entry: dict) -> dict:
        """Evaluate a single surface entry and return the promotion verdict.

        Parameters
        ----------
        key : str
            Composite key in the form ``"{bucket}|{impact}"``.
        horizon : int
            Forward horizon (seconds).
        entry : dict
            Surface entry dictionary with at minimum fields:
            ``aer``, ``spread_multiple``, ``n``, ``exceed_prob``,
            and optionally ``stability`` (computed on-the-fly if absent).

        Returns
        -------
        dict
            With keys:

            - ``key``              — the input key
            - ``horizon``          — the forward horizon (seconds)
            - ``aer``              — amplitude excess ratio
            - ``spread_multiple``  — mean_abs / mean_spread
            - ``n``                — observation count
            - ``p2x``              — P(|move| > 2 × spread)
            - ``p3x``              — P(|move| > 3 × spread)
            - ``tier``             — 1, 2, or 0 (rejected)
            - ``promoted``         — True if tier > 0
            - ``reasons``          — list of human-readable strings
              explaining why each criterion passed
            - ``failures``         — list of human-readable strings
              explaining which criteria failed
        """
        reasons: List[str] = []
        failures: List[str] = []

        aer = entry.get("aer", 0.0)
        spread_multiple = entry.get("spread_multiple", 0.0)
        n = entry.get("n", 0)
        exceed_prob: dict = entry.get("exceed_prob", {})
        p2x = exceed_prob.get(2.0, 0.0)
        p3x = exceed_prob.get(3.0, 0.0)
        stability = entry.get("stability", _compute_stability(exceed_prob))

        # ------------------- Tier 1 checks -------------------
        tier1_aer = aer >= 2.5
        tier1_p3x = p3x >= 0.45
        tier1_n = n >= 50
        tier1_stability = stability >= 0.75

        tier1_criteria = {
            "aer >= 2.5": tier1_aer,
            "P(|move| > 3*spread) >= 0.45": tier1_p3x,
            "n >= 50": tier1_n,
            f"stability >= 0.75 (got {stability:.3f})": tier1_stability,
        }

        tier1_pass = all(tier1_criteria.values())

        # ------------------- Tier 2 checks -------------------
        tier2_aer = aer >= 1.8
        tier2_p2x = p2x >= 0.35
        tier2_n = n >= 30
        tier2_stability = stability >= 0.65

        tier2_criteria = {
            "aer >= 1.8": tier2_aer,
            "P(|move| > 2*spread) >= 0.35": tier2_p2x,
            "n >= 30": tier2_n,
            f"stability >= 0.65 (got {stability:.3f})": tier2_stability,
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

        return {
            "key": key,
            "horizon": horizon,
            "aer": aer,
            "spread_multiple": spread_multiple,
            "n": n,
            "p2x": p2x,
            "p3x": p3x,
            "tier": tier,
            "promoted": tier > 0,
            "reasons": reasons,
            "failures": failures,
        }

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    def evaluate_all(self, surface_dict: dict) -> pl.DataFrame:
        """Evaluate all entries in a surface dictionary.

        Parameters
        ----------
        surface_dict : dict
            Nested dictionary of the form::

                {
                    "bucket|impact": {
                        horizon_sec: { ... entry fields ... },
                        ...
                    },
                    ...
                }

            This matches the internal ``_surface`` structure produced by
            :class:`EventSurface`.

        Returns
        -------
        pl.DataFrame
            Columns:
                key             str
                horizon         i64
                aer             f64
                spread_multiple f64
                n               i64
                p2x             f64
                p3x             f64
                tier            i64  (0, 1, or 2)
                promoted        bool
                reasons         str  (semicolon-joined list)
                failures        str  (semicolon-joined list)
        """
        rows = []
        for key, horizons in surface_dict.items():
            for horizon, entry in horizons.items():
                result = self.evaluate(key, horizon, entry)
                rows.append(
                    {
                        "key": result["key"],
                        "horizon": result["horizon"],
                        "aer": result["aer"],
                        "spread_multiple": result["spread_multiple"],
                        "n": result["n"],
                        "p2x": result["p2x"],
                        "p3x": result["p3x"],
                        "tier": result["tier"],
                        "promoted": result["promoted"],
                        "reasons": "; ".join(result["reasons"]),
                        "failures": "; ".join(result["failures"]),
                    }
                )

        if not rows:
            return pl.DataFrame(
                {
                    "key": pl.Series([], dtype=pl.Utf8),
                    "horizon": pl.Series([], dtype=pl.Int64),
                    "aer": pl.Series([], dtype=pl.Float64),
                    "spread_multiple": pl.Series([], dtype=pl.Float64),
                    "n": pl.Series([], dtype=pl.Int64),
                    "p2x": pl.Series([], dtype=pl.Float64),
                    "p3x": pl.Series([], dtype=pl.Float64),
                    "tier": pl.Series([], dtype=pl.Int64),
                    "promoted": pl.Series([], dtype=pl.Boolean),
                    "reasons": pl.Series([], dtype=pl.Utf8),
                    "failures": pl.Series([], dtype=pl.Utf8),
                }
            )

        return pl.DataFrame(rows)
