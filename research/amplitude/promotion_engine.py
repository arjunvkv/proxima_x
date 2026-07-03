"""
Phase V — Amplitude Promotion Engine

Determines whether an amplitude-surface entry meets deployability criteria
and assigns a promotion tier (Tier 1 / Tier 2 / Reject).
"""

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Any

import polars as pl

from research.amplitude.schemas import AmplitudeSurfaceEntry

sys.path.insert(0, ".")
sys.path.insert(0, "proxima_x")


# ---------------------------------------------------------------------------
# Promotion thresholds
# ---------------------------------------------------------------------------

TIER_1_REQUIREMENTS = {
    "aer": 3.0,
    "spread_multiple": 4.0,
    "n": 100,
    "stability": 0.80,
}

TIER_2_REQUIREMENTS = {
    "aer": 2.0,
    "spread_multiple": 2.5,
    "n": 50,
    "stability": 0.70,
}


# ---------------------------------------------------------------------------
# Promotion engine
# ---------------------------------------------------------------------------


class AmplitudePromotionEngine:
    """Evaluates amplitude-surface entries and assigns promotion tiers."""

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, entry: AmplitudeSurfaceEntry) -> dict:
        """Return a promotion verdict dict for a single entry.

        Returns
        -------
        dict with keys: state_hash, horizon, aer, spread_multiple, n,
                        tier, promoted, reasons, failures
        """
        stability = self._compute_stability(entry)

        # -- Tier 1 check ---------------------------------------------------
        tier1_pass, tier1_reasons, tier1_failures = self._check_tier(
            entry, stability, tier=1, reqs=TIER_1_REQUIREMENTS,
        )

        if tier1_pass:
            return self._build_result(
                entry, tier=1, promoted=True,
                reasons=tier1_reasons, failures=[],
            )

        # -- Tier 2 check ---------------------------------------------------
        tier2_pass, tier2_reasons, tier2_failures = self._check_tier(
            entry, stability, tier=2, reqs=TIER_2_REQUIREMENTS,
        )

        if tier2_pass:
            return self._build_result(
                entry, tier=2, promoted=True,
                reasons=tier2_reasons, failures=[],
            )

        # -- Reject ---------------------------------------------------------
        combined_failures = list(
            dict.fromkeys(tier1_failures + tier2_failures)
        )
        return self._build_result(
            entry, tier=0, promoted=False,
            reasons=[], failures=combined_failures,
        )

    def evaluate_batch(
        self, entries: list[AmplitudeSurfaceEntry],
    ) -> pl.DataFrame:
        """Evaluate every entry in *entries* and return a DataFrame.

        The result DataFrame contains one row per entry with columns:

            state_hash, horizon, aer, spread_multiple, n,
            tier, promoted, n_reasons, n_failures
        """
        results = [self.evaluate(e) for e in entries]
        return pl.from_dicts(results)

    def summarize(self, results: pl.DataFrame) -> str:
        """Return a human-readable summary of a batch evaluation *results* DataFrame."""
        total = len(results)

        if total == 0:
            return "No entries evaluated."

        promoted_df = results.filter(pl.col("promoted"))
        n_promoted = len(promoted_df)
        n_tier1 = len(promoted_df.filter(pl.col("tier") == 1))
        n_tier2 = len(promoted_df.filter(pl.col("tier") == 2))
        n_rejected = total - n_promoted

        lines = [
            "=" * 60,
            "Amplitude Promotion Summary",
            "=" * 60,
            f"  Total entries evaluated  : {total}",
            f"  Promoted (Tier 1)        : {n_tier1}",
            f"  Promoted (Tier 2)        : {n_tier2}",
            f"  Rejected                 : {n_rejected}",
        ]

        if n_promoted > 0:
            # Show top entries by AER
            top = (
                promoted_df.sort("aer", descending=True)
                .head(5)
                .select(["state_hash", "horizon", "aer", "tier"])
            )
            lines.append("")
            lines.append("Top promoted entries (by AER):")
            for row in top.to_dicts():
                lines.append(
                    f"    {row['state_hash']}  h={row['horizon']}s  "
                    f"AER={row['aer']:.2f}  tier={row['tier']}"
                )

        if n_rejected > 0:
            # Most common failure reasons
            rejected = results.filter(~pl.col("promoted"))
            all_failures = rejected.select(
                pl.col("failures").list.explode().alias("reason")
            )
            if len(all_failures) > 0:
                top_fails = (
                    all_failures.group_by("reason")
                    .agg(pl.len().alias("count"))
                    .sort("count", descending=True)
                    .head(3)
                )
                lines.append("")
                lines.append("Top rejection reasons:")
                for row in top_fails.to_dicts():
                    lines.append(f"    [{row['count']}] {row['reason']}")

        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_stability(self, entry: AmplitudeSurfaceEntry) -> float:
        """Fraction of exceed-probability values above 0.3.

        Returns 0.0 if *exceed_prob* is empty.
        """
        if not entry.exceed_prob:
            return 0.0
        n_total = len(entry.exceed_prob)
        n_above = sum(1 for p in entry.exceed_prob.values() if p > 0.3)
        return n_above / n_total

    def _check_tier(
        self,
        entry: AmplitudeSurfaceEntry,
        stability: float,
        tier: int,
        reqs: dict,
    ) -> tuple[bool, list[str], list[str]]:
        """Check whether *entry* satisfies all requirements for *tier*.

        Returns
        -------
        (passed, reasons, failures)
        """
        reasons: list[str] = []
        failures: list[str] = []

        # AER
        aer_min = reqs["aer"]
        if entry.aer >= aer_min:
            reasons.append(
                f"AER meets threshold ({entry.aer:.2f} >= {aer_min})"
            )
        else:
            failures.append(
                f"AER too low ({entry.aer:.2f} < {aer_min})"
            )

        # Spread multiple
        sm_min = reqs["spread_multiple"]
        if entry.spread_multiple >= sm_min:
            reasons.append(
                f"spread_multiple meets threshold "
                f"({entry.spread_multiple:.2f} >= {sm_min})"
            )
        else:
            failures.append(
                f"spread_multiple too low "
                f"({entry.spread_multiple:.2f} < {sm_min})"
            )

        # Sample size
        n_min = reqs["n"]
        if entry.n >= n_min:
            reasons.append(
                f"n meets threshold ({entry.n} >= {n_min})"
            )
        else:
            failures.append(
                f"n too low ({entry.n} < {n_min})"
            )

        # Stability
        stab_min = reqs["stability"]
        if stability >= stab_min:
            reasons.append(
                f"stability meets threshold ({stability:.2f} >= {stab_min})"
            )
        else:
            failures.append(
                f"stability too low ({stability:.2f} < {stab_min})"
            )

        passed = len(failures) == 0
        return passed, reasons, failures

    @staticmethod
    def _build_result(
        entry: AmplitudeSurfaceEntry,
        tier: int,
        promoted: bool,
        reasons: list[str],
        failures: list[str],
    ) -> dict:
        """Assemble the result dict for *entry*."""
        return {
            "state_hash": entry.state_hash,
            "horizon": entry.horizon,
            "aer": entry.aer,
            "spread_multiple": entry.spread_multiple,
            "n": entry.n,
            "tier": tier,
            "promoted": promoted,
            "reasons": reasons,
            "failures": failures,
        }
