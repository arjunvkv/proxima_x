"""
Phase V — Spread Exceedance Model
==================================
Models P(|move| > k * spread | state) — the probability that forward absolute
price movement exceeds a multiple of the quoted spread, conditioned on market
state.

This is the primary monetisation model for Phase V. It answers: given the
current market state, how likely is a price move large enough to cover
transaction costs (spread) and still produce profit?
"""

import sys

sys.path.insert(0, ".")
sys.path.insert(0, "proxima_x")

from typing import Any

import numpy as np
import polars as pl

from research.amplitude.schemas import AmplitudeSurfaceEntry
from research.amplitude.amplitude_surface import AmplitudeSurfaceEngine


class SpreadExceedanceModel:
    """Models P(|move| > k * spread | state) for configurable multiples of
    the quoted spread.

    The model is fitted on a DataFrame of observations (state_hash, abs_move,
    spread) and stores per-state exceedance probabilities for a set of k
    values.  These probabilities can be queried directly, interpolated to
    find breakeven k thresholds, or used to rank states by monetisation
    potential.
    """

    def __init__(self) -> None:
        # state_hash -> k -> probability
        self._exceed_probs: dict[str, dict[float, float]] = {}
        # state_hash -> number of observations
        self._state_n: dict[str, int] = {}
        # Sorted list of fitted k values
        self._k_values: list[float] = []

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pl.DataFrame,
        k_values: list[float] | None = None,
    ) -> None:
        """Compute per-state exceedance probabilities from observation data.

        Parameters
        ----------
        df : pl.DataFrame
            Observation table with at least the columns:

            - ``state_hash`` : str   — composite state key
            - ``abs_move``   : float — absolute forward price move (bps)
            - ``spread``     : float — quoted spread at observation time (bps)

        k_values : list[float] | None
            Spread multiples to evaluate.  Defaults to
            ``[1.0, 1.5, 2.0, 3.0, 4.0]``.

        Effects
        -------
        Populates :attr:`_exceed_probs`, :attr:`_state_n`, and
        :attr:`_k_values`.
        """
        if k_values is None:
            k_values = [1.0, 1.5, 2.0, 3.0, 4.0]
        self._k_values = sorted(k_values)

        required = {"state_hash", "abs_move", "spread"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"DataFrame missing required columns: {missing}"
            )

        # Build aggregation expressions: count + exceed probability per k
        aggs: list[pl.Expr] = [pl.count().alias("n")]
        for k in self._k_values:
            aggs.append(
                (pl.col("abs_move") > k * pl.col("spread"))
                .mean()
                .alias(f"exceed_{k}")
            )

        grouped = df.group_by("state_hash").agg(aggs)

        self._exceed_probs = {}
        self._state_n = {}

        for row in grouped.iter_rows(named=True):
            state_hash: str = row["state_hash"]
            n: int = row["n"]
            self._state_n[state_hash] = n

            probs: dict[float, float] = {}
            for k in self._k_values:
                val = row.get(f"exceed_{k}")
                probs[k] = float(val) if val is not None else 0.0

            self._exceed_probs[state_hash] = probs

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def probability(self, state_hash: str, k: float) -> float:
        """Return P(|move| > k * spread | state_hash).

        If *k* does not exactly match a stored value, the probability is
        linearly interpolated from the two nearest stored k values.

        Parameters
        ----------
        state_hash : str
            Composite state key.
        k : float
            Spread multiple.

        Returns
        -------
        float
            Probability estimate, or 0.0 if the state_hash is unknown.
        """
        probs = self._exceed_probs.get(state_hash)
        if probs is None:
            return 0.0
        if k in probs:
            return probs[k]
        return self._interpolate_k(probs, k)

    def probability_by_state(
        self,
        oss_bucket: int,
        regime_id: str,
        k: float,
    ) -> float:
        """Return P(|move| > k * spread) for a coarse (bucket, regime) key.

        Constructs a ``StateHasher`` digest from the two parameters (with
        placeholder values for the remaining state fields) and delegates to
        :meth:`probability`.

        Parameters
        ----------
        oss_bucket : int
            Over-/under-shot bucket (0–9).
        regime_id : str
            Regime identifier (e.g. ``"high_velocity"``).
        k : float
            Spread multiple.

        Returns
        -------
        float
            Probability estimate, or 0.0 if the composite key is unknown.
        """
        from research.amplitude.schemas import AmplitudeState, StateHasher

        stub_state = AmplitudeState(
            symbol="",
            ts=0.0,
            oss_bucket=oss_bucket,
            tross_delta=0,
            sal_score=0.0,
            entropy_slope=0.0,
            compression_density=0.0,
            tick_velocity=0.0,
            spread=0.0,
            regime_id=regime_id,
        )
        h = StateHasher.hash(stub_state)
        return self.probability(h, k)

    # ------------------------------------------------------------------
    # Sweep & interpolation
    # ------------------------------------------------------------------

    def sweep_k(self, state_hash: str) -> dict[float, float]:
        """Return the full exceedance curve for a state.

        Parameters
        ----------
        state_hash : str
            Composite state key.

        Returns
        -------
        dict[float, float]
            ``{k: probability}`` for all fitted k values, or an empty dict
            if the state_hash is unknown.
        """
        return dict(self._exceed_probs.get(state_hash, {}))

    def find_breakeven_k(
        self,
        state_hash: str,
        target_prob: float = 0.5,
    ) -> float:
        """Linearly interpolate to find the k where
        P(|move| > k * spread) equals *target_prob*.

        A *higher* breakeven k is better — it means the state produces
        larger moves relative to the spread, so even at a large multiple
        the exceedance probability remains high.

        Parameters
        ----------
        state_hash : str
            Composite state key.
        target_prob : float
            Target probability (default 0.5).

        Returns
        -------
        float
            The interpolated k value.  Returns ``k_values[0]`` (smallest
            fitted k) if the probability is below *target_prob* even at
            the smallest multiple, or ``k_values[-1]`` (largest fitted k)
            if the probability never drops to *target_prob*.
        """
        probs = self._exceed_probs.get(state_hash)
        if probs is None or not probs:
            return 0.0

        sorted_ks = sorted(probs.keys())
        sorted_probs = [probs[k] for k in sorted_ks]

        # Probability decreases as k increases.  If target_prob is above
        # the highest observed probability (smallest k), clamp to k[0].
        if target_prob >= sorted_probs[0]:
            return sorted_ks[0]
        # If target_prob is below the lowest observed probability
        # (largest k), clamp to k[-1].
        if target_prob <= sorted_probs[-1]:
            return sorted_ks[-1]

        # Linear interpolation between adjacent stored k values.
        for i in range(len(sorted_ks) - 1):
            p_left = sorted_probs[i]
            p_right = sorted_probs[i + 1]
            if p_left >= target_prob >= p_right:
                k_left = sorted_ks[i]
                k_right = sorted_ks[i + 1]
                if p_right == p_left:
                    return (k_left + k_right) / 2.0
                fraction = (target_prob - p_left) / (p_right - p_left)
                return k_left + fraction * (k_right - k_left)

        return sorted_ks[-1]

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def exceedance_curve(self) -> pl.DataFrame:
        """Return the full exceedance curve as a DataFrame.

        Returns
        -------
        pl.DataFrame
            Columns:

            - ``state_hash`` — composite state key
            - ``k``          — spread multiple
            - ``probability``— P(|move| > k * spread | state_hash)
            - ``n``          — number of observations for this state_hash

            Sorted by (state_hash, k).
        """
        rows: list[dict[str, Any]] = []
        for state_hash, probs in self._exceed_probs.items():
            n = self._state_n.get(state_hash, 0)
            for k in sorted(probs.keys()):
                rows.append({
                    "state_hash": state_hash,
                    "k": k,
                    "probability": probs[k],
                    "n": n,
                })
        result = pl.DataFrame(rows)
        if len(result) > 0:
            result = result.sort(["state_hash", "k"])
        return result

    def best_states(
        self,
        k: float = 2.0,
        min_prob: float = 0.3,
        min_n: int = 20,
    ) -> pl.DataFrame:
        """Return states that meet minimum exceedance probability and
        observation count thresholds at a given spread multiple.

        Parameters
        ----------
        k : float
            Spread multiple to evaluate (default 2.0).
        min_prob : float
            Minimum acceptable probability (default 0.3).
        min_n : int
            Minimum observation count (default 20).

        Returns
        -------
        pl.DataFrame
            Columns:

            - ``state_hash`` — composite state key
            - ``k``          — spread multiple
            - ``probability``— P(|move| > k * spread | state_hash)
            - ``n``          — observation count

            Sorted by probability descending.
        """
        rows: list[dict[str, Any]] = []
        for state_hash, probs in self._exceed_probs.items():
            n = self._state_n.get(state_hash, 0)
            if n < min_n:
                continue
            prob = probs.get(k, 0.0)
            if prob >= min_prob:
                rows.append({
                    "state_hash": state_hash,
                    "k": k,
                    "probability": prob,
                    "n": n,
                })
        result = pl.DataFrame(rows)
        if len(result) > 0:
            result = result.sort("probability", descending=True)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _interpolate_k(
        probs: dict[float, float],
        target_k: float,
    ) -> float:
        """Linearly interpolate P(|move| > k * spread) for a k value that
        is not in the fitted set.

        Parameters
        ----------
        probs : dict[float, float]
            Fitted ``{k: probability}`` map (must be non-empty).
        target_k : float
            Spread multiple to interpolate.

        Returns
        -------
        float
            Interpolated probability.
        """
        sorted_ks = sorted(probs.keys())
        if not sorted_ks:
            return 0.0

        if target_k <= sorted_ks[0]:
            return probs[sorted_ks[0]]
        if target_k >= sorted_ks[-1]:
            return probs[sorted_ks[-1]]

        for i in range(len(sorted_ks) - 1):
            k_left = sorted_ks[i]
            k_right = sorted_ks[i + 1]
            if k_left <= target_k <= k_right:
                p_left = probs[k_left]
                p_right = probs[k_right]
                if k_right == k_left:
                    return (p_left + p_right) / 2.0
                fraction = (target_k - k_left) / (k_right - k_left)
                return p_left + fraction * (p_right - p_left)

        return probs[sorted_ks[-1]]
