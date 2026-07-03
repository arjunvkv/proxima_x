"""Program VI — Session-conditioned amplitude distribution mapper.

Records exogenous observations and produces session-grouped and
key-grouped amplitude statistics for downstream surface construction.
"""

import sys

sys.path.insert(0, ".")
sys.path.insert(0, "proxima_x")

from typing import List

import numpy as np
import polars as pl

from research.exogenous.schemas import ExogenousObservation, ExogenousState, make_key

_EPS = 1e-12


# ── Session-conditioned amplitude mapper ──────────────────────────────────


class SessionAmplitudeMapper:
    """Maps session-conditioned amplitude distributions for Program VI.

    Accumulates labelled observations (state + forward move) and exposes
    grouped amplitude summaries keyed by session or by the full exogenous
    feature vector.
    """

    def __init__(self) -> None:
        self._records: List[ExogenousObservation] = []

    # ── record ─────────────────────────────────────────────────────────────

    def record(self, state: ExogenousState, future: ExogenousObservation) -> None:
        """Store an observation pairing a state snapshot with a forward move.

        Parameters
        ----------
        state : ExogenousState
            The state vector at the observation time (used for validation /
            debugging; the observation already carries its own state).
        future : ExogenousObservation
            The labelled forward observation whose ``abs_move``, ``state``
            and derived fields enter the summary statistics.
        """
        _ = state  # kept for api symmetry; obs.state carries equivalent info
        self._records.append(future)

    # ── summarize ──────────────────────────────────────────────────────────

    def summarize(self) -> pl.DataFrame:
        """Group all observations by session and return amplitude statistics.

        Returns
        -------
        pl.DataFrame
            Columns:
                session              str   – TOKYO / LONDON / NEWYORK / DEADZONE
                n                    i64   – observation count
                mean_abs             f64   – mean of absolute forward move
                median_abs           f64   – median of absolute forward move
                std_abs              f64   – sample standard deviation of abs_move
                p90_abs              f64   – 90th percentile of abs_move
                mean_spread_multiple f64   – mean of (abs_move / state.spread)
                aer                  f64   – amplitude excess ratio (mean_abs / median_abs)
        """
        if not self._records:
            return self._empty_summarize()

        sessions: List[str] = []
        abs_moves: List[float] = []
        spread_multiples: List[float] = []

        for obs in self._records:
            sessions.append(obs.state.session)
            abs_moves.append(obs.abs_move)
            spread = obs.state.spread
            spread_multiples.append(
                obs.abs_move / spread if spread > _EPS else 0.0
            )

        df = pl.DataFrame(
            {
                "session": sessions,
                "abs_move": abs_moves,
                "spread_multiple": spread_multiples,
            }
        )

        result = df.group_by("session", maintain_order=True).agg(
            pl.count().alias("n"),
            pl.mean("abs_move").alias("mean_abs"),
            pl.median("abs_move").alias("median_abs"),
            pl.std("abs_move").alias("std_abs"),
            pl.quantile("abs_move", 0.90).alias("p90_abs"),
            pl.mean("spread_multiple").alias("mean_spread_multiple"),
        )

        result = result.with_columns(
            pl.when(pl.col("median_abs") > _EPS)
            .then(pl.col("mean_abs") / pl.col("median_abs"))
            .otherwise(1.0)
            .alias("aer")
        )

        return result

    # ── summarize_by_key ───────────────────────────────────────────────────

    def summarize_by_key(self) -> pl.DataFrame:
        """Group observations by the exogenous composite key.

        The composite key encodes the full feature vector:
        ``session | fixing_window | rollover | liquidity_void | news_proxy``.

        Returns
        -------
        pl.DataFrame
            Columns:
                exogenous_key   str  – pipe-delimited composite key
                session         str  – session label (first in group)
                fixing_window   str  – "None" or fixing window label
                rollover        bool – rollover flag
                liquidity_void  bool – liquidity void flag
                news_proxy      bool – news proxy flag
                n               i64  – observation count
                mean_abs        f64  – mean of absolute forward move
                aer             f64  – amplitude excess ratio
                spread_multiple f64  – mean of (abs_move / state.spread)
        """
        if not self._records:
            return self._empty_summarize_by_key()

        keys: List[str] = []
        sessions: List[str] = []
        fixing_windows: List[str] = []
        rollovers: List[bool] = []
        liquidity_voids: List[bool] = []
        news_proxies: List[bool] = []
        abs_moves: List[float] = []
        spread_multiples: List[float] = []

        for obs in self._records:
            st = obs.state
            keys.append(make_key(st))
            sessions.append(st.session)
            fixing_windows.append(
                "None" if st.fixing_window is None else st.fixing_window
            )
            rollovers.append(st.rollover)
            liquidity_voids.append(st.liquidity_void)
            news_proxies.append(st.news_proxy)
            abs_moves.append(obs.abs_move)
            spread = st.spread
            spread_multiples.append(
                obs.abs_move / spread if spread > _EPS else 0.0
            )

        df = pl.DataFrame(
            {
                "exogenous_key": keys,
                "session": sessions,
                "fixing_window": fixing_windows,
                "rollover": rollovers,
                "liquidity_void": liquidity_voids,
                "news_proxy": news_proxies,
                "abs_move": abs_moves,
                "spread_multiple": spread_multiples,
            }
        )

        # First-pass aggregate: group by key, pull first label values
        # and compute mean_abs + spread_multiple
        result = df.group_by("exogenous_key", maintain_order=True).agg(
            pl.first("session").alias("session"),
            pl.first("fixing_window").alias("fixing_window"),
            pl.first("rollover").alias("rollover"),
            pl.first("liquidity_void").alias("liquidity_void"),
            pl.first("news_proxy").alias("news_proxy"),
            pl.count().alias("n"),
            pl.mean("abs_move").alias("mean_abs"),
            pl.mean("spread_multiple").alias("spread_multiple"),
        )

        # Compute median per key for AER
        median_df = df.group_by("exogenous_key").agg(
            pl.median("abs_move").alias("median_abs"),
        )

        result = result.join(median_df, on="exogenous_key", how="left")

        result = result.with_columns(
            pl.when(pl.col("median_abs") > _EPS)
            .then(pl.col("mean_abs") / pl.col("median_abs"))
            .otherwise(1.0)
            .alias("aer")
        ).drop("median_abs")

        result = result.select(
            "exogenous_key",
            "session",
            "fixing_window",
            "rollover",
            "liquidity_void",
            "news_proxy",
            "n",
            "mean_abs",
            "aer",
            "spread_multiple",
        )

        return result

    # ── reset ──────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all previously recorded observations."""
        self._records.clear()

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_summarize() -> pl.DataFrame:
        return pl.DataFrame(
            {
                "session": pl.Series([], dtype=pl.Utf8),
                "n": pl.Series([], dtype=pl.Int64),
                "mean_abs": pl.Series([], dtype=pl.Float64),
                "median_abs": pl.Series([], dtype=pl.Float64),
                "std_abs": pl.Series([], dtype=pl.Float64),
                "p90_abs": pl.Series([], dtype=pl.Float64),
                "mean_spread_multiple": pl.Series([], dtype=pl.Float64),
                "aer": pl.Series([], dtype=pl.Float64),
            }
        )

    @staticmethod
    def _empty_summarize_by_key() -> pl.DataFrame:
        return pl.DataFrame(
            {
                "exogenous_key": pl.Series([], dtype=pl.Utf8),
                "session": pl.Series([], dtype=pl.Utf8),
                "fixing_window": pl.Series([], dtype=pl.Utf8),
                "rollover": pl.Series([], dtype=pl.Boolean),
                "liquidity_void": pl.Series([], dtype=pl.Boolean),
                "news_proxy": pl.Series([], dtype=pl.Boolean),
                "n": pl.Series([], dtype=pl.Int64),
                "mean_abs": pl.Series([], dtype=pl.Float64),
                "aer": pl.Series([], dtype=pl.Float64),
                "spread_multiple": pl.Series([], dtype=pl.Float64),
            }
        )
