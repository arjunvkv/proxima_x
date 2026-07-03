"""
Program VI.5 — Event Amplitude Surface
========================================
Builds conditional amplitude surfaces grouped by event proximity buckets.
For each (event_bucket, event_impact, horizon_sec) cell, computes distributional
statistics of forward absolute moves for downstream projection and promotion.

Input columns (from event_amplitude_cache):
    event_bucket, event_impact, currency_match, horizon_sec, abs_move, spread
"""

import sys; sys.path.insert(0, "."); sys.path.insert(0, "proxima_x")

import os
import pickle
from typing import Any, Optional

import polars as pl

# Exceed-probability thresholds used by the surface engine.
_EXCEED_KS: tuple = (2.0, 3.0)
_STABILITY_EXCEED_THRESHOLD: float = 0.3


def _compute_stability(exceed_prob: dict) -> float:
    """Fraction of k values where exceed_prob[k] > 0.3.
    Mirrors the helper in ``research.exogenous.promotion_engine``.
    """
    if not exceed_prob:
        return 0.0
    passing = sum(
        1 for v in exceed_prob.values() if v > _STABILITY_EXCEED_THRESHOLD
    )
    return passing / len(exceed_prob)


class EventSurface:
    """Build, query, persist and analyse event amplitude surfaces.

    Examples
    --------
    >>> surface = EventSurface()
    >>> surface.fit(df)                          # doctest: +SKIP
    >>> entry = surface.lookup("PRE_15M", "HIGH", 60)
    >>> surface.save("cache/exogenous/events/surface.pkl")
    """

    def __init__(self) -> None:
        # key (bucket|impact) -> horizon_sec -> dict of stats
        self._surface: dict[str, dict[int, dict]] = {}
        self._unconditional_std_map: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Fit — build the surface from raw observations
    # ------------------------------------------------------------------

    def fit(self, df: pl.DataFrame) -> None:
        """Build the event amplitude surface from observation records.

        Parameters
        ----------
        df : pl.DataFrame
            Flat observation table with at least the columns:
            - ``event_bucket``    : str   — proximity bucket label
            - ``event_impact``    : str   — HIGH / MEDIUM / LOW
            - ``currency_match``  : bool  — event currency matches trading pair
            - ``horizon_sec``     : int   — forward horizon (seconds)
            - ``abs_move``        : float — absolute forward price move (bps)
            - ``spread``          : float — quoted spread at observation (bps)

        Effects
        -------
        Populates :attr:`_surface` and :attr:`_unconditional_std_map`.
        """
        required = {
            "event_bucket", "event_impact", "currency_match",
            "horizon_sec", "abs_move", "spread",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"DataFrame missing required columns: {missing}"
            )

        # ---- 1. Unconditional std per horizon (used for AER) -------------
        unconditional_std = (
            df.group_by("horizon_sec")
            .agg(pl.col("abs_move").std().alias("unconditional_std_abs"))
        )
        self._unconditional_std_map = dict(
            zip(
                unconditional_std["horizon_sec"].to_list(),
                unconditional_std["unconditional_std_abs"].to_list(),
            )
        )

        # ---- 2. Build aggregation expressions ----------------------------
        aggs: list[pl.Expr] = [
            pl.count().alias("n"),
            pl.col("abs_move").mean().alias("mean_abs"),
            pl.col("abs_move").median().alias("median_abs"),
            pl.col("abs_move").std().alias("std_abs"),
            pl.col("abs_move").quantile(0.90).alias("p90_abs"),
            pl.col("spread").mean().alias("mean_spread"),
        ]
        for k in _EXCEED_KS:
            aggs.append(
                (pl.col("abs_move") > k * pl.col("spread"))
                .mean()
                .alias(f"exceed_{k}")
            )

        grouped = df.group_by(
            ["event_bucket", "event_impact", "horizon_sec"]
        ).agg(aggs)

        # ---- 3. Assemble surface entries ---------------------------------
        self._surface = {}

        for row in grouped.iter_rows(named=True):
            bucket: str = row["event_bucket"]
            impact: str = row["event_impact"]
            horizon: int = row["horizon_sec"]
            n: int = row["n"]

            mean_abs: float = (
                row["mean_abs"] if row["mean_abs"] is not None else 0.0
            )
            median_abs: float = (
                row["median_abs"] if row["median_abs"] is not None else 0.0
            )
            std_abs: float = (
                row["std_abs"] if row["std_abs"] is not None else 0.0
            )
            p90_abs: float = (
                row["p90_abs"] if row["p90_abs"] is not None else 0.0
            )
            mean_spread: float = (
                row["mean_spread"] if row["mean_spread"] is not None else 0.0
            )

            # spread_multiple = mean_abs / mean_spread
            spread_multiple = (
                mean_abs / mean_spread if mean_spread > 0 else 0.0
            )

            # AER = conditional std / unconditional std (at same horizon)
            unconditional_std_abs = self._unconditional_std_map.get(
                horizon, 1.0
            )
            if unconditional_std_abs is not None and unconditional_std_abs > 0:
                aer = std_abs / unconditional_std_abs
            else:
                aer = 1.0

            exceed_prob: dict[float, float] = {}
            for k in _EXCEED_KS:
                val = row.get(f"exceed_{k}")
                exceed_prob[k] = float(val) if val is not None else 0.0

            stability = _compute_stability(exceed_prob)

            # Build entry dictionary
            entry: dict[str, Any] = {
                "event_bucket": bucket,
                "event_impact": impact,
                "horizon": horizon,
                "n": n,
                "mean_abs": mean_abs,
                "median_abs": median_abs,
                "std_abs": std_abs,
                "p90_abs": p90_abs,
                "mean_spread": mean_spread,
                "spread_multiple": spread_multiple,
                "aer": aer,
                "exceed_prob": exceed_prob,
                "exceed_2": exceed_prob.get(2.0, 0.0),
                "exceed_3": exceed_prob.get(3.0, 0.0),
                "stability": stability,
            }

            key = f"{bucket}|{impact}"
            if key not in self._surface:
                self._surface[key] = {}
            self._surface[key][horizon] = entry

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(
        self, event_bucket: str, impact: str, horizon: int
    ) -> dict:
        """Return the surface entry for a given (bucket, impact, horizon).

        Parameters
        ----------
        event_bucket : str
            Proximity bucket label (e.g. PRE_15M, EVENT_0_2M, POST_5M).
        impact : str
            Event impact level (e.g. HIGH, MEDIUM, LOW).
        horizon : int
            Forward-looking horizon in seconds.

        Returns
        -------
        dict
            Entry fields or an empty dict when the key is not present.
        """
        key = f"{event_bucket}|{impact}"
        horizons = self._surface.get(key)
        if horizons is None:
            return {}
        entry = horizons.get(horizon)
        if entry is None:
            return {}
        return dict(entry)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self, path: str = "cache/exogenous/events/surface.pkl"
    ) -> None:
        """Pickle the surface data to disk.

        Parameters
        ----------
        path : str
            Destination path.  Parent directories are created if they do
            not exist.
        """
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "surface": self._surface,
                    "unconditional_std_map": self._unconditional_std_map,
                },
                fh,
            )

    def load(
        self, path: str = "cache/exogenous/events/surface.pkl"
    ) -> None:
        """Load a previously saved surface from disk.

        Parameters
        ----------
        path : str
            Source path.
        """
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        self._surface = data["surface"]
        self._unconditional_std_map = data.get("unconditional_std_map", {})

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def get_bucket_summary(self) -> pl.DataFrame:
        """Return a DataFrame grouped by ``event_bucket`` across all impacts.

        Returns
        -------
        pl.DataFrame
            Columns: event_bucket, n_entries, mean_aer, mean_spread_multiple,
            mean_n, n_impacts, mean_stability.
        """
        rows: list[dict] = []
        bucket_groups: dict[str, list[dict]] = {}

        for key, horizons in self._surface.items():
            bucket = key.split("|")[0]
            for horizon, entry in horizons.items():
                bucket_groups.setdefault(bucket, []).append(entry)

        for bucket, entries in bucket_groups.items():
            n_entries = len(entries)
            mean_aer = sum(e["aer"] for e in entries) / n_entries
            mean_sm = sum(
                e["spread_multiple"] for e in entries
            ) / n_entries
            mean_n = sum(e["n"] for e in entries) / n_entries
            mean_stability = sum(
                e["stability"] for e in entries
            ) / n_entries
            impacts = set(
                e["event_impact"] for e in entries
            )
            rows.append(
                {
                    "event_bucket": bucket,
                    "n_entries": n_entries,
                    "mean_aer": round(mean_aer, 4),
                    "mean_spread_multiple": round(mean_sm, 4),
                    "mean_n": round(mean_n, 1),
                    "n_impacts": len(impacts),
                    "mean_stability": round(mean_stability, 4),
                }
            )

        if not rows:
            return pl.DataFrame(
                {
                    "event_bucket": pl.Series([], dtype=pl.Utf8),
                    "n_entries": pl.Series([], dtype=pl.Int64),
                    "mean_aer": pl.Series([], dtype=pl.Float64),
                    "mean_spread_multiple": pl.Series([], dtype=pl.Float64),
                    "mean_n": pl.Series([], dtype=pl.Float64),
                    "n_impacts": pl.Series([], dtype=pl.Int64),
                    "mean_stability": pl.Series([], dtype=pl.Float64),
                }
            )

        return pl.DataFrame(rows)

    def get_best_buckets(self, min_n: int = 10) -> pl.DataFrame:
        """Return surface entries sorted by ``spread_multiple`` descending.

        Only entries with ``n >= min_n`` are included.

        Parameters
        ----------
        min_n : int
            Minimum observation count for an entry to qualify.

        Returns
        -------
        pl.DataFrame
            All entry columns, sorted by spread_multiple descending.
        """
        rows: list[dict] = []
        for key, horizons in self._surface.items():
            bucket = key.split("|")[0]
            impact = key.split("|")[1] if "|" in key else ""
            for horizon, entry in horizons.items():
                if entry["n"] >= min_n:
                    rows.append(
                        {
                            "event_bucket": bucket,
                            "event_impact": impact,
                            "horizon": horizon,
                            **{k: v for k, v in entry.items()
                               if k not in ("event_bucket", "event_impact",
                                            "horizon", "exceed_prob")},
                        }
                    )

        if not rows:
            return pl.DataFrame(
                {
                    "event_bucket": pl.Series([], dtype=pl.Utf8),
                    "event_impact": pl.Series([], dtype=pl.Utf8),
                    "horizon": pl.Series([], dtype=pl.Int64),
                    "n": pl.Series([], dtype=pl.Int64),
                    "mean_abs": pl.Series([], dtype=pl.Float64),
                    "median_abs": pl.Series([], dtype=pl.Float64),
                    "std_abs": pl.Series([], dtype=pl.Float64),
                    "p90_abs": pl.Series([], dtype=pl.Float64),
                    "mean_spread": pl.Series([], dtype=pl.Float64),
                    "spread_multiple": pl.Series([], dtype=pl.Float64),
                    "aer": pl.Series([], dtype=pl.Float64),
                    "mean_spread": pl.Series([], dtype=pl.Float64),
                    "exceed_2": pl.Series([], dtype=pl.Float64),
                    "exceed_3": pl.Series([], dtype=pl.Float64),
                    "stability": pl.Series([], dtype=pl.Float64),
                }
            )

        df = pl.DataFrame(rows)
        if not df.is_empty() and "spread_multiple" in df.columns:
            df = df.sort("spread_multiple", descending=True)
        return df
