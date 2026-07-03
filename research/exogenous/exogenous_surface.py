"""
Program VI — Exogenous Amplitude Surface Construction
======================================================
Builds conditional amplitude surfaces grouped by exogenous session-state keys.
For each (exogenous_key, horizon_sec) bucket, computes distributional statistics
of forward absolute moves and stores them as ``ExogenousSurfaceEntry`` objects
for fast downstream projection and promotion.
"""

import sys

sys.path.insert(0, ".")
sys.path.insert(0, "proxima_x")

import os
import pickle
from typing import Any, Optional

import polars as pl

from research.exogenous.schemas import (
    ExogenousState,
    ExogenousSurfaceEntry,
    ExogenousStats,
    make_key,
)

# Exceed-probability thresholds used by the surface engine.
_EXCEED_KS: tuple = (2.0, 3.0)


class ExogenousAmplitudeSurface:
    """Build, query, persist and analyse exogenous amplitude surfaces.

    The engine ingests a flat DataFrame of observation records (the output
    format produced by ``SessionAmplitudeMapper``), groups by
    *(exogenous_key, horizon_sec)*, and computes a rich set of statistics
    for each bucket.  Results are stored in-memory as
    ``ExogenousSurfaceEntry`` and can be saved / loaded via pickle.

    Examples
    --------
    >>> surface = ExogenousAmplitudeSurface()
    >>> surface.fit(df)                          # doctest: +SKIP
    >>> entry = surface.lookup("TOKYO|None|False|False|False", 60)
    >>> surface.save("cache/exogenous/surface.pkl")
    """

    def __init__(self) -> None:
        # exogenous_key -> horizon_sec -> ExogenousSurfaceEntry
        self._surface: dict[str, dict[int, ExogenousSurfaceEntry]] = {}
        self._stats: Optional[ExogenousStats] = None

    # ------------------------------------------------------------------
    # Fit — build the surface from raw observations
    # ------------------------------------------------------------------

    def fit(self, df: pl.DataFrame) -> None:
        """Build the exogenous amplitude surface from observation records.

        Parameters
        ----------
        df : pl.DataFrame
            Flat observation table with **at least** the columns:

            - ``exogenous_key``   : str   — composite key from ``make_key()``
            - ``session``         : str   — TOKYO / LONDON / NEWYORK / DEADZONE
            - ``fixing_window``   : str   — ``"None"`` or fixing-window label
            - ``rollover``        : bool  — inside daily rollover window
            - ``liquidity_void``  : bool  — liquidity-void flag
            - ``news_proxy``      : bool  — news-proxy flag
            - ``horizon_sec``     : int   — forward horizon (seconds)
            - ``abs_move``        : float — absolute forward price move (bps)
            - ``spread``          : float — quoted spread at observation time (bps)

            Additional columns are silently ignored.

        Effects
        -------
        Populates :attr:`_surface` and :attr:`_stats`.
        """
        required = {
            "exogenous_key", "horizon_sec", "abs_move", "spread",
            "session", "fixing_window", "rollover",
            "liquidity_void", "news_proxy",
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
        std_map: dict[int, float] = dict(
            zip(
                unconditional_std["horizon_sec"].to_list(),
                unconditional_std["unconditional_std_abs"].to_list(),
            )
        )

        # ---- 2. Build aggregation expressions ----------------------------
        aggs: list[pl.Expr] = [
            pl.first("session").alias("session"),
            pl.first("fixing_window").alias("fixing_window"),
            pl.first("rollover").alias("rollover"),
            pl.first("liquidity_void").alias("liquidity_void"),
            pl.first("news_proxy").alias("news_proxy"),
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

        grouped = df.group_by(["exogenous_key", "horizon_sec"]).agg(aggs)

        # ---- 3. Assemble surface entries ---------------------------------
        self._surface = {}

        for row in grouped.iter_rows(named=True):
            key: str = row["exogenous_key"]
            horizon: int = row["horizon_sec"]
            n: int = row["n"]
            session: str = row["session"]
            fixing_window_raw: Any = row["fixing_window"]
            rollover: bool = row["rollover"]
            liquidity_void: bool = row["liquidity_void"]
            news_proxy: bool = row["news_proxy"]

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
            unconditional_std_abs = std_map.get(horizon, 1.0)
            if unconditional_std_abs is not None and unconditional_std_abs > 0:
                aer = std_abs / unconditional_std_abs
            else:
                aer = 1.0

            exceed_prob: dict[float, float] = {}
            for k in _EXCEED_KS:
                val = row.get(f"exceed_{k}")
                exceed_prob[k] = float(val) if val is not None else 0.0

            # Convert "None" string back to Python None for the entry
            fixing_window: Optional[str] = (
                None if fixing_window_raw == "None"
                else str(fixing_window_raw) if fixing_window_raw is not None
                else None
            )

            entry = ExogenousSurfaceEntry(
                session=session,
                fixing_window=fixing_window,
                rollover=rollover,
                liquidity_void=liquidity_void,
                news_proxy=news_proxy,
                horizon=horizon,
                n=n,
                mean_abs=mean_abs,
                median_abs=median_abs,
                std_abs=std_abs,
                p90_abs=p90_abs,
                spread_multiple=spread_multiple,
                aer=aer,
                exceed_prob=exceed_prob,
            )

            if key not in self._surface:
                self._surface[key] = {}
            self._surface[key][horizon] = entry

        # ---- 4. Compute ExogenousStats -----------------------------------
        all_horizons: set[int] = set()
        for horizons in self._surface.values():
            all_horizons.update(horizons.keys())

        self._stats = ExogenousStats(
            total_observations=len(df),
            n_sessions=len(self._surface),
            n_horizons=len(all_horizons),
            symbols=[],
        )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(self, exogenous_key: str, horizon: int) -> dict:
        """Return the surface entry for a given *(exogenous_key, horizon)*.

        Parameters
        ----------
        exogenous_key : str
            Composite key from ``make_key()`` or ``_build_key()``.
        horizon : int
            Forward-looking horizon in seconds.

        Returns
        -------
        dict
            Fields match ``ExogenousSurfaceEntry`` attribute names, or an
            empty dict when the key is not present.
        """
        horizons = self._surface.get(exogenous_key)
        if horizons is None:
            return {}
        entry = horizons.get(horizon)
        if entry is None:
            return {}
        return self._entry_to_dict(entry)

    def lookup_by_components(
        self,
        session: str,
        fixing_window: Optional[str],
        rollover: bool,
        liquidity_void: bool,
        news_proxy: bool,
        horizon: int,
    ) -> dict:
        """Build the exogenous composite key from components and look up.

        Parameters
        ----------
        session : str
            Session label (TOKYO / LONDON / NEWYORK / DEADZONE).
        fixing_window : str or None
            Fixing-window label or None.
        rollover : bool
            Rollover flag.
        liquidity_void : bool
            Liquidity-void flag.
        news_proxy : bool
            News-proxy flag.
        horizon : int
            Forward horizon in seconds.

        Returns
        -------
        dict
            Surface entry dict or empty dict if not found.
        """
        key = self._build_key(
            session, fixing_window, rollover, liquidity_void, news_proxy
        )
        return self.lookup(key, horizon)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str = "cache/exogenous/surface.pkl") -> None:
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
                {"surface": self._surface, "stats": self._stats}, fh
            )

    def load(self, path: str = "cache/exogenous/surface.pkl") -> None:
        """Load a previously saved surface from disk.

        Parameters
        ----------
        path : str
            Source path.
        """
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        self._surface = data["surface"]
        self._stats = data["stats"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _entry_to_dict(entry: ExogenousSurfaceEntry) -> dict:
        """Convert a frozen dataclass to a plain dictionary."""
        return {
            "session": entry.session,
            "fixing_window": entry.fixing_window,
            "rollover": entry.rollover,
            "liquidity_void": entry.liquidity_void,
            "news_proxy": entry.news_proxy,
            "horizon": entry.horizon,
            "n": entry.n,
            "mean_abs": entry.mean_abs,
            "median_abs": entry.median_abs,
            "std_abs": entry.std_abs,
            "p90_abs": entry.p90_abs,
            "spread_multiple": entry.spread_multiple,
            "aer": entry.aer,
            "exceed_prob": dict(entry.exceed_prob),
        }

    @staticmethod
    def _build_key(
        session: str,
        fixing_window: Optional[str],
        rollover: bool,
        liquidity_void: bool,
        news_proxy: bool,
    ) -> str:
        """Deterministic composite key from exogenous components.

        Matches the format produced by ``make_key()`` in :mod:`schemas`.
        """
        fixing = "None" if fixing_window is None else fixing_window
        return "|".join(
            [
                session,
                fixing,
                str(rollover),
                str(liquidity_void),
                str(news_proxy),
            ]
        )
