"""
Phase V — Amplitude Surface Construction
=========================================
Builds conditional amplitude surfaces from cached amplitude records.
For each (state_hash, horizon) bucket, computes distributional statistics
of forward amplitude moves and stores them as ``AmplitudeSurfaceEntry``
objects for fast downstream query.
"""

import sys

sys.path.insert(0, ".")
sys.path.insert(0, "proxima_x")

import os
import pickle
from typing import Any, Dict

import numpy as np
import polars as pl

from research.amplitude.schemas import (
    AmplitudeState,
    AmplitudeSurfaceEntry,
    StateHasher,
    SurfaceStats,
)

# AmplitudeCache may not exist at import time (created by a sibling module).
# We attempt the import so that any code relying on it can still reference
# the class, but gracefully degrade if the cache module is absent.
try:
    from research.amplitude.amplitude_cache import AmplitudeCache  # noqa: F401
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Amplitude Surface Engine
# ---------------------------------------------------------------------------


class AmplitudeSurfaceEngine:
    """Build, query, persist and analyse conditional amplitude surfaces.

    The engine ingests a flat DataFrame of observations (the output format
    produced by ``AmplitudeCache``), groups by *(state_hash, horizon)*, and
    computes a rich set of statistics for each bucket.  Results are stored
    in-memory as ``AmplitudeSurfaceEntry`` and can be saved / loaded via
    pickle.
    """

    # Threshold values for exceed-probability computation.
    _EXCEED_KS: tuple = (1.0, 1.5, 2.0, 3.0, 4.0)

    def __init__(self) -> None:
        # state_hash -> horizon -> AmplitudeSurfaceEntry
        self._surfaces: dict[str, dict[int, AmplitudeSurfaceEntry]] = {}
        self._stats: SurfaceStats = SurfaceStats(
            total_observations=0,
            n_buckets=0,
            n_horizons=0,
            symbols=[],
            coverage_pct=0.0,
        )

    # ------------------------------------------------------------------
    # Fit — build the surface from raw observations
    # ------------------------------------------------------------------

    def fit(self, df: pl.DataFrame) -> None:
        """Build the amplitude surface from cached observation records.

        Parameters
        ----------
        df : pl.DataFrame
            Flat observation table with *at least* the columns:

            - ``state_hash`` : str      — composite key from ``StateHasher``
            - ``horizon``    : int      — forward-looking horizon (seconds)
            - ``abs_move``   : float    — absolute price move (bps)
            - ``spread``     : float    — quoted spread at observation time (bps)
            - ``symbol``     : str      — instrument identifier (optional)

            Additional columns are silently ignored.

        Effects
        -------
        Populates :attr:`_surfaces` and :attr:`_stats`.
        """
        required = {"state_hash", "horizon", "abs_move", "spread"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"DataFrame missing required columns: {missing}"
            )

        # ---- 1. Unconditional std per horizon (used for AER) -------------
        unconditional_std = (
            df.group_by("horizon")
            .agg(pl.col("abs_move").std().alias("unconditional_std_abs"))
        )
        std_map: dict[int, float] = dict(
            zip(
                unconditional_std["horizon"].to_list(),
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
        for k in self._EXCEED_KS:
            aggs.append(
                (pl.col("abs_move") > k * pl.col("spread"))
                .mean()
                .alias(f"exceed_{k}")
            )

        grouped = df.group_by(["state_hash", "horizon"]).agg(aggs)

        # ---- 3. Assemble surface entries ---------------------------------
        self._surfaces = {}

        for row in grouped.iter_rows(named=True):
            state_hash: str = row["state_hash"]
            horizon: int = row["horizon"]
            n: int = row["n"]
            mean_abs: float = row["mean_abs"] if row["mean_abs"] is not None else 0.0
            median_abs: float = row["median_abs"] if row["median_abs"] is not None else 0.0
            std_abs: float = row["std_abs"] if row["std_abs"] is not None else 0.0
            p90_abs: float = row["p90_abs"] if row["p90_abs"] is not None else 0.0
            mean_spread: float = row["mean_spread"] if row["mean_spread"] is not None else 0.0

            # spread_multiple = mean_abs / mean_spread
            spread_multiple = mean_abs / mean_spread if mean_spread > 0 else 0.0

            # AER = conditional std / unconditional std (at same horizon)
            unconditional_std_abs = std_map.get(horizon, 1.0)
            if unconditional_std_abs is not None and unconditional_std_abs > 0:
                aer = std_abs / unconditional_std_abs
            else:
                aer = 1.0

            exceed_prob: dict[float, float] = {}
            for k in self._EXCEED_KS:
                val = row.get(f"exceed_{k}")
                exceed_prob[k] = float(val) if val is not None else 0.0

            entry = AmplitudeSurfaceEntry(
                state_hash=state_hash,
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

            if state_hash not in self._surfaces:
                self._surfaces[state_hash] = {}
            self._surfaces[state_hash][horizon] = entry

        # ---- 4. Compute SurfaceStats -------------------------------------
        symbols: list[str] = (
            sorted(df["symbol"].unique().to_list())
            if "symbol" in df.columns
            else []
        )
        total_obs: int = len(df)
        n_buckets: int = len(self._surfaces)

        all_horizons: set[int] = set()
        for horizons in self._surfaces.values():
            all_horizons.update(horizons.keys())
        n_horizons: int = len(all_horizons)

        # Coverage: fraction of (state_hash × horizon) cells that are populated
        possible_cells = n_buckets * n_horizons if n_horizons > 0 else 1
        populated_cells = sum(len(h) for h in self._surfaces.values())
        coverage_pct = populated_cells / possible_cells

        self._stats = SurfaceStats(
            total_observations=total_obs,
            n_buckets=n_buckets,
            n_horizons=n_horizons,
            symbols=symbols,
            coverage_pct=coverage_pct,
        )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(self, state_hash: str, horizon: int) -> dict:
        """Return the surface entry for a given *(state_hash, horizon)*.

        Parameters
        ----------
        state_hash : str
            Composite state key.
        horizon : int
            Forward-looking horizon in seconds.

        Returns
        -------
        dict
            Fields match ``AmplitudeSurfaceEntry`` attribute names, or an
            empty dict when the key is not present.
        """
        horizons = self._surfaces.get(state_hash)
        if horizons is None:
            return {}
        entry = horizons.get(horizon)
        if entry is None:
            return {}
        return self._entry_to_dict(entry)

    def lookup_by_state(
        self, oss_bucket: int, regime_id: str, horizon: int
    ) -> dict:
        """Look up a surface entry using a coarse *(bucket, regime)* key.

        Constructs a ``StateHasher`` digest from the two parameters (with
        placeholder values for the remaining state fields) and delegates to
        :meth:`lookup`.

        Parameters
        ----------
        oss_bucket : int
            Over-/under-shot bucket (0–9).
        regime_id : str
            Regime identifier (e.g. ``"high_velocity"``).
        horizon : int
            Forward-looking horizon in seconds.

        Returns
        -------
        dict
            Surface entry dict or empty dict if not found.
        """
        # Build a minimal AmplitudeState; only oss_bucket and regime_id carry
        # real signal — the remaining fields are zeroed out so that lookups
        # are reproducible for any caller that provides the same two values.
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
        return self.lookup(h, horizon)

    def get_stats(self) -> dict:
        """Return the surface-level roll-up statistics as a plain dict.

        Returns
        -------
        dict
            Keys match ``SurfaceStats`` field names.
        """
        return {
            "total_observations": self._stats.total_observations,
            "n_buckets": self._stats.n_buckets,
            "n_horizons": self._stats.n_horizons,
            "symbols": self._stats.symbols,
            "coverage_pct": self._stats.coverage_pct,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str = "cache/amplitude/surface.pkl") -> None:
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
                {"surfaces": self._surfaces, "stats": self._stats}, fh
            )

    def load(self, path: str = "cache/amplitude/surface.pkl") -> None:
        """Load a previously saved surface from disk.

        Parameters
        ----------
        path : str
            Source path.
        """
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        self._surfaces = data["surfaces"]
        self._stats = data["stats"]

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def coverage_report(self) -> pl.DataFrame:
        """Produce a per-state-hash coverage overview.

        Returns
        -------
        pl.DataFrame
            Columns:

            - ``state_hash``        — composite state key
            - ``n_horizons``        — number of distinct horizons observed
            - ``n_observations``    — total observations across all horizons
            - ``avg_aer``           — mean AER across horizons
            - ``avg_spread_multiple`` — mean spread-multiple across horizons

            Rows are sorted descending by ``n_observations``.
        """
        rows: list[dict[str, Any]] = []
        for state_hash, horizons in self._surfaces.items():
            entries = list(horizons.values())
            n_horizons = len(entries)
            n_obs = sum(e.n for e in entries)
            avg_aer = sum(e.aer for e in entries) / n_horizons
            avg_spread_multiple = (
                sum(e.spread_multiple for e in entries) / n_horizons
            )
            rows.append(
                {
                    "state_hash": state_hash,
                    "n_horizons": n_horizons,
                    "n_observations": n_obs,
                    "avg_aer": avg_aer,
                    "avg_spread_multiple": avg_spread_multiple,
                }
            )

        result = pl.DataFrame(rows)
        if len(result) > 0:
            result = result.sort("n_observations", descending=True)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _entry_to_dict(entry: AmplitudeSurfaceEntry) -> dict:
        """Convert a frozen dataclass to a plain dictionary."""
        return {
            "state_hash": entry.state_hash,
            "horizon": entry.horizon,
            "n": entry.n,
            "mean_abs": entry.mean_abs,
            "median_abs": entry.median_abs,
            "std_abs": entry.std_abs,
            "p90_abs": entry.p90_abs,
            "spread_multiple": entry.spread_multiple,
            "aer": entry.aer,
            "exceed_prob": entry.exceed_prob,
        }
