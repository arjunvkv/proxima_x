"""
Program VI — Exogenous Amplitude Discovery Sweeper
====================================================
Runs parameter sweeps over exogenous surface construction and promotion
evaluation.  The sweeper automates the end-to-end pipeline from cache
building through surface fitting to promotion scoring across multiple
experimental configurations.

Typical usage::

    from research.exogenous.exogenous_sweeper import ExogenousSweeper

    sweeper = ExogenousSweeper(output_dir="outputs")
    results = sweeper.run_first_test()
    print(results)
"""

import sys

sys.path.insert(0, ".")
sys.path.insert(0, "proxima_x")

import os
import time as time_module
from typing import Optional

import numpy as np
import polars as pl

from research.exogenous.exogenous_cache import ExogenousCache
from research.exogenous.exogenous_surface import ExogenousAmplitudeSurface
from research.exogenous.promotion_engine import ExogenousPromotionEngine
from research.exogenous.schemas import ExogenousSurfaceEntry

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
_FIRST_TEST_WINDOWS: list[dict] = [
    {
        "label": "London Open",
        "session": "LONDON",
        "fixing_window": "None",
        "rollover": False,
        "liquidity_void": False,
        "news_proxy": False,
    },
    {
        "label": "WM Fix",
        "session": "LONDON",
        "fixing_window": "WM_FIX",
        "rollover": False,
        "liquidity_void": False,
        "news_proxy": False,
    },
    {
        "label": "Rollover",
        "session": "TOKYO",
        "fixing_window": "None",
        "rollover": True,
        "liquidity_void": False,
        "news_proxy": False,
    },
    {
        "label": "News Proxy",
        "session": "NEWYORK",
        "fixing_window": "None",
        "rollover": False,
        "liquidity_void": False,
        "news_proxy": True,
    },
]

_FIRST_TEST_HORIZONS: list[int] = [60, 300, 900]


# ===================================================================
# ExogenousSweeper
# ===================================================================


class ExogenousSweeper:
    """Parameter sweeper for Program VI exogenous amplitude discovery.

    Parameters
    ----------
    max_workers : int
        Maximum number of parallel workers (reserved for future use;
        currently all runs are sequential).
    output_dir : str
        Directory where result parquet files are saved.
    """

    def __init__(
        self, max_workers: int = 4, output_dir: str = "outputs"
    ) -> None:
        self.max_workers = max_workers
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # First-pass test
    # ------------------------------------------------------------------

    def run_first_test(self) -> pl.DataFrame:
        """Run the first-pass exogenous amplitude discovery test.

        Steps
        -----
        1. Build exogenous cache for EURJPY (April 1–7, 50000 ticks).
           Falls back to synthetic data if the real replay environment
           is unavailable.
        2. Fit :class:`ExogenousAmplitudeSurface` on the cached data.
        3. For every surface entry, run :class:`ExogenousPromotionEngine`.
        4. Save the combined results to ``{output_dir}/exogenous_first_test.parquet``.
        5. Return the results DataFrame.

        The synthetic fallback generates observations for four exogenous
        windows (London Open, WM Fix, Rollover, News Proxy) at three
        horizons [60, 300, 900] — a total of 12 experiments.

        Returns
        -------
        pl.DataFrame
            Promotion evaluation results with surface entry metadata.
        """
        t_start = time_module.perf_counter()
        print("[ExogenousSweeper] run_first_test — begin")

        # ---- 1. Build exogenous cache ------------------------------------
        df_obs = self._build_cache_or_fallback()

        if df_obs.is_empty():
            print(
                "[ExogenousSweeper] WARNING: observation DataFrame is empty; "
                "cannot fit surface."
            )
            return pl.DataFrame()

        print(
            f"[ExogenousSweeper] Cache returned {len(df_obs)} observations "
            f"with {df_obs['exogenous_key'].n_unique()} unique keys"
        )

        # ---- 2. Fit surface ----------------------------------------------
        surface = ExogenousAmplitudeSurface()
        surface.fit(df_obs)
        print("[ExogenousSweeper] Surface fitted")

        # ---- 3. Collect all surface entries + evaluate --------------------
        entries: list[ExogenousSurfaceEntry] = []
        for key, horizons in surface._surface.items():
            for horizon, entry in horizons.items():
                entries.append(entry)

        if not entries:
            print("[ExogenousSweeper] WARNING: no surface entries to evaluate")
            return pl.DataFrame()

        engine = ExogenousPromotionEngine()
        promo_df = engine.evaluate_batch(entries)
        print(
            f"[ExogenousSweeper] Evaluated {len(promo_df)} entries, "
            f"{promo_df['promoted'].sum()} promoted"
        )

        # ---- 4. Combine with surface entry metadata -----------------------
        results = self._enrich_with_surface_fields(promo_df, surface)

        # ---- 5. Save ------------------------------------------------------
        save_path = os.path.join(
            self.output_dir, "exogenous_first_test.parquet"
        )
        results.write_parquet(save_path, compression="zstd")
        print(f"[ExogenousSweeper] Results saved -> {save_path}")

        elapsed = time_module.perf_counter() - t_start
        print(f"[ExogenousSweeper] run_first_test completed in {elapsed:.1f}s")
        return results

    # ------------------------------------------------------------------
    # Full sweep (placeholder)
    # ------------------------------------------------------------------

    def run_full(self) -> pl.DataFrame:
        """Run the full parameter sweep across all symbols, folds, and
        window configurations.

        .. note::

            This is a placeholder for future implementation.
        """
        print("[ExogenousSweeper] run_full — not yet implemented")
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
                "session": pl.Series([], dtype=pl.Utf8),
                "mean_abs": pl.Series([], dtype=pl.Float64),
                "median_abs": pl.Series([], dtype=pl.Float64),
                "std_abs": pl.Series([], dtype=pl.Float64),
                "p90_abs": pl.Series([], dtype=pl.Float64),
            }
        )

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _build_cache_or_fallback(self) -> pl.DataFrame:
        """Try to build the real ExogenousCache; fall back to synthetic data.

        Returns
        -------
        pl.DataFrame
            Observation DataFrame matching the surface ``fit()`` schema.
        """
        try:
            cache = ExogenousCache(base_path="cache/exogenous")
            df = cache.build(
                symbols=["EURJPY"],
                start="2025-04-01",
                end="2025-04-07",
                horizons=_FIRST_TEST_HORIZONS,
                n_ticks=50000,
                warmup=5000,
                seed=42,
            )
            if not df.is_empty():
                return df
            print(
                "[ExogenousSweeper] ExogenousCache returned empty DF; "
                "falling back to synthetic data"
            )
        except Exception as exc:
            print(
                f"[ExogenousSweeper] ExogenousCache.build failed ({exc}); "
                f"falling back to synthetic data"
            )

        return self._generate_synthetic_observations()

    def _generate_synthetic_observations(
        self, n_per_window: int = 200
    ) -> pl.DataFrame:
        """Generate synthetic observation data for a deterministic test.

        Produces observations for the four first-test windows at each of
        the three horizons with controlled statistical properties.
        """
        rng = np.random.default_rng(42)

        records: list[dict] = []

        # Base distribution parameters per window label
        window_params = {
            "London Open": {"mean_abs": 3.0, "spread": 0.75, "std_abs": 0.8},
            "WM Fix":      {"mean_abs": 6.0, "spread": 1.00, "std_abs": 1.5},
            "Rollover":    {"mean_abs": 1.5, "spread": 1.00, "std_abs": 0.4},
            "News Proxy":  {"mean_abs": 5.0, "spread": 1.20, "std_abs": 1.2},
        }

        for win_spec in _FIRST_TEST_WINDOWS:
            label = win_spec["label"]
            session = win_spec["session"]
            fixing_window = win_spec["fixing_window"]
            rollover = win_spec["rollover"]
            liquidity_void = win_spec["liquidity_void"]
            news_proxy = win_spec["news_proxy"]

            # Build the exogenous key
            key = "|".join([
                session,
                fixing_window,
                str(rollover),
                str(liquidity_void),
                str(news_proxy),
            ])

            params = window_params.get(label, {"mean_abs": 3.0, "spread": 1.0, "std_abs": 1.0})

            for horizon in _FIRST_TEST_HORIZONS:
                # Sample gamma-distributed absolute moves (positive, right-skewed)
                # Scale so that mean approximates the target mean_abs
                shape = 4.0
                scale = params["mean_abs"] / shape
                abs_moves = rng.gamma(shape, scale, size=n_per_window)

                # Sample spreads with some noise
                spreads = params["spread"] + rng.normal(
                    0, params["spread"] * 0.15, size=n_per_window
                )
                spreads = np.clip(spreads, params["spread"] * 0.3, None)

                # Base timestamp: April 1-7 2025 in epoch seconds
                base_ts = 1743465600.0  # 2025-04-01 00:00:00 UTC

                for i in range(n_per_window):
                    state_ts = base_ts + i * 60.0 + rng.uniform(0, 10.0)
                    records.append({
                        "symbol": "EURJPY",
                        "state_ts": state_ts,
                        "session": session,
                        "fixing_window": fixing_window,
                        "rollover": rollover,
                        "liquidity_void": liquidity_void,
                        "news_proxy": news_proxy,
                        "spread": float(spreads[i]),
                        "tick_velocity": float(rng.exponential(0.5)),
                        "exogenous_key": key,
                        "horizon_sec": horizon,
                        "abs_move": float(abs_moves[i]),
                        "signed_move": float(rng.normal(0, abs_moves[i] / 2)),
                    })

        df = pl.DataFrame(records)
        print(
            f"[ExogenousSweeper] Generated {len(df)} synthetic observations "
            f"({df['exogenous_key'].n_unique()} unique keys)"
        )
        return df

    @staticmethod
    def _enrich_with_surface_fields(
        promo_df: pl.DataFrame, surface: ExogenousAmplitudeSurface
    ) -> pl.DataFrame:
        """Add surface entry detail columns to the promotion results.

        Parses the ``exogenous_key`` to extract session and merges surface
        entry statistics (mean_abs, median_abs, std_abs, p90_abs) onto
        the promotion evaluation output.
        """
        if promo_df.is_empty():
            return promo_df

        # Parse session from exogenous_key (first pipe-delimited field)
        rows = promo_df.to_dicts()
        enriched_rows: list[dict] = []

        for row in rows:
            key: str = row["exogenous_key"]
            horizon: int = row["horizon"]

            # Extract session from key
            parts = key.split("|")
            session = parts[0] if len(parts) >= 1 else "UNKNOWN"

            # Look up surface entry for additional stats
            entry_dict = surface.lookup(key, horizon)

            enriched_rows.append({
                "exogenous_key": key,
                "horizon": horizon,
                "session": session,
                "fixing_window": parts[1] if len(parts) >= 2 else "None",
                "rollover": parts[2] == "True" if len(parts) >= 3 else False,
                "liquidity_void": parts[3] == "True" if len(parts) >= 4 else False,
                "news_proxy": parts[4] == "True" if len(parts) >= 5 else False,
                "aer": row["aer"],
                "spread_multiple": row["spread_multiple"],
                "n": row["n"],
                "tier": row["tier"],
                "promoted": row["promoted"],
                "reasons": row["reasons"],
                "failures": row["failures"],
                "mean_abs": entry_dict.get("mean_abs", 0.0),
                "median_abs": entry_dict.get("median_abs", 0.0),
                "std_abs": entry_dict.get("std_abs", 0.0),
                "p90_abs": entry_dict.get("p90_abs", 0.0),
            })

        return pl.DataFrame(enriched_rows)


# ===================================================================
# __main__ — run first-pass test
# ===================================================================
if __name__ == "__main__":
    t0 = time_module.perf_counter()
    sweeper = ExogenousSweeper()
    results = sweeper.run_first_test()
    print(f"Completed in {time_module.perf_counter() - t0:.1f}s")
    print(results)
