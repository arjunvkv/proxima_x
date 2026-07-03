"""
Tick Time Machine integration for Phase V — orchestrates replay for amplitude
data collection across walk-forward folds.

The :class:`AmplitudeReplayRunner` drives replay environments for each fold
in a cross-validation scheme, delegates amplitude computation to
:class:`~research.amplitude.amplitude_cache.AmplitudeCache`, and persists
the resulting DataFrames as Parquet files for downstream surface-building
and regime analysis.

Typical usage::

    from research.amplitude.replay_runner import AmplitudeReplayRunner

    runner = AmplitudeReplayRunner()
    folds = AmplitudeReplayRunner.create_folds_from_daterange(
        "2025-01-01", "2025-06-01", n_folds=3, train_ratio=0.5
    )
    results = runner.run_all_folds(
        folds, symbols=["EURJPY", "USDJPY"],
        horizons=[60, 300, 600], n_ticks=80000,
    )
    for fold_id, df in results.items():
        print(f"Fold {fold_id}: {len(df)} observations")
"""

from __future__ import annotations

import sys
import os
import logging
import time
from typing import Optional

import polars as pl

sys.path.insert(0, ".")
sys.path.insert(0, "proxima_x")

from replay.environment import build_replay_environment, ReplayConfig
from replay.replay_feed import ReplayFeed
from replay.clock_patcher import patch_clock, unpatch_clock
from research.amplitude.amplitude_cache import AmplitudeCache

logger = logging.getLogger("proxima.amplitude.replay_runner")

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
_DEFAULT_HORIZONS = [60, 300, 600, 1800, 3600]
_DEFAULT_SYMBOLS = ["EURJPY", "USDJPY", "GBPJPY", "XAUUSD", "EURUSD"]


# ===================================================================
# AmplitudeReplayRunner
# ===================================================================


class AmplitudeReplayRunner:
    """Orchestrates replay-driven amplitude data collection across walk-forward
    folds.

    Parameters
    ----------
    base_cache_path : str
        Root directory where per-fold per-symbol Parquet files are written.
        Defaults to ``"cache/amplitude"``.
    """

    def __init__(self, base_cache_path: str = "cache/amplitude") -> None:
        self.base_cache_path = base_cache_path
        os.makedirs(self.base_cache_path, exist_ok=True)
        logger.info(
            "AmplitudeReplayRunner initialized – cache=%s",
            self.base_cache_path,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_fold(
        self,
        fold_id: int,
        symbols: list[str],
        start: str,
        end: str,
        horizons: Optional[list[int]] = None,
        n_ticks: int = 80000,
        warmup: int = 5000,
        seed: int = 42,
    ) -> pl.DataFrame:
        """Run amplitude collection for a single walk-forward fold.

        Steps
        -----
        1. Call :meth:`AmplitudeCache.build` with the provided parameters.
        2. Save the result to ``{base_cache_path}/{symbol}_{fold_id}.parquet``
           for each symbol present in the returned DataFrame.
        3. Return the concatenated DataFrame.

        Parameters
        ----------
        fold_id : int
            Fold index (used for file naming and metadata).
        symbols : list of str
            Symbols to process.
        start : str
            Fold start date in ``YYYY-MM-DD`` format.
        end : str
            Fold end date in ``YYYY-MM-DD`` format.
        horizons : list of int, optional
            Forward-looking horizons in seconds.
        n_ticks : int
            Maximum ticks to process (passed to replay).
        warmup : int
            Tick warmup before collecting observations.
        seed : int
            Random seed for replay determinism.

        Returns
        -------
        pl.DataFrame
            Amplitude observations with columns documented in
            :meth:`AmplitudeCache.build`.
        """
        if horizons is None:
            horizons = list(_DEFAULT_HORIZONS)

        logger.info(
            "run_fold – fold=%d symbols=%s [%s → %s] "
            "horizons=%s n_ticks=%d warmup=%d seed=%d",
            fold_id, symbols, start, end, horizons, n_ticks, warmup, seed,
        )

        t_start = time.perf_counter()

        # Build replay config and validate
        try:
            cfg = self.build_replay_config(
                symbol=symbols[0],
                start=start,
                end=end,
                n_ticks=n_ticks,
            )
        except Exception as exc:
            logger.error(
                "Failed to build ReplayConfig for fold %d: %s", fold_id, exc
            )
            return pl.DataFrame()

        # Delegate to AmplitudeCache
        try:
            df = AmplitudeCache.build(
                fold_id=fold_id,
                symbols=symbols,
                start=start,
                end=end,
                horizons=horizons,
                n_ticks=n_ticks,
                warmup=warmup,
                seed=seed,
            )
        except Exception as exc:
            logger.error(
                "AmplitudeCache.build failed for fold %d: %s", fold_id, exc
            )
            return pl.DataFrame()

        # Save per-symbol Parquet files
        if df.is_empty():
            logger.warning("Empty DataFrame for fold %d – skipping save", fold_id)
        else:
            self._save_fold_results(df, fold_id)

        elapsed = time.perf_counter() - t_start
        logger.info(
            "run_fold – fold=%d completed in %.2fs, collected %d rows",
            fold_id, elapsed, len(df),
        )

        return df

    def run_all_folds(
        self,
        folds: list[dict],
        symbols: list[str],
        **kwargs,
    ) -> dict[int, pl.DataFrame]:
        """Run amplitude collection for multiple folds sequentially.

        Folds are executed one at a time to avoid clock-patching conflicts
        (only one replay clock can be patched globally at any moment).

        Parameters
        ----------
        folds : list of dict
            Each dict must contain at least ``"fold_id"``, ``"start"``, and
            ``"end"``.  Example::

                [
                    {"fold_id": 0, "start": "2025-01-01", "end": "2025-03-01"},
                    {"fold_id": 1, "start": "2025-02-01", "end": "2025-04-01"},
                    {"fold_id": 2, "start": "2025-03-01", "end": "2025-06-01"},
                ]

        symbols : list of str
            Symbols to process for every fold.
        **kwargs
            Additional keyword arguments forwarded to :meth:`run_fold`
            (e.g. ``horizons``, ``n_ticks``, ``warmup``, ``seed``).

        Returns
        -------
        dict[int, pl.DataFrame]
            Mapping from ``fold_id`` to the collected DataFrame.  Folds that
            fail will have an empty DataFrame.
        """
        results: dict[int, pl.DataFrame] = {}

        logger.info(
            "run_all_folds – %d folds, symbols=%s, kwargs=%s",
            len(folds), symbols, kwargs,
        )

        for fold_spec in folds:
            fold_id: int = fold_spec["fold_id"]
            start: str = fold_spec["start"]
            end: str = fold_spec["end"]

            try:
                df = self.run_fold(
                    fold_id=fold_id,
                    symbols=symbols,
                    start=start,
                    end=end,
                    **kwargs,
                )
            except Exception as exc:
                logger.error(
                    "Unhandled exception in fold %d: %s", fold_id, exc
                )
                df = pl.DataFrame()

            results[fold_id] = df

        successful = sum(1 for df in results.values() if not df.is_empty())
        logger.info(
            "run_all_folds – %d/%d folds completed successfully",
            successful, len(folds),
        )

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def build_replay_config(
        symbol: str,
        start: str,
        end: str,
        n_ticks: int = 80000,
    ) -> ReplayConfig:
        """Construct a :class:`~replay.environment.ReplayConfig` for the
        given symbol and date range.

        The returned config uses accelerated replay with burst mode, minimal
        latency/slippage overhead, and a generous speed factor so that large
        tick windows complete quickly.

        Parameters
        ----------
        symbol : str
            Primary symbol.  Additional symbols can be added to the config's
            ``symbols`` list after construction if needed.
        start : str
            Start date in ``YYYY-MM-DD`` format.
        end : str
            End date in ``YYYY-MM-DD`` format.
        n_ticks : int
            Target number of ticks (stored as ``warmup_ticks`` on the config
            for reference by the replay environment).

        Returns
        -------
        ReplayConfig
        """
        cfg = ReplayConfig(
            symbols=[symbol],
            start=start,
            end=end,
            speed=500_000,
            mode="ACCELERATED",
            burst=True,
            latency=False,
            slippage=False,
            seed=42,
            warmup_ticks=n_ticks,
        )
        logger.debug(
            "ReplayConfig built – symbol=%s [%s → %s] n_ticks=%d",
            symbol, start, end, n_ticks,
        )
        return cfg

    # ------------------------------------------------------------------
    # Classmethods
    # ------------------------------------------------------------------

    @classmethod
    def create_folds_from_daterange(
        cls,
        start: str,
        end: str,
        n_folds: int = 3,
        train_ratio: float = 0.5,
    ) -> list[dict]:
        """Create overlapping walk-forward fold definitions from a date range.

        Generates ``n_folds`` folds where each fold has a training period and
        a test period.  Folds are overlapping so that earlier data is reused
        across folds (purposely *not* a strict time-series split — the goal is
        to test regime stability under different temporal contexts).

        For ``n_folds=3`` and ``train_ratio=0.5``::

            Fold 0:  train [start ........... 50%]
                     test  [........................ 75%]
            Fold 1:  train [      25% .............. 75%]
                     test  [........................ 75% ........ 100%]
            Fold 2:  train [                    50% ............... end]
                     test  [........................ 75% ............... end]

        In the illustration ``.`` represents the full date range and ``----``
        the fold's coverage.  The ``train_ratio`` controls the width of the
        training window (as a fraction of the total range); the test window
        is half the training window and always starts at the midpoint of the
        fold's coverage.

        Parameters
        ----------
        start : str
            Overall start date in ``YYYY-MM-DD``.
        end : str
            Overall end date in ``YYYY-MM-DD``.
        n_folds : int
            Number of folds to create (default 3).
        train_ratio : float
            Fraction of the full range consumed by each fold's training
            window (default 0.5).

        Returns
        -------
        list of dict
            Each dict has keys ``fold_id``, ``start``, ``end`` representing
            the date range for that fold.  These are the *full* fold ranges
            (spanning both train and test periods); the train/test split is
            left to downstream consumers who can further sub-divide.
        """
        if n_folds < 1:
            raise ValueError("n_folds must be >= 1")
        if not 0.0 < train_ratio < 1.0:
            raise ValueError("train_ratio must be in (0, 1)")

        # Parse dates as ordinal floats for arithmetic
        from datetime import datetime, timedelta

        dt_start = datetime.strptime(start, "%Y-%m-%d")
        dt_end = datetime.strptime(end, "%Y-%m-%d")
        total_days = (dt_end - dt_start).days

        if total_days <= 0:
            raise ValueError(f"end ({end}) must be after start ({start})")

        # Each fold steps forward by (1 - train_ratio) / (n_folds - 1) of the
        # total range, so the folds tile the range with even overlap.
        # When n_folds == 1, the single fold covers the entire range.
        if n_folds == 1:
            step_days = 0
        else:
            step_days = int((1.0 - train_ratio) * total_days / (n_folds - 1))

        train_window_days = int(train_ratio * total_days)

        folds: list[dict] = []
        for i in range(n_folds):
            fold_start = dt_start + timedelta(days=i * step_days)
            fold_end = fold_start + timedelta(days=train_window_days)

            # Clamp to the overall range
            if fold_start < dt_start:
                fold_start = dt_start
            if fold_end > dt_end:
                fold_end = dt_end

            folds.append({
                "fold_id": i,
                "start": fold_start.strftime("%Y-%m-%d"),
                "end": fold_end.strftime("%Y-%m-%d"),
            })

        logger.info(
            "create_folds_from_daterange – %s → %s, %d folds, "
            "train_ratio=%.2f, step=%d days, window=%d days",
            start, end, n_folds, train_ratio, step_days, train_window_days,
        )

        return folds

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_fold_results(self, df: pl.DataFrame, fold_id: int) -> None:
        """Save the amplitude DataFrame to per-symbol Parquet files.

        Writes one file per symbol to ``{base_cache_path}/{symbol}_{fold_id}.parquet``.
        """
        if df.is_empty():
            return

        try:
            symbols_in_df = df["symbol"].unique().to_list()
        except Exception:
            symbols_in_df = []

        for sym in symbols_in_df:
            sym_df = df.filter(pl.col("symbol") == sym)
            if sym_df.is_empty():
                continue
            path = os.path.join(self.base_cache_path, f"{sym}_{fold_id}.parquet")
            try:
                sym_df.write_parquet(path, compression="zstd")
                logger.debug("Saved %d rows to %s", len(sym_df), path)
            except Exception as exc:
                logger.error("Failed to save %s: %s", path, exc)

    def _fold_cache_path(self, symbol: str, fold_id: int) -> str:
        """Return the expected Parquet path for a (symbol, fold) pair."""
        return os.path.join(self.base_cache_path, f"{symbol}_{fold_id}.parquet")
