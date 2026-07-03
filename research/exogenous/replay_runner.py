"""
Program VI — Exogenous Amplitude Discovery Replay Runner
=========================================================
Orchestrates replay-driven exogenous data collection across walk-forward folds.

The :class:`ExogenousReplayRunner` drives replay environments for each fold,
delegates cache-building to :class:`ExogenousCache`, and returns the
resulting DataFrames for downstream surface construction and promotion
analysis.

Typical usage::

    from research.exogenous.replay_runner import ExogenousReplayRunner

    runner = ExogenousReplayRunner()
    folds = ExogenousReplayRunner.create_folds_from_daterange(
        "2025-04-01", "2025-04-07", n_folds=3, train_ratio=0.5
    )
    results = runner.run_all_folds(
        folds, symbols=["EURJPY"],
        horizons=[60, 300, 900], n_ticks=50000,
    )
    for fold_id, df in results.items():
        print(f"Fold {fold_id}: {len(df)} observations")
"""

import sys

sys.path.insert(0, ".")
sys.path.insert(0, "proxima_x")

import logging
import os
import time
from typing import Optional

import polars as pl

from research.exogenous.exogenous_cache import ExogenousCache

logger = logging.getLogger("proxima.exogenous.replay_runner")

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
_DEFAULT_HORIZONS: list[int] = [60, 300, 900, 1800]


# ===================================================================
# ExogenousReplayRunner
# ===================================================================


class ExogenousReplayRunner:
    """Orchestrates replay-driven exogenous data collection across
    walk-forward folds.

    Parameters
    ----------
    base_cache_path : str
        Root directory where per-fold per-symbol Parquet files are written.
        Defaults to ``"cache/exogenous"``.
    """

    def __init__(self, base_cache_path: str = "cache/exogenous") -> None:
        self.base_cache_path = base_cache_path
        os.makedirs(self.base_cache_path, exist_ok=True)
        logger.info(
            "ExogenousReplayRunner initialized – cache=%s", self.base_cache_path
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
        """Run exogenous collection for a single walk-forward fold.

        Delegates to :meth:`ExogenousCache.build` with the provided
        parameters and returns the concatenated DataFrame.

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
            Default: ``[60, 300, 900, 1800]``.
        n_ticks : int
            Maximum ticks to process (passed to replay).
        warmup : int
            Tick warmup before collecting observations.
        seed : int
            Random seed for replay determinism.

        Returns
        -------
        pl.DataFrame
            Exogenous observations with columns documented in
            :meth:`ExogenousCache.build`.
        """
        if horizons is None:
            horizons = list(_DEFAULT_HORIZONS)

        logger.info(
            "run_fold – fold=%d symbols=%s [%s → %s] "
            "horizons=%s n_ticks=%d warmup=%d seed=%d",
            fold_id, symbols, start, end, horizons, n_ticks, warmup, seed,
        )

        t_start = time.perf_counter()

        # Delegate to ExogenousCache
        try:
            cache = ExogenousCache(base_path=self.base_cache_path)
            df = cache.build(
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
                "ExogenousCache.build failed for fold %d: %s", fold_id, exc
            )
            return pl.DataFrame()

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
        """Run exogenous collection for multiple folds sequentially.

        Folds are executed one at a time to avoid clock-patching conflicts
        (only one replay clock can be patched globally at any moment).

        Parameters
        ----------
        folds : list of dict
            Each dict must contain at least ``"fold_id"``, ``"start"``, and
            ``"end"``.  Example::

                [
                    {"fold_id": 0, "start": "2025-04-01", "end": "2025-04-04"},
                    {"fold_id": 1, "start": "2025-04-03", "end": "2025-04-06"},
                    {"fold_id": 2, "start": "2025-04-05", "end": "2025-04-07"},
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
    # Classmethods
    # ------------------------------------------------------------------

    @staticmethod
    def create_folds_from_daterange(
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

        For ``n_folds=3`` and ``train_ratio=0.5`` (percentiles of the full
        date range)::

            Fold 0:  overall [start .............. 75%]
                     train   [start ... 50%]
                     test    [           50% .. 75%]

            Fold 1:  overall [   25% .............. 100%]
                     train   [   25% .. 75%]
                     test    [           50% .. 100%]

            Fold 2:  overall [        50% ............... end]
                     train   [        50% ............... end]
                     test    [             75% ............... end]

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
            Each dict has keys ``fold_id``, ``start``, ``end``,
            ``train_start``, ``train_end``, ``test_start``, ``test_end``
            representing the date ranges for that fold.
        """
        if n_folds < 1:
            raise ValueError("n_folds must be >= 1")
        if not 0.0 < train_ratio < 1.0:
            raise ValueError("train_ratio must be in (0, 1)")

        from datetime import datetime, timedelta

        dt_start = datetime.strptime(start, "%Y-%m-%d")
        dt_end = datetime.strptime(end, "%Y-%m-%d")
        total_days = (dt_end - dt_start).days

        if total_days <= 0:
            raise ValueError(f"end ({end}) must be after start ({start})")

        # Compute percentile anchor points over the full date range.
        fracs = [0.0, 0.25, 0.50, 0.75, 1.0]
        anchors = [
            dt_start + timedelta(seconds=int(f * total_days * 86400))
            for f in fracs
        ]
        p0, p25, p50, p75, p100 = anchors

        # Fold definitions based on spec for n_folds=3.
        # For other n_folds values the pattern generalises.
        if n_folds == 3:
            fold_specs = [
                # (overall_start, overall_end, train_start, train_end,
                #  test_start, test_end)
                (p0, p75, p0, p50, p50, p75),
                (p25, p100, p25, p75, p50, p100),
                (p50, p100, p50, p100, p75, p100),
            ]
        elif n_folds == 1:
            fold_specs = [(p0, p100, p0, p50, p50, p100)]
        elif n_folds == 2:
            fold_specs = [
                (p0, p75, p0, p50, p50, p75),
                (p50, p100, p50, p100, p75, p100),
            ]
        else:
            # Generalised: n folds spaced evenly over the range.
            step_frac = (1.0 - train_ratio) / (n_folds - 1)
            train_window_frac = train_ratio
            half_train = train_window_frac / 2.0
            fold_specs = []
            for i in range(n_folds):
                f_start_frac = i * step_frac
                f_mid_frac = f_start_frac + half_train
                f_test_end_frac = min(f_mid_frac + half_train, 1.0)
                f_train_end_frac = min(f_start_frac + train_window_frac, 1.0)
                f_test_start_frac = f_mid_frac
                fold_specs.append(
                    (
                        dt_start + timedelta(seconds=int(f_start_frac * total_days * 86400)),
                        dt_start + timedelta(seconds=int(max(f_train_end_frac, f_test_end_frac) * total_days * 86400)),
                        dt_start + timedelta(seconds=int(f_start_frac * total_days * 86400)),
                        dt_start + timedelta(seconds=int(f_train_end_frac * total_days * 86400)),
                        dt_start + timedelta(seconds=int(f_test_start_frac * total_days * 86400)),
                        dt_start + timedelta(seconds=int(f_test_end_frac * total_days * 86400)),
                    )
                )

        folds: list[dict] = []
        for i, (ov_start, ov_end, tr_start, tr_end, te_start, te_end) in enumerate(fold_specs):
            folds.append({
                "fold_id": i,
                "start": ov_start.strftime("%Y-%m-%d"),
                "end": ov_end.strftime("%Y-%m-%d"),
                "train_start": tr_start.strftime("%Y-%m-%d"),
                "train_end": tr_end.strftime("%Y-%m-%d"),
                "test_start": te_start.strftime("%Y-%m-%d"),
                "test_end": te_end.strftime("%Y-%m-%d"),
            })

        logger.info(
            "create_folds_from_daterange – %s → %s, %d folds, "
            "train_ratio=%.2f",
            start, end, n_folds, train_ratio,
        )

        return folds
