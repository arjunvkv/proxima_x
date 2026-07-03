"""WFVValidator — 3-fold walk-forward validation for cross-asset propagation.

Validates the stability of cross-asset propagation edges across three
overlapping train/test folds, reporting win rate (WR), profit factor (PF),
amplitude expansion ratio (AER), and multiple stability metrics.

Usage::

    from research.dpl_x.wfv_validator import WFVValidator

    validator = WFVValidator()
    result = validator.validate(
        source_symbol="EURUSD",
        target_symbol="GBPUSD",
        horizon=300,
        start="2025-01-01",
        end="2025-03-01",
    )
    print(result["pf_stability"], result["overall_stability"])
"""
import sys; sys.path.insert(0, ".")

import math
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import polars as pl

from research.dpl_x.state_extractor import StateExtractor
from research.dpl_x.forward_surface import ForwardSurface
from research.dpl_x.propagation_mapper import PropagationMapper


# ---------------------------------------------------------------------------
# Default fold definitions (as fractions of the total date range)
#   Fold 1: train [  0%,  50%], test [ 50%,  75%]
#   Fold 2: train [ 25%,  75%], test [ 50%, 100%]
#   Fold 3: train [ 50%, 100%], test [ 75%, 100%]
# ---------------------------------------------------------------------------
_DEFAULT_FOLD_DEFS: list[tuple[float, float, float, float]] = [
    (0.0, 0.5, 0.5, 0.75),
    (0.25, 0.75, 0.5, 1.0),
    (0.5, 1.0, 0.75, 1.0),
]


# ---------------------------------------------------------------------------
# WFVValidator
# ---------------------------------------------------------------------------

class WFVValidator:
    """3-fold walk-forward validator for cross-asset propagation.

    Parameters
    ----------
    n_folds : int
        Number of overlapping folds to use (default 3).  When fewer than 3,
        only the first ``n_folds`` fold definitions are applied.
    train_ratio : float
        Fraction of the total date range used as each fold's training window
        (default 0.5).  This parameter is exposed for consistency but the
        built-in fold definitions are fixed; override ``_build_folds`` for
        custom behaviour.
    """

    def __init__(self, n_folds: int = 3, train_ratio: float = 0.5):
        self.n_folds = n_folds
        self.train_ratio = train_ratio
        self._mapper = PropagationMapper()
        self._forward_surface = ForwardSurface()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        source_symbol: str,
        target_symbol: str,
        horizon: int,
        start: str,
        end: str,
        n_ticks: int = 80000,
    ) -> dict:
        """Run 3-fold walk-forward validation for a source -> target pair.

        For each fold:
          1. Train/run state extraction on the training window.
          2. Build a forward surface for the training window.
          3. Compute propagation metrics (PF, WR, AER) on training data.
          4. Repeat (1–3) for the test window.
          5. Record per-fold results.

        After all folds, stability metrics are computed across test folds.

        Parameters
        ----------
        source_symbol : str
            Source asset symbol (whose state drives propagation).
        target_symbol : str
            Target asset symbol (whose forward returns are affected).
        horizon : int
            Forward-return horizon in seconds (e.g. ``300`` for 5 min).
        start : str
            Start date in ``YYYY-MM-DD`` format.
        end : str
            End date in ``YYYY-MM-DD`` format.
        n_ticks : int
            Maximum number of replay ticks to process per symbol per fold
            (default 80 000).

        Returns
        -------
        dict
            Keys:
            - ``source``, ``target``, ``horizon`` — input identifiers.
            - ``fold_results`` — list of per-fold dicts, each containing
              ``fold``, ``train_start``, ``train_end``, ``test_start``,
              ``test_end``, ``train`` (dict with ``n``, ``pf``, ``wr``,
              ``aer``), and ``test`` (same structure).
            - ``pf_mean``, ``pf_std`` — mean and std of test PF across folds.
            - ``wr_mean``, ``wr_std`` — mean and std of test WR across folds.
            - ``aer_mean``, ``aer_std`` — mean and std of test AER across folds.
            - ``pf_stability`` — fraction of folds where test PF > 1.0.
            - ``wr_stability`` — std of test WR across folds.
            - ``aer_stability`` — std of test AER across folds.
            - ``overall_stability`` — fraction of folds where
              ``PF_test > PF_train * 0.8`` (test not drastically worse
              than train).
            - ``n_total`` — total observations across all folds
              (train + test).
        """
        # --- 1. Build fold date boundaries --------------------------------
        folds = self._build_folds(start, end)

        fold_results: list[dict[str, Any]] = []

        # --- 2. Walk each fold --------------------------------------------
        for fold_idx, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
            fold_result = self._evaluate_fold(
                source_symbol=source_symbol,
                target_symbol=target_symbol,
                horizon=horizon,
                train_start=tr_s,
                train_end=tr_e,
                test_start=te_s,
                test_end=te_e,
                n_ticks=n_ticks,
            )
            fold_result["fold"] = fold_idx
            fold_results.append(fold_result)

        # --- 3. Aggregate stability metrics -------------------------------
        test_pfs  = [fr["test"]["pf"] for fr in fold_results]
        test_wrs  = [fr["test"]["wr"] for fr in fold_results]
        test_aers = [fr["test"]["aer"] for fr in fold_results]

        # PF stability: fraction of folds where test PF > 1.0
        n_folds = len(fold_results)
        pf_stability = (
            sum(1 for pf in test_pfs if pf > 1.0) / n_folds if n_folds > 0 else 0.0
        )

        # WR stability: standard deviation of test WR across folds
        wr_stability = float(np.std(test_wrs, ddof=0)) if len(test_wrs) > 1 else 0.0

        # AER stability: standard deviation of test AER across folds
        aer_stability = float(np.std(test_aers, ddof=0)) if len(test_aers) > 1 else 0.0

        # Overall stability: fraction where PF_test > PF_train * 0.8
        overall_stable = 0
        for fr in fold_results:
            pf_train = fr["train"]["pf"]
            pf_test  = fr["test"]["pf"]
            if pf_train > 0 and pf_test > pf_train * 0.8:
                overall_stable += 1
            elif pf_train <= 0 and pf_test > 0:
                # No train edge but test shows positive edge — also stable
                overall_stable += 1
        overall_stability = (
            overall_stable / n_folds if n_folds > 0 else 0.0
        )

        # Total observations
        n_total = sum(
            fr["train"]["n"] + fr["test"]["n"] for fr in fold_results
        )

        return {
            "source": source_symbol,
            "target": target_symbol,
            "horizon": horizon,
            "fold_results": fold_results,
            "pf_mean": float(np.mean(test_pfs)) if test_pfs else 0.0,
            "pf_std": float(np.std(test_pfs, ddof=0)) if test_pfs else 0.0,
            "wr_mean": float(np.mean(test_wrs)) if test_wrs else 0.0,
            "wr_std": float(np.std(test_wrs, ddof=0)) if len(test_wrs) > 1 else 0.0,
            "aer_mean": float(np.mean(test_aers)) if test_aers else 0.0,
            "aer_std": float(np.std(test_aers, ddof=0)) if len(test_aers) > 1 else 0.0,
            "pf_stability": pf_stability,
            "wr_stability": wr_stability,
            "aer_stability": aer_stability,
            "overall_stability": overall_stability,
            "n_total": n_total,
        }

    def validate_batch(
        self,
        pairs: list[tuple],
        **kwargs,
    ) -> list[dict]:
        """Run validation for multiple source -> target pairs.

        Parameters
        ----------
        pairs : list[tuple]
            Each element is ``(source_symbol, target_symbol, horizon)``.
        **kwargs
            Forwarded to :meth:`validate` (e.g. ``start``, ``end``,
            ``n_ticks``).

        Returns
        -------
        list[dict]
            One result dict (see :meth:`validate`) per input pair.
        """
        results: list[dict] = []
        for source, target, horizon in pairs:
            result = self.validate(
                source_symbol=source,
                target_symbol=target,
                horizon=horizon,
                **kwargs,
            )
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Single-fold evaluation
    # ------------------------------------------------------------------

    def _evaluate_fold(
        self,
        source_symbol: str,
        target_symbol: str,
        horizon: int,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
        n_ticks: int,
    ) -> dict[str, Any]:
        """Run state extraction, forward-surface build, and propagation
        mapping for one train/test fold.

        Returns a dict with keys ``train_start``, ``train_end``,
        ``test_start``, ``test_end``, ``train`` (nested dict with ``n``,
        ``pf``, ``wr``, ``aer``), ``test`` (same).
        """
        # --- Train period ------------------------------------------------
        train_extractor = StateExtractor(
            symbols=[source_symbol, target_symbol],
            start=train_start,
            end=train_end,
            n_ticks=n_ticks,
        )
        train_state = train_extractor.extract()

        train_source_state = train_state.filter(
            pl.col("symbol") == source_symbol
        )
        train_target_fwd = self._forward_surface.build(
            target_symbol,
            start=train_start,
            end=train_end,
            n_ticks=n_ticks,
        )

        train_result = self._mapper.map(
            train_source_state,
            train_target_fwd,
            source_symbol,
            target_symbol,
            horizon,
        )

        # --- Test period -------------------------------------------------
        test_extractor = StateExtractor(
            symbols=[source_symbol, target_symbol],
            start=test_start,
            end=test_end,
            n_ticks=n_ticks,
        )
        test_state = test_extractor.extract()

        test_source_state = test_state.filter(
            pl.col("symbol") == source_symbol
        )
        test_target_fwd = self._forward_surface.build(
            target_symbol,
            start=test_start,
            end=test_end,
            n_ticks=n_ticks,
        )

        test_result = self._mapper.map(
            test_source_state,
            test_target_fwd,
            source_symbol,
            target_symbol,
            horizon,
        )

        return {
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "train": {
                "n": train_result.n,
                "pf": train_result.pf,
                "wr": train_result.wr,
                "aer": train_result.aer,
            },
            "test": {
                "n": test_result.n,
                "pf": test_result.pf,
                "wr": test_result.wr,
                "aer": test_result.aer,
            },
        }

    # ------------------------------------------------------------------
    # Fold boundary construction
    # ------------------------------------------------------------------

    def _build_folds(self, start: str, end: str) -> list[tuple[str, str, str, str]]:
        """Build overlapping fold date boundaries.

        Parameters
        ----------
        start : str
            Start date in ``YYYY-MM-DD`` format.
        end : str
            End date in ``YYYY-MM-DD`` format.

        Returns
        -------
        list[tuple[str, str, str, str]]
            Each tuple is ``(train_start, train_end, test_start, test_end)``
            as ``YYYY-MM-DD`` strings.
        """
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt   = datetime.strptime(end, "%Y-%m-%d")
        total_days = (end_dt - start_dt).days

        if total_days <= 0:
            # Degenerate range — return a single zero-width fold
            return [(start, start, start, start)]

        folds: list[tuple[str, str, str, str]] = []

        # Use the first n_folds definitions from the default set
        fold_defs = _DEFAULT_FOLD_DEFS[: min(len(_DEFAULT_FOLD_DEFS), self.n_folds)]

        for tr_sp, tr_ep, te_sp, te_ep in fold_defs:
            tr_s = start_dt + timedelta(days=round(total_days * tr_sp))
            tr_e = start_dt + timedelta(days=round(total_days * tr_ep))
            te_s = start_dt + timedelta(days=round(total_days * te_sp))
            te_e = start_dt + timedelta(days=round(total_days * te_ep))

            # Ensure at least one day per period
            if tr_s >= tr_e:
                tr_e = tr_s + timedelta(days=1)
            if te_s >= te_e:
                te_e = te_s + timedelta(days=1)

            folds.append((
                tr_s.strftime("%Y-%m-%d"),
                tr_e.strftime("%Y-%m-%d"),
                te_s.strftime("%Y-%m-%d"),
                te_e.strftime("%Y-%m-%d"),
            ))

        return folds


# ---------------------------------------------------------------------------
# Quick smoke test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    v = WFVValidator(n_folds=3)
    folds = v._build_folds("2025-01-01", "2025-04-01")
    print("Fold boundaries for 2025-01-01 -> 2025-04-01:")
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
        print(f"  Fold {i + 1}: train [{tr_s}, {tr_e}]  test [{te_s}, {te_e}]")
    print(f"\nn_folds={v.n_folds}, train_ratio={v.train_ratio}")
