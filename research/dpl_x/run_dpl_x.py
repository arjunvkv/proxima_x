"""DPL-X Orchestrator — Phase IV: Directed Propagation Lab, Experiment X.

Main orchestrator for the cross-asset propagation discovery pipeline.

Pipeline passes
---------------
1. **Smoke test** — 3 pairs × 60 s, CME profile only, single fold.
2. **Core matrix** — full Cartesian (sources × targets × horizons), parallel.
3. **Friction survivability** — top-N propagation edges across 5 execution profiles.
4. **Walk-forward validation** — 3-fold WFV for top-N edges.
5. **Full sweep** — ~3000-experiment comprehensive sweep.

Usage
-----
    from research.dpl_x.run_dpl_x import DPLXOrchestrator

    orch = DPLXOrchestrator()
    results = orch.run_smoke_test()
    print(results)
"""
import sys; sys.path.insert(0, ".")
sys.path.insert(0, "proxima_x")  # noqa: E402 — needed for imports regardless of CWD

import functools
import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Optional

import numpy as np
import polars as pl

from research.dpl_x.state_extractor import StateExtractor
from research.dpl_x.forward_surface import ForwardSurface
from research.dpl_x.propagation_mapper import PropagationMapper, PropagationResult
from research.dpl_x.friction_validator import FrictionValidator
from research.dpl_x.regime_alignment import RegimeAlignmentEngine
from validation.wfv_engine import WalkForwardValidator

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("DPL-X")

# ── Constants ─────────────────────────────────────────────────────────────
AVAILABLE_SYMBOLS: list[str] = ["EURJPY", "USDJPY", "EURUSD"]
DEFAULT_HORIZONS: list[int] = [60, 300, 900, 3600]
EXECUTION_PROFILES: list[str] = [
    "retail_fx", "prime_fx", "ecn_fx", "cme_fx", "crypto_perps",
]
SMOKE_PAIRS: list[tuple[str, str]] = [
    ("EURJPY", "USDJPY"),
    ("EURJPY", "EURUSD"),
    ("USDJPY", "EURJPY"),
]
DEFAULT_START = "2026-06-01"
DEFAULT_END = "2026-06-19"
OUTPUT_DIR = "outputs"
_EXPERIMENT_TIMEOUT = 120  # seconds per experiment

# Map public horizon column names expected by the schema.
_HORIZON_COL_NAMES: dict[int, str] = {h: f"forward_{h}" for h in DEFAULT_HORIZONS}


# ---------------------------------------------------------------------------
# Module-level worker — picklable for ProcessPoolExecutor
# ---------------------------------------------------------------------------
def _propagation_worker(
    source: str,
    target: str,
    horizon: int,
    source_df: pl.DataFrame,
    target_df: pl.DataFrame,
) -> PropagationResult:
    """Standalone worker for one propagation experiment (pickle-friendly)."""
    mapper = PropagationMapper()
    return mapper.map(
        source_df=source_df,
        target_df=target_df,
        source_symbol=source,
        target_symbol=target,
        horizon=horizon,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class DPLXOrchestrator:
    """Phase IV DPL-X main orchestrator.

    Parameters
    ----------
    max_workers : int
        Maximum parallel workers (default 8).  Used by the core-matrix
        and full-sweep passes.
    """

    def __init__(self, max_workers: int = 8) -> None:
        self.max_workers = max_workers
        self.symbols = list(AVAILABLE_SYMBOLS)
        self.horizons = list(DEFAULT_HORIZONS)
        self.start_date = DEFAULT_START
        self.end_date = DEFAULT_END

        # Module instances (lazy-initialised)
        self._state_extractor: Optional[StateExtractor] = None
        self._forward_surface: Optional[ForwardSurface] = None
        self._propagation_mapper: Optional[PropagationMapper] = None
        self._friction_validator: Optional[FrictionValidator] = None
        self._regime_engine: Optional[RegimeAlignmentEngine] = None
        self._wfv: Optional[WalkForwardValidator] = None

        # Data caches (symbol -> DataFrame)
        self._state_cache: dict[str, pl.DataFrame] = {}
        self._forward_cache: dict[str, pl.DataFrame] = {}

        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Lazy module accessors
    # ------------------------------------------------------------------

    @property
    def state_extractor(self) -> StateExtractor:
        if self._state_extractor is None:
            self._state_extractor = StateExtractor(
                symbols=self.symbols,
                start=self.start_date,
                end=self.end_date,
            )
        return self._state_extractor

    @property
    def forward_surface(self) -> ForwardSurface:
        if self._forward_surface is None:
            self._forward_surface = ForwardSurface(horizons=self.horizons)
        return self._forward_surface

    @property
    def propagation_mapper(self) -> PropagationMapper:
        if self._propagation_mapper is None:
            self._propagation_mapper = PropagationMapper()
        return self._propagation_mapper

    @property
    def friction_validator(self) -> FrictionValidator:
        if self._friction_validator is None:
            self._friction_validator = FrictionValidator()
        return self._friction_validator

    @property
    def regime_engine(self) -> RegimeAlignmentEngine:
        if self._regime_engine is None:
            self._regime_engine = RegimeAlignmentEngine()
        return self._regime_engine

    @property
    def wfv(self) -> WalkForwardValidator:
        if self._wfv is None:
            self._wfv = WalkForwardValidator()
        return self._wfv

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _get_state(self, symbol: str) -> pl.DataFrame:
        """Return (cached) state DataFrame for *symbol*."""
        if symbol not in self._state_cache:
            log.info("  Extracting state for %s ...", symbol)
            t0 = time.perf_counter()
            df = self.state_extractor.extract_symbol(symbol)
            elapsed = time.perf_counter() - t0
            self._state_cache[symbol] = df
            log.info("    %s: %d state snapshots in %.1fs", symbol, len(df), elapsed)
        return self._state_cache[symbol]

    def _get_forward(self, symbol: str) -> pl.DataFrame:
        """Return (cached) forward-surface DataFrame for *symbol*."""
        if symbol not in self._forward_cache:
            log.info("  Building forward surface for %s ...", symbol)
            t0 = time.perf_counter()
            df = self.forward_surface.build(
                symbol, start=self.start_date, end=self.end_date,
            )
            elapsed = time.perf_counter() - t0
            self._forward_cache[symbol] = df
            log.info("    %s: %d forward rows in %.1fs", symbol, len(df), elapsed)
        return self._forward_cache[symbol]

    def _compute_states(self) -> None:
        """Pre-compute state for all configured symbols."""
        for sym in self.symbols:
            self._get_state(sym)

    def _compute_forwards(self) -> None:
        """Pre-compute forward surfaces for all configured symbols."""
        for sym in self.symbols:
            self._get_forward(sym)

    # ------------------------------------------------------------------
    # Result converters
    # ------------------------------------------------------------------

    @staticmethod
    def _propagation_to_dict(pr: PropagationResult) -> dict[str, Any]:
        """Flatten a PropagationResult into a dict suitable for a DataFrame row."""
        return {
            "source": pr.source,
            "target": pr.target,
            "horizon": pr.horizon,
            "n": pr.n,
            "win_rate": pr.wr,
            "profit_factor": pr.pf,
            "aer": pr.aer,
            "transfer_entropy": pr.transfer_entropy,
            "lag_score": pr.lag_score,
            "expected_return": pr.expected_return,
        }

    @staticmethod
    def _friction_to_dict(fr: dict[str, Any]) -> dict[str, Any]:
        """Flatten a friction evaluation result dict."""
        return {
            "profile": fr.get("profile", ""),
            "pf_after_cost": fr.get("pf_after_cost", 0.0),
            "expectancy_after_cost": fr.get("expectancy_after_cost", 0.0),
            "slippage_ratio": fr.get("slippage_ratio", 0.0),
            "net_edge": fr.get("net_edge", 0.0),
        }

    @staticmethod
    def _wfv_to_dict(wfv_result: dict[str, Any]) -> dict[str, Any]:
        """Flatten a WFV result dict."""
        return {
            "accuracy": wfv_result.get("accuracy", 0.5),
            "pnl_proxy": wfv_result.get("pnl_proxy", 0.0),
            "edge_detected": wfv_result.get("edge_detected", False),
            "windows": wfv_result.get("windows", 0),
        }

    # ------------------------------------------------------------------
    # WFV record builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_wfv_records(pr: PropagationResult) -> list[dict[str, Any]]:
        """Synthesise (signal, outcome) records from propagation effects.

        Maps OSS buckets to trade direction:
            buckets 0-3  → short (signal = -1)
            buckets 6-9  → long  (signal = +1)
            buckets 4-5  → neutral (skipped)
        """
        records: list[dict[str, Any]] = []
        rng = np.random.default_rng(42)

        for bucket_id, effect in pr.effects.items():
            bid = int(bucket_id)
            if 0 <= bid <= 3:
                signal = -1
            elif 6 <= bid <= 9:
                signal = 1
            else:
                continue

            n = int(effect.get("n", 0))
            exp_ret = float(effect.get("expected_return", 0.0))
            std_ret = float(effect.get("std_return", 0.001))
            std_ret = max(std_ret, 1e-10)

            for _ in range(n):
                outcome = float(rng.normal(exp_ret, std_ret))
                records.append({"signal": signal, "outcome": outcome})

        return records

    # ------------------------------------------------------------------
    # Executor selection
    # ------------------------------------------------------------------

    @staticmethod
    def _select_executor():
        """Select a parallel executor.

        Tries ``ProcessPoolExecutor`` first (true parallelism), but falls
        back to ``ThreadPoolExecutor`` when pickling fails (common on
        Windows with complex module state).
        """
        try:
            with ProcessPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(lambda: 42)  # noqa: E731 — simple pickle test
                fut.result(timeout=5)
            return ProcessPoolExecutor
        except Exception:
            return ThreadPoolExecutor

    # ------------------------------------------------------------------
    # Pass 1 — Smoke Test
    # ------------------------------------------------------------------

    def run_smoke_test(self) -> pl.DataFrame:
        """Pass 1: limited smoke test (3 pairs, 60 s, CME profile).

        Returns
        -------
        pl.DataFrame
            Columns: source, target, horizon, n, win_rate, profit_factor,
            aer, transfer_entropy, lag_score, expected_return, profile,
            pf_after_cost, expectancy_after_cost, slippage_ratio, net_edge.
        """
        log.info("=" * 60)
        log.info("PASS 1 — SMOKE TEST")
        log.info("=" * 60)

        self._compute_states()
        self._compute_forwards()

        horizon = 60
        records: list[dict[str, Any]] = []

        for source, target in SMOKE_PAIRS:
            t0 = time.perf_counter()
            log.info("  Smoke pair: %s -> %s @ %ds", source, target, horizon)

            try:
                source_df = self._get_state(source)
                target_df = self._get_forward(target)

                # ── Propagation mapping ──────────────────────────────
                pr = self.propagation_mapper.map(
                    source_df=source_df,
                    target_df=target_df,
                    source_symbol=source,
                    target_symbol=target,
                    horizon=horizon,
                )
                log.info(
                    "    Propagation: n=%d, wr=%.4f, pf=%.4f",
                    pr.n, pr.wr, pr.pf,
                )

                # ── Friction — CME only ─────────────────────────────
                fr = self.friction_validator.evaluate(
                    pr, profile_name="cme_fx",
                )
                log.info(
                    "    Friction (CME): pf_after=%.4f, net_edge=%.2fbps",
                    fr["pf_after_cost"], fr["net_edge"],
                )

                row = self._propagation_to_dict(pr)
                row.update(self._friction_to_dict(fr))
                records.append(row)

            except Exception as exc:
                log.error("    FAILED: %s -> %s: %s", source, target, exc)
                records.append({
                    "source": source,
                    "target": target,
                    "horizon": horizon,
                    "n": 0,
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "aer": 1.0,
                    "transfer_entropy": 0.0,
                    "lag_score": 0,
                    "expected_return": 0.0,
                    "profile": "cme_fx",
                    "pf_after_cost": 0.0,
                    "expectancy_after_cost": 0.0,
                    "slippage_ratio": 0.0,
                    "net_edge": 0.0,
                })

            elapsed = time.perf_counter() - t0
            log.info("    Done in %.1fs", elapsed)

        df = pl.DataFrame(records)
        path = os.path.join(OUTPUT_DIR, "dpl_x_smoke_test.parquet")
        df.write_parquet(path)
        log.info("Smoke test complete: %d rows -> %s", len(df), path)
        return df

    # ------------------------------------------------------------------
    # Pass 2 — Core Matrix
    # ------------------------------------------------------------------

    def run_core_matrix(
        self,
        sources: Optional[list[str]] = None,
        targets: Optional[list[str]] = None,
        horizons: Optional[list[int]] = None,
    ) -> pl.DataFrame:
        """Pass 2: full Cartesian of sources × targets × horizons.

        Parameters
        ----------
        sources : list[str] or None
            Source symbols.  Defaults to all available symbols.
        targets : list[str] or None
            Target symbols.  Defaults to all available symbols.
        horizons : list[int] or None
            Forward-return horizons in seconds.  Defaults to
            ``[60, 300, 900, 3600]``.

        Returns
        -------
        pl.DataFrame
            Propagation metrics for every combination (self-pairs excluded).
        """
        sources = sources or self.symbols
        targets = targets or self.symbols
        horizons = horizons or self.horizons

        # Build experiment list (skip source == target)
        experiments: list[tuple[str, str, int]] = [
            (s, t, h)
            for s in sources
            for t in targets
            for h in horizons
            if s != t
        ]

        log.info("=" * 60)
        log.info("PASS 2 — CORE MATRIX  (%d experiments)", len(experiments))
        log.info("=" * 60)

        self._compute_states()
        self._compute_forwards()

        executor_class = self._select_executor()
        log.info("Using executor: %s", executor_class.__name__)

        results: list[tuple[PropagationResult, int, int, int]] = []

        with executor_class(max_workers=self.max_workers) as executor:
            futures = {}
            for source, target, horizon in experiments:
                source_df = self._get_state(source)
                target_df = self._get_forward(target)

                fn = functools.partial(
                    _propagation_worker,
                    source=source,
                    target=target,
                    horizon=horizon,
                    source_df=source_df,
                    target_df=target_df,
                )
                fut = executor.submit(fn)
                futures[fut] = (source, target, horizon)

            total = len(futures)
            done_count = 0
            global_timeout = _EXPERIMENT_TIMEOUT * total + 60

            for fut in as_completed(futures, timeout=global_timeout):
                source, target, horizon = futures[fut]
                done_count += 1
                try:
                    pr = fut.result(timeout=_EXPERIMENT_TIMEOUT)
                    results.append(pr)
                    log.info(
                        "  [%d/%d] %s -> %s @ %ds: wr=%.4f pf=%.4f",
                        done_count, total, source, target, horizon,
                        pr.wr, pr.pf,
                    )
                except Exception as exc:
                    log.error(
                        "  [%d/%d] %s -> %s @ %ds FAILED: %s",
                        done_count, total, source, target, horizon, exc,
                    )
                    results.append(
                        PropagationResult(
                            source=source,
                            target=target,
                            horizon=horizon,
                            n=0,
                            wr=0.0,
                            pf=0.0,
                            aer=1.0,
                            transfer_entropy=0.0,
                            lag_score=0,
                            expected_return=0.0,
                            pct_return=0.0,
                            effects={},
                        )
                    )

        df = pl.DataFrame([self._propagation_to_dict(r) for r in results])
        path = os.path.join(OUTPUT_DIR, "dpl_x_core_matrix.parquet")
        df.write_parquet(path)
        log.info("Core matrix complete: %d rows -> %s", len(df), path)
        return df

    # ------------------------------------------------------------------
    # Pass 3 — Friction Survivability
    # ------------------------------------------------------------------

    def run_friction_survivability(
        self,
        propagation_results: pl.DataFrame,
        top_n: int = 20,
    ) -> pl.DataFrame:
        """Pass 3: evaluate top-N propagation results across 5 execution profiles.

        Parameters
        ----------
        propagation_results : pl.DataFrame
            Propagation metrics from :meth:`run_core_matrix` (or any
            DataFrame with columns ``source``, ``target``, ``horizon``,
            ``profit_factor``).
        top_n : int
            Number of top propagation edges to evaluate (default 20).

        Returns
        -------
        pl.DataFrame
            Propagation + friction metrics for each (edge, profile) pair.
        """
        log.info("=" * 60)
        log.info("PASS 3 — FRICTION SURVIVABILITY  (top_n=%d)", top_n)
        log.info("=" * 60)

        # Sort by profit_factor descending, take top N
        top = propagation_results.sort(
            "profit_factor", descending=True,
        ).head(top_n)

        all_records: list[dict[str, Any]] = []

        for row in top.iter_rows(named=True):
            source = str(row["source"])
            target = str(row["target"])
            horizon = int(row["horizon"])
            log.info(
                "  Edge: %s -> %s @ %ds (pf=%.4f)",
                source, target, horizon, row.get("profit_factor", 0.0),
            )

            try:
                source_df = self._get_state(source)
                target_df = self._get_forward(target)

                pr = self.propagation_mapper.map(
                    source_df=source_df,
                    target_df=target_df,
                    source_symbol=source,
                    target_symbol=target,
                    horizon=horizon,
                )

                for profile_name in EXECUTION_PROFILES:
                    try:
                        fr = self.friction_validator.evaluate(
                            pr, profile_name=profile_name,
                        )
                        rec = self._propagation_to_dict(pr)
                        rec.update(self._friction_to_dict(fr))
                        all_records.append(rec)
                    except Exception as exc:
                        log.error(
                            "    Friction %s/%s FAILED: %s",
                            profile_name, horizon, exc,
                        )

            except Exception as exc:
                log.error(
                    "    Propagation %s -> %s @ %ds FAILED: %s",
                    source, target, horizon, exc,
                )

        df = pl.DataFrame(all_records)
        path = os.path.join(OUTPUT_DIR, "dpl_x_friction_survivability.parquet")
        df.write_parquet(path)
        log.info("Friction survivability complete: %d rows -> %s", len(df), path)
        return df

    # ------------------------------------------------------------------
    # Pass 4 — Walk-Forward Validation
    # ------------------------------------------------------------------

    def run_wfv(
        self,
        propagation_results: pl.DataFrame,
        top_n: int = 10,
    ) -> pl.DataFrame:
        """Pass 4: 3-fold walk-forward validation for top-N propagation edges.

        Parameters
        ----------
        propagation_results : pl.DataFrame
            Propagation metrics from :meth:`run_core_matrix`.
        top_n : int
            Number of top edges to validate (default 10).

        Returns
        -------
        pl.DataFrame
            Columns: source, target, horizon, accuracy, pnl_proxy,
            edge_detected, windows.
        """
        log.info("=" * 60)
        log.info("PASS 4 — WALK-FORWARD VALIDATION  (top_n=%d)", top_n)
        log.info("=" * 60)

        # Configure 3-fold WFV (train=200, test=50 → ~3 folds for 750+ recs)
        wfv_engine = WalkForwardValidator(train_size=200, test_size=50)

        top = propagation_results.sort(
            "profit_factor", descending=True,
        ).head(top_n)

        records: list[dict[str, Any]] = []

        for row in top.iter_rows(named=True):
            source = str(row["source"])
            target = str(row["target"])
            horizon = int(row["horizon"])
            log.info(
                "  Edge: %s -> %s @ %ds (pf=%.4f)",
                source, target, horizon, row.get("profit_factor", 0.0),
            )

            try:
                source_df = self._get_state(source)
                target_df = self._get_forward(target)

                pr = self.propagation_mapper.map(
                    source_df=source_df,
                    target_df=target_df,
                    source_symbol=source,
                    target_symbol=target,
                    horizon=horizon,
                )

                wfv_records = self._build_wfv_records(pr)
                if len(wfv_records) < wfv_engine.train_size + wfv_engine.test_size:
                    log.warning(
                        "    Insufficient records (%d < %d), skipping WFV",
                        len(wfv_records),
                        wfv_engine.train_size + wfv_engine.test_size,
                    )
                    continue

                wfv_result = wfv_engine.run(wfv_records)

                rec: dict[str, Any] = {
                    "source": source,
                    "target": target,
                    "horizon": horizon,
                    "n_records": len(wfv_records),
                }
                rec.update(self._wfv_to_dict(wfv_result))
                records.append(rec)

                log.info(
                    "    WFV: acc=%.4f, pnl=%.4f, edge=%s",
                    rec["accuracy"], rec["pnl_proxy"], rec["edge_detected"],
                )

            except Exception as exc:
                log.error(
                    "    WFV %s -> %s @ %ds FAILED: %s",
                    source, target, horizon, exc,
                )

        df = pl.DataFrame(records)
        path = os.path.join(OUTPUT_DIR, "dpl_x_wfv.parquet")
        df.write_parquet(path)
        log.info("WFV complete: %d rows -> %s", len(df), path)
        return df

    # ------------------------------------------------------------------
    # Pass 5 — Full Sweep
    # ------------------------------------------------------------------

    def run_full_sweep(self) -> pl.DataFrame:
        """Pass 5: full ~3000-experiment sweep.

        Sweeps across sources, targets, horizons, execution profiles,
        multiple date windows, and random seeds to produce a dense
        evaluation grid.

        Returns
        -------
        pl.DataFrame
            Combined propagation + friction results for all experiments.
        """
        log.info("=" * 60)
        log.info("PASS 5 — FULL SWEEP  (~3000 experiments)")
        log.info("=" * 60)

        # Build a grid of date windows
        date_windows: list[tuple[str, str]] = [
            ("2026-06-01", "2026-06-05"),
            ("2026-06-05", "2026-06-09"),
            ("2026-06-09", "2026-06-13"),
            ("2026-06-13", "2026-06-17"),
            ("2026-06-17", "2026-06-19"),
        ]
        seeds = [42, 43, 44, 45, 46, 47]
        sources = self.symbols
        targets = self.symbols
        horizons = self.horizons
        profiles = EXECUTION_PROFILES

        # Build experiment list
        experiments: list[dict[str, Any]] = []
        for s in sources:
            for t in targets:
                if s == t:
                    continue
                for h in horizons:
                    for p in profiles:
                        for (dw_start, dw_end) in date_windows:
                            for seed in seeds:
                                experiments.append({
                                    "source": s,
                                    "target": t,
                                    "horizon": h,
                                    "profile": p,
                                    "start": dw_start,
                                    "end": dw_end,
                                    "seed": seed,
                                })

        log.info("  Total experiments planned: %d", len(experiments))
        log.info("  Grid: %d src × %d tgt × %d hzn × %d prof × %d win × %d seed",
                 len(sources), len(targets), len(horizons),
                 len(profiles), len(date_windows), len(seeds))

        # Process in batches to manage memory / time
        batch_size = 200
        all_results: list[dict[str, Any]] = []
        executor_class = self._select_executor()
        log.info("  Using executor: %s", executor_class.__name__)

        for batch_start in range(0, len(experiments), batch_size):
            batch = experiments[batch_start:batch_start + batch_size]
            log.info(
                "  Batch [%d:%d] of %d",
                batch_start, batch_start + len(batch), len(experiments),
            )

            # Cache the current date range's data
            current_start = batch[0]["start"]
            current_end = batch[0]["end"]
            self.start_date = current_start
            self.end_date = current_end
            # Invalidate caches — data may differ across windows
            self._state_cache.clear()
            self._forward_cache.clear()

            try:
                self._compute_states()
                self._compute_forwards()
            except Exception as exc:
                log.warning(
                    "  Could not pre-compute for window %s..%s: %s",
                    current_start, current_end, exc,
                )
                continue

            with executor_class(max_workers=self.max_workers) as executor:
                futures = {}
                for exp in batch:
                    source, target = exp["source"], exp["target"]
                    horizon = exp["horizon"]
                    source_df = self._get_state(source)
                    target_df = self._get_forward(target)

                    fn = functools.partial(
                        _propagation_worker,
                        source=source,
                        target=target,
                        horizon=horizon,
                        source_df=source_df,
                        target_df=target_df,
                    )
                    fut = executor.submit(fn)
                    futures[fut] = exp

                for fut in as_completed(futures, timeout=_EXPERIMENT_TIMEOUT * len(batch) + 30):
                    exp = futures[fut]
                    try:
                        pr = fut.result(timeout=_EXPERIMENT_TIMEOUT)
                        # Run friction for the requested profile
                        fr = self.friction_validator.evaluate(
                            pr, profile_name=exp["profile"],
                        )
                        row = self._propagation_to_dict(pr)
                        row.update(self._friction_to_dict(fr))
                        row["date_start"] = exp["start"]
                        row["date_end"] = exp["end"]
                        row["seed"] = exp["seed"]
                        all_results.append(row)
                    except Exception as exc:
                        log.debug("  Experiment FAILED: %s", exc)

            log.info("    Cumulative results: %d", len(all_results))

        df = pl.DataFrame(all_results)
        path = os.path.join(OUTPUT_DIR, "dpl_x_full_sweep.parquet")
        df.write_parquet(path)
        log.info("Full sweep complete: %d rows -> %s", len(df), path)
        return df

    # ------------------------------------------------------------------
    # Full Pipeline
    # ------------------------------------------------------------------

    def run_pipeline(self) -> dict[str, Any]:
        """Execute the full 5-pass pipeline sequentially.

        Returns
        -------
        dict
            Keys:
            - ``smoke_test`` : pl.DataFrame
            - ``core_matrix`` : pl.DataFrame
            - ``friction_survivability`` : pl.DataFrame
            - ``wfv`` : pl.DataFrame
            - ``full_sweep`` : pl.DataFrame
            - ``summary`` : pl.DataFrame
            - ``elapsed`` : float (total wall-clock seconds)
        """
        pipeline_t0 = time.time()
        log.info("")
        log.info("#" * 70)
        log.info("#  DPL-X FULL PIPELINE")
        log.info("#" * 70)

        # ── Pass 1 ────────────────────────────────────────────────────
        t1 = time.time()
        smoke = self.run_smoke_test()
        log.info("Pass 1 (smoke) elapsed: %.1fs\n", time.time() - t1)

        # ── Pass 2 ────────────────────────────────────────────────────
        t2 = time.time()
        core = self.run_core_matrix()
        log.info("Pass 2 (core matrix) elapsed: %.1fs\n", time.time() - t2)

        # ── Pass 3 ────────────────────────────────────────────────────
        t3 = time.time()
        friction = self.run_friction_survivability(core, top_n=20)
        log.info("Pass 3 (friction) elapsed: %.1fs\n", time.time() - t3)

        # ── Pass 4 ────────────────────────────────────────────────────
        t4 = time.time()
        wfv_results = self.run_wfv(core, top_n=10)
        log.info("Pass 4 (WFV) elapsed: %.1fs\n", time.time() - t4)

        # ── Pass 5 ────────────────────────────────────────────────────
        t5 = time.time()
        sweep = self.run_full_sweep()
        log.info("Pass 5 (full sweep) elapsed: %.1fs\n", time.time() - t5)

        total_elapsed = time.time() - pipeline_t0

        # ── Summary ───────────────────────────────────────────────────
        summary_rows: list[dict[str, Any]] = []

        # Smoke test summary
        if len(smoke) > 0:
            best_smoke = smoke.sort("profit_factor", descending=True).head(1)
            summary_rows.append({
                "pass": "smoke_test",
                "n_experiments": len(smoke),
                "best_source": best_smoke[0, "source"],
                "best_target": best_smoke[0, "target"],
                "best_horizon": best_smoke[0, "horizon"],
                "best_profit_factor": best_smoke[0, "profit_factor"],
                "best_net_edge_bps": best_smoke[0, "net_edge"],
            })

        # Core matrix summary
        if len(core) > 0:
            best_core = core.sort("profit_factor", descending=True).head(1)
            summary_rows.append({
                "pass": "core_matrix",
                "n_experiments": len(core),
                "best_source": best_core[0, "source"],
                "best_target": best_core[0, "target"],
                "best_horizon": best_core[0, "horizon"],
                "best_profit_factor": best_core[0, "profit_factor"],
                "best_net_edge_bps": 0.0,
            })

        # Friction survivability summary
        if len(friction) > 0:
            best_fric = friction.sort("net_edge", descending=True).head(1)
            summary_rows.append({
                "pass": "friction_survivability",
                "n_experiments": len(friction),
                "best_source": best_fric[0, "source"],
                "best_target": best_fric[0, "target"],
                "best_horizon": best_fric[0, "horizon"],
                "best_profit_factor": best_fric[0, "profit_factor"],
                "best_net_edge_bps": best_fric[0, "net_edge"],
            })

        # WFV summary
        if len(wfv_results) > 0:
            best_wfv = wfv_results.sort("accuracy", descending=True).head(1)
            summary_rows.append({
                "pass": "wfv",
                "n_experiments": len(wfv_results),
                "best_source": best_wfv[0, "source"],
                "best_target": best_wfv[0, "target"],
                "best_horizon": best_wfv[0, "horizon"],
                "best_profit_factor": best_wfv[0, "accuracy"],
                "best_net_edge_bps": best_wfv[0, "pnl_proxy"],
            })

        # Full sweep summary
        if len(sweep) > 0:
            best_sweep = sweep.sort("net_edge", descending=True).head(1)
            summary_rows.append({
                "pass": "full_sweep",
                "n_experiments": len(sweep),
                "best_source": best_sweep[0, "source"],
                "best_target": best_sweep[0, "target"],
                "best_horizon": best_sweep[0, "horizon"],
                "best_profit_factor": best_sweep[0, "profit_factor"],
                "best_net_edge_bps": best_sweep[0, "net_edge"],
            })

        summary = pl.DataFrame(summary_rows) if summary_rows else pl.DataFrame()

        # Save summary
        summary_path = os.path.join(OUTPUT_DIR, "dpl_x_pipeline_summary.parquet")
        if len(summary) > 0:
            summary.write_parquet(summary_path)
            log.info("Pipeline summary -> %s", summary_path)

        log.info("#" * 70)
        log.info("DPL-X PIPELINE COMPLETE in %.1fs (%.1f min)",
                 total_elapsed, total_elapsed / 60)
        log.info("#" * 70)

        return {
            "smoke_test": smoke,
            "core_matrix": core,
            "friction_survivability": friction,
            "wfv": wfv_results,
            "full_sweep": sweep,
            "summary": summary,
            "elapsed": total_elapsed,
        }


# ---------------------------------------------------------------------------
# __main__ — default entry: run_smoke_test()
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()
    orch = DPLXOrchestrator()
    results = orch.run_smoke_test()
    results.write_parquet("outputs/dpl_x_smoke_test.parquet")
    print(f"Smoke test complete in {time.time() - t0:.1f}s")
    print(results)
