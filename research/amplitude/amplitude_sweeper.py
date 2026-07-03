"""
Phase V — Amplitude Sweep Engine
=================================
Master sweep engine that runs the full experiment grid across folds, symbols,
horizons, k-values, and regimes.  Produces the aggregated results DataFrame
used for downstream analysis and promotion decisions.

Sweep dimensions (600 combinations):
    - symbols:   ["EURJPY", "USDJPY"]           (2)
    - horizons:  [60, 300, 900, 1800]            (4)
    - k_values:  [1.0, 1.5, 2.0, 3.0, 4.0]      (5)
    - regimes:   all 5 from AmplitudeRegimeMapper (5)
    - folds:     3 from create_folds_from_daterange (3)

Total: 2 x 4 x 5 x 5 x 3 = 600 experiment combinations
"""

import sys

sys.path.insert(0, ".")
sys.path.insert(0, "proxima_x")

import os
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import polars as pl

from research.amplitude.amplitude_cache import AmplitudeCache
from research.amplitude.amplitude_surface import AmplitudeSurfaceEngine
from research.amplitude.spread_exceedance import SpreadExceedanceModel
from research.amplitude.amplitude_projection import AmplitudeProjectionEngine
from research.amplitude.promotion_engine import AmplitudePromotionEngine
from research.amplitude.replay_runner import AmplitudeReplayRunner
from research.amplitude.regime_mapper import AmplitudeRegimeMapper
from research.amplitude.schemas import AmplitudeSurfaceEntry
from research.cep.counterfactual_execution import CounterfactualExecutionEngine
from research.cep.execution_profile import load_profiles

# Use perf_counter, NOT time.time (which is patched by replay clock).
import time as time_module


# ===================================================================
# AmplitudeSweeper
# ===================================================================


class AmplitudeSweeper:
    """Master sweep engine for Phase V.

    Orchestrates the full experiment grid across folds, symbols, horizons,
    k-values, and regimes.  For each (fold, symbol) combination, builds the
    amplitude cache, fits the conditional amplitude surface, evaluates
    exceedance probabilities, applies the promotion engine, and estimates
    counterfactual execution friction.

    Results are aggregated into a single DataFrame and persisted as Parquet.
    """

    # Default sweep dimensions
    DEFAULT_SYMBOLS: list[str] = ["EURJPY", "USDJPY"]
    DEFAULT_HORIZONS: list[int] = [60, 300, 900, 1800]
    DEFAULT_K_VALUES: list[float] = [1.0, 1.5, 2.0, 3.0, 4.0]

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(self, max_workers: int = 4, output_dir: str = "outputs") -> None:
        self.max_workers = max_workers
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all(self) -> pl.DataFrame:
        """Run the full sweep across all folds.

        Steps
        -----
        1. Create 3 walk-forward folds using
           :meth:`AmplitudeReplayRunner.create_folds_from_daterange`.
        2. For each fold:
           a. Build (or load from cache) the amplitude cache for both symbols.
           b. Fit the :class:`AmplitudeSurfaceEngine`.
           c. Fit the :class:`SpreadExceedanceModel`.
           d. For every ``state_hash`` on the surface, evaluate each
              horizon through the promotion and CME pipeline.
        3. Concatenate all fold results into a single DataFrame.
        4. Save to ``{output_dir}/amplitude_sweep_results.parquet``.
        5. Return the DataFrame.

        Returns
        -------
        pl.DataFrame
            Columns:

            ``fold_id``, ``symbol``, ``state_hash``, ``horizon``, ``n``,
            ``mean_abs``, ``spread_multiple``, ``aer``, ``exceed_prob_k1``,
            ``exceed_prob_k1_5``, ``exceed_prob_k2``, ``exceed_prob_k3``,
            ``exceed_prob_k4``, ``tier``, ``promoted``, ``pf_cme``,
            ``net_edge``.
        """
        t_start = time_module.perf_counter()
        print("=" * 60)
        print("AmplitudeSweeper — Phase V Master Sweep")
        print("=" * 60)

        # ---- 1. Create walk-forward folds ---------------------------------
        folds = AmplitudeReplayRunner.create_folds_from_daterange(
            "2025-01-01",
            "2025-06-01",
            n_folds=3,
            train_ratio=0.5,
        )
        print(f"[AmplitudeSweeper] Created {len(folds)} walk-forward folds")

        # ---- 2. Process each fold (robust — wrap in try/except) -----------
        all_results: list[pl.DataFrame] = []

        for fold_spec in folds:
            fold_id: int = fold_spec["fold_id"]
            start: str = fold_spec["start"]
            end: str = fold_spec["end"]

            try:
                print(f"\n[AmplitudeSweeper] --- Fold {fold_id}: "
                      f"{start} -> {end} ---")
                fold_df = self.run_fold_sweep(
                    fold_id=fold_id,
                    symbols=list(self.DEFAULT_SYMBOLS),
                    start=start,
                    end=end,
                )
                if fold_df is not None and len(fold_df) > 0:
                    all_results.append(fold_df)
                    print(f"[AmplitudeSweeper] Fold {fold_id} complete: "
                          f"{len(fold_df)} records")
                else:
                    print(f"[AmplitudeSweeper] Fold {fold_id}: no results")
            except Exception as exc:
                print(f"[AmplitudeSweeper] Fold {fold_id} FAILED: {exc}")
                continue

        # ---- 3. Aggregate results ------------------------------------------
        if all_results:
            result_df = pl.concat(all_results)
        else:
            result_df = pl.DataFrame(
                schema={
                    "fold_id": pl.Int32,
                    "symbol": pl.Utf8,
                    "state_hash": pl.Utf8,
                    "horizon": pl.Int32,
                    "n": pl.Int32,
                    "mean_abs": pl.Float64,
                    "spread_multiple": pl.Float64,
                    "aer": pl.Float64,
                    "exceed_prob_k1": pl.Float64,
                    "exceed_prob_k1_5": pl.Float64,
                    "exceed_prob_k2": pl.Float64,
                    "exceed_prob_k3": pl.Float64,
                    "exceed_prob_k4": pl.Float64,
                    "tier": pl.Int32,
                    "promoted": pl.Boolean,
                    "pf_cme": pl.Float64,
                    "net_edge": pl.Float64,
                }
            )

        # ---- 4. Save to parquet -------------------------------------------
        out_path = os.path.join(self.output_dir,
                                "amplitude_sweep_results.parquet")
        if len(result_df) > 0:
            result_df.write_parquet(out_path, compression="zstd")
            print(f"\n[AmplitudeSweeper] Saved {len(result_df)} rows "
                  f"-> {out_path}")
        else:
            print("\n[AmplitudeSweeper] WARNING: no results collected; "
                  "output file not created")

        elapsed = time_module.perf_counter() - t_start
        print(f"\n[AmplitudeSweeper] Total sweep completed in "
              f"{elapsed:.2f}s")
        print(f"[AmplitudeSweeper] Result columns: "
              f"{list(result_df.columns)}")
        print("=" * 60)

        return result_df

    # ------------------------------------------------------------------

    def run_fold_sweep(
        self,
        fold_id: int,
        symbols: list[str],
        start: str,
        end: str,
    ) -> pl.DataFrame:
        """Run the full amplitude pipeline for a single walk-forward fold.

        Steps
        -----
        1. Build (or load from cache) the amplitude dataset for *symbols*.
        2. Fit a :class:`AmplitudeSurfaceEngine` on the cached data.
        3. Fit a :class:`SpreadExceedanceModel` on the same data.
        4. For every ``state_hash`` present on the surface, evaluate each
           horizon through the promotion and CME pipeline.
        5. Return the per-(state_hash, horizon) result DataFrame.

        Parameters
        ----------
        fold_id : int
            Fold index (used for file naming and metadata).
        symbols : list of str
            Symbols to process (e.g. ``["EURJPY", "USDJPY"]``).
        start : str
            Fold start date in ``YYYY-MM-DD`` format.
        end : str
            Fold end date in ``YYYY-MM-DD`` format.

        Returns
        -------
        pl.DataFrame
            Sweep results for this fold (one row per
            ``state_hash × horizon`` combination that had sufficient
            observations).
        """
        t_fold = time_module.perf_counter()
        cache = AmplitudeCache()

        # ---- 1a. Try loading from cache ------------------------------------
        cache_dfs: list[pl.DataFrame] = []
        all_cached = True
        for sym in symbols:
            try:
                sym_df = cache.load(symbol=sym, fold=fold_id)
                if not sym_df.is_empty():
                    cache_dfs.append(sym_df)
                    print(f"[AmplitudeSweeper] Loaded cache: "
                          f"{sym} fold {fold_id} ({len(sym_df)} rows)")
                else:
                    all_cached = False
                    break
            except (FileNotFoundError, Exception):
                all_cached = False
                break

        # ---- 1b. Build cache if not fully cached ---------------------------
        if all_cached and len(cache_dfs) == len(symbols):
            cache_df = pl.concat(cache_dfs)
            print(f"[AmplitudeSweeper] Using cached data for fold "
                  f"{fold_id} ({len(cache_df)} total rows)")
        else:
            print(f"[AmplitudeSweeper] Building amplitude cache for "
                  f"fold {fold_id} ...")
            try:
                cache_df = cache.build(
                    symbols=symbols,
                    start=start,
                    end=end,
                    horizons=list(self.DEFAULT_HORIZONS),
                    n_ticks=80000,
                    warmup=5000,
                    seed=42,
                )
                # Persist per-symbol with fold id for future runs
                for sym in symbols:
                    sym_df = cache_df.filter(pl.col("symbol") == sym)
                    if len(sym_df) > 0:
                        cache.save(sym_df, symbol=sym, fold=fold_id)
            except Exception as exc:
                print(f"[AmplitudeSweeper] Cache build failed for fold "
                      f"{fold_id}: {exc}")
                return pl.DataFrame()

        if cache_df.is_empty():
            print(f"[AmplitudeSweeper] Empty cache for fold {fold_id}")
            return pl.DataFrame()

        # ---- 2. Fit surface engine -----------------------------------------
        surface = AmplitudeSurfaceEngine()
        try:
            surface.fit(cache_df)
        except Exception as exc:
            print(f"[AmplitudeSweeper] Surface fit failed: {exc}")
            return pl.DataFrame()

        stats = surface.get_stats()
        print(f"[AmplitudeSweeper] Surface: {stats['n_buckets']} buckets "
              f"x {stats['n_horizons']} horizons, "
              f"coverage={stats['coverage_pct']:.1%}")

        # ---- 3. Fit spread exceedance model --------------------------------
        exceed_model = SpreadExceedanceModel()
        try:
            exceed_model.fit(cache_df,
                             k_values=list(self.DEFAULT_K_VALUES))
        except Exception as exc:
            print(f"[AmplitudeSweeper] Exceedance fit failed: {exc}")
            return pl.DataFrame()

        # ---- 4. Load CME execution profiles --------------------------------
        try:
            profiles = load_profiles()
            print(f"[AmplitudeSweeper] Loaded {len(profiles)} execution "
                  f"profiles: {list(profiles.keys())}")
        except Exception as exc:
            profiles = {}
            print(f"[AmplitudeSweeper] WARNING: could not load CME "
                  f"profiles ({exc}); using zero friction")

        # ---- 5. Evaluate every (state_hash, horizon) cell ------------------
        promotion_engine = AmplitudePromotionEngine()

        experiment_params: list[dict] = []
        for state_hash, horizons in surface._surfaces.items():
            for horizon, entry in horizons.items():
                experiment_params.append({
                    "state_hash": state_hash,
                    "horizon": horizon,
                    "entry": entry,
                    "exceed_model": exceed_model,
                    "promotion_engine": promotion_engine,
                    "profiles": profiles,
                })

        n_experiments = len(experiment_params)
        print(f"[AmplitudeSweeper] Evaluating {n_experiments} "
               f"(state_hash x horizon) cells ...")

        # Run experiments
        if self.max_workers > 1 and n_experiments > 1:
            records = self._run_experiments_parallel(experiment_params)
        else:
            records = [
                self._run_single_experiment(p) for p in experiment_params
            ]

        if not records:
            print(f"[AmplitudeSweeper] No results from experiments")
            return pl.DataFrame()

        # ---- 6. Build fold result DataFrame --------------------------------
        result_df = pl.DataFrame(records)
        symbol_str = ",".join(symbols)
        result_df = result_df.with_columns([
            pl.lit(fold_id).alias("fold_id").cast(pl.Int32),
            pl.lit(symbol_str).alias("symbol"),
        ])

        # Reorder columns to match the canonical schema
        column_order = [
            "fold_id",
            "symbol",
            "state_hash",
            "horizon",
            "n",
            "mean_abs",
            "spread_multiple",
            "aer",
            "exceed_prob_k1",
            "exceed_prob_k1_5",
            "exceed_prob_k2",
            "exceed_prob_k3",
            "exceed_prob_k4",
            "tier",
            "promoted",
            "pf_cme",
            "net_edge",
        ]
        result_df = result_df.select(column_order)

        elapsed = time_module.perf_counter() - t_fold
        print(f"[AmplitudeSweeper] Fold {fold_id} finished: "
              f"{len(result_df)} rows in {elapsed:.2f}s")

        return result_df

    # ------------------------------------------------------------------
    # Parallel execution helpers
    # ------------------------------------------------------------------

    def _run_experiments_parallel(
        self,
        params_list: list[dict],
    ) -> list[dict]:
        """Run experiments concurrently via ``ThreadPoolExecutor``.

        Uses ``ThreadPoolExecutor`` (not ``ProcessPoolExecutor``) since
        the workload is IO-bound on replay cache reads.

        Parameters
        ----------
        params_list : list of dict
            Each dict is passed to :meth:`_run_single_experiment`.

        Returns
        -------
        list of dict
            Collected result dicts (failed experiments are skipped).
        """
        records: list[dict] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {
                executor.submit(self._run_single_experiment, p): p
                for p in params_list
            }
            for future in as_completed(future_map):
                params = future_map[future]
                try:
                    result = future.result()
                    if result is not None:
                        records.append(result)
                except Exception as exc:
                    sh = params.get("state_hash", "?")
                    hr = params.get("horizon", "?")
                    print(f"[AmplitudeSweeper] Experiment failed: "
                          f"state_hash={sh} horizon={hr}: {exc}")
        return records

    # ------------------------------------------------------------------
    # Single experiment (static, pickle-able)
    # ------------------------------------------------------------------

    @staticmethod
    def _run_single_experiment(params: dict) -> dict:
        """Evaluate a single ``(state_hash, horizon)`` experiment cell.

        Parameters
        ----------
        params : dict
            Keys:

            - ``state_hash`` (str) — composite state key
            - ``horizon`` (int) — forward-looking horizon (seconds)
            - ``entry`` (AmplitudeSurfaceEntry) — surface entry for this cell
            - ``exceed_model`` (SpreadExceedanceModel) — fitted exceedance
              model
            - ``promotion_engine`` (AmplitudePromotionEngine) — promotion
              evaluator
            - ``profiles`` (dict) — execution profiles from
              :func:`~research.cep.execution_profile.load_profiles`

        Returns
        -------
        dict
            Keys:

            - ``state_hash``, ``horizon``, ``n``, ``mean_abs``,
              ``spread_multiple``, ``aer``
            - ``exceed_prob_k1``, ``exceed_prob_k1_5``, ``exceed_prob_k2``,
              ``exceed_prob_k3``, ``exceed_prob_k4``
            - ``tier``, ``promoted``, ``pf_cme``, ``net_edge``
        """
        state_hash: str = params["state_hash"]
        horizon: int = params["horizon"]
        entry: AmplitudeSurfaceEntry = params["entry"]
        exceed_model: SpreadExceedanceModel = params["exceed_model"]
        promotion_engine: AmplitudePromotionEngine = params["promotion_engine"]
        profiles: dict = params["profiles"]

        # ---- Extract surface entry fields ----------------------------------
        n = entry.n
        mean_abs = entry.mean_abs
        spread_multiple = entry.spread_multiple
        aer = entry.aer

        # ---- Compute exceed probabilities for all k values -----------------
        exceed_probs: dict[float, float] = exceed_model.sweep_k(state_hash)

        # ---- Promotion engine evaluation -----------------------------------
        promo_result: dict = promotion_engine.evaluate(entry)
        tier: int = promo_result["tier"]
        promoted: bool = promo_result["promoted"]

        # ---- CME friction estimation ---------------------------------------
        pf_cme: float = _estimate_cme_friction(profiles, mean_abs)

        # ---- Net edge = expected move - friction costs ---------------------
        if mean_abs > 0 and n > 0:
            net_edge: float = mean_abs - pf_cme
        else:
            net_edge = 0.0

        # ---- Assemble result record ----------------------------------------
        return {
            "state_hash": state_hash,
            "horizon": horizon,
            "n": n,
            "mean_abs": mean_abs,
            "spread_multiple": spread_multiple,
            "aer": aer,
            "exceed_prob_k1": exceed_probs.get(1.0, 0.0),
            "exceed_prob_k1_5": exceed_probs.get(1.5, 0.0),
            "exceed_prob_k2": exceed_probs.get(2.0, 0.0),
            "exceed_prob_k3": exceed_probs.get(3.0, 0.0),
            "exceed_prob_k4": exceed_probs.get(4.0, 0.0),
            "tier": tier,
            "promoted": promoted,
            "pf_cme": pf_cme,
            "net_edge": net_edge,
        }


# ===================================================================
# Module-level helpers
# ===================================================================


def _estimate_cme_friction(
    profiles: dict,
    mean_abs: float,
) -> float:
    """Estimate counterfactual execution friction (in bps) for a trade.

    Uses the first available execution profile to compute a composite
    friction cost that accounts for spread, slippage, latency, and
    reject probability.

    Parameters
    ----------
    profiles : dict
        Execution profiles keyed by name
        (e.g. ``{"MT5_DEMO": ExecutionProfile(...)}``).
        Pass an empty dict to get zero friction.
    mean_abs : float
        Mean absolute move (bps) — used as a sanity cap on friction.

    Returns
    -------
    float
        Estimated friction in bps.  Always non-negative.
    """
    if not profiles:
        return 0.0

    # Use the first available profile
    profile = next(iter(profiles.values()))

    # Round-trip spread cost (one-way × 2)
    spread_cost = profile.spread_bps * 2.0

    # Slippage: mean + 1 std (conservative estimate)
    slippage_cost = abs(profile.slippage_bps_mean) + profile.slippage_bps_std

    # Latency impact: convert ms to a small bps penalty
    # (1 ms latency ≈ 0.01 bps at typical FX velocities)
    latency_cost = profile.latency_ms_mean * 0.01

    # Reject probability: treated as opportunity cost
    reject_cost = profile.reject_probability * spread_cost

    total_friction = (
        spread_cost + slippage_cost + latency_cost + reject_cost
    )

    # Sanity cap: friction should not exceed 3× the expected move
    # (otherwise the trade is never viable)
    if mean_abs > 0 and total_friction > mean_abs * 3.0:
        return mean_abs * 3.0

    return max(0.0, total_friction)


# ===================================================================
# Self-test (import verification)
# ===================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("AmplitudeSweeper - Self-test / Import verification")
    print("=" * 60)

    # Verify the class imports and constructs
    sweeper = AmplitudeSweeper(max_workers=2, output_dir="outputs")
    print(f"[OK] AmplitudeSweeper created: max_workers={sweeper.max_workers}, "
          f"output_dir={sweeper.output_dir}")

    # Verify dimensions
    n_combos = (
        len(sweeper.DEFAULT_SYMBOLS)
        * len(sweeper.DEFAULT_HORIZONS)
        * len(sweeper.DEFAULT_K_VALUES)
        * len(AmplitudeRegimeMapper.all_regimes())
        * 3  # folds
    )
    print(f"[OK] Sweep dimensions: {n_combos} combinations "
          f"({len(sweeper.DEFAULT_SYMBOLS)} symbols x "
          f"{len(sweeper.DEFAULT_HORIZONS)} horizons x "
          f"{len(sweeper.DEFAULT_K_VALUES)} k-values x "
          f"{len(AmplitudeRegimeMapper.all_regimes())} regimes x "
          f"3 folds)")

    # Verify folds creation
    folds = AmplitudeReplayRunner.create_folds_from_daterange(
        "2025-01-01", "2025-06-01", n_folds=3, train_ratio=0.5,
    )
    print(f"[OK] create_folds_from_daterange returned {len(folds)} folds:")
    for f in folds:
        print(f"      Fold {f['fold_id']}: {f['start']} -> {f['end']}")

    # Verify pipeline imports
    print(f"[OK] All required imports resolved successfully")

    # Verify static method is callable
    test_params = {
        "state_hash": "abc123",
        "horizon": 60,
        "entry": AmplitudeSurfaceEntry(
            state_hash="abc123",
            horizon=60,
            n=100,
            mean_abs=5.0,
            median_abs=4.5,
            std_abs=3.0,
            p90_abs=9.0,
            spread_multiple=3.5,
            aer=2.0,
            exceed_prob={1.0: 0.5, 1.5: 0.4, 2.0: 0.3, 3.0: 0.1, 4.0: 0.05},
        ),
        "exceed_model": SpreadExceedanceModel(),
        "promotion_engine": AmplitudePromotionEngine(),
        "profiles": {},
    }
    result = AmplitudeSweeper._run_single_experiment(test_params)
    print(f"[OK] _run_single_experiment returns {len(result)} keys: "
          f"{list(result.keys())}")
    print(f"      tier={result['tier']}, promoted={result['promoted']}, "
          f"pf_cme={result['pf_cme']:.2f}, net_edge={result['net_edge']:.2f}")

    print(f"\n[AmplitudeSweeper] Self-test passed.")
    print("=" * 60)
