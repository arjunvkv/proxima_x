"""
AmplitudeCache — converts Tick Time Machine replay into amplitude training
records for Phase V.

Walks a merged replay stream across symbols, captures per-minute microstructure
state (amplitude features), and computes forward-looking amplitude observations
at multiple horizons.  The resulting labelled dataset is cached as per-symbol
parquet files for downstream surface training.
"""

import sys; sys.path.insert(0, "."); sys.path.insert(0, "proxima_x")

import os
import math
import pickle
import hashlib
import time as time_module
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from replay.environment import build_replay_environment, ReplayConfig
from replay.clock_patcher import patch_clock
from research.replay_cache import ReplayCache
from signals.outcome_surface_signal import OutcomeSurfaceSignal
from signals.transition_oss import TransitionOSS
from signals.sal_mapper import SignalAggregationLayer
from features.ecdf_transform import PerSymbolECDF
from research.amplitude.schemas import (
    AmplitudeState,
    StateHasher,
    ForwardAmplitude,
    AmplitudeObservation,
)
from research.amplitude.burst_detector import BurstDetector
from research.amplitude.regime_mapper import AmplitudeRegimeMapper
from research.cep.session_partition import SessionPartitioner


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_HORIZONS: List[int] = [60, 300, 900, 1800]
_OSS_TRAIN_LOOKAHEAD_TICKS: int = 20
_PRICE_BUF_MAXLEN: int = 50
_ENTROPY_SLOPE_WINDOW_SEC: float = 300.0  # 5 minutes


# ===================================================================
# AmplitudeCache
# ===================================================================


class AmplitudeCache:
    """Build, cache, and load amplitude training records.

    Typical usage::

        ac = AmplitudeCache("cache/amplitude")
        df = ac.build(symbols=["EURJPY", "USDJPY"], start="2025-01-01",
                      end="2025-02-01")
        # df is a Polars DataFrame with all observations
        # Per-symbol parquet files are saved automatically
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __init__(self, base_path: str = "cache/amplitude") -> None:
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    # ------------------------------------------------------------------

    def build(
        self,
        symbols: list[str],
        start: str,
        end: str,
        horizons: Optional[list[int]] = None,
        n_ticks: int = 80000,
        warmup: int = 5000,
        seed: int = 42,
    ) -> pl.DataFrame:
        """Run the full amplitude-cache build pipeline.

        Steps
        -----
        1. Train OSS from a ReplayCache warm-up pass.
        2. Set up signal pipeline: OSS, TrOSS, SAL, ECDF, BurstDetector,
           RegimeMapper, SessionPartitioner.
        3. Build a merged replay environment for all *symbols*.
        4. Walk ticks round-robin, updating features online.
        5. At each minute boundary (every 60 s of replay time) after
           *warmup*, emit an AmplitudeState.
        6. For every emitted state, look forward across each *horizon* and
           compute realised amplitude observations.
        7. Persist per-symbol parquet files and return the full DataFrame.

        Parameters
        ----------
        symbols : list of str
            Instrument tickers (e.g. ``["EURJPY", "USDJPY"]``).
        start : str
            ISO date ``"YYYY-MM-DD"`` — start of replay window.
        end : str
            ISO date ``"YYYY-MM-DD"`` — end of replay window.
        horizons : list of int, optional
            Forward look-ahead windows in **seconds**.
            Default: ``[60, 300, 900, 1800]``.
        n_ticks : int
            Total number of merged ticks to consume.
        warmup : int
            Number of initial ticks to skip before recording states.
        seed : int
            Deterministic replay seed.

        Returns
        -------
        pl.DataFrame
            Schema described in _observations_to_df.
        """
        if horizons is None:
            horizons = list(_DEFAULT_HORIZONS)

        t_start = time_module.perf_counter()
        print(
            f"[AmplitudeCache] Build start  symbols={symbols}  "
            f"n_ticks={n_ticks}  warmup={warmup}  seed={seed}"
        )

        # ---- 1. Train OSS from ReplayCache --------------------------------
        oss = self._train_oss(symbols, start, end, n_ticks, seed)

        # ---- 2. Set up signal pipeline ------------------------------------
        tross = TransitionOSS(oss, cross_threshold=2)
        sal = SignalAggregationLayer()
        ecdf = PerSymbolECDF(window_size=2000)
        burst_detector = BurstDetector(window=20)
        regime_mapper = AmplitudeRegimeMapper()
        partitioner = SessionPartitioner()

        # ---- 3. Build replay environment ----------------------------------
        print("[AmplitudeCache] Building replay environment ...")
        cfg = ReplayConfig(
            symbols=symbols,
            start=start,
            end=end,
            speed=500000,
            burst=True,
            latency=False,
            slippage=False,
            seed=seed,
            warmup_ticks=warmup,
        )
        env = build_replay_environment(cfg)
        patch_clock(env.clock)

        resolved_symbols = (
            list(env.replay_feed._symbols)
            if hasattr(env, "replay_feed") and env.replay_feed is not None
            else symbols
        )

        # ---- 4+5. Walk ticks & emit states --------------------------------
        print(
            f"[AmplitudeCache] Walking up to {n_ticks} merged ticks "
            f"(warmup={warmup}) ..."
        )

        state_records: List[AmplitudeState] = []
        total_ticks: int = 0
        idle_rounds: int = 0

        # Per-symbol state
        price_bufs: Dict[str, List[float]] = {}
        entropy_hist: Dict[str, List[Tuple[float, float]]] = {}
        prev_bucket: Dict[str, int] = {}
        last_minute_key: Dict[str, int] = {}
        tick_history: Dict[str, List[Tuple[float, float]]] = {}

        for sym in resolved_symbols:
            price_bufs[sym] = []
            entropy_hist[sym] = []
            prev_bucket[sym] = -1
            last_minute_key[sym] = -1
            tick_history[sym] = []

        walk_t0 = time_module.perf_counter()

        while total_ticks < n_ticks:
            made_progress = False

            for sym in resolved_symbols:
                tick_obj = env.tick_source.get_tick(sym)
                if tick_obj is None:
                    continue

                made_progress = True
                total_ticks += 1

                ask = float(tick_obj.get("ask", 0) or 0)
                bid = float(tick_obj.get("bid", 0) or 0)
                price = ask if ask > 0 else (bid if bid > 0 else 0.0)
                ts = float(
                    tick_obj.get("time_sec") or
                    tick_obj.get("timestamp_ns", 0) / 1_000_000_000 or
                    tick_obj.get("timestamp", 0) or
                    time_module.perf_counter()
                )
                current_spread = 0.2
                if ask > 0 and bid > 0:
                    current_spread = (ask - bid) / ((ask + bid) / 2) * 10000.0
                    if current_spread <= 0:
                        current_spread = 0.2

                # Record tick for forward scan
                tick_history[sym].append((ts, price))

                # Update ECDF
                ecdf_rank = ecdf.update(sym, price)

                # Price buffer for entropy
                price_bufs[sym].append(price)
                if len(price_bufs[sym]) > _PRICE_BUF_MAXLEN:
                    price_bufs[sym] = price_bufs[sym][-_PRICE_BUF_MAXLEN:]

                # Burst detector
                burst_result = burst_detector.update(sym, tick_obj)

                # ---- Minute-boundary state emission (after warmup) ----
                if total_ticks >= warmup:
                    minute_key = int(ts // 60)
                    if minute_key > last_minute_key[sym]:
                        last_minute_key[sym] = minute_key

                        state = self._build_state(
                            sym=sym,
                            ts=ts,
                            price=price,
                            ecdf_rank=ecdf_rank,
                            price_buf=price_bufs[sym],
                            entropy_hist=entropy_hist[sym],
                            prev_bucket=prev_bucket[sym],
                            burst_result=burst_result,
                            oss=oss,
                            tross=tross,
                            sal=sal,
                            regime_mapper=regime_mapper,
                            resolved_symbols=resolved_symbols,
                            spread=current_spread,
                        )
                        state_records.append(state)

                        # Update prev_bucket after state creation
                        prev_bucket[sym] = int(ecdf_rank * 10)

                if total_ticks >= n_ticks:
                    break

            # Safety valve: no progress in a full round means the feed
            # is exhausted for all symbols.
            if not made_progress:
                idle_rounds += 1
                if idle_rounds >= 3:
                    print(
                        f"[AmplitudeCache] Feed exhausted after "
                        f"{total_ticks} ticks (idle rounds={idle_rounds})."
                    )
                    break
            else:
                idle_rounds = 0

        walk_elapsed = time_module.perf_counter() - walk_t0
        print(
            f"[AmplitudeCache] Walk complete: {total_ticks} ticks, "
            f"{len(state_records)} states in {walk_elapsed:.2f}s"
        )

        # ---- 6. Compute forward amplitudes --------------------------------
        if not state_records:
            print("[AmplitudeCache] WARNING: no states captured; returning empty DF")
            return self._empty_df()

        print(f"[AmplitudeCache] Computing forward amplitudes ...")
        observations: List[AmplitudeObservation] = []

        fwd_t0 = time_module.perf_counter()
        for state in state_records:
            sym = state.symbol
            hist = tick_history.get(sym, [])
            start_price = self._price_before(hist, state.ts)
            if start_price is None:
                continue

            for h in horizons:
                future_ts = state.ts + h
                # Collect price ticks in (state.ts, future_ts]
                window_prices: List[float] = [
                    p for t, p in hist if state.ts < t <= future_ts
                ]
                if not window_prices:
                    continue

                max_p = float(np.max(window_prices))
                min_p = float(np.min(window_prices))
                end_price = window_prices[-1]

                # All moves in basis-points relative to start_price
                abs_move = max(abs(max_p - start_price), abs(min_p - start_price))
                abs_move_bps = abs_move / start_price * 10000.0

                signed_move = end_price - start_price
                signed_move_bps = signed_move / start_price * 10000.0

                max_exc_bps = abs(max_p - start_price) / start_price * 10000.0
                min_exc_bps = abs(min_p - start_price) / start_price * 10000.0

                future = ForwardAmplitude(
                    horizon_sec=h,
                    abs_move=abs_move_bps,
                    signed_move=signed_move_bps,
                    max_excursion=max_exc_bps,
                    min_excursion=min_exc_bps,
                )
                observations.append(AmplitudeObservation(state=state, future=future))

        fwd_elapsed = time_module.perf_counter() - fwd_t0
        print(
            f"[AmplitudeCache] Forward pass: {len(observations)} observations "
            f"in {fwd_elapsed:.2f}s"
        )

        # ---- Convert & persist --------------------------------------------
        df = self._observations_to_df(observations)

        # Save per-symbol parquet (fold=0)
        for sym in resolved_symbols:
            sym_df = df.filter(pl.col("symbol") == sym)
            if len(sym_df) > 0:
                self.save(sym_df, symbol=sym, fold=0)

        total_elapsed = time_module.perf_counter() - t_start
        print(
            f"[AmplitudeCache] Build complete in {total_elapsed:.2f}s  "
            f"total_rows={len(df)}"
        )
        return df

    # ------------------------------------------------------------------

    def load(self, symbol: str, fold: int = 0) -> pl.DataFrame:
        """Load a cached amplitude parquet file for *symbol* / *fold*."""
        path = self._fold_path(symbol, fold)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"AmplitudeCache: no cached data for symbol={symbol!r} "
                f"fold={fold} at {path}"
            )
        return pl.read_parquet(path)

    # ------------------------------------------------------------------

    def save(self, df: pl.DataFrame, symbol: str, fold: int = 0) -> None:
        """Save *df* as a compressed parquet file for *symbol* / *fold*."""
        path = self._fold_path(symbol, fold)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.write_parquet(path, compression="zstd")
        print(f"[AmplitudeCache] Saved {len(df)} rows -> {path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fold_path(self, symbol: str, fold: int) -> str:
        return os.path.join(self.base_path, f"{symbol}_fold{fold}.parquet")

    # ------------------------------------------------------------------

    def _train_oss(
        self,
        symbols: list[str],
        start: str,
        end: str,
        n_ticks: int,
        seed: int,
    ) -> OutcomeSurfaceSignal:
        """Build a warm-up ReplayCache, derive outcomes, and train OSS."""
        print(f"[AmplitudeCache] Training OSS from ReplayCache ...")
        cache = ReplayCache(
            symbols=symbols,
            start=start,
            end=end,
            tick_limit=n_ticks,
            seed=seed,
        )
        cached_ticks = cache.compute()

        if not cached_ticks:
            print(
                "[AmplitudeCache] WARNING: ReplayCache returned no ticks; "
                "OSS will be untrained."
            )
            return OutcomeSurfaceSignal(ev_threshold=0.05)

        train_records: List[Dict] = []
        n = len(cached_ticks)
        la = _OSS_TRAIN_LOOKAHEAD_TICKS

        for i, tick in enumerate(cached_ticks):
            future_idx = min(i + la, n - 1)
            future_tick = cached_ticks[future_idx]
            price_now = tick.get("price", 0.0)
            price_fwd = future_tick.get("price", 0.0)
            if price_now > 0:
                outcome_bps = (price_fwd - price_now) / price_now * 10000.0
            else:
                outcome_bps = 0.0

            train_records.append(
                {
                    "ecdf": tick.get("ecdf", 0.5),
                    "outcome": outcome_bps,
                }
            )

        oss = OutcomeSurfaceSignal.from_pipeline_records(train_records, ev_threshold=0.05)
        print(
            f"[AmplitudeCache] OSS trained on {len(train_records)} records, "
            f"{oss.bucket_count()} buckets, density={oss.signal_density():.3f}"
        )
        return oss

    # ------------------------------------------------------------------

    @staticmethod
    def _build_state(
        sym: str,
        ts: float,
        price: float,
        ecdf_rank: float,
        price_buf: List[float],
        entropy_hist: List[Tuple[float, float]],
        prev_bucket: int,
        burst_result: dict,
        oss: OutcomeSurfaceSignal,
        tross: "TransitionOSS",
        sal: SignalAggregationLayer,
        regime_mapper: AmplitudeRegimeMapper,
        resolved_symbols: List[str],
        spread: float = 0.2,
    ) -> AmplitudeState:
        """Build a single AmplitudeState from current features."""

        # --- entropy & entropy slope ----------------------------------------
        ent = _entropy(price_buf)
        entropy_hist.append((ts, ent))

        # Prune history older than 5 minutes
        cutoff = ts - _ENTROPY_SLOPE_WINDOW_SEC
        pruned = [(t, e) for t, e in entropy_hist if t >= cutoff]
        entropy_hist.clear()
        entropy_hist.extend(pruned)

        ent_slope = _compute_entropy_slope(entropy_hist)

        # --- OSS bucket (ECDF bucket 0-9) ----------------------------------
        bucket = int(ecdf_rank * 10)
        if bucket > 9:
            bucket = 9
        oss_signal = oss.predict(ecdf_rank)

        # --- TrOSS delta (signed bucket change, clamped to +/-2) -----------
        if prev_bucket >= 0:
            tross_delta = max(-2, min(2, bucket - prev_bucket))
        else:
            tross_delta = 0

        # --- SAL score ------------------------------------------------------
        # Build an all_scores dict for consensus
        all_scores: Dict[str, float] = {s: 0.0 for s in resolved_symbols}
        all_scores[sym] = float(oss_signal)

        sal.update(sym, oss_signal, confidence=1.0, price=price, all_scores=all_scores)
        raw_score = sal.agg_score()
        # Normalise sal_score to [0, 1] from [-1, 1]
        if raw_score is None:
            sal_score = 0.0
        else:
            sal_score = abs(float(raw_score))

        # --- AmplitudeState -------------------------------------------------
        state = AmplitudeState(
            symbol=sym,
            ts=ts,
            oss_bucket=bucket,
            tross_delta=tross_delta,
            sal_score=sal_score,
            entropy_slope=ent_slope,
            compression_density=float(burst_result.get("compression_density", 0.0)),
            tick_velocity=float(burst_result.get("tick_velocity", 0.0)),
            spread=spread,
            regime_id=regime_mapper.map_from_features(
                burst_score=float(burst_result.get("burst_score", 0.0)),
                compression_density=float(burst_result.get("compression_density", 0.0)),
                tick_velocity=float(burst_result.get("tick_velocity", 0.0)),
                entropy_slope=ent_slope,
                spread=spread,
            ),
        )
        return state

    # ------------------------------------------------------------------

    @staticmethod
    def _price_before(
        hist: List[Tuple[float, float]], ts: float
    ) -> Optional[float]:
        """Return the most recent price at or before *ts*."""
        for t, p in reversed(hist):
            if t <= ts:
                return p
        return None

    # ------------------------------------------------------------------

    @staticmethod
    def _empty_df() -> pl.DataFrame:
        """Return an empty DataFrame with the expected schema."""
        return pl.DataFrame(
            schema={
                "symbol": pl.Utf8,
                "state_ts": pl.Float64,
                "oss_bucket": pl.Int32,
                "tross_delta": pl.Int32,
                "sal_score": pl.Float64,
                "entropy_slope": pl.Float64,
                "compression_density": pl.Float64,
                "tick_velocity": pl.Float64,
                "spread": pl.Float64,
                "regime_id": pl.Utf8,
                "state_hash": pl.Utf8,
                "horizon_sec": pl.Int32,
                "horizon": pl.Int32,
                "abs_move": pl.Float64,
                "signed_move": pl.Float64,
                "max_excursion": pl.Float64,
                "min_excursion": pl.Float64,
            }
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _observations_to_df(
        observations: List[AmplitudeObservation],
    ) -> pl.DataFrame:
        """Convert a list of observations to a Polars DataFrame."""
        if not observations:
            return AmplitudeCache._empty_df()

        records: List[Dict] = []
        for obs in observations:
            s = obs.state
            f = obs.future
            records.append(
                {
                    "symbol": s.symbol,
                    "state_ts": s.ts,
                    "oss_bucket": s.oss_bucket,
                    "tross_delta": s.tross_delta,
                    "sal_score": s.sal_score,
                    "entropy_slope": s.entropy_slope,
                    "compression_density": s.compression_density,
                    "tick_velocity": s.tick_velocity,
                    "spread": s.spread,
                    "regime_id": s.regime_id,
                    "state_hash": StateHasher.hash(s),
                    "horizon_sec": f.horizon_sec,
                    "horizon": f.horizon_sec,
                    "abs_move": f.abs_move,
                    "signed_move": f.signed_move,
                    "max_excursion": f.max_excursion,
                    "min_excursion": f.min_excursion,
                }
            )
        return pl.DataFrame(records)


# ===================================================================
# Module-level helpers (stateless, testable)
# ===================================================================


def _entropy(prices: List[float]) -> float:
    """Compute normalised entropy of *prices* over a 10-bin histogram.

    Returns a value in *[0, 1]* where 1 means maximally dispersed and
    0 means all values are identical.  Falls back to 0.5 when fewer
    than 10 samples are available.
    """
    if len(prices) < 10:
        return 0.5
    mn = min(prices)
    mx = max(prices)
    if mx == mn:
        return 0.0
    nb = 10
    n = len(prices)
    hist = [0] * nb
    for p in prices:
        idx = int((p - mn) / (mx - mn) * nb)
        if idx >= nb:
            idx = nb - 1
        hist[idx] += 1
    ent = 0.0
    for h in hist:
        if h > 0:
            pv = h / n
            ent -= pv * math.log2(pv)
    return min(ent / math.log2(nb), 1.0)


def _compute_entropy_slope(
    entropy_hist: List[Tuple[float, float]],
) -> float:
    """Fit a linear regression over *(timestamp, entropy)* pairs.

    Returns the slope (entropy per second).  Returns 0.0 if fewer
    than 2 points are available, or if the time span is zero.
    """
    if len(entropy_hist) < 2:
        return 0.0
    ts_arr = np.array([t for t, _ in entropy_hist], dtype=np.float64)
    ev_arr = np.array([e for _, e in entropy_hist], dtype=np.float64)

    t_range = ts_arr[-1] - ts_arr[0]
    if t_range < 1e-9:
        return 0.0

    # Centre for numerical stability
    t_mean = np.mean(ts_arr)
    ts_centred = ts_arr - t_mean
    ev_mean = np.mean(ev_arr)

    cov = np.sum(ts_centred * (ev_arr - ev_mean))
    var = np.sum(ts_centred ** 2)
    if var < 1e-12:
        return 0.0
    return float(cov / var)


# ===================================================================
# Self-test
# ===================================================================
if __name__ == "__main__":
    # Quick smoke-test: construct cache, verify import + schema.
    ac = AmplitudeCache(base_path="cache/amplitude_test")
    print(f"AmplitudeCache created at {ac.base_path}")

    # Build a tiny dataset (single symbol, few ticks)
    df = ac.build(
        symbols=["EURJPY"],
        start="2025-01-02",
        end="2025-01-03",
        n_ticks=5000,
        warmup=1000,
        seed=42,
    )
    print(f"\nResult DataFrame: {len(df)} rows x {len(df.columns)} columns")
    if len(df) > 0:
        print(df.head())
        print("\nColumn dtypes:")
        print(df.dtypes)
    print("\n[AmplitudeCache] Self-test passed.")
