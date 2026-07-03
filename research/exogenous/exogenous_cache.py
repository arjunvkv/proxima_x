"""
Program VI — Exogenous Amplitude Discovery
============================================
ExogenousCache — converts Tick Time Machine replay into exogenous amplitude
training records for Program VI.

Walks a merged replay stream across symbols, captures per-minute exogenous
state vectors and computes forward amplitude observations at multiple
horizons.  The resulting labelled dataset is cached as per-symbol parquet
files for downstream surface training.

Program VI is ORTHOGONAL to microstructure state — no OSS, no ECDF, no
entropy features are computed.
"""

import sys; sys.path.insert(0, "."); sys.path.insert(0, "proxima_x")

import os
import time as time_module  # use perf_counter — time.time gets patched by replay
from collections import defaultdict
from typing import Optional

import numpy as np
import polars as pl

from replay.environment import build_replay_environment, ReplayConfig
from replay.clock_patcher import patch_clock

from research.exogenous.schemas import ExogenousState, ExogenousObservation, make_key
from research.exogenous.event_clock import EventClock
from research.exogenous.session_open_detector import SessionOpenDetector
from research.exogenous.fixing_windows import FixingWindowDetector
from research.exogenous.rollover_detector import RolloverDetector
from research.exogenous.news_shock_proxy import NewsShockProxy
from research.exogenous.liquidity_void_detector import LiquidityVoidDetector


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_HORIZONS: list[int] = [60, 300, 900, 1800]


# ===================================================================
# ExogenousCache
# ===================================================================


class ExogenousCache:
    """Build, cache, and load exogenous amplitude training records.

    Typical usage::

        ec = ExogenousCache("cache/exogenous")
        df = ec.build(symbols=["EURJPY", "USDJPY"], start="2025-01-01",
                      end="2025-02-01")
        # df is a Polars DataFrame with all observations
        # Per-symbol parquet files are saved automatically
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __init__(self, base_path: str = "cache/exogenous") -> None:
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
        """Run the full exogenous-cache build pipeline.

        Steps
        -----
        1. Build a merged replay environment for all *symbols*.
        2. Set up EventClock, SessionOpenDetector, FixingWindowDetector,
           RolloverDetector, NewsShockProxy, LiquidityVoidDetector.
        3. Walk ticks round-robin, updating NewsShockProxy and
           LiquidityVoidDetector on EVERY tick.
        4. At each minute boundary (every 60 s of replay time) after
           *warmup*, emit an ExogenousState.
        5. For every emitted state, look forward across each *horizon* and
           compute realised amplitude observations.
        6. Persist per-symbol parquet files and return the full DataFrame.

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
            Schema: symbol, state_ts, session, fixing_window, rollover,
            liquidity_void, news_proxy, spread, tick_velocity,
            exogenous_key, horizon_sec, abs_move, signed_move.
        """
        if horizons is None:
            horizons = list(_DEFAULT_HORIZONS)

        t_start = time_module.perf_counter()
        print(
            f"[ExogenousCache] Build start  symbols={symbols}  "
            f"n_ticks={n_ticks}  warmup={warmup}  seed={seed}"
        )

        # ---- 1. Build replay environment --------------------------------
        print("[ExogenousCache] Building replay environment ...")
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

        # ---- 2. Set up exogenous detectors ------------------------------
        event_clock = EventClock()
        _session_detector = SessionOpenDetector()  # kept for interface completeness
        _fixing_detector = FixingWindowDetector()
        _rollover_detector = RolloverDetector()
        news_shock_proxy = NewsShockProxy()
        liquidity_void_detector = LiquidityVoidDetector()

        # ---- 3+4. Walk ticks & emit states ------------------------------
        print(
            f"[ExogenousCache] Walking up to {n_ticks} merged ticks "
            f"(warmup={warmup}) ..."
        )

        state_records: list[ExogenousState] = []
        total_ticks: int = 0
        idle_rounds: int = 0

        # Per-symbol tracking
        last_minute_key: dict[str, int] = {}
        tick_history: dict[str, list[tuple[float, float]]] = {}
        last_news_output: dict[str, dict] = {}
        last_void_output: dict[str, dict] = {}

        for sym in resolved_symbols:
            last_minute_key[sym] = -1
            tick_history[sym] = []
            last_news_output[sym] = {"shock_detected": False}
            last_void_output[sym] = {"void_detected": False}

        walk_t0 = time_module.perf_counter()

        while total_ticks < n_ticks:
            made_progress = False

            for sym in resolved_symbols:
                tick_obj = env.tick_source.get_tick(sym)
                if tick_obj is None:
                    continue

                made_progress = True
                total_ticks += 1

                # Extract raw tick fields
                ts = float(
                    tick_obj.get("time_sec") or tick_obj.get("timestamp", 0)
                )
                bid = float(tick_obj.get("bid", 0) or 0)
                ask = float(tick_obj.get("ask", 0) or 0)

                # Mid price (used as reference price)
                if ask > 0 and bid > 0:
                    mid_price = (bid + ask) / 2.0
                elif ask > 0:
                    mid_price = ask
                elif bid > 0:
                    mid_price = bid
                else:
                    mid_price = 0.0

                # Spread in bps
                current_spread = 0.2
                if ask > 0 and bid > 0:
                    mid = (bid + ask) / 2.0
                    if mid > 0:
                        current_spread = (ask - bid) / mid * 10000.0
                        if current_spread <= 0:
                            current_spread = 0.2

                # Build a compatible tick dict for the detectors
                formatted_tick = {
                    "ts": ts,
                    "price": mid_price,
                    "bid": bid,
                    "ask": ask,
                }

                # ---- Update detectors on EVERY tick --------------------
                news_output = news_shock_proxy.update(sym, formatted_tick)
                last_news_output[sym] = news_output

                void_output = liquidity_void_detector.update(sym, formatted_tick)
                last_void_output[sym] = void_output

                # Record tick for forward amplitude lookup
                tick_history[sym].append((ts, mid_price))

                # ---- Minute-boundary state emission (after warmup) ----
                if total_ticks >= warmup:
                    minute_key = int(ts // 60)
                    if minute_key > last_minute_key[sym]:
                        last_minute_key[sym] = minute_key

                        state = self._build_exogenous_state(
                            sym=sym,
                            ts=ts,
                            spread=current_spread,
                            event_clock=event_clock,
                            news_shock_proxy=news_shock_proxy,
                            liquidity_void_detector=liquidity_void_detector,
                            last_news_output=last_news_output[sym],
                            last_void_output=last_void_output[sym],
                        )
                        state_records.append(state)

                if total_ticks >= n_ticks:
                    break

            # Safety valve — feed exhausted
            if not made_progress:
                idle_rounds += 1
                if idle_rounds >= 3:
                    print(
                        f"[ExogenousCache] Feed exhausted after "
                        f"{total_ticks} ticks (idle_rounds={idle_rounds})."
                    )
                    break
            else:
                idle_rounds = 0

        walk_elapsed = time_module.perf_counter() - walk_t0
        print(
            f"[ExogenousCache] Walk complete: {total_ticks} ticks, "
            f"{len(state_records)} states in {walk_elapsed:.2f}s"
        )

        # ---- 5. Compute forward amplitudes ------------------------------
        if not state_records:
            print("[ExogenousCache] WARNING: no states captured; returning empty DF")
            return self._empty_df()

        print(f"[ExogenousCache] Computing forward amplitudes ...")
        observations: list[ExogenousObservation] = []

        fwd_t0 = time_module.perf_counter()
        for state in state_records:
            sym = state.symbol
            hist = tick_history.get(sym, [])
            start_price = self._price_before(hist, state.ts)
            if start_price is None or start_price == 0.0:
                continue

            for h in horizons:
                future_ts = state.ts + h
                # Collect price ticks in (state.ts, future_ts]
                window_prices: list[float] = [
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

                observations.append(
                    ExogenousObservation(
                        state=state,
                        horizon_sec=h,
                        abs_move=abs_move_bps,
                        signed_move=signed_move_bps,
                    )
                )

        fwd_elapsed = time_module.perf_counter() - fwd_t0
        print(
            f"[ExogenousCache] Forward pass: {len(observations)} observations "
            f"in {fwd_elapsed:.2f}s"
        )

        # ---- Convert & persist ------------------------------------------
        df = self._observations_to_df(observations)

        # Save per-symbol parquet (fold=0)
        for sym in resolved_symbols:
            sym_df = df.filter(pl.col("symbol") == sym)
            if len(sym_df) > 0:
                self.save(sym_df, symbol=sym, fold=0)

        total_elapsed = time_module.perf_counter() - t_start
        print(
            f"[ExogenousCache] Build complete in {total_elapsed:.2f}s  "
            f"total_rows={len(df)}"
        )
        return df

    # ------------------------------------------------------------------

    def load(self, symbol: str, fold: int = 0) -> pl.DataFrame:
        """Load a cached exogenous parquet file for *symbol* / *fold*."""
        path = self._fold_path(symbol, fold)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"ExogenousCache: no cached data for symbol={symbol!r} "
                f"fold={fold} at {path}"
            )
        return pl.read_parquet(path)

    # ------------------------------------------------------------------

    def save(self, df: pl.DataFrame, symbol: str, fold: int = 0) -> None:
        """Save *df* as a compressed parquet file for *symbol* / *fold*."""
        path = self._fold_path(symbol, fold)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.write_parquet(path, compression="zstd")
        print(f"[ExogenousCache] Saved {len(df)} rows -> {path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fold_path(self, symbol: str, fold: int) -> str:
        return os.path.join(self.base_path, f"{symbol}_fold{fold}.parquet")

    # ------------------------------------------------------------------

    @staticmethod
    def _build_exogenous_state(
        sym: str,
        ts: float,
        spread: float,
        event_clock: EventClock,
        news_shock_proxy: NewsShockProxy,
        liquidity_void_detector: LiquidityVoidDetector,
        last_news_output: dict,
        last_void_output: dict,
    ) -> ExogenousState:
        """Build a single ExogenousState from current features.

        Parameters
        ----------
        sym : str
            Symbol identifier.
        ts : float
            Epoch timestamp in seconds.
        spread : float
            Current spread in bps.
        event_clock : EventClock
            Used for session, fix, and rollover classification.
        news_shock_proxy : NewsShockProxy
            Used to extract shock flag and tick velocity.
        liquidity_void_detector : LiquidityVoidDetector
            Used to extract void flag.
        last_news_output : dict
            The most recent result from ``news_shock_proxy.update()``.
        last_void_output : dict
            The most recent result from ``liquidity_void_detector.update()``.

        Returns
        -------
        ExogenousState
        """
        # Session classification
        session = event_clock.get_session(ts)
        fixing_window = event_clock.in_fix(ts)
        rollover = event_clock.in_rollover(ts)

        # Exogenous flags from the detectors
        news_proxy = bool(last_news_output.get("shock_detected", False))
        liquidity_void = bool(last_void_output.get("void_detected", False))

        # Tick velocity from NewsShockProxy internals
        tick_velocity = _extract_tick_velocity(news_shock_proxy, sym)

        return ExogenousState(
            symbol=sym,
            ts=ts,
            session=session,
            fixing_window=fixing_window,
            rollover=rollover,
            liquidity_void=liquidity_void,
            news_proxy=news_proxy,
            spread=spread,
            tick_velocity=tick_velocity,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _price_before(
        hist: list[tuple[float, float]], ts: float
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
                "session": pl.Utf8,
                "fixing_window": pl.Utf8,
                "rollover": pl.Boolean,
                "liquidity_void": pl.Boolean,
                "news_proxy": pl.Boolean,
                "spread": pl.Float64,
                "tick_velocity": pl.Float64,
                "exogenous_key": pl.Utf8,
                "horizon_sec": pl.Int32,
                "abs_move": pl.Float64,
                "signed_move": pl.Float64,
            }
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _observations_to_df(
        observations: list[ExogenousObservation],
    ) -> pl.DataFrame:
        """Convert a list of observations to a Polars DataFrame."""
        if not observations:
            return ExogenousCache._empty_df()

        records: list[dict] = []
        for obs in observations:
            s = obs.state
            key = make_key(s)
            records.append(
                {
                    "symbol": s.symbol,
                    "state_ts": s.ts,
                    "session": s.session,
                    "fixing_window": (
                        s.fixing_window if s.fixing_window is not None else "None"
                    ),
                    "rollover": s.rollover,
                    "liquidity_void": s.liquidity_void,
                    "news_proxy": s.news_proxy,
                    "spread": s.spread,
                    "tick_velocity": s.tick_velocity,
                    "exogenous_key": key,
                    "horizon_sec": obs.horizon_sec,
                    "abs_move": obs.abs_move,
                    "signed_move": obs.signed_move,
                }
            )
        return pl.DataFrame(records)


# ===================================================================
# Module-level helpers
# ===================================================================


def _extract_tick_velocity(
    proxy: NewsShockProxy, symbol: str
) -> float:
    """Extract the latest tick velocity from NewsShockProxy internals.

    Accesses the per-symbol ``velocities`` deque which stores the most
    recent tick velocity computed during ``update()``.

    Returns
    -------
    float
        The last recorded tick velocity in price-units/second, or 0.0
        if no velocity has been recorded yet.
    """
    try:
        state = proxy._state.get(symbol)
        if state is None:
            return 0.0
        velocities = state.get("velocities")
        if velocities and len(velocities) > 0:
            return float(velocities[-1])
    except Exception:
        pass
    return 0.0


# ===================================================================
# Self-test
# ===================================================================
if __name__ == "__main__":
    # Quick smoke-test: construct cache, verify import + schema.
    ec = ExogenousCache(base_path="cache/exogenous_test")
    print(f"ExogenousCache created at {ec.base_path}")

    # Build a tiny dataset (single symbol, few ticks)
    df = ec.build(
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
    print("\n[ExogenousCache] Self-test passed.")
