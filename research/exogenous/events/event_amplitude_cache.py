import sys

sys.path.insert(0, ".")
sys.path.insert(0, "proxima_x")

import os
import time
from typing import Optional

import numpy as np
import polars as pl

from replay.environment import build_replay_environment, ReplayConfig
from replay.clock_patcher import patch_clock
from research.exogenous.events.schemas import (
    MacroEvent,
    EventProximityState,
    EventAmplitudeObservation,
)
from research.exogenous.events.event_loader import EventLoader
from research.exogenous.events.event_window_mapper import EventWindowMapper


# ── helpers ──────────────────────────────────────────────────────────────────

def _currencies_for_symbol(symbol: str) -> list[str]:
    """Derive relevant currencies from an FX symbol.

    Examples
    --------
    EURJPY  -> ['EUR', 'JPY']
    USDJPY  -> ['USD', 'JPY']
    GBPJPY  -> ['GBP', 'JPY']
    XAUUSD  -> ['USD']
    EURUSD  -> ['EUR', 'USD']
    """
    mapping = {
        "EURJPY": ["EUR", "JPY", "USD"],  # USD events affect EURJPY via USDJPY correlation
        "USDJPY": ["USD", "JPY"],
        "GBPJPY": ["GBP", "JPY"],
        "XAUUSD": ["USD"],
        "EURUSD": ["EUR", "USD"],
    }
    return mapping.get(symbol.upper(), [symbol[:3]])


def _extract_ts(tick_obj: dict) -> float:
    """Extract the unix-epoch timestamp from a tick dictionary."""
    return float(tick_obj.get("time_sec") or tick_obj.get("timestamp", 0))


def _extract_spread(tick_obj: dict) -> float:
    """Extract the bid–ask spread from a tick dictionary."""
    ask = float(tick_obj.get("ask", 0) or 0)
    bid = float(tick_obj.get("bid", 0) or 0)
    return ask - bid


# ── main class ───────────────────────────────────────────────────────────────

class EventAmplitudeCache:
    """Walk replay ticks, classify event proximity, and record amplitude
    observations for Program VI.5 (Tick Time Machine)."""

    def __init__(self, base_path: str = "cache/exogenous/events") -> None:
        self._base_path = base_path
        self._loader = EventLoader()
        self._mapper = EventWindowMapper()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def build(
        self,
        symbols: list[str],
        start: str,
        end: str,
        events: Optional[list[MacroEvent]] = None,
        horizons: Optional[list[int]] = None,
        n_ticks: int = 80000,
        warmup: int = 5000,
        seed: int = 42,
    ) -> pl.DataFrame:
        """Walk merged replay ticks, record event-proximity states, compute
        forward amplitudes, and persist per-symbol results.

        Parameters
        ----------
        symbols : list[str]
            FX symbols to replay (e.g. ``["EURJPY", "USDJPY"]``).
        start : str
            Replay start date ``"YYYY-MM-DD"``.
        end : str
            Replay end date ``"YYYY-MM-DD"``.
        events : list[MacroEvent] | None
            If *None*, built-in events are loaded via ``EventLoader``.
        horizons : list[int] | None
            Forward-amplitude lookback windows in seconds.
            Default: ``[60, 300, 900, 1800]``.
        n_ticks : int
            Maximum number of merged ticks to walk (default 80 000).
        warmup : int
            Number of ticks to skip before recording states (default 5000).
        seed : int
            Random seed for reproducibility (default 42).

        Returns
        -------
        pl.DataFrame
            Columns:
                symbol, state_ts, event_bucket, event_impact, currency_match,
                event_name, seconds_to_event, spread, horizon_sec,
                abs_move, signed_move
        """
        t_start = time.perf_counter()

        # ── 1.  Events ────────────────────────────────────────────────
        if events is None:
            events = self._loader.load_builtin_events()

        # Collect currencies that matter for the requested symbols
        relevant_currencies: set[str] = set()
        for sym in symbols:
            relevant_currencies.update(_currencies_for_symbol(sym))

        filtered_events = self._loader.filter_by_currency(
            events, list(relevant_currencies)
        )

        if not filtered_events:
            # No relevant events — return empty DataFrame with the
            # expected schema.
            t_elapsed = time.perf_counter() - t_start
            self._log_timing("build (no events) — returning empty", t_elapsed)
            return pl.DataFrame(
                schema={
                    "symbol": pl.Utf8,
                    "state_ts": pl.Float64,
                    "event_bucket": pl.Utf8,
                    "event_impact": pl.Utf8,
                    "currency_match": pl.Boolean,
                    "event_name": pl.Utf8,
                    "seconds_to_event": pl.Float64,
                    "spread": pl.Float64,
                    "horizon_sec": pl.Int32,
                    "abs_move": pl.Float64,
                    "signed_move": pl.Float64,
                }
            )

        # ── 2.  Window mapper ────────────────────────────────────────
        # We will classify for each symbol using its relevant currencies.
        # The mapper is stateless so we can re-use one instance.

        # ── 3.  Build replay environment ─────────────────────────────
        config = ReplayConfig(
            symbols=symbols,
            start=start,
            end=end,
            speed=0.0,  # manual stepping; no wall-clock acceleration
            mode="ACCELERATED",
            seed=seed,
            warmup_ticks=warmup,
        )

        env = build_replay_environment(config)
        patch_clock(env.clock)

        feed = env.replay_feed
        # feed is guaranteed non-None for a replay environment
        _feed = feed  # keep mypy happy

        # ── 4.  Walk merged ticks ───────────────────────────────────
        # We keep a sliding history per symbol keyed by (ts, symbol) for
        # forward-amplitude computation.
        #
        # States are recorded at **minute boundaries** after warmup.

        # Per-symbol state records: list[dict]
        state_records: list[dict] = []

        # Per-symbol tick history: symbol -> list[(ts, price, spread)]
        # price is mid = (bid + ask) / 2
        tick_history: dict[str, list[tuple[float, float, float]]] = {
            sym: [] for sym in symbols
        }

        tick_count = 0
        last_minute = -1

        while tick_count < n_ticks:
            tick_obj = _feed.next()
            if tick_obj is None:
                break

            tick_count += 1
            ts = _extract_ts(tick_obj)
            sym = str(tick_obj.get("symbol", ""))
            if sym not in symbols:
                continue

            # mid price
            bid = float(tick_obj.get("bid", 0) or 0)
            ask = float(tick_obj.get("ask", 0) or 0)
            mid = (bid + ask) / 2.0
            spread = ask - bid

            # Append to tick history (trim later if needed)
            tick_history[sym].append((ts, mid, spread))

            # ── classify at minute boundaries after warmup ───────
            if tick_count >= warmup:
                minute = int(ts // 60)
                if minute != last_minute:
                    last_minute = minute

                    # Determine the currencies for this symbol
                    sym_currencies = _currencies_for_symbol(sym)

                    # Classify against events for each currency;
                    # take the nearest event across all relevant currencies.
                    best_state: Optional[EventProximityState] = None
                    for cur in sym_currencies:
                        state = self._mapper.classify(
                            ts=ts, events=filtered_events, currency=cur
                        )
                        if state.bucket != "NONE":
                            if best_state is None:
                                best_state = state
                            else:
                                # Keep the one with the smaller absolute distance
                                if abs(state.seconds_to_event) < abs(
                                    best_state.seconds_to_event
                                ):
                                    best_state = state
                                # If tied, prefer currency_match=True
                                elif (
                                    abs(state.seconds_to_event)
                                    == abs(best_state.seconds_to_event)
                                    and state.currency_match
                                    and not best_state.currency_match
                                ):
                                    best_state = state

                    if best_state is not None:
                        state_records.append(
                            {
                                "symbol": sym,
                                "state_ts": ts,
                                "mid": mid,
                                "spread": spread,
                                "event_bucket": best_state.bucket,
                                "event_impact": best_state.impact or "",
                                "currency_match": best_state.currency_match,
                                "event_name": best_state.nearest_event_name or "",
                                "seconds_to_event": best_state.seconds_to_event,
                                # Stash the index of this tick in history for
                                # forward-amplitude lookup.
                                "_history_idx": len(tick_history[sym]) - 1,
                            }
                        )

        # ── 5.  Compute forward amplitudes ──────────────────────────
        if horizons is None:
            horizons = [60, 300, 900, 1800]

        rows: list[dict] = []

        for rec in state_records:
            sym = rec["symbol"]
            state_ts = rec["state_ts"]
            entry_price = rec["mid"]
            entry_spread = rec["spread"]
            hist_idx = rec["_history_idx"]
            history = tick_history[sym]

            for horizon in horizons:
                # Find the farthest tick within the horizon window
                end_ts = state_ts + horizon
                # Search forward from the recorded index
                max_idx = hist_idx
                for j in range(hist_idx, len(history)):
                    if history[j][0] <= end_ts:
                        max_idx = j
                    else:
                        break

                # Compute moves in basis points (1 bp = 0.0001 for most fx,
                # but we compute as (price / entry_price - 1) * 10000)
                #  |  For JPY pairs mid ~ 100-200, this still yields bps.
                max_price = max(history[k][1] for k in range(hist_idx, max_idx + 1))
                min_price = min(history[k][1] for k in range(hist_idx, max_idx + 1))
                end_price = history[max_idx][1]

                abs_move = (max_price - min_price) / entry_price * 10000.0
                signed_move = (end_price - entry_price) / entry_price * 10000.0

                rows.append(
                    {
                        "symbol": sym,
                        "state_ts": state_ts,
                        "event_bucket": rec["event_bucket"],
                        "event_impact": rec["event_impact"],
                        "currency_match": rec["currency_match"],
                        "event_name": rec["event_name"],
                        "seconds_to_event": rec["seconds_to_event"],
                        "spread": entry_spread,
                        "horizon_sec": horizon,
                        "abs_move": abs_move,
                        "signed_move": signed_move,
                    }
                )

        result = pl.DataFrame(
            rows,
            schema={
                "symbol": pl.Utf8,
                "state_ts": pl.Float64,
                "event_bucket": pl.Utf8,
                "event_impact": pl.Utf8,
                "currency_match": pl.Boolean,
                "event_name": pl.Utf8,
                "seconds_to_event": pl.Float64,
                "spread": pl.Float64,
                "horizon_sec": pl.Int32,
                "abs_move": pl.Float64,
                "signed_move": pl.Float64,
            },
        )

        # ── 6.  Persist per symbol ──────────────────────────────────
        for sym in symbols:
            sym_df = result.filter(pl.col("symbol") == sym)
            if sym_df.height > 0:
                self.save(sym_df, symbol=sym, fold=0)

        t_elapsed = time.perf_counter() - t_start
        self._log_timing(
            f"build ({len(symbols)} symbols, {len(result)} observations)",
            t_elapsed,
        )

        return result

    # ------------------------------------------------------------------
    # load / save
    # ------------------------------------------------------------------

    def load(self, symbol: str, fold: int = 0) -> pl.DataFrame:
        """Load a previously cached amplitude table for *symbol*.

        If the file does not exist, returns an empty DataFrame with the
        expected schema.
        """
        path = self._path_for(symbol, fold)
        if os.path.exists(path):
            return pl.read_parquet(path)
        return pl.DataFrame(
            schema={
                "symbol": pl.Utf8,
                "state_ts": pl.Float64,
                "event_bucket": pl.Utf8,
                "event_impact": pl.Utf8,
                "currency_match": pl.Boolean,
                "event_name": pl.Utf8,
                "seconds_to_event": pl.Float64,
                "spread": pl.Float64,
                "horizon_sec": pl.Int32,
                "abs_move": pl.Float64,
                "signed_move": pl.Float64,
            }
        )

    def save(self, df: pl.DataFrame, symbol: str, fold: int = 0) -> None:
        """Persist *df* to the cache directory under *symbol*."""
        path = self._path_for(symbol, fold)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.write_parquet(path, compression="zstd")

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _path_for(self, symbol: str, fold: int = 0) -> str:
        return os.path.join(
            self._base_path, symbol, f"amplitude_fold_{fold}.parquet"
        )

    @staticmethod
    def _log_timing(message: str, elapsed: float) -> None:
        """Log a timing message using the module logger."""
        import logging

        logger = logging.getLogger("proxima.exogenous.event_amplitude_cache")
        logger.info(f"{message}  [{elapsed:.3f}s]")
