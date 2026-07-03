"""StateExtractor — extracts microstructure state streams from replay data.

Produces a Polars DataFrame with per-minute state features for each symbol
in the given date range, using OSS (Outcome Surface Signal), TransitionOSS,
and SAL (Signal Aggregation Layer).
"""
import sys; sys.path.insert(0, ".")
import math
import os
from typing import Optional

import polars as pl

from replay.environment import build_replay_environment, ReplayConfig
from replay.clock_patcher import patch_clock
from features.ecdf_transform import PerSymbolECDF
from signals.outcome_surface_signal import OutcomeSurfaceSignal
from signals.transition_oss import TransitionOSS
from signals.sal_mapper import SignalAggregationLayer
from research.replay_cache import ReplayCache
from evaluation.delayed_outcome_engine import DelayedOutcomeEngine
from research.cep.session_partition import SessionPartitioner


class StateExtractor:
    """Extracts microstructure state snapshots at minute boundaries from replay data.

    Parameters
    ----------
    symbols : list[str]
        List of symbol names to extract states for.
    start : str
        Start date in ``YYYY-MM-DD`` format.
    end : str
        End date in ``YYYY-MM-DD`` format.
    n_ticks : int, optional
        Maximum number of replay ticks to process per symbol (default 80 000).
    seed : int, optional
        Random seed for replay determinism (default 42).
    """

    def __init__(
        self,
        symbols: list[str],
        start: str,
        end: str,
        n_ticks: int = 80000,
        seed: int = 42,
    ):
        self.symbols = symbols
        self.start = start
        self.end = end
        self.n_ticks = n_ticks
        self.seed = seed
        self._oss: Optional[OutcomeSurfaceSignal] = None
        self._partitioner = SessionPartitioner()

    # ------------------------------------------------------------------
    # OSS training
    # ------------------------------------------------------------------

    def _train_oss(self, ev_threshold: float = 0.05) -> OutcomeSurfaceSignal:
        """Train an OutcomeSurfaceSignal from April 2026 data via ReplayCache."""
        cache = ReplayCache(
            symbols=self.symbols,
            start="2026-04-01",
            end="2026-04-30",
            tick_limit=self.n_ticks,
            seed=self.seed,
        )
        records = cache.compute()
        oss = OutcomeSurfaceSignal.from_pipeline_records(
            records, ev_threshold=ev_threshold
        )
        self._oss = oss
        return oss

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self) -> pl.DataFrame:
        """Extract state for all configured symbols and return concatenated DataFrame.

        Returns
        -------
        pl.DataFrame
            Columns: ts, symbol, oss_bucket, oss_ev, tross_cross, sal_score,
            entropy_decile, session_regime, price, oss_signal.
        """
        frames: list[pl.DataFrame] = []
        for sym in self.symbols:
            frames.append(self.extract_symbol(sym))
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames)

    def extract_symbol(self, symbol: str) -> pl.DataFrame:
        """Run state extraction for a single symbol.

        Parameters
        ----------
        symbol : str
            Symbol to extract (must be one of ``self.symbols``).

        Returns
        -------
        pl.DataFrame
            Per-minute state snapshots for *symbol*.
        """
        if self._oss is None:
            self._train_oss()

        # --- build replay environment ---------------------------------
        cfg = ReplayConfig(
            symbols=[symbol],
            start=self.start,
            end=self.end,
            speed=500_000,
            burst=True,
            latency=False,
            slippage=False,
            seed=self.seed,
        )
        env = build_replay_environment(cfg)
        patch_clock(env.clock)

        # --- streaming state objects ----------------------------------
        ecdf = PerSymbolECDF(window_size=2000)
        tross = TransitionOSS(self._oss, cross_threshold=2)
        sal = SignalAggregationLayer()

        # --- per-symbol mutable state ---------------------------------
        price_buf: list[float] = []
        prev_bucket: Optional[int] = None
        prev_minute: Optional[int] = None
        warmup = 5000
        tick_count = 0
        records: list[dict] = []

        # --- walk ticks -----------------------------------------------
        for tick in env.replay_feed.stream():
            sym: str = tick.get("symbol", "")
            if sym != symbol:
                continue

            tick_count += 1
            price = tick.get("ask", 0) or tick.get("price", 0) or 0.0
            ts = int(tick.get("time_sec", tick.get("timestamp", 0)))
            current_minute = ts // 60

            # 1. Update ECDF
            ecdf_rank = ecdf.update(sym, price)

            # 2. Price buffer for entropy
            price_buf.append(price)
            if len(price_buf) > 50:
                price_buf = price_buf[-50:]

            # 3. Current ECDF bucket
            bucket = min(int(ecdf_rank * 10), 9)

            # 4. TransitionOSS cross detection
            _ = tross.update(sym, ecdf_rank)

            # 5. Cross level since previous tick
            cross_level = 0
            if prev_bucket is not None:
                diff = abs(bucket - prev_bucket)
                if diff >= 2:
                    cross_level = 2
                elif diff >= 1:
                    cross_level = 1
            prev_bucket = bucket

            # 6. Raw OSS signal
            oss_signal = self._oss.predict(ecdf_rank)

            # 7. Update SAL rolling accumulator
            oss_info = self._oss.predict_with_info(ecdf_rank)
            sal.update(sym, oss_signal, oss_info.get("confidence", 1.0), price)

            # 8. Record snapshot at minute boundaries (after warmup)
            if (
                tick_count >= warmup
                and prev_minute is not None
                and current_minute != prev_minute
            ):
                entropy = self._entropy(price_buf)
                entropy_decile = min(int(entropy * 10), 9)
                session = self._partitioner.classify(ts)
                oss_ev = self._oss.bucket_ev(bucket)

                records.append(
                    {
                        "ts": prev_minute * 60,
                        "symbol": symbol,
                        "oss_bucket": bucket,
                        "oss_ev": oss_ev,
                        "tross_cross": cross_level,
                        "sal_score": sal.agg_score(),
                        "entropy_decile": entropy_decile,
                        "session_regime": session,
                        "price": price,
                        "oss_signal": oss_signal,
                    }
                )

            prev_minute = current_minute

            if tick_count >= self.n_ticks:
                break

        if not records:
            return pl.DataFrame(
                schema={
                    "ts": pl.Int64,
                    "symbol": pl.Utf8,
                    "oss_bucket": pl.Int32,
                    "oss_ev": pl.Float64,
                    "tross_cross": pl.Int32,
                    "sal_score": pl.Float64,
                    "entropy_decile": pl.Int32,
                    "session_regime": pl.Utf8,
                    "price": pl.Float64,
                    "oss_signal": pl.Int32,
                }
            )

        return pl.DataFrame(records)

    # ------------------------------------------------------------------
    # Entropy (directional-change entropy, 0–1)
    # ------------------------------------------------------------------

    @staticmethod
    def _entropy(prices: list[float]) -> float:
        """Compute directional-change entropy from a price buffer.

        Returns a value in [0, 1] where 1.0 means equal up/down moves
        and 0.0 means all moves are in the same direction.
        """
        if len(prices) < 3:
            return 0.5
        n = len(prices)
        diffs = [prices[i] - prices[i - 1] for i in range(1, n)]
        pos = sum(1 for d in diffs if d > 0)
        neg = sum(1 for d in diffs if d < 0)
        total = pos + neg
        if total == 0:
            return 0.5
        pp = pos / total
        pn = neg / total
        if pp <= 0 or pn <= 0:
            return 0.0
        return -(pp * math.log2(pp) + pn * math.log2(pn))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, df: pl.DataFrame, path: str = None) -> None:
        """Save extracted state DataFrame to Parquet.

        Parameters
        ----------
        df : pl.DataFrame
            Data to persist.
        path : str, optional
            File path.  When *None*, a name is auto-generated from the
            extractor's symbol list and date range.
        """
        if path is None:
            sym_str = "_".join(sorted(self.symbols))
            path = f"state_{sym_str}_{self.start}_{self.end}.parquet"
        df.write_parquet(path)
        print(f"StateExtractor: saved {len(df)} rows to {path}")
