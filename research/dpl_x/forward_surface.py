"""ForwardSurface — builds forward return surfaces from replay data.

For a given symbol and date range, this module replays historical ticks,
collects (timestamp, price) pairs, then computes forward returns at multiple
horizons (in bps).  The result is a Polars DataFrame suitable for downstream
cross-sectional or time-series analysis.

Usage::

    fs = ForwardSurface()
    df = fs.build("EURJPY", start="2025-01-01", end="2025-02-01")
    fs.save(df)
"""
import sys; sys.path.insert(0, ".")
import time

import polars as pl

from replay.environment import build_replay_environment, ReplayConfig
from replay.clock_patcher import patch_clock


class ForwardSurface:
    """Compute forward return surfaces for one or more symbols.

    Parameters
    ----------
    horizons : list[int], optional
        Forward-looking windows in seconds.  Defaults to
        ``[60, 300, 900, 3600]`` (1 min, 5 min, 15 min, 1 hour).
    """

    def __init__(self, horizons: list[int] = None):
        self.horizons = sorted(horizons) if horizons else [60, 300, 900, 3600]

    # ------------------------------------------------------------------
    # Build — single symbol
    # ------------------------------------------------------------------

    def build(self, symbol: str, start: str, end: str,
              n_ticks: int = 80000) -> pl.DataFrame:
        """Replay ticks for *symbol* and compute forward returns.

        Parameters
        ----------
        symbol : str
            Instrument symbol (e.g. ``"EURJPY"``).
        start : str
            Start date in ``YYYY-MM-DD`` format.
        end : str
            End date in ``YYYY-MM-DD`` format.
        n_ticks : int, optional
            Maximum number of ticks to process (default 80 000).

        Returns
        -------
        pl.DataFrame
            Columns: ``ts``, ``symbol``, ``price``, ``forward_60``,
            ``forward_300``, ``forward_900``, ``forward_3600``
            (the last four in bps).
        """
        t0 = time.perf_counter()
        print(f"[ForwardSurface] Building surface for {symbol} "
              f"{start}->{end} (max {n_ticks:,} ticks) ...")

        # --- setup replay environment ----------------------------------
        cfg = ReplayConfig(
            symbols=[symbol],
            start=start,
            end=end,
            speed=500_000,
            burst=True,
            latency=False,
            slippage=False,
        )
        env = build_replay_environment(cfg)
        patch_clock(env.clock)

        # --- walk ticks, recording (ts, price) -------------------------
        tick_data: list[tuple[int, float]] = []
        tick_count = 0

        for tick in env.replay_feed.stream():
            sym: str = tick.get("symbol", "")
            if sym != symbol:
                continue

            tick_count += 1
            price = tick.get("ask", 0) or tick.get("price", 0) or 0.0
            ts = int(tick.get("time_sec", tick.get("timestamp", 0)))
            tick_data.append((ts, price))

            if tick_count >= n_ticks:
                break

        n = len(tick_data)
        if n == 0:
            print(f"[ForwardSurface] No ticks loaded for {symbol}")
            return self._empty_df(symbol)

        t1 = time.perf_counter()
        print(f"[ForwardSurface] Collected {n:,} ticks in {t1 - t0:.2f}s — "
              f"computing forward returns ...")

        # --- compute forward returns (sample every 60th tick) ----------
        # Maintain a pointer per horizon so the overall scan stays O(N).
        pointers: dict[int, int] = {h: 0 for h in self.horizons}
        records: list[dict] = []

        for i in range(0, n, 60):
            ts_i, price_i = tick_data[i]
            row: dict = {
                "ts": ts_i,
                "symbol": symbol,
                "price": price_i,
            }

            for h in self.horizons:
                target_ts = ts_i + h
                ptr = pointers[h]
                # advance pointer to first tick at or after target_ts
                while ptr < n and tick_data[ptr][0] < target_ts:
                    ptr += 1
                pointers[h] = ptr

                future_price = tick_data[ptr][1] if ptr < n else price_i

                # forward return in basis points (1 bps = 0.01%)
                if price_i != 0.0:
                    fwd_ret = (future_price - price_i) / price_i * 10000.0
                else:
                    fwd_ret = 0.0

                row[f"forward_{h}"] = fwd_ret

            records.append(row)

        t2 = time.perf_counter()
        print(f"[ForwardSurface] Computed {len(records):,} rows "
              f"in {t2 - t1:.2f}s  (total {t2 - t0:.2f}s)")

        if not records:
            return self._empty_df(symbol)

        return pl.DataFrame(records)

    # ------------------------------------------------------------------
    # Build — multiple symbols
    # ------------------------------------------------------------------

    def build_all(self, symbols: list[str], **kwargs) -> pl.DataFrame:
        """Build forward surfaces for all *symbols* and concatenate.

        Parameters
        ----------
        symbols : list[str]
            One or more instrument symbols.
        **kwargs
            Forwarded to :meth:`build` (e.g. ``start``, ``end``,
            ``n_ticks``).

        Returns
        -------
        pl.DataFrame
            Concatenated result for all symbols.
        """
        frames: list[pl.DataFrame] = []
        for sym in symbols:
            frames.append(self.build(sym, **kwargs))
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, df: pl.DataFrame, path: str = None) -> None:
        """Save forward-surface DataFrame to Parquet.

        Parameters
        ----------
        df : pl.DataFrame
            Data to persist.
        path : str, optional
            File path.  When *None*, a default name is used.
        """
        if path is None:
            path = "forward_surface.parquet"
        df.write_parquet(path)
        print(f"[ForwardSurface] Saved {len(df):,} rows -> {path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_df(symbol: str) -> pl.DataFrame:
        """Return a zero-row DataFrame with the correct schema."""
        cols: dict[str, type] = {
            "ts": pl.Int64,
            "symbol": pl.Utf8,
            "price": pl.Float64,
        }
        # Order matters — keep horizons sorted for reproducible columns
        for h in [60, 300, 900, 3600]:
            cols[f"forward_{h}"] = pl.Float64
        return pl.DataFrame(schema=cols)


# ------------------------------------------------------------------
# Quick smoke test when run directly
# ------------------------------------------------------------------
if __name__ == "__main__":
    fs = ForwardSurface()
    df = fs.build("EURJPY", start="2025-01-01", end="2025-01-02", n_ticks=5000)
    print(df)
    print(f"\nShape: {df.shape}")
