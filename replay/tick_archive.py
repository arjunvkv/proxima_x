import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

logger = logging.getLogger("proxima.replay.tick_archive")

TICK_SCHEMA = pa.schema([
    pa.field("timestamp_ns", pa.int64()),
    pa.field("time_sec", pa.int64()),
    pa.field("time_msc", pa.int64()),
    pa.field("bid", pa.float64()),
    pa.field("ask", pa.float64()),
    pa.field("spread", pa.float64()),
    pa.field("last", pa.float64()),
    pa.field("volume", pa.float64()),
    pa.field("volume_real", pa.float64()),
    pa.field("flags", pa.int32()),
    pa.field("symbol", pa.utf8()),
    pa.field("point", pa.float64()),
    pa.field("digits", pa.int32()),
])

BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "ticks")


def _partition_path(symbol: str, year: int, month: int, day: int) -> str:
    return os.path.join(BASE_PATH, symbol, str(year), f"{month:02d}", f"{day:02d}.parquet")


class TickArchive:
    def __init__(self, base_path: str = None):
        self._base = base_path or BASE_PATH

    def _ensure_dir(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def store_ticks(self, symbol: str, ticks: list[dict]):
        if not ticks:
            return
        enriched = []
        for t in ticks:
            t["spread"] = t.get("spread", t.get("ask", 0) - t.get("bid", 0))
            # Backfill point/digits for legacy writers (default 5-digit);
            # real ingester rows carry broker truth (e.g. EURJPY point=0.001).
            t["point"] = t.get("point") or 1e-5
            t["digits"] = int(t.get("digits") or 5)
            enriched.append(t)
        # Add per-second sequence counter to preserve sub-second ticks
        from collections import Counter
        seq = Counter()
        for t in enriched:
            key = t.get("time_sec", 0)
            t["_seq"] = seq[key]
            seq[key] += 1
        by_day: dict[str, list[dict]] = {}
        for t in enriched:
            ts = t.get("time_sec", t.get("timestamp", 0))
            dt = datetime.fromtimestamp(ts)
            key = f"{dt.year}-{dt.month:02d}-{dt.day:02d}"
            if key not in by_day:
                by_day[key] = []
            by_day[key].append(t)
        for day_key, day_ticks in by_day.items():
            parts = day_key.split("-")
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            path = _partition_path(symbol, year, month, day)
            self._write_partition(path, day_ticks)

    def _write_partition(self, path: str, ticks: list[dict]):
        self._ensure_dir(path)
        try:
            existing = pl.read_parquet(path) if os.path.exists(path) else None
        except Exception:
            existing = None
        df = pl.from_dicts(ticks, schema={
            "timestamp_ns": pl.Int64, "time_sec": pl.Int64, "time_msc": pl.Int64,
            "bid": pl.Float64, "ask": pl.Float64, "spread": pl.Float64, "last": pl.Float64,
            "volume": pl.Float64, "volume_real": pl.Float64,
            "flags": pl.Int32, "symbol": pl.Utf8, "_seq": pl.Int64,
            "point": pl.Float64, "digits": pl.Int32,
        })
        if existing is not None:
            dedup_cols = ["timestamp_ns", "symbol"]
            if "_seq" in existing.columns:
                dedup_cols.append("_seq")
            df = pl.concat([existing, df]).unique(subset=dedup_cols, keep="first")
            df = df.sort(["timestamp_ns", "_seq"])
        table = df.to_arrow()
        pq.write_table(table, path, compression="zstd")

    def load_range(self, symbol: str, start: datetime, end: datetime) -> pl.LazyFrame:
        paths = []
        d = start
        while d <= end:
            path = _partition_path(symbol, d.year, d.month, d.day)
            if os.path.exists(path):
                paths.append(path)
            d += timedelta(days=1)
        if not paths:
            return pl.LazyFrame()
        df = pl.scan_parquet(paths)
        if "spread" not in df.collect_schema().names():
            df = df.with_columns((pl.col("ask") - pl.col("bid")).alias("spread"))
        return df

    def load_day(self, symbol: str, date: datetime) -> pl.LazyFrame:
        path = _partition_path(symbol, date.year, date.month, date.day)
        if os.path.exists(path):
            return pl.scan_parquet(path)
        return pl.LazyFrame()

    def load_random_window(self, symbol: str, days: int, seed: int = 42) -> pl.LazyFrame:
        rng = np.random.default_rng(seed)
        all_days = self._available_days(symbol)
        if len(all_days) < days:
            return pl.LazyFrame()
        start_idx = rng.integers(0, len(all_days) - days + 1)
        selected = all_days[start_idx:start_idx + days]
        return pl.concat([pl.scan_parquet(d) for d in selected if os.path.exists(d)])

    def load_random_session(self, symbol: str, session: str, seed: int = 42) -> pl.LazyFrame:
        SESSION_HOURS = {
            "ASIA": (0, 9), "LONDON": (8, 17), "NY": (13, 22),
            "OVERLAP": (13, 17), "DEAD": (22, 24),
        }
        hrs = SESSION_HOURS.get(session.upper(), (0, 24))
        all_days = self._available_days(symbol)
        if not all_days:
            return pl.LazyFrame()
        rng = np.random.default_rng(seed)
        path = rng.choice(all_days)
        df = pl.scan_parquet(path)
        return df.filter(
            (pl.col("time_sec") % 86400 // 3600 >= hrs[0]) &
            (pl.col("time_sec") % 86400 // 3600 < hrs[1])
        )

    def load_random_regime(self, symbol: str, regime: str, seed: int = 42) -> pl.LazyFrame:
        return pl.LazyFrame()

    def load_random_volatility(self, symbol: str, bucket: str, seed: int = 42) -> pl.LazyFrame:
        return pl.LazyFrame()

    def _available_days(self, symbol: str) -> list[str]:
        base = os.path.join(self._base, symbol)
        paths = []
        if not os.path.exists(base):
            return paths
        for year in sorted(os.listdir(base)):
            year_path = os.path.join(base, year)
            if not os.path.isdir(year_path):
                continue
            for month in sorted(os.listdir(year_path)):
                month_path = os.path.join(year_path, month)
                if not os.path.isdir(month_path):
                    continue
                for day in sorted(os.listdir(month_path)):
                    if day.endswith(".parquet"):
                        paths.append(os.path.join(month_path, day))
        return paths

    def get_date_range(self, symbol: str) -> tuple:
        all_days = self._available_days(symbol)
        if not all_days:
            return (None, None)
        dates = []
        for p in all_days:
            parts = p.replace("\\", "/").split("/")
            day_file = parts[-1].replace(".parquet", "")
            month = parts[-2]
            year = parts[-3]
            dates.append(datetime(int(year), int(month), int(day_file)))
        return (min(dates), max(dates))

    def ingest_parquet_files(self, source_dir: str, symbol: str):
        import glob as glob_mod
        seen = set()
        files = sorted(glob_mod.glob(os.path.join(source_dir, f"{symbol}_*.parquet")))
        for fpath in files:
            logger.info(f"Ingesting {fpath} into archive for {symbol}")
            df = pl.read_parquet(fpath)
            cols = df.columns
            ts_col = None
            for c in ["timestamp", "time", "timestamp_ns"]:
                if c in cols:
                    ts_col = c
                    break
            if ts_col:
                ts_sample = df.select(pl.col(ts_col)).head(1).item()
                if ts_sample > 1e17:
                    df = df.with_columns([
                        (pl.col(ts_col) // 1_000_000_000).alias("time_sec"),
                        (pl.col(ts_col) // 1_000_000).alias("time_msc"),
                        pl.col(ts_col).alias("timestamp_ns"),
                    ])
                elif ts_sample > 1e14:
                    df = df.with_columns([
                        (pl.col(ts_col) // 1_000_000).alias("time_sec"),
                        (pl.col(ts_col) // 1_000).alias("time_msc"),
                        (pl.col(ts_col) * 1_000).alias("timestamp_ns"),
                    ])
                elif ts_sample > 1e11:
                    df = df.with_columns([
                        (pl.col(ts_col) // 1_000).alias("time_sec"),
                        (pl.col(ts_col) // 1).alias("time_msc"),
                        (pl.col(ts_col) * 1_000_000).alias("timestamp_ns"),
                    ])
                else:
                    df = df.with_columns([
                        pl.col(ts_col).alias("time_sec"),
                        pl.col(ts_col).alias("time_msc"),
                        (pl.col(ts_col) * 1_000_000_000).alias("timestamp_ns"),
                    ])
            rename = {}
            if "timestamp" in cols and "timestamp_ns" not in df.columns:
                rename["timestamp"] = "timestamp_ns"
            if rename:
                df = df.rename(rename)
            missing = [f.name for f in TICK_SCHEMA if f.name not in df.columns]
            for m in missing:
                df = df.with_columns(pl.lit(0).alias(m))
            existing_cols = [c for c in [f.name for f in TICK_SCHEMA] if c in df.columns]
            ticks = df.select(existing_cols).to_dicts()
            self.store_ticks(symbol, ticks)
            logger.info(f"Ingested {len(ticks)} ticks from {fpath}")
