"""MarketSeedLoader — MT5 parquet bootstrap with date-verified freshness.

Loads latest M1 bars from MT5 at engine start to seed ECDF, entropy,
and spread baselines for all symbols (execution + shadow).
"""
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("proxima_ops.bootstrap")

MARKET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "market")
FRESHNESS_HOURS = 24
DEFAULT_BARS = 1440


def _ensure_market_dir():
    os.makedirs(MARKET_DIR, exist_ok=True)


class MarketSeedLoader:
    def __init__(self, mt5_connector):
        self.mt5 = mt5_connector
        _ensure_market_dir()

    def _parquet_path(self, symbol: str) -> str:
        return os.path.join(MARKET_DIR, f"{symbol}.parquet")

    def _is_fresh(self, symbol: str) -> bool:
        path = self._parquet_path(symbol)
        if not os.path.isfile(path):
            return False
        try:
            df = pd.read_parquet(path)
            if df.empty:
                return False
            last_ts = df["timestamp"].max()
            if hasattr(last_ts, "tz") and last_ts.tz is not None:
                last_ts = last_ts.tz_localize(None)
            age_hours = (datetime.utcnow() - pd.Timestamp(last_ts).to_pydatetime()).total_seconds() / 3600
            return age_hours < FRESHNESS_HOURS
        except Exception as e:
            logger.warning(f"Parquet check failed for {symbol}: {e}")
            return False

    def seed_symbol(self, symbol: str, bars: int = DEFAULT_BARS) -> dict:
        path = self._parquet_path(symbol)
        if self._is_fresh(symbol):
            df = pd.read_parquet(path)
            logger.info(f"[BOOTSTRAP] {symbol}: loaded fresh cache ({len(df)} bars, {os.path.getsize(path)} bytes)")
        else:
            logger.info(f"[BOOTSTRAP] {symbol}: fetching {bars} M1 bars from MT5...")
            rates = self.mt5.get_rates(symbol, count=bars, timeframe="M1")
            if rates is None or len(rates) == 0:
                logger.warning(f"[BOOTSTRAP] {symbol}: MT5 returned no data, trying existing cache")
                if os.path.isfile(path):
                    df = pd.read_parquet(path)
                else:
                    return {"symbol": symbol, "bars": 0, "entropy_seed": 0.5, "spread_seed": 0, "ecdf_seed": 0.5, "error": "no data"}
            else:
                df = pd.DataFrame(rates)
                df["timestamp"] = pd.to_datetime(df["time"], unit="s")
                df = df.sort_values("timestamp").reset_index(drop=True)
                df.to_parquet(path, index=False)
                logger.info(f"[BOOTSTRAP] {symbol}: saved {len(df)} bars -> {path}")

        return self._compute_seed(symbol, df)

    def _compute_seed(self, symbol: str, df: pd.DataFrame) -> dict:
        if df.empty:
            return {"symbol": symbol, "bars": 0, "entropy_seed": 0.5, "spread_seed": 0, "ecdf_seed": 0.5}

        closes = df["close"].values
        n = len(closes)

        # ECDF seed: last close rank in trailing distribution
        last_close = closes[-1]
        ecdf_seed = float(np.mean(closes <= last_close))

        # Entropy seed: approximate from log returns distribution
        log_returns = np.diff(np.log(closes[closes > 0]))
        if len(log_returns) > 1:
            hist, _ = np.histogram(log_returns, bins=20, density=True)
            hist = hist[hist > 0]
            entropy_seed = float(-np.sum(hist * np.log2(hist)) / np.log2(20))
        else:
            entropy_seed = 0.5

        # Spread seed: approximate from high-low range
        spreads = (df["high"].values - df["low"].values) / (closes + 1e-10)
        spread_p50 = float(np.median(spreads)) * 10000 if len(spreads) > 0 else 0
        spread_p95 = float(np.percentile(spreads, 95)) * 10000 if len(spreads) > 0 else 0

        return {
            "symbol": symbol,
            "bars": n,
            "closes": closes.tolist(),
            "entropy_seed": round(entropy_seed, 4),
            "spread_seed": round(spread_p50, 2),
            "spread_p95": round(spread_p95, 2),
            "ecdf_seed": round(ecdf_seed, 4),
            "last_close": float(closes[-1]),
            "last_timestamp": str(df["timestamp"].iloc[-1]),
        }

    def seed_all(self, symbols: list[str], bars: int = DEFAULT_BARS) -> dict[str, dict]:
        results = {}
        for sym in symbols:
            try:
                results[sym] = self.seed_symbol(sym, bars=bars)
            except Exception as e:
                logger.error(f"[BOOTSTRAP] {sym} failed: {e}")
                results[sym] = {"symbol": sym, "bars": 0, "entropy_seed": 0.5, "spread_seed": 0, "ecdf_seed": 0.5, "error": str(e)}
        fresh = sum(1 for r in results.values() if r.get("bars", 0) > 0)
        stale = len(symbols) - fresh
        logger.info(f"[BOOTSTRAP] seed_all complete: {fresh}/{len(symbols)} symbols seeded ({stale} stale/missing)")
        return results
