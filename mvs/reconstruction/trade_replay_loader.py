from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import polars as pl

from mvs.reconstruction.state_rebuilder import StateRebuilder
from mvs.analysis.latency_profiler import LatencyProfiler


class TradeReplayLoader:
    __slots__ = (
        "signal_path", "_signals", "profiler",
    )

    def __init__(
        self,
        signal_path: str = "proxima_ops/data/signals_export.csv",
    ) -> None:
        self.signal_path = Path(signal_path)
        self._signals: Optional[pl.DataFrame] = None
        self.profiler = LatencyProfiler()

    def load_signals(self) -> pl.DataFrame:
        if self._signals is not None:
            return self._signals
        suf = self.signal_path.suffix.lower()
        if suf == ".csv":
            df = pl.read_csv(str(self.signal_path), try_parse_dates=True)
        elif suf == ".parquet":
            df = pl.read_parquet(str(self.signal_path))
        else:
            raise ValueError(f"Unsupported signal path format: {suf}")
        for src, dst in [
            ("timestamp_generated", "ts_gen"),
            ("timestamp_opened", "ts_open"),
            ("timestamp_closed", "ts_close"),
        ]:
            if src in df.columns and df[src].dtype == pl.Utf8:
                df = df.with_columns(pl.col(src).str.strptime(pl.Datetime, strict=False).alias(dst))
            elif src in df.columns:
                df = df.with_columns(pl.col(src).alias(dst))
        self._signals = df
        return df

    def get_closed_trades(self, symbol: Optional[str] = None) -> pl.DataFrame:
        df = self.load_signals()
        closed = df.filter(
            pl.col("final_state") == "POSITION_CLOSED",
            pl.col("pnl_points") != 0.0,
        )
        if symbol:
            closed = closed.filter(pl.col("symbol") == symbol)
        return closed

    def load_tick_path(
        self, symbol: str, entry_us: int, exit_us: int,
        buffer_ticks: int = 300,
    ) -> np.ndarray:
        import MetaTrader5 as mt5

        if not mt5.initialize():
            return np.array([])

        from_ts = int(entry_us / 1_000_000) - buffer_ticks
        to_ts = int(exit_us / 1_000_000) + buffer_ticks

        ticks = mt5.copy_ticks_range(symbol, from_ts, to_ts, mt5.COPY_TICKS_ALL)
        mt5.shutdown()

        if ticks is None or len(ticks) == 0:
            return np.array([])

        return np.array(ticks, copy=False)

    def replay_trade(
        self, trade: Dict, tick_path: np.ndarray,
    ) -> List[Dict]:
        symbol = trade["symbol"]
        rebuilder = StateRebuilder(symbol)
        timeline = []
        prev_mid = None

        for i in range(len(tick_path)):
            t = tick_path[i]
            if "time_msc" in t.dtype.names:
                ts_us = int(t["time_msc"]) * 1000
            else:
                ts_us = int(t["time"]) * 1_000_000
            bid = float(t["bid"])
            ask = float(t["ask"])
            mid = (bid + ask) * 0.5

            if prev_mid is None:
                delta = 0.0
            else:
                delta = mid - prev_mid
            prev_mid = mid

            tick_data = {
                "mid": mid,
                "bid": bid,
                "ask": ask,
                "spread": ask - bid,
                "delta": delta,
                "velocity": 0.0,
                "acceleration": 0.0,
                "jerk": 0.0,
            }

            open_trades = []
            if trade.get("timestamp_opened"):
                open_us = int(trade["timestamp_opened"].timestamp() * 1_000_000)
                if ts_us >= open_us:
                    open_trades = [{
                        "ticket": trade.get("mt5_ticket", 0),
                        "symbol": symbol,
                        "entry_ts_ns": open_us,
                        "volume": 0.1,
                    }]

            state = rebuilder.on_tick(i, symbol, ts_us, tick_data, open_trades)
            state["tick_idx"] = i
            state["mid"] = mid
            state["ts_us"] = ts_us
            timeline.append(state)

        return timeline

    def analyze_trade_truth(
        self, trade: Dict, timeline: List[Dict],
    ) -> Dict:
        entry_us = int(trade.get("timestamp_opened", datetime.now()).timestamp() * 1_000_000) if isinstance(trade.get("timestamp_opened"), datetime) else 0
        exit_us = int(trade.get("timestamp_closed", datetime.now()).timestamp() * 1_000_000) if isinstance(trade.get("timestamp_closed"), datetime) else 0
        direction = 1 if trade.get("pnl_points", 0) > 0 else -1

        latency = self.profiler.analyze(timeline, entry_us, exit_us, direction)

        result = {
            "trade_id": trade.get("signal_id", ""),
            "symbol": trade.get("symbol", ""),
            "pnl_points": trade.get("pnl_points", 0.0),
            "pnl_money": trade.get("pnl_money", 0.0),
            "total_ticks": len(timeline),
            **latency,
        }

        return result

    def run(self, symbol: Optional[str] = None) -> List[Dict]:
        trades = self.get_closed_trades(symbol)
        results = []

        for trade in trades.rows(named=True):
            ts_open = trade.get("ts_open")
            ts_close = trade.get("ts_close")
            if ts_open is None or ts_close is None:
                continue

            entry_us = int(ts_open.timestamp() * 1_000_000)
            exit_us = int(ts_close.timestamp() * 1_000_000)

            tick_path = self.load_tick_path(trade["symbol"], entry_us, exit_us)
            if len(tick_path) == 0:
                continue

            timeline = self.replay_trade(trade, tick_path)
            analysis = self.analyze_trade_truth(trade, timeline)
            results.append(analysis)

        return results
