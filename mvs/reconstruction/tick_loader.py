from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

from proxima_ops.execution.mt5_connector import MT5Connector
from mvs.utils.time_sync import TimeSync


class TickLoader:
    __slots__ = (
        "symbol", "connector", "replay_root",
        "_prev_mid", "_prev_velocity", "_prev_acceleration",
        "_prev_ts_ns", "_tick_id",
    )

    def __init__(self, symbol: str, replay_root: str = "data/ticks") -> None:
        self.symbol = symbol
        self.connector = MT5Connector()
        self.replay_root = Path(replay_root)
        self._prev_mid: Optional[float] = None
        self._prev_velocity: float = 0.0
        self._prev_acceleration: float = 0.0
        self._prev_ts_ns: Optional[int] = None
        self._tick_id = 0

    def _load_replay_tick(self) -> Dict[str, Any]:
        raise RuntimeError("Replay fallback not implemented in live mode. Use explicit replay adapter.")

    def _fetch_raw(self) -> Dict[str, Any]:
        tick = self.connector.get_tick(self.symbol)
        if tick is None:
            return self._load_replay_tick()
        return tick

    def next(self) -> Dict[str, Any]:
        raw = self._fetch_raw()
        bid = float(raw["bid"])
        ask = float(raw["ask"])
        ts_ns = int(raw.get("time_msc", raw["time"] * 1_000_000)) * 1000
        mid = (bid + ask) * 0.5
        spread = ask - bid

        if self._prev_mid is None:
            delta = 0.0; velocity = 0.0; acceleration = 0.0; jerk = 0.0
        else:
            delta = mid - self._prev_mid
            dt_ns = max(ts_ns - self._prev_ts_ns, 1)
            dt_sec = dt_ns / 1_000_000_000.0
            velocity = delta / dt_sec
            acceleration = velocity - self._prev_velocity
            jerk = acceleration - self._prev_acceleration

        tick_data = {
            "tick_id": self._tick_id,
            "symbol": self.symbol,
            "ts_ns": ts_ns,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread": spread,
            "delta": delta,
            "velocity": velocity,
            "acceleration": acceleration,
            "jerk": jerk,
            "entropy": 0.0,
            "d_entropy": 0.0,
            "compression_ratio": 0.0,
            "burst_density": 0.0,
            "regime_hint": "",
            "liquidity_proxy": spread,
            "pressure_proxy": abs(delta),
            "vol_cluster": abs(acceleration),
        }

        self._prev_mid = mid
        self._prev_velocity = velocity
        self._prev_acceleration = acceleration
        self._prev_ts_ns = ts_ns
        self._tick_id += 1
        return tick_data
