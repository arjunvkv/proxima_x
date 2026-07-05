"""
DPL-19: Tick Thermodynamics Engine.

Extracts hidden directional information from tick-time structure:
  - Tick-arrival acceleration (tempo deformation before moves)
  - Burst clustering (density anomalies)
  - Directional density asymmetry (uptick vs downtick pressure)
  - Composite pressure score

Operates on rolling tick windows from parquet or live buffer.
"""
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

PRESSURE_WINDOW = 500
BURST_BASELINE_WINDOW = 2000
MIN_TICKS = 100


class TickThermodynamicsEngine:
    def __init__(self, synthetic_symbols: Optional[set] = None):
        self._tick_data: Dict[str, dict] = {}
        self._pressure_history: Dict[str, list] = defaultdict(list)
        self._burst_history: Dict[str, list] = defaultdict(list)
        self._asymmetry_history: Dict[str, list] = defaultdict(list)
        self._accel_history: Dict[str, list] = defaultdict(list)
        self._max_history = 500
        # P2.2: Synthetic thermodynamics for symbols without tick data
        self._synthetic_symbols = synthetic_symbols or {"XAUUSD"}
        self._bar_buffer: Dict[str, list] = defaultdict(list)  # symbol -> [{"o":, "h":, "l":, "c":}]

    def load_offline(self, symbol: str) -> bool:
        """Load tick data from parquet for a symbol. Normalizes timestamps to seconds."""
        try:
            import polars as pl
            path = f"C:/Trading/Agentic_Trading/data/ticks/{symbol}_ticks.parquet"
            df = pl.read_parquet(path)
            arr = df.to_numpy()
            bid = arr[:, 1].astype(np.float64)
            ask = arr[:, 2].astype(np.float64)
            ts = arr[:, 0].astype(np.float64)
            # Normalize to microseconds: parquet may store ns, us, or ms
            if ts.max() > 1e18:
                ts = ts / 1_000.0           # ns → us
            elif ts.max() > 1e15:
                pass                        # already us
            elif ts.max() > 1e12:
                ts = ts * 1_000.0           # ms → us
            else:
                ts = ts * 1_000_000.0       # s → us
            # Invariant: all timestamps must be in microseconds
            if ts.max() < 1e14 or ts.max() > 1e17:
                raise ValueError(f"Invalid timestamp domain for {symbol}: max={ts.max()} (expected us)")
            mid = (bid + ask) / 2.0
            self._tick_data[symbol] = {"bid": bid, "ask": ask, "ts": ts, "mid": mid}
            self._offline_timestamps_normalized = True
            return True
        except Exception as e:
            self._offline_timestamps_normalized = False
            return False

    def feed_ticks(self, symbol: str, bid: float, ask: float, timestamp: float) -> None:
        """Feed a single tick into the engine (live mode)."""
        if symbol not in self._tick_data:
            self._tick_data[symbol] = {"bid": [], "ask": [], "ts": [], "mid": []}
        d = self._tick_data[symbol]
        # Convert numpy arrays to lists on first live feed (offline preload uses arrays)
        for k in ("bid", "ask", "ts", "mid"):
            if isinstance(d.get(k), np.ndarray):
                d[k] = d[k].tolist()
        d["bid"].append(bid)
        d["ask"].append(ask)
        d["ts"].append(timestamp * 1_000_000.0)  # seconds → microseconds
        d["mid"].append((bid + ask) / 2.0)
        excess = len(d["bid"]) - PRESSURE_WINDOW * 10
        if excess > 0:
            for k in ("bid", "ask", "ts", "mid"):
                d[k] = d[k][excess:]

    def feed_bar(self, symbol: str, o: float, h: float, l: float, c: float) -> None:
        """Feed a 1s bar into the synthetic buffer for thermodynamics proxy computation."""
        buf = self._bar_buffer[symbol]
        buf.append({"o": o, "h": h, "l": l, "c": c})
        if len(buf) > 200:
            buf.pop(0)

    def _compute_synthetic_pressure(self, symbol: str) -> dict:
        """Compute thermodynamics proxies from 1s bar data (no ticks available)."""
        bars = self._bar_buffer.get(symbol, [])
        if len(bars) < 10:
            return {"state": "INSUFFICIENT_DATA", "n_bars": len(bars)}

        arr = np.array([(b["o"], b["h"], b["l"], b["c"]) for b in bars])
        opens, highs, lows, closes = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

        # 1. Range compression
        ranges = highs - lows
        compression = np.std(ranges) / max(np.mean(ranges), 1e-10)
        compression = min(compression, 1.0)

        # 2. Directional asymmetry
        deltas = closes - opens
        asym = (np.sum(deltas > 0) - np.sum(deltas < 0)) / max(len(deltas), 1)

        # 3. Spread elasticity proxy: bar range / close
        spread_ratios = ranges / np.maximum(closes, 1e-10)
        mean_range = np.mean(spread_ratios)
        recent_range = np.mean(spread_ratios[-20:]) if len(spread_ratios) >= 20 else mean_range
        spread_elasticity = recent_range / max(mean_range, 1e-10)

        # 4. Burst density: large bar moves
        abs_delta = np.abs(deltas)
        threshold = np.percentile(abs_delta, 80) if len(abs_delta) >= 5 else 0.0
        burst = np.sum(abs_delta > threshold) / max(len(abs_delta), 1)

        # Composite synthetic pressure
        pressure = (
            0.35 * (1.0 - min(compression, 1.0))
            + 0.25 * min(abs(asym), 1.0)
            + 0.20 * burst
            + 0.20 * min(spread_elasticity, 2.0) / 2.0
        )
        pressure = min(max(pressure, 0.0), 1.0)
        pressure_direction = 1 if asym > 0.05 else (-1 if asym < -0.05 else 0)

        result = {
            "state": "SYNTHETIC",
            "n_bars": len(bars),
            "pressure": round(pressure, 4),
            "pressure_direction": pressure_direction,
            "tempo_compression": round(compression, 4),
            "burst_ratio": round(burst, 3),
            "asymmetry": round(asym, 4),
            "uptick_pct": round(np.sum(deltas > 0) / max(len(deltas), 1) * 100, 1),
            "downtick_pct": round(np.sum(deltas < 0) / max(len(deltas), 1) * 100, 1),
            "mean_accel": 0.0,
            "median_dt_us": 0.0,
        }
        self._pressure_history[symbol].append(pressure)
        self._burst_history[symbol].append(burst)
        self._asymmetry_history[symbol].append(asym)
        self._accel_history[symbol].append(0.0)
        for lst in [self._pressure_history, self._burst_history,
                     self._asymmetry_history, self._accel_history]:
            if len(lst[symbol]) > self._max_history:
                lst[symbol].pop(0)
        return result

    def compute_pressure(self, symbol: str) -> dict:
        """Compute full thermodynamics snapshot for a symbol.
        Routes to synthetic mode if symbol has no tick data."""
        if symbol in self._synthetic_symbols:
            return self._compute_synthetic_pressure(symbol)
        data = self._tick_data.get(symbol)
        if data is None:
            return {"state": "NO_DATA"}
        bid = data.get("bid", [])
        ts = data.get("ts", [])
        if len(bid) < MIN_TICKS:
            return {"state": "INSUFFICIENT_DATA", "n_ticks": len(bid)}

        bid_arr = np.array(bid[-PRESSURE_WINDOW:], dtype=np.float64)
        ts_arr = np.array(ts[-PRESSURE_WINDOW:], dtype=np.float64)

        # Tick-arrival acceleration
        dt = np.diff(ts_arr)
        dt = np.maximum(dt, 1)  # avoid zeros
        accel = np.diff(dt)
        median_dt = float(np.median(dt))
        mean_dt = float(np.mean(dt))
        recent_dt_mean = float(np.mean(dt[-50:])) if len(dt) >= 50 else mean_dt
        tempo_compression = max(0.0, 1.0 - recent_dt_mean / max(mean_dt, 1))
        mean_accel = float(np.mean(accel)) if len(accel) > 0 else 0.0

        # Burst clustering
        baseline_start = max(0, len(ts) - BURST_BASELINE_WINDOW)
        baseline_ticks = ts[baseline_start:]
        if len(baseline_ticks) >= 100:
            baseline_duration = float(baseline_ticks[-1] - baseline_ticks[0]) / 1e6
            baseline_rate = len(baseline_ticks) / max(baseline_duration, 1)
        else:
            baseline_rate = 1.0
        recent_ticks = ts[-100:]
        recent_duration = float(recent_ticks[-1] - recent_ticks[0]) / 1e6
        recent_rate = len(recent_ticks) / max(recent_duration, 1)
        burst_ratio = recent_rate / max(baseline_rate, 0.01)

        # Directional density asymmetry
        bid_changes = np.diff(bid_arr)
        upticks = np.sum(bid_changes > 1e-8)
        downticks = np.sum(bid_changes < -1e-8)
        total_directional = upticks + downticks
        if total_directional > 0:
            asymmetry = (upticks - downticks) / total_directional
        else:
            asymmetry = 0.0
        uptick_density = upticks / max(len(bid_changes), 1)
        downtick_density = downticks / max(len(bid_changes), 1)

        # Composite pressure score
        pressure = (
            0.35 * tempo_compression
            + 0.25 * min(burst_ratio / 3.0, 1.0)
            + 0.25 * abs(asymmetry)
            + 0.15 * min(abs(mean_accel) / 1000.0, 1.0)
        )
        pressure = min(pressure, 1.0)

        # Directional bias from asymmetry
        pressure_direction = 1 if asymmetry > 0.05 else (-1 if asymmetry < -0.05 else 0)

        result = {
            "state": "ACTIVE",
            "n_ticks": len(bid),
            "pressure": round(pressure, 4),
            "pressure_direction": pressure_direction,
            "tempo_compression": round(tempo_compression, 4),
            "burst_ratio": round(burst_ratio, 3),
            "asymmetry": round(asymmetry, 4),
            "uptick_pct": round(uptick_density * 100, 1),
            "downtick_pct": round(downtick_density * 100, 1),
            "mean_accel": round(mean_accel, 2),
            "median_dt_us": round(median_dt, 1),
        }

        self._pressure_history[symbol].append(pressure)
        self._burst_history[symbol].append(burst_ratio)
        self._asymmetry_history[symbol].append(asymmetry)
        self._accel_history[symbol].append(mean_accel)
        for lst in [self._pressure_history, self._burst_history,
                     self._asymmetry_history, self._accel_history]:
            if len(lst[symbol]) > self._max_history:
                lst[symbol].pop(0)

        return result

    def compute_lift(self, symbol: str, tpi_signals: list) -> dict:
        compare_total = len(tpi_signals)
        if compare_total < 10:
            return {"lift_pct": None, "n_samples": compare_total}
        tpi_correct = sum(1 for s in tpi_signals if s.get("tpi_direction") == s.get("actual_direction"))
        tpi_accuracy = tpi_correct / max(compare_total, 1)
        pressure_high = [s for s in tpi_signals if s.get("pressure", 0) > 0.5]
        if len(pressure_high) >= 5:
            p_correct = sum(1 for s in pressure_high if s.get("tpi_direction") == s.get("actual_direction"))
            p_accuracy = p_correct / max(len(pressure_high), 1)
            lift = p_accuracy - tpi_accuracy
        else:
            lift = None
        return {
            "tpi_accuracy": round(tpi_accuracy, 3),
            "high_pressure_accuracy": round(p_accuracy, 3) if lift is not None else None,
            "lift_pct": round(lift * 100, 1) if lift is not None else None,
            "n_samples": compare_total,
            "n_high_pressure": len(pressure_high) if lift is not None else 0,
        }

    def summary(self, symbols: List[str]) -> str:
        lines = []
        lines.append("  DPL-19: TICK THERMODYNAMICS")
        lines.append("-" * 52)
        lines.append(f"  {'Symbol':<8s} {'Pres':<7s} {'Dir':<5s} {'Tempo':<7s} {'Burst':<7s} {'Asym':<7s} {'Up%':<6s} {'Dn%':<6s} {'Accel':<7s}")
        for sym in symbols:
            r = self.compute_pressure(sym)
            if r.get("state") not in ("ACTIVE", "SYNTHETIC"):
                lines.append(f"  {sym:<8s} {r.get('state', '?'):<20s}")
                continue
            p = f"{r['pressure']:.3f}"
            d = f"{'LONG' if r['pressure_direction']==1 else 'SHORT' if r['pressure_direction']==-1 else 'FLAT':>4s}"
            tc = f"{r['tempo_compression']:.3f}"
            br = f"{r['burst_ratio']:.2f}x"
            asym = f"{r['asymmetry']:+.3f}"
            up = f"{r['uptick_pct']:.1f}%"
            dn = f"{r['downtick_pct']:.1f}%"
            acc = f"{r['mean_accel']:.1f}"
            lines.append(f"  {sym:<8s} {p:<7s} {d:<5s} {tc:<7s} {br:<7s} {asym:<7s} {up:<6s} {dn:<6s} {acc:<7s}")
        lines.append("")
        lines.append("  Key: Pres=Pressure(0-1), Tempo=Compression, Burst=density ratio, Asym=uptick-downtick bias")
        return "\n".join(lines)

    def reset(self, symbol=None):
        if symbol is not None:
            self._tick_data.pop(symbol, None)
            self._pressure_history.pop(symbol, None)
            self._burst_history.pop(symbol, None)
            self._asymmetry_history.pop(symbol, None)
            self._accel_history.pop(symbol, None)
            self._bar_buffer.pop(symbol, None)
        else:
            self._tick_data.clear()
            self._pressure_history.clear()
            self._burst_history.clear()
            self._asymmetry_history.clear()
            self._accel_history.clear()
            self._bar_buffer.clear()
