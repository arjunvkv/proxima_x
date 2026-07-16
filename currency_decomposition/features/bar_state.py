"""5-min bar trend from real MT5 M5 bars + real-time forming bar via M1 stream."""
import time
import numpy as np
from collections import deque
from config.settings import CURRENCY_LIST, BASE_CURRENCY_MAP, SYMBOLS
from currency.wls_solver import WLSSolver


_EPS = 1e-12
_M5 = 5


class BarStateEngine:
    def __init__(self, mt5, store):
        self.mt5 = mt5
        self.store = store
        self.solver = WLSSolver()
        # Completed M5 bar OHLC (from MT5 preload) — (open, high, low, close)
        self._bar_ohlc: dict[str, deque] = {
            s: deque(maxlen=30) for s in SYMBOLS
        }
        self._last_completed_ts: dict[str, float] = {}
        # Current forming M5 bar — updated via M1 stream
        self._forming_m5_start: dict[str, float] = {}
        self._forming_open: dict[str, float] = {}
        self._forming_close: dict[str, float] = {}
        # WLS strength trajectory (one per completed M5 bar)
        self._strength_history: dict[str, deque] = {
            c: deque(maxlen=5) for c in CURRENCY_LIST
        }
        self._preloaded = False
        self._ready = False
        self._state: dict[str, dict] = {}
        self._processed_bars: int = 0

    def _preload(self) -> None:
        for symbol in SYMBOLS:
            try:
                rates = self.mt5.get_rates_from(symbol, _M5, 1, 30)
                if rates is not None:
                    for r in rates:
                        self._bar_ohlc[symbol].append((
                            float(r[1]), float(r[2]), float(r[3]), float(r[4])
                        ))
                    latest_ts = float(rates[-1][0])
                    self._last_completed_ts[symbol] = latest_ts
                    self._forming_m5_start[symbol] = latest_ts + 300
                    self._forming_open[symbol] = float(rates[-1][1])
                    self._forming_close[symbol] = float(rates[-1][4])
            except Exception:
                pass
        self._preloaded = True
        populated = [v for v in self._bar_ohlc.values() if len(v) > 0]
        if len(populated) >= 6 and min(len(v) for v in populated) >= 2:
            self._compute_from_cache()

    def _handle_m1_stream(self) -> bool:
        """Check M1 bars for M5 close detection + forming bar update. Returns True if a new M5 bar closed."""
        new_bar_closed = False
        for symbol in SYMBOLS:
            m1_bars = list(self.store._bars.get(symbol, []))
            if not m1_bars:
                continue
            latest = m1_bars[-1]
            ts = latest.timestamp
            m1_mid = latest.mid
            m5_start = self._forming_m5_start.get(symbol, 0.0)
            new_m5_start = (int(ts) // 300) * 300.0
            if new_m5_start > m5_start:
                new_bar_closed = True
                self._forming_m5_start[symbol] = new_m5_start
                self._forming_open[symbol] = m1_mid
            self._forming_close[symbol] = m1_mid
        return new_bar_closed

    def _compute_from_cache(self) -> None:
        populated = [v for v in self._bar_ohlc.values() if len(v) > 0]
        if len(populated) < 2:
            return
        min_bars = min(len(v) for v in populated)
        if min_bars < 2:
            return
        symbols_with_data = [s for s in SYMBOLS if len(self._bar_ohlc[s]) > 0]
        start = max(self._processed_bars, 0)
        for i in range(start, min_bars - 1):
            returns = {}
            for symbol in symbols_with_data:
                ohlcs = list(self._bar_ohlc[symbol])
                prev = ohlcs[i][3]
                curr = ohlcs[i + 1][3]
                if prev > 0 and curr > 0:
                    returns[symbol] = float(np.log(curr / prev))
                else:
                    returns[symbol] = 0.0
            active = sum(1 for v in returns.values() if abs(v) > _EPS)
            if active < 10:
                continue
            strengths = self.solver.solve(returns, lam=0.05)
            for ccy in CURRENCY_LIST:
                self._strength_history[ccy].append(strengths.get(ccy, 0.0))
        self._processed_bars = max(min_bars - 1, 0)
        min_samples = min(len(h) for h in self._strength_history.values())
        self._ready = min_samples >= 3
        if self._ready:
            self._compute_state()

    def update(self) -> bool:
        if not self._preloaded:
            self._preload()
            return self._ready
        if self._handle_m1_stream():
            self._compute_from_cache()
        return self._ready

    def forming_return(self, symbol: str) -> float:
        """Return of the current forming M5 bar vs last completed M5 bar close."""
        if symbol not in SYMBOLS:
            return 0.0
        ohlcs = list(self._bar_ohlc.get(symbol, []))
        if len(ohlcs) < 1:
            return 0.0
        last_completed = ohlcs[-1][3]
        current = self._forming_close.get(symbol, last_completed)
        if last_completed > 0 and current > 0:
            return float(np.log(current / last_completed))
        return 0.0

    def forming_return_from_open(self, symbol: str) -> float | None:
        """Return of the current forming bar relative to its M5 open (None if not tracked)."""
        if symbol not in SYMBOLS:
            return None
        if symbol not in self._forming_open:
            ohlcs = list(self._bar_ohlc.get(symbol, []))
            if len(ohlcs) < 1:
                return None
            self._forming_open[symbol] = ohlcs[-1][3]
        open_ = self._forming_open.get(symbol)
        current = self._forming_close.get(symbol, open_)
        if open_ and open_ > 0 and current > 0:
            return float(np.log(current / open_))
        return None

    def forming_price_displacement(self, symbol: str) -> float | None:
        """Price displacement of the forming bar in price units (not log)."""
        if symbol not in SYMBOLS:
            return None
        if symbol not in self._forming_open:
            ohlcs = list(self._bar_ohlc.get(symbol, []))
            if len(ohlcs) < 1:
                return None
            self._forming_open[symbol] = ohlcs[-1][3]
        open_ = self._forming_open.get(symbol)
        current = self._forming_close.get(symbol, open_)
        if open_ and current:
            return float(current - open_)
        return None

    def get_swing_stats(self, symbol: str, lookback: int = 10) -> dict | None:
        """Return avg downside/upside/range from completed M5 bars in price units."""
        if symbol not in SYMBOLS:
            return None
        ohlcs = list(self._bar_ohlc.get(symbol, []))
        if len(ohlcs) < lookback:
            return None
        ohlcs = ohlcs[-lookback:]
        downsides = []
        upsides = []
        for o, h, l, c in ohlcs:
            if o > 0:
                downsides.append(l - o)
                upsides.append(h - o)
        if not downsides:
            return None
        avg_dn = sum(downsides) / len(downsides)
        avg_up = sum(upsides) / len(upsides)
        return {
            "avg_downside": avg_dn,
            "avg_upside": avg_up,
            "avg_range": avg_up - avg_dn,
            "samples": len(downsides),
        }

    def get_structural_swing_position(self, symbol: str, current_price: float, lookback: int = 20) -> dict | None:
        """Multi-bar structural location (SSP) and volatility context."""
        if symbol not in SYMBOLS:
            return None
        ohlcs = list(self._bar_ohlc.get(symbol, []))
        if len(ohlcs) < lookback:
            return None
        bars = ohlcs[-lookback:]
        lows = [b[2] for b in bars]
        highs = [b[1] for b in bars]
        swing_low = min(lows)
        swing_high = max(highs)
        width = swing_high - swing_low
        if width <= 0:
            return None
        buy_ssp = (current_price - swing_low) / width
        sell_ssp = (swing_high - current_price) / width
        bar_ranges = [b[1] - b[2] for b in bars]
        median_range = sorted(bar_ranges)[len(bar_ranges) // 2] if bar_ranges else width
        range_expansion = width / median_range if median_range > 0 else 1.0
        current_bar_range = None
        if symbol in self._forming_open and symbol in self._forming_close:
            current_bar_range = abs(self._forming_close[symbol] - self._forming_open[symbol])
        median_vol = sorted(bar_ranges)[len(bar_ranges) // 2] if bar_ranges else None
        vol_expansion = 1.0
        if current_bar_range is not None and median_vol and median_vol > 0:
            bar_start = self._forming_m5_start.get(symbol)
            bar_age = max(0, time.time() - bar_start) if bar_start else 0
            age_confidence = min(bar_age / 60.0, 1.0)
            if age_confidence >= 0.5:
                vol_expansion = current_bar_range / median_vol
        return {
            "buy_ssp": buy_ssp,
            "sell_ssp": sell_ssp,
            "swing_low": swing_low,
            "swing_high": swing_high,
            "range_price": width,
            "range_expansion": range_expansion,
            "vol_expansion": vol_expansion,
        }

    def get_micro_swing_positions(self, symbol: str, avg_up: float, avg_dn: float) -> dict | None:
        """Micro swing position (MSP) for both directions. Uses price displacement (not log)."""
        if symbol not in SYMBOLS:
            return None
        if symbol not in self._forming_open or symbol not in self._forming_close:
            return None
        open_ = self._forming_open[symbol]
        current = self._forming_close[symbol]
        displacement = current - open_
        buy_msp = displacement / avg_up if avg_up > 0 else 0.0
        sell_msp = abs(displacement) / abs(avg_dn) if abs(avg_dn) > 0 else 0.0
        bar_start = self._forming_m5_start.get(symbol)
        now = time.time()
        bar_age = max(0, now - bar_start) if bar_start else 0
        # Estimate age from forming displacement as fallback
        min_conf = 0.0
        if symbol in self._forming_open and symbol in self._forming_close:
            disp = abs(self._forming_close[symbol] - self._forming_open[symbol])
            if disp > 0:
                min_conf = 0.55
        confidence = max(min_conf, min(bar_age / 60.0, 1.0))
        return {
            "buy_msp": buy_msp,
            "sell_msp": sell_msp,
            "bar_age_seconds": round(bar_age),
            "confidence": round(confidence, 3),
        }

    @staticmethod
    def classify_swing_state(direction: float, ssp_data: dict | None, msp_data: dict | None) -> dict:
        """Classify swing state for a given direction using SSP + MSP data."""
        if ssp_data is None or msp_data is None:
            return {"swing_state": "INSUFFICIENT_DATA", "position_state": "UNKNOWN", "decision": "NO_CLASSIFICATION"}
        if msp_data.get("confidence", 0) < 0.45:
            return {"swing_state": "UNCONFIRMED", "position_state": "EARLY_BAR", "decision": "ALLOW"}
        buy_ssp = ssp_data["buy_ssp"]
        sell_ssp = ssp_data["sell_ssp"]
        vol_exp = ssp_data.get("vol_expansion", 1.0)
        range_exp = ssp_data.get("range_expansion", 1.0)
        if range_exp < 0.5:
            position_state = "COMPRESSED_RANGE"
        elif buy_ssp > 1.0:
            position_state = "BREAKOUT_UP"
        elif sell_ssp > 1.0:
            position_state = "BREAKOUT_DOWN"
        else:
            position_state = "INSIDE_RANGE"
        if vol_exp > 3.0 and direction > 0 and buy_ssp > 0.85:
            return {"swing_state": "EXHAUSTION_SUPPRESSED_BY_VOLATILITY", "position_state": position_state, "decision": "CAUTION"}
        if vol_exp > 3.0 and direction < 0 and sell_ssp > 0.85:
            return {"swing_state": "EXHAUSTION_SUPPRESSED_BY_VOLATILITY", "position_state": position_state, "decision": "CAUTION"}
        if direction > 0:
            msp = msp_data.get("buy_msp", 0)
            if buy_ssp > 0.85 and msp > 0.70:
                return {"swing_state": "EXHAUSTED", "position_state": position_state, "decision": "BLOCK"}
            elif buy_ssp > 0.65:
                return {"swing_state": "LATE", "position_state": position_state, "decision": "CAUTION"}
            else:
                return {"swing_state": "HEALTHY", "position_state": position_state, "decision": "ALLOW"}
        else:
            msp = msp_data.get("sell_msp", 0)
            if sell_ssp > 0.85 and msp > 0.70:
                return {"swing_state": "EXHAUSTED", "position_state": position_state, "decision": "BLOCK"}
            elif sell_ssp > 0.65:
                return {"swing_state": "LATE", "position_state": position_state, "decision": "CAUTION"}
            else:
                return {"swing_state": "HEALTHY", "position_state": position_state, "decision": "ALLOW"}

    def _compute_state(self) -> None:
        for ccy in CURRENCY_LIST:
            vals = list(self._strength_history[ccy])
            current = vals[-1]
            x = np.arange(len(vals))
            y = np.array(vals)
            slope = float(np.polyfit(x, y, 1)[0]) if len(vals) >= 2 else 0.0
            direction = 1 if slope > _EPS else (-1 if slope < -_EPS else 0)
            if direction == 0:
                direction = 1 if vals[-1] > _EPS else (-1 if vals[-1] < -_EPS else 0)
            aligned = sum(1 for i in range(1, len(vals))
                          if (vals[i] - vals[i - 1]) * direction > 0)
            consistency = aligned / max(len(vals) - 1, 1) if direction != 0 else 0.0
            var_ = float(np.var(vals)) + _EPS
            stability = 1.0 / (1.0 + var_ * 100)
            low, high = min(vals), max(vals)
            pos = (current - low) / (high - low + _EPS) if high > low else 0.5
            prev2 = vals[-3] if len(vals) >= 3 else vals[0]
            momentum = (current - prev2) / max(abs(prev2), _EPS)
            momentum = max(-1.0, min(1.0, momentum))
            self._state[ccy] = {
                "direction": direction,
                "consistency": round(consistency, 4),
                "slope": round(slope, 6),
                "current": round(current, 6),
                "momentum": round(momentum, 4),
                "stability": round(stability, 4),
                "position": round(pos, 4),
            }

    def alignment(self, symbol: str, direction: float) -> float:
        if not self._ready:
            return 1.0
        if symbol not in BASE_CURRENCY_MAP:
            return 1.0
        # Use forming bar return from open — direct price movement check
        bar_ret = self.forming_return_from_open(symbol)
        if bar_ret is None:
            return 1.0
        hyp_dir = 1 if direction > 0 else -1
        bar_dir = 1 if bar_ret > 0 else -1
        agree = 1.0 if hyp_dir == bar_dir else 0.0
        if agree == 0.0:
            return max(0.05, min(0.30, abs(bar_ret) * 500))
        magnitude = min(abs(bar_ret) * 1000, 1.0)
        return max(0.50, magnitude)

    def get_state(self) -> dict:
        return dict(self._state)

    def get_strength_history(self) -> dict[str, list]:
        """Return last 5 bar-level strength vectors per currency."""
        return {c: list(h) for c, h in self._strength_history.items()}

    def get_summary(self) -> str:
        if not self._ready:
            return "not-ready"
        parts = []
        top = sorted(self._state.items(), key=lambda x: abs(x[1].get("slope", 0)), reverse=True)
        for ccy, s in top[:4]:
            arrow = "▲" if s["direction"] > 0 else "▼" if s["direction"] < 0 else "○"
            parts.append(f"{ccy}={arrow}c={s['consistency']:.2f}m={s['momentum']:+.2f}")
        return "  ".join(parts)

    def reset(self) -> None:
        for h in self._strength_history.values():
            h.clear()
        for b in self._bar_ohlc.values():
            b.clear()
        self._last_completed_ts.clear()
        self._forming_m5_start.clear()
        self._forming_open.clear()
        self._forming_close.clear()
        self._preloaded = False
        self._ready = False
        self._state.clear()
        self._processed_bars = 0
