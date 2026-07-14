"""5-min bar trend from real MT5 M5 bars + real-time forming bar via M1 stream."""
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
        # Completed M5 bar closes (from MT5 preload)
        self._bar_closes: dict[str, deque] = {
            s: deque(maxlen=30) for s in SYMBOLS
        }
        self._last_completed_ts: dict[str, float] = {}
        # Current forming M5 bar — updated via M1 stream
        self._forming_m5_start: dict[str, float] = {}
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
                        self._bar_closes[symbol].append(float(r[4]))
                    latest_ts = float(rates[-1][0])
                    self._last_completed_ts[symbol] = latest_ts
                    self._forming_m5_start[symbol] = latest_ts + 300
                    self._forming_close[symbol] = float(rates[-1][4])
            except Exception:
                pass
        self._preloaded = True
        populated = [v for v in self._bar_closes.values() if len(v) > 0]
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
            self._forming_close[symbol] = m1_mid
        return new_bar_closed

    def _compute_from_cache(self) -> None:
        populated = [v for v in self._bar_closes.values() if len(v) > 0]
        if len(populated) < 2:
            return
        min_bars = min(len(v) for v in populated)
        if min_bars < 2:
            return
        symbols_with_data = [s for s in SYMBOLS if len(self._bar_closes[s]) > 0]
        start = max(self._processed_bars, 0)
        for i in range(start, min_bars - 1):
            returns = {}
            for symbol in symbols_with_data:
                closes = list(self._bar_closes[symbol])
                prev = closes[i]
                curr = closes[i + 1]
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
        """Return of the current forming M5 bar vs last completed M5 bar close.
        This updates every M1 tick for real-time comparison with tick WLS."""
        if symbol not in SYMBOLS:
            return 0.0
        closes = list(self._bar_closes.get(symbol, []))
        if len(closes) < 1:
            return 0.0
        last_completed = closes[-1]
        current = self._forming_close.get(symbol, last_completed)
        if last_completed > 0 and current > 0:
            return float(np.log(current / last_completed))
        return 0.0

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
        base, quote = BASE_CURRENCY_MAP[symbol]
        bs = self._state.get(base, {})
        qs = self._state.get(quote, {})
        expected = 1 if direction > 0 else -1
        base_aligned = 1 if bs.get("direction", 0) == expected else 0
        quote_aligned = 1 if qs.get("direction", 0) == -expected else 0
        direction_score = 0.6 * base_aligned + 0.4 * quote_aligned
        base_w = bs.get("consistency", 0.5)
        quote_w = qs.get("consistency", 0.5)
        avg_weight = (base_w + quote_w) / 2.0
        base_ext = abs(bs.get("position", 0.5) - 0.5) * 2
        quote_ext = abs(qs.get("position", 0.5) - 0.5) * 2
        extreme_penalty = 1.0 - max(base_ext, quote_ext) * 0.3
        result = direction_score * avg_weight * extreme_penalty
        if result > 0.6:
            boost = 1.0 + min(base_w, quote_w) * 0.3
            result *= boost
        return max(0.05, min(result, 1.3))

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
        for b in self._bar_closes.values():
            b.clear()
        self._last_completed_ts.clear()
        self._forming_m5_start.clear()
        self._forming_close.clear()
        self._preloaded = False
        self._ready = False
        self._state.clear()
        self._processed_bars = 0
