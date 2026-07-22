"""Phase Dislocation — Cross-rate triangle dislocation recovery.

Core concept:
  FX cross rates are mathematically linked:
    EUR/JPY ≈ EUR/USD × USD/JPY    (multiplication triangle)
    GBP/USD ≈ EUR/USD / EUR/GBP    (division triangle)

  When the traded cross diverges from its synthetic rate, the triangle
  MUST resolve (arbitrage constraint). The pair with the LEAST recent
  momentum is the "path of least resistance" — it will do the work.

Why this is hard for the market to exploit:
  1. Not obvious — most trade the moving pair, we trade the "silent" one
  2. Structural, not statistical — mathematical identity, not a pattern
  3. Low frequency — dislocations are rare (3-8/day in liquid hours)
  4. Multi-triangle — 4 independent triangles diversify the edge
  5. Cannot be faked — the market can't permanently dislocate a triangle

Triangles:
  1. (EURUSD, USDJPY, EURJPY)  mult: C = A × B
  2. (EURUSD, EURGBP, GBPUSD)  div:  C = A / B
  3. (GBPUSD, USDJPY, GBPJPY)  mult: C = A × B
  4. (AUDUSD, USDJPY, AUDJPY)  mult: C = A × B

Signal rules:
  For each triangle:
  1. Compute log returns of A, B, C over lookback window
  2. Compute synthetic cross return and dislocation d
  3. Track rolling mean + std of d (100-sample window)
  4. If |d| > 2σ → dislocation detected
  5. Pick pair with smallest |return| = path of least resistance
  6. Direction = opposite of impact × sign(d)
"""
import numpy as np
from collections import deque

PAIRS = [
    "EURUSD", "USDJPY", "EURJPY",
    "EURGBP", "GBPUSD", "GBPJPY",
    "AUDUSD", "AUDJPY",
]

# Triangle defs: (a, b, c, type)
# mult: C = A × B  => r_C ≈ r_A + r_B
# div:  C = A / B  => r_C ≈ r_A - r_B
TRIANGLES = [
    ("EURUSD", "USDJPY", "EURJPY", "mult"),
    ("EURUSD", "EURGBP", "GBPUSD", "div"),
    ("GBPUSD", "USDJPY", "GBPJPY", "mult"),
    ("AUDUSD", "USDJPY", "AUDJPY", "mult"),
]

# Impact of each role going UP on dislocation d.
# mult: d = r_C - r_A - r_B   => A:-1, B:-1, C:+1
# div:  d = r_C - r_A + r_B   => A:-1, B:+1, C:+1
PAIR_IMPACT = {
    "mult": {"a": -1, "b": -1, "c": 1},
    "div":  {"a": -1, "b":  1, "c": 1},
}

ROLE_INDEX = {"a": 0, "b": 1, "c": 2}

LOOKBACK_TICKS = 60
D_THRESHOLD_SIGMA = 2.0
MIN_HISTORY = 30
MIN_TICK_HISTORY = 10


class PhaseDislocation:
    def __init__(self, lookback=LOOKBACK_TICKS, threshold_sigma=D_THRESHOLD_SIGMA):
        self.lookback = lookback
        self.threshold_sigma = threshold_sigma
        self.price_history = {}   # pair -> deque of (timestamp, mid)
        self.dislocation_history = {}  # tri_key -> deque of d values

    def update_prices(self, data):
        """data: {pair: {bid, ask, time}} from a bar/tick snapshot."""
        import time
        now = int(time.time())
        for pair in PAIRS:
            if pair not in self.price_history:
                self.price_history[pair] = deque(maxlen=self.lookback)
            values = data.get(pair)
            if values is not None:
                mid = (values.get("bid", 0) + values.get("ask", 0)) / 2
                if mid > 0:
                    self.price_history[pair].append((now, mid))

    def tri_key(self, a, b, c, tt):
        return f"{a}_{b}_{c}_{tt}"

    def log_return(self, arr):
        if len(arr) < 2:
            return 0.0
        return float(np.log(arr[-1] / arr[0]))

    def generate_signal(self, data):
        self.update_prices(data)
        signals = []

        for a, b, c, tt in TRIANGLES:
            ha = self.price_history.get(a)
            hb = self.price_history.get(b)
            hc = self.price_history.get(c)
            if not ha or not hb or not hc:
                continue
            if len(ha) < MIN_TICK_HISTORY or len(hb) < MIN_TICK_HISTORY or len(hc) < MIN_TICK_HISTORY:
                continue

            pa = np.array([p[1] for p in ha])
            pb = np.array([p[1] for p in hb])
            pc = np.array([p[1] for p in hc])

            r_a = self.log_return(pa)
            r_b = self.log_return(pb)
            r_c = self.log_return(pc)

            if r_a == 0.0 or r_b == 0.0 or r_c == 0.0:
                continue

            r_c_synth = r_a + r_b if tt == "mult" else r_a - r_b
            d = r_c - r_c_synth
            key = self.tri_key(a, b, c, tt)

            if key not in self.dislocation_history:
                self.dislocation_history[key] = deque(maxlen=100)
            self.dislocation_history[key].append(d)

            if len(self.dislocation_history[key]) < MIN_HISTORY:
                continue

            d_arr = np.array(self.dislocation_history[key])
            d_mean = np.mean(d_arr)
            d_std = np.std(d_arr)
            if d_std < 1e-12:
                continue

            d_z = (d - d_mean) / d_std

            if abs(d_z) < self.threshold_sigma:
                continue

            impacts = PAIR_IMPACT[tt]
            returns = {"a": r_a, "b": r_b, "c": r_c}
            roles = ["a", "b", "c"]

            best_role = min(roles, key=lambda r: abs(returns[r]))
            best_pair = {"a": a, "b": b, "c": c}[best_role]
            impact = impacts[best_role]

            direction = -impact if d > 0 else impact

            pair_h = [ha, hb, hc][ROLE_INDEX[best_role]]
            t0 = pair_h[0][0]
            t1 = pair_h[-1][0]
            elapsed = t1 - t0 if t1 > t0 else self.lookback

            confidence = min(0.95, abs(d_z) / 4.0)

            signals.append({
                "pair": best_pair,
                "direction": 1 if direction > 0 else -1,
                "confidence": round(confidence, 4),
                "triangle": f"{a}/{b}/{c}",
                "triangle_type": tt,
                "dislocation_z": round(float(d_z), 2),
                "dislocation": round(float(d), 8),
                "best_role": best_role,
                "r_a_bp": round(r_a * 10000, 1),
                "r_b_bp": round(r_b * 10000, 1),
                "r_c_bp": round(r_c * 10000, 1),
                "lookback_elapsed_s": elapsed,
            })

        return signals if signals else None
