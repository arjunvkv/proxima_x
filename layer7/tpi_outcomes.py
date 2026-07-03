"""TPI Outcome Tracker — Tick→Bar causal resolution.

For every TPI shadow observation, maps tick-time to bar-open,
then resolves forward H1 and H3 directional hit rates.

Also tracks per-symbol decay distributions, rolling confidence,
persistence (consecutive same-direction prints), and curvature.
"""
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from collections import defaultdict
from layer7.types import TPIObservation

DECAY_EMA_ALPHA = 0.3
PERSISTENCE_SESSION_VOL_WINDOW = 20


# ═══════════════════════════════════════════════════════════
# P3: TPIOutcomeTracker — enhanced with symbol decay + rolling confidence
# ═══════════════════════════════════════════════════════════

class TPIOutcomeTracker:
    def __init__(self):
        self.observations: List[TPIObservation] = []
        self._max_obs = 500

        # per-symbol decay distributions
        self._symbol_decay: Dict[str, dict] = defaultdict(lambda: {
            "h1_hits": 0, "h1_total": 0, "h3_hits": 0, "h3_total": 0,
            "ema_hit_rate": None,  # rolling EMA of H1 hit rate
        })

    def register(self, obs: TPIObservation) -> None:
        self.observations.append(obs)
        if len(self.observations) > self._max_obs:
            self.observations.pop(0)

    def resolve_bar(self, symbol: str, bar_close_time: datetime, close_price: float) -> int:
        """Resolve all pending observations for symbol where bar_close_time >= bar_open_time + 1h/3h."""
        resolved_count = 0
        symd = self._symbol_decay[symbol]

        for obs in self.observations:
            if obs.symbol != symbol:
                continue
            elapsed = (bar_close_time - obs.bar_open_time).total_seconds() / 3600.0

            # H1 resolution
            if not obs.resolved_h1 and elapsed >= 1.0:
                raw_ret = (close_price - obs.entry_price) / obs.entry_price
                if obs.direction == "SHORT":
                    raw_ret = -raw_ret
                obs.h1_return = raw_ret
                obs.h1_hit = raw_ret > 0
                obs.resolved_h1 = True
                resolved_count += 1
                symd["h1_total"] += 1
                if obs.h1_hit:
                    symd["h1_hits"] += 1
                # EMA update
                hit_val = 1.0 if obs.h1_hit else 0.0
                if symd["ema_hit_rate"] is None:
                    symd["ema_hit_rate"] = hit_val
                else:
                    symd["ema_hit_rate"] = DECAY_EMA_ALPHA * hit_val + (1 - DECAY_EMA_ALPHA) * symd["ema_hit_rate"]

            # H3 resolution
            if not obs.resolved_h3 and elapsed >= 3.0:
                raw_ret = (close_price - obs.entry_price) / obs.entry_price
                if obs.direction == "SHORT":
                    raw_ret = -raw_ret
                obs.h3_return = raw_ret
                obs.h3_hit = raw_ret > 0
                obs.resolved_h3 = True
                resolved_count += 1
                symd["h3_total"] += 1
                if obs.h3_hit:
                    symd["h3_hits"] += 1

        return resolved_count

    def live_decay_stats(self) -> dict:
        h1_resolved = [o for o in self.observations if o.resolved_h1]
        h3_resolved = [o for o in self.observations if o.resolved_h3]

        h1_hits = sum(1 for o in h1_resolved if o.h1_hit)
        h3_hits = sum(1 for o in h3_resolved if o.h3_hit)
        h1_total = len(h1_resolved)
        h3_total = len(h3_resolved)

        # Signal age
        ages = []
        for o in self.observations:
            if o.resolved_h1:
                age_hrs = (datetime.now() - o.timestamp).total_seconds() / 3600.0
                ages.append(age_hrs)
        avg_age = sum(ages) / len(ages) if ages else 0.0

        # Half-life: median time from observation to H1 resolution
        half_life = None
        resolved_times = []
        for o in self.observations:
            if o.resolved_h1:
                hours_to_resolve = (o.bar_open_time - o.timestamp).total_seconds() / 3600.0
                resolved_times.append(hours_to_resolve)
        if len(resolved_times) >= 5:
            sorted_rt = sorted(resolved_times)
            mid = len(sorted_rt) // 2
            if len(sorted_rt) % 2 == 1:
                half_life = round(sorted_rt[mid], 2)
            else:
                half_life = round((sorted_rt[mid - 1] + sorted_rt[mid]) / 2.0, 2)

        # Per-symbol decay
        per_symbol = {}
        for sym, sd in sorted(self._symbol_decay.items()):
            h1r = round(sd["h1_hits"] / sd["h1_total"] * 100, 1) if sd["h1_total"] > 0 else None
            h3r = round(sd["h3_hits"] / sd["h3_total"] * 100, 1) if sd["h3_total"] > 0 else None
            ema = round(sd["ema_hit_rate"] * 100, 1) if sd["ema_hit_rate"] is not None else None
            per_symbol[sym] = {
                "h1_hit_rate": h1r, "h1_n": sd["h1_total"],
                "h3_hit_rate": h3r, "h3_n": sd["h3_total"],
                "ema_confidence": ema,
            }

        # Rolling confidence (global EMA-based)
        global_ema = None
        sorted_by_age = sorted(self.observations, key=lambda o: o.timestamp)
        all_h1 = [1.0 if o.h1_hit else 0.0 for o in sorted_by_age if o.resolved_h1]
        for v in all_h1:
            if global_ema is None:
                global_ema = v
            else:
                global_ema = DECAY_EMA_ALPHA * v + (1 - DECAY_EMA_ALPHA) * global_ema
        rolling_conf = round(global_ema * 100, 1) if global_ema is not None else None

        return {
            "h1_hit_rate": round(h1_hits / h1_total * 100, 1) if h1_total > 0 else None,
            "h3_hit_rate": round(h3_hits / h3_total * 100, 1) if h3_total > 0 else None,
            "h1_resolved": h1_total,
            "h3_resolved": h3_total,
            "avg_signal_age_bars": round(avg_age, 1),
            "half_life_bars": half_life,
            "rolling_decay_confidence": rolling_conf,
            "per_symbol": per_symbol,
        }


# ═══════════════════════════════════════════════════════════
# P4: TPI Persistence Tracker
# ═══════════════════════════════════════════════════════════

class TPIPersistenceTracker:
    """Tracks consecutive same-direction TPI prints per symbol,
    normalised by session volatility, with persistence rank."""

    def __init__(self):
        self._streaks: Dict[str, int] = defaultdict(int)
        self._directions: Dict[str, Optional[int]] = {}
        self._history: Dict[str, list] = defaultdict(list)  # recent TPI values per sym
        self._vol_window: Dict[str, list] = defaultdict(list)  # rolling TPI vol
        self._max_history = 50

    def update(self, symbol: str, tpi: float, direction: int) -> dict:
        """Update persistence streak and rank. Returns persistence state dict."""
        prev_dir = self._directions.get(symbol)
        if direction == 0:
            self._streaks[symbol] = 0
        elif direction == prev_dir and prev_dir is not None:
            self._streaks[symbol] += 1
        else:
            self._streaks[symbol] = 1

        self._directions[symbol] = direction
        self._history[symbol].append(tpi)
        if len(self._history[symbol]) > self._max_history:
            self._history[symbol].pop(0)

        # Session volatility: rolling std of TPI
        self._vol_window[symbol].append(abs(tpi))
        if len(self._vol_window[symbol]) > PERSISTENCE_SESSION_VOL_WINDOW:
            self._vol_window[symbol].pop(0)

        vol = 0.0
        if len(self._vol_window[symbol]) >= 5:
            import numpy as np
            vol = float(np.std(self._vol_window[symbol]))

        # Normalized persistence: streak / max(vol, 0.05)
        norm = self._streaks[symbol] / max(vol, 0.05)

        # Persistence rank: assign based on norm value (0-100 scale)
        rank = min(100.0, norm * 10.0)

        return {
            "streak": self._streaks[symbol],
            "direction": direction,
            "direction_label": "LONG" if direction == 1 else ("SHORT" if direction == -1 else "FLAT"),
            "session_vol": round(vol, 6),
            "normalized_persistence": round(norm, 4),
            "persistence_rank": round(rank, 1),
        }

    def state(self, symbol: str) -> dict:
        dir = self._directions.get(symbol, 0)
        return {
            "streak": self._streaks.get(symbol, 0),
            "direction": dir,
            "direction_label": "LONG" if dir == 1 else ("SHORT" if dir == -1 else "FLAT"),
        }


# ═══════════════════════════════════════════════════════════
# P5: TPI Curvature Classifier
# ═══════════════════════════════════════════════════════════

class TPICurvatureTracker:
    """Computes dTPI, d2TPI and classifies the curvature state.

    States:
      - acceleration     : dTPI > 0, d2TPI > 0  (trend strengthening)
      - exhaustion       : dTPI > 0, d2TPI < 0  (trend weakening)
      - reversal_tension : dTPI < 0, d2TPI > 0  (potential reversal up)
      - decay            : dTPI < 0, d2TPI < 0  (trend decay accelerating down)
      - neutral          : insufficient history
    """

    def __init__(self):
        self._history: Dict[str, list] = defaultdict(list)

    def update(self, symbol: str, tpi: float) -> dict:
        self._history[symbol].append(tpi)
        if len(self._history[symbol]) > 10:
            self._history[symbol].pop(0)

        h = self._history[symbol]
        if len(h) < 3:
            return {"state": "NEUTRAL", "dTPI": None, "d2TPI": None, "n": len(h)}

        # dTPI = TPI(t) - TPI(t-1)
        # d2TPI = dTPI(t) - dTPI(t-1)
        dtpi = h[-1] - h[-2]
        d2tpi = (h[-1] - h[-2]) - (h[-2] - h[-3]) if len(h) >= 3 else 0.0

        if dtpi > 0 and d2tpi > 0:
            state = "ACCELERATION"
        elif dtpi > 0 and d2tpi < 0:
            state = "EXHAUSTION"
        elif dtpi < 0 and d2tpi > 0:
            state = "REVERSAL_TENSION"
        elif dtpi < 0 and d2tpi < 0:
            state = "DECAY"
        else:
            state = "NEUTRAL"

        return {
            "state": state,
            "dTPI": round(dtpi, 6),
            "d2TPI": round(d2tpi, 6),
            "n": len(h),
        }

    def state(self, symbol: str) -> dict:
        h = self._history.get(symbol, [])
        if len(h) < 3:
            return {"state": "NEUTRAL", "dTPI": None, "d2TPI": None, "n": len(h)}
        dtpi = h[-1] - h[-2]
        d2tpi = (h[-1] - h[-2]) - (h[-2] - h[-3])
        if dtpi > 0 and d2tpi > 0:
            state = "ACCELERATION"
        elif dtpi > 0 and d2tpi < 0:
            state = "EXHAUSTION"
        elif dtpi < 0 and d2tpi > 0:
            state = "REVERSAL_TENSION"
        elif dtpi < 0 and d2tpi < 0:
            state = "DECAY"
        else:
            state = "NEUTRAL"
        return {"state": state, "dTPI": round(dtpi, 6), "d2TPI": round(d2tpi, 6), "n": len(h)}

    def is_supportive(self, symbol: str, position_direction: int) -> bool:
        """Whether curvature supports adding to a position in given direction.
        
        Args:
            symbol: Instrument symbol
            position_direction: 1=LONG, -1=SHORT
        Returns:
            True if curvature state supports continuation in that direction.
        """
        s = self.state(symbol)
        st = s.get("state", "NEUTRAL")
        if position_direction == 1:  # LONG
            return st in ("ACCELERATION", "EXHAUSTION", "REVERSAL_TENSION")
        elif position_direction == -1:  # SHORT
            return st in ("DECAY", "EXHAUSTION", "REVERSAL_TENSION")
        return False
