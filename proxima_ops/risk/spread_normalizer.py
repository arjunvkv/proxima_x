"""
P1.3: Adaptive Spread Normalization.

Replaces static spread limits with:
  spread / ATR      → volatility-adjusted friction
  spread / entropy  → information-state-adjusted friction
  spread / session  → expected session baseline

All three produce dimensionless ratios.
High ratio → expensive entry → block.
"""
import numpy as np
from collections import deque
from typing import Optional


class SpreadNormalizer:
    def __init__(self):
        self._session_baselines: dict[str, dict[str, float]] = {}
        self._spread_memory: dict[str, deque] = {}  # P7: per-symbol spread history
        self._point_sizes = {
            "EURUSD": 1e-5, "USDJPY": 1e-3, "GBPUSD": 1e-5,
            "EURJPY": 1e-3, "GBPJPY": 1e-3, "XAUUSD": 1e-2,
        }

    def _to_price(self, symbol: str, spread_points: float) -> float:
        return spread_points * self._point_sizes.get(symbol, 1e-5)

    def compute_atr(self, rates: list, period: int = 14) -> float:
        if not rates or len(rates) < period + 1:
            return 0.0
        trs = []
        for i in range(-period, 0):
            if i - 1 < -len(rates):
                continue
            high = rates[i].get("high", rates[i].get("close", 0))
            low = rates[i].get("low", rates[i].get("close", 0))
            prev_close = rates[i - 1].get("close", rates[i].get("close", 0))
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)
        if not trs:
            return 0.0
        return float(np.mean(trs))

    def spread_atr_ratio(self, symbol: str, spread_points: float, atr: float) -> float:
        if atr <= 0 or spread_points <= 0:
            return 1.0
        spread_price = self._to_price(symbol, spread_points)
        return spread_price / atr

    def spread_entropy_ratio(self, symbol: str, spread_points: float, entropy_score: float) -> float:
        if entropy_score <= 0 or spread_points <= 0:
            return 1.0
        entropy_eff = max(entropy_score, 1e-6)
        spread_price = self._to_price(symbol, spread_points)
        return spread_price / (entropy_eff * 1000 + 1e-10)

    MAX_SPREAD_BPS = 0.75
    MAX_ECON_RATIO = 0.45  # P9: spread must consume <45% of expected move
    # deprecated: MAX_SPREAD_EV_RATIO replaced by expected_move economics

    MIN_EXPECTED_MOVE = 1e-5

    def spread_econ_gate(self, spread_price: float, expected_move: float) -> dict:
        """P9: Economic gate — block if spread > % of expected move."""
        if expected_move <= 0 or spread_price <= 0:
            return {"passed": True, "econ_ratio": 0.0, "econ_ok": True}
        adj_move = max(expected_move, self.MIN_EXPECTED_MOVE)
        econ_ratio = spread_price / adj_move
        econ_ok = econ_ratio <= self.MAX_ECON_RATIO
        return {"passed": econ_ok, "econ_ratio": round(econ_ratio, 4), "econ_ok": econ_ok}

    def spread_session_ratio(self, symbol: str, spread_points: float,
                             session: str) -> float:
        baseline = self._session_baselines.get(symbol, {}).get(session)
        if baseline is None or baseline <= 0:
            return 0.0
        return spread_points / baseline  # both in points

    def update_session_baseline(self, symbol: str, session: str,
                                spread_points: float, alpha: float = 0.05):
        if symbol not in self._session_baselines:
            self._session_baselines[symbol] = {}
        baselines = self._session_baselines[symbol]
        if session not in baselines:
            baselines[session] = spread_points
        else:
            baselines[session] = (1 - alpha) * baselines[session] + alpha * spread_points

    def session_baseline_summary(self) -> str:
        if not self._session_baselines:
            return ""
        lines = ["  SPREAD SESSION BASELINES:"]
        for sym, sessions in sorted(self._session_baselines.items()):
            parts = [f"    {sym}:"]
            for sess, baseline in sorted(sessions.items()):
                parts.append(f" {sess}={baseline:.1f}")
            lines.append("".join(parts))
        return "\n".join(lines)

    def reset(self):
        self._session_baselines.clear()

    def evaluate(self, symbol: str, spread: float, rates: list,
                 entropy_score: float, session: str,
                 es_rank: float = 0.0,
                 expected_move: float = 0.0) -> dict:
        # P7: Track spread history for decay calculation
        if symbol not in self._spread_memory:
            self._spread_memory[symbol] = deque(maxlen=10)
        self._spread_memory[symbol].append(spread)
        mem = list(self._spread_memory[symbol])
        spread_decay = (sum(mem) / len(mem) / spread) if len(mem) >= 3 and spread > 0 else 1.0

        # Stale spread: cannot evaluate economics on zero spread
        if spread <= 0:
            return {
                "passed": True,
                "reason": "STALE_SPREAD",
                "spread_price": 0.0,
                "expected_move": expected_move,
                "atr": 0, "atr_ratio": 0, "atr_ok": True,
                "entropy_ratio": 0, "entropy_ok": True,
                "session_ratio": 0, "session_ok": True,
                "es_factor": 1.0, "spread_decay": 1.0,
                "econ_ratio": 0.0, "econ_ok": True,
                "reasons": [],
            }

        # P9: Economic gate — spread vs expected_move (highest priority)
        spread_price = self._to_price(symbol, spread)
        econ_gate = self.spread_econ_gate(spread_price, expected_move)
        if not econ_gate["econ_ok"]:
            return {
                "passed": False,
                "reason": "ECONOMICALLY_UNVIABLE",
                "econ_ratio": econ_gate["econ_ratio"],
                "expected_move": expected_move,
                "spread_price": spread_price,
                "atr": 0, "atr_ratio": 0, "atr_ok": False,
                "entropy_ratio": 0, "entropy_ok": False,
                "session_ratio": 0, "session_ok": False,
                "es_factor": 0, "spread_decay": round(spread_decay, 4),
                "reasons": [f"econ_ratio={econ_gate['econ_ratio']:.4f} > max={self.MAX_ECON_RATIO}"],
            }

        atr = self.compute_atr(rates)
        atr_ratio = self.spread_atr_ratio(symbol, spread, atr)
        ent_ratio = self.spread_entropy_ratio(symbol, spread, entropy_score)
        sess_ratio = self.spread_session_ratio(symbol, spread, session)

        # Adaptive thresholds
        atr_threshold = 0.5    # spread should be < 50% of ATR
        ent_threshold = 2.0    # spread should be < 2x entropy scaling
        sess_threshold = 3.0   # spread should be < 3x session baseline

        # Relax thresholds with high ES rank
        es_factor = 1.0 + max(0.0, es_rank - 0.5) * 2.0
        atr_ok = atr_ratio <= atr_threshold * es_factor
        ent_ok = ent_ratio <= ent_threshold * es_factor
        sess_ok = sess_ratio <= sess_threshold * es_factor if sess_ratio > 0 else True

        passed = atr_ok and ent_ok and sess_ok

        return {
            "passed": passed,
            "atr": atr,
            "atr_ratio": round(atr_ratio, 6),
            "atr_ok": atr_ok,
            "entropy_ratio": round(ent_ratio, 6),
            "entropy_ok": ent_ok,
            "session_ratio": round(sess_ratio, 4),
            "session_ok": sess_ok,
            "es_factor": round(es_factor, 2),
            "spread_decay": round(spread_decay, 4),  # P7
            "econ_ratio": round(econ_gate["econ_ratio"], 4),
            "expected_move": expected_move,
            "reasons": [],
        }
