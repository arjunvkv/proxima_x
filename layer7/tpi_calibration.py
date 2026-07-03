"""
TPI Calibration Layer — OSS bucket-regime-aware adaptive thresholding.

Replaces entropy-regime conditioning with OSS ECDF-bucket regimes:

Regimes (from OSS bucket):
  STRONG_EDGE_BUCKET (bucket 0, 9) → extreme ECDF, strongest directional signal
  WEAK_EDGE_BUCKET  (bucket 4, 5) → neutral ECDF, weakest signal
  MODERATE_EDGE_BUCKET (other) → moderate signal, default thresholds

Research basis: ECDF calibration curve shows bucket 0→55% UP, bucket 4-5→95% DOWN.
"""
from collections import defaultdict
from typing import Dict, Optional


OSS_BUCKET_REGIMES = {
    0: "STRONG_EDGE_BUCKET",
    1: "MODERATE_EDGE_BUCKET",
    2: "MODERATE_EDGE_BUCKET",
    3: "MODERATE_EDGE_BUCKET",
    4: "WEAK_EDGE_BUCKET",
    5: "WEAK_EDGE_BUCKET",
    6: "MODERATE_EDGE_BUCKET",
    7: "MODERATE_EDGE_BUCKET",
    8: "MODERATE_EDGE_BUCKET",
    9: "STRONG_EDGE_BUCKET",
}


class TPICalibrationLayer:
    def __init__(self, mode: str = "HARD_GATE"):
        self.mode = mode
        self._regime_cache: Dict[str, str] = {}
        self._shadow_log: list[dict] = []
        self._gate_stats: Dict[str, int] = defaultdict(int)
        self._shadow_opportunities: int = 0

    def set_mode(self, mode: str) -> None:
        if mode not in ("HARD_GATE", "SOFT_SCORE"):
            raise ValueError(f"Invalid TPI_MODE: {mode}")
        self.mode = mode

    def update_regime(self, symbol: str, ecdf: float) -> str:
        bucket = min(int(ecdf * 10), 9)
        regime = OSS_BUCKET_REGIMES.get(bucket, "MODERATE_EDGE_BUCKET")
        self._regime_cache[symbol] = regime
        return regime

    def regime(self, symbol: str) -> str:
        return self._regime_cache.get(symbol, "MODERATE_EDGE_BUCKET")

    def required_persistence(self, symbol: str) -> int:
        r = self.regime(symbol)
        if r == "STRONG_EDGE_BUCKET":
            return 1
        elif r == "WEAK_EDGE_BUCKET":
            return 3
        return 2

    def is_curvature_supportive(self, state: str, position_dir: int, symbol: str) -> bool:
        r = self.regime(symbol)
        if r == "STRONG_EDGE_BUCKET":
            return True
        elif r == "WEAK_EDGE_BUCKET":
            if position_dir == 1:
                return state == "ACCELERATION"
            else:
                return state == "DECAY"
        if position_dir == 1:
            return state in ("ACCELERATION", "EXHAUSTION", "REVERSAL_TENSION")
        else:
            return state in ("DECAY", "EXHAUSTION", "REVERSAL_TENSION")

    def evaluate(self, symbol: str, tpi_sign: int, persistence_streak: int,
                 curvature_state: str, position_dir: int) -> dict:
        """Evaluate all TPI gates with regime awareness.
        Returns dict with gate results and whether execution is blocked.
        """
        reasons = []
        blocked = False

        # TPI sign gate
        sign_ok = (tpi_sign == position_dir)
        if not sign_ok:
            reasons.append("TPI_GATE")
            blocked = True

        # Persistence gate (regime-aware)
        min_streak = self.required_persistence(symbol)
        pers_ok = (persistence_streak >= min_streak)
        if not pers_ok:
            reasons.append("PERSISTENCE_GATE")
            blocked = True

        # Curvature gate (regime-aware)
        curv_ok = self.is_curvature_supportive(curvature_state, position_dir, symbol)
        if not curv_ok:
            reasons.append("CURVATURE_GATE")
            blocked = True

        # Record statistics
        if blocked:
            self._shadow_opportunities += 1
            for r in reasons:
                self._gate_stats[r] += 1

        # In SOFT_SCORE mode, never block — just log
        effective_blocked = blocked if self.mode == "HARD_GATE" else False

        result = {
            "blocked": effective_blocked,
            "gate_blocked": blocked,  # True if any gate would trigger in HARD mode
            "reasons": reasons,
            "regime": self.regime(symbol),
            "min_persistence": min_streak,
            "sign_ok": sign_ok,
            "persistence_ok": pers_ok,
            "curvature_ok": curv_ok,
            "mode": self.mode,
        }

        # Shadow log entry
        if blocked:
            self._shadow_log.append(result.copy())

        return result

    def gate_stats(self) -> dict:
        total = sum(self._gate_stats.values())
        return {
            "total_triggers_blocked": self._shadow_opportunities,
            "total_gate_firings": total,
            "by_gate": dict(self._gate_stats),
            "shadow_opportunities": len(self._shadow_log),
            "mode": self.mode,
        }

    def reset_stats(self) -> None:
        self._gate_stats.clear()
        self._shadow_opportunities = 0
        self._shadow_log.clear()
        self._regime_cache.clear()
