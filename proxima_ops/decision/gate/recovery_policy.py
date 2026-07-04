from __future__ import annotations

from typing import Any


REGIME_DEGRADED = "DEGRADED"


def check_regime_failure(ucf_alignment: float, signal_entropy: float | None = None) -> str:
    if ucf_alignment <= 0.01:
        return REGIME_DEGRADED
    if signal_entropy is not None and signal_entropy < 0.05:
        return REGIME_DEGRADED
    return "NORMAL"


class RecoveryPolicy:
    STRUCTURAL = "STRUCTURAL"
    TRANSIENT = "TRANSIENT"
    NORMAL = "NORMAL"

    def __init__(self, veto_threshold_base: float = 0.50) -> None:
        self._symbol_rv: dict[str, float] = {}
        self._symbol_rc: dict[str, float] = {}
        self._symbol_regime_vol: dict[str, float] = {}
        self._veto_threshold_base = veto_threshold_base

    def update_rv(self, symbol: str, recovery_velocity: float) -> None:
        self._symbol_rv[symbol] = recovery_velocity

    def update_rc(self, symbol: str, recovery_confidence: float) -> None:
        self._symbol_rc[symbol] = recovery_confidence

    def set_regime_volatility(self, symbol: str, vol: float) -> None:
        self._symbol_regime_vol[symbol] = vol

    def classify_rv(self, recovery_velocity: float) -> str:
        if recovery_velocity < 0.30:
            return self.STRUCTURAL
        elif recovery_velocity < 0.55:
            return self.TRANSIENT
        return self.NORMAL

    def resolve(self, symbol: str) -> dict[str, Any]:
        rv = self._symbol_rv.get(symbol, 1.0)
        rc = self._symbol_rc.get(symbol, 1.0)
        vol = self._symbol_regime_vol.get(symbol, 0.5)
        vol_factor = max(0.7, min(1.3, 1.0 + (vol - 0.5) * 0.4))
        dynamic_threshold = self._veto_threshold_base * vol_factor
        rv_class = self.classify_rv(rv)
        if rv_class == self.STRUCTURAL and rc > dynamic_threshold:
            final_class = self.TRANSIENT
            veto_applied = True
        else:
            final_class = rv_class
            veto_applied = False
        return {
            "classification": final_class,
            "rv_score": round(rv, 4),
            "rc_score": round(rc, 4),
            "rv_classification": rv_class,
            "veto_applied": veto_applied,
            "veto_threshold": round(dynamic_threshold, 4),
            "regime_vol_factor": round(vol_factor, 4),
        }

    def get_cooldown_override(self, classification: str) -> int:
        if classification == self.STRUCTURAL:
            return 25
        if classification == self.TRANSIENT:
            return 12
        return 5
