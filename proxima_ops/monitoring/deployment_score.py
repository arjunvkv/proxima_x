import numpy as np
from datetime import datetime, date
from typing import Optional
from proxima_ops.config.settings import SETTINGS
from proxima_ops.monitoring.evidence_state import evidence_phase


class DeploymentScore:
    def __init__(self, confidence_threshold: int = 20, prior: float = 0.5):
        self._history: list[float] = []
        self._last_components: dict = {
            "Performance": 0.0,
            "Execution": 0.0,
            "Persistence": 0.0,
            "Frequency": 0.0,
            "Trade Count": 0.0
        }
        self._confidence_threshold = confidence_threshold
        self._prior = prior

    def compute(self, sharpe: Optional[float], pp: Optional[float], dd: Optional[float],
                freq_cv: float, trade_count: int, signal_count: int,
                effective_n: Optional[int] = None) -> float:
        sharpe_val = float(sharpe) if sharpe is not None else 0.0
        pp_val = float(pp) if pp is not None else 0.5
        dd_val = float(dd) if dd is not None else 0.0
        freq_cv = float(freq_cv) if freq_cv is not None else 0.0

        sharpe_score = min(max((sharpe_val - 0.5) / max(2.0, 1e-12), 0.0), 1.0) if sharpe_val > 0 else 0.0
        pp_score = min(max((pp_val - 0.45) / max(0.20, 1e-12), 0.0), 1.0)
        dd_score = 1.0 - min(dd_val / max(0.12, 1e-12), 1.0) if dd_val >= 0 else 0.0
        freq_score = 1.0 - min(freq_cv / max(0.50, 1e-12), 1.0) if freq_cv > 0 else 1.0
        volume_score = min(trade_count / max(100.0, 1e-12), 1.0) if trade_count > 0 else 0.0
        raw_score = (
            0.25 * sharpe_score + 0.20 * pp_score +
            0.20 * dd_score + 0.20 * freq_score +
            0.15 * volume_score)
        if effective_n is None:
            effective_n = trade_count
        confidence = min(effective_n / max(self._confidence_threshold, 1), 1.0)
        score = self._prior * (1.0 - confidence) + raw_score * confidence
        score = float(np.clip(score, 0.0, 1.0))
        self._history.append(score)

        self._last_components = {
            "Performance": sharpe_score,
            "Execution": pp_score,
            "Persistence": dd_score,
            "Frequency": freq_score,
            "Trade Count": volume_score,
            "Confidence": round(confidence, 3),
            "Prior": self._prior,
        }

        return score

    @property
    def current(self) -> float:
        return self._history[-1] if self._history else self._prior

    @property
    def classification(self) -> str:
        c = self.current
        if c >= 0.75:
            return "LIVE_HEALTHY"
        elif c >= 0.50:
            return "LIVE_WARNING"
        return "LIVE_CRITICAL"

    @property
    def trend(self) -> str:
        if len(self._history) < 7:
            return "STABLE"
        recent = self._history[-7:]
        slope = float(np.polyfit(range(len(recent)), recent, 1)[0])
        if slope > 0.01:
            return "IMPROVING"
        elif slope < -0.01:
            return "DEGRADING"
        return "STABLE"

    def summary(self) -> dict:
        return {
            "current_score": round(self.current, 3),
            "classification": self.classification,
            "trend": self.trend,
            "target": SETTINGS.deployment_score_target,
            "hits_target": self.current >= SETTINGS.deployment_score_target,
            "components": self._last_components,
            "evidence_phase": evidence_phase(len(self._history)),
            "confidence_threshold": self._confidence_threshold,
            "prior": self._prior,
        }
