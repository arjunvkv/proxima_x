from __future__ import annotations

from typing import Dict
import numpy as np

from mvs.models.honesty_model import HonestyScore
from mvs.models.conflict_model import ConflictResult
from mvs.models.market_plane import MarketRealityPlane
from mvs.models.perception_plane import PerceptionStatePlane
from mvs.models.action_plane import ActionStatePlane
from mvs.models.outcome_plane import OutcomeStatePlane


class LayerHonestyEngine:
    __slots__ = ("alpha", "beta", "gamma", "delta", "epsilon")

    def __init__(self) -> None:
        self.alpha = 0.35; self.beta = 0.25; self.gamma = 0.20; self.delta = 0.10; self.epsilon = 0.10

    @staticmethod
    def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 2 or len(b) < 2: return 0.0
        if np.std(a) == 0 or np.std(b) == 0: return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    @staticmethod
    def _clip01(x: float) -> float:
        return float(np.clip(x, 0.0, 1.0))

    def _contradiction_penalty(self, layer_name: str, conflicts: ConflictResult) -> float:
        if not conflicts.conflicts: return 0.0
        total = len(conflicts.conflicts)
        own = sum(1 for c in conflicts.conflicts if c.layer == layer_name)
        return own / total

    def score_layer(self, layer_name: str, market: MarketRealityPlane, perception: PerceptionStatePlane, action: ActionStatePlane, outcome: OutcomeStatePlane, conflicts: ConflictResult) -> HonestyScore:
        mw = market.window(100); pw = perception.window(100); aw = action.window(100); ow = outcome.window(100)
        A = 0.0; T = 0.0; P = 0.0; D = 0.0; C = self._contradiction_penalty(layer_name, conflicts)
        if len(mw) == 0 or len(pw) == 0:
            return HonestyScore(layer_name=layer_name, score=0.0, directional_accuracy=0.0, timing_precision=0.0, path_alignment=0.0, delay_penalty=0.0, contradiction_penalty=C, sample_count=0, timestamp=0)
        price_delta = mw["delta"]

        if layer_name == "tpi":
            signs_pred = np.sign(pw["tpi"]); signs_real = np.sign(price_delta)
            A = np.sum(signs_pred == signs_real) / max(len(signs_pred), 1)
            avg_decay_lag = float(np.mean(np.abs(pw["tpi_decay"] - pw["tpi"])))
            T = self._clip01(1.0 - avg_decay_lag)
            P = self._clip01(abs(self._safe_corr(pw["tpi"], price_delta)))
        elif layer_name == "entropy":
            volatility = np.abs(price_delta)
            A = self._clip01(abs(self._safe_corr(pw["entropy"], volatility)))
            T = self._clip01(1.0 - np.mean(np.abs(pw["entropy"] - np.roll(pw["entropy"], 1))))
            P = A
        elif layer_name == "regime":
            stable = np.sum(pw["regime"] == pw["regime"][-1])
            A = stable / len(pw)
            T = self._clip01(1.0 - np.mean(pw["regime_transition_prob"]))
            P = A
        elif layer_name == "vpl":
            stable = np.mean(pw["vpl_stability"])
            A = stable; T = stable; P = stable
        elif layer_name == "observer":
            A = np.mean(pw["observer_confidence"])
            T = self._clip01(1.0 - np.std(pw["observer_confidence"]))
            if len(ow) > 0:
                P = self._clip01(abs(self._safe_corr(pw["observer_confidence"][-len(ow):], ow["mfe"])))
        elif layer_name == "calibration":
            threshold = pw["calibration_threshold"]
            A = self._clip01(np.mean(np.abs(pw["tpi"]) >= threshold))
            T = self._clip01(1.0 - np.std(threshold))
            P = self._clip01(abs(self._safe_corr(threshold, np.abs(price_delta))))
        elif layer_name == "drift":
            A = self._clip01(1.0 - np.mean(pw["drift_score"])); T = A; P = A
        elif layer_name == "action":
            if len(ow) > 0:
                wins = np.sum(ow["mfe"] > ow["mae"]); A = wins / len(ow)
                T = self._clip01(1.0 - np.mean(np.abs(ow["actual_exit_price"] - ow["model_exit_price"])))
                P = self._clip01(1.0 - np.mean(ow["mae"] / np.maximum(ow["mfe"], 1e-9)))
        elif layer_name == "market":
            A = 1.0; T = 1.0; P = 1.0

        avg_conflict_latency = np.mean([c.severity for c in conflicts.conflicts]) if conflicts.conflicts else 0.0
        D = 1.0 / (1.0 + avg_conflict_latency)
        H = self.alpha * A + self.beta * T + self.gamma * P - self.delta * D - self.epsilon * C
        score = float(np.clip(H * 100.0, 0.0, 100.0))
        return HonestyScore(layer_name=layer_name, score=score, directional_accuracy=A, timing_precision=T, path_alignment=P, delay_penalty=D, contradiction_penalty=C, sample_count=len(pw), timestamp=int(pw[-1]["ts_ns"]))
