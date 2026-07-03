from __future__ import annotations

from typing import List
import numpy as np

from mvs.models.conflict_model import ConflictType, ConflictRecord, ConflictResult
from mvs.models.market_plane import MarketRealityPlane
from mvs.models.perception_plane import PerceptionStatePlane
from mvs.models.action_plane import ActionStatePlane
from mvs.models.outcome_plane import OutcomeStatePlane


class TruthConflictEngine:
    __slots__ = ("latency_threshold_ticks", "timing_threshold_ns")

    def __init__(self, latency_threshold_ticks: int = 10, timing_threshold_ns: int = 5_000_000_000) -> None:
        self.latency_threshold_ticks = latency_threshold_ticks
        self.timing_threshold_ns = timing_threshold_ns

    @staticmethod
    def _sign(x: float) -> int:
        if x > 0: return 1
        if x < 0: return -1
        return 0

    @staticmethod
    def _safe_latest(arr: np.ndarray):
        if len(arr) == 0:
            return None
        return arr[-1]

    def detect(self, market: MarketRealityPlane, perception: PerceptionStatePlane, action: ActionStatePlane, outcome: OutcomeStatePlane, tick_lookback: int = 100) -> ConflictResult:
        conflicts: List[ConflictRecord] = []
        market_w = market.window(tick_lookback)
        perception_w = perception.window(tick_lookback)
        action_w = action.window(tick_lookback)
        outcome_w = outcome.window(min(10, tick_lookback))
        if len(market_w) == 0 or len(perception_w) == 0:
            return ConflictResult(conflicts=[], aggregate_score=0.0, timestamp=0)
        latest_ts = int(perception_w[-1]["ts_ns"])
        latest_action = self._safe_latest(action_w)
        latest_outcome = self._safe_latest(outcome_w)

        # FALSE_BELIEF
        if latest_action is not None and latest_outcome is not None:
            tpi = float(perception_w[-1]["tpi"])
            mfe = float(latest_outcome["mfe"])
            if self._sign(tpi) != self._sign(mfe):
                mfe_norm = mfe / max(abs(mfe), 1e-9)
                severity = min(1.0, abs(tpi - mfe_norm))
                conflicts.append(ConflictRecord(tick_id=int(perception_w[-1]["tick_id"]), conflict_type=ConflictType.FALSE_BELIEF, severity=severity, description="TPI sign disagrees with realized MFE", layer="tpi", timestamp=latest_ts))

        # DELAYED_BELIEF
        if latest_action is not None:
            signals = np.where(perception_w["tpi"] >= perception_w["calibration_threshold"])[0]
            if len(signals) > 0:
                first_signal_idx = int(signals[0])
                signal_tick = int(perception_w[first_signal_idx]["tick_id"])
                entry_tick = int(latest_action["trade_id"])
                latency = entry_tick - signal_tick
                if latency > self.latency_threshold_ticks:
                    severity = min(1.0, latency / 50.0)
                    conflicts.append(ConflictRecord(tick_id=entry_tick, conflict_type=ConflictType.DELAYED_BELIEF, severity=severity, description="Signal fired too late", layer="observer", timestamp=latest_ts))

        # CORRUPTED_BELIEF
        entropy = float(perception_w[-1]["entropy"])
        regime = str(perception_w[-1]["regime"])
        if entropy < 0.3 and regime == "TREND":
            severity = (0.3 - entropy) / 0.3
            conflicts.append(ConflictRecord(tick_id=int(perception_w[-1]["tick_id"]), conflict_type=ConflictType.CORRUPTED_BELIEF, severity=severity, description="Entropy collapse while regime reports TREND", layer="regime", timestamp=latest_ts))

        # SUPPRESSED_GOOD
        if latest_outcome is not None:
            latest_tpi = float(perception_w[-1]["tpi"])
            threshold = float(perception_w[-1]["calibration_threshold"])
            if latest_tpi >= threshold and len(action_w) == 0:
                mfe = float(latest_outcome["mfe"])
                if mfe > 0:
                    severity = min(1.0, mfe * 2.0)
                    conflicts.append(ConflictRecord(tick_id=int(perception_w[-1]["tick_id"]), conflict_type=ConflictType.SUPPRESSED_GOOD, severity=severity, description="Valid profitable signal suppressed", layer="observer", timestamp=latest_ts))

        # ALLOWED_BAD
        if latest_action is not None and latest_outcome is not None:
            latest_tpi = float(perception_w[-1]["tpi"])
            threshold = float(perception_w[-1]["calibration_threshold"])
            mae = float(latest_outcome["mae"])
            spread = float(market_w[-1]["spread"])
            if latest_tpi < threshold and mae > (spread * 3.0):
                severity = min(1.0, mae / max(spread * 3.0, 1e-9))
                conflicts.append(ConflictRecord(tick_id=int(latest_action["trade_id"]), conflict_type=ConflictType.ALLOWED_BAD, severity=severity, description="Weak signal allowed and failed", layer="calibration", timestamp=latest_ts))

        # PATH_DISSONANCE
        if latest_outcome is not None:
            actual_exit = float(latest_outcome["actual_exit_price"])
            model_exit = float(latest_outcome["model_exit_price"])
            avg_spread = float(np.mean(market_w["spread"]))
            diff = abs(actual_exit - model_exit)
            if diff > (2.0 * avg_spread):
                severity = min(1.0, diff / max(2.0 * avg_spread, 1e-9))
                conflicts.append(ConflictRecord(tick_id=int(latest_outcome["trade_id"]), conflict_type=ConflictType.PATH_DISSONANCE, severity=severity, description="Actual exit diverged from model exit", layer="action", timestamp=latest_ts))

        # TIMING_MISALIGNMENT
        if latest_action is not None:
            signal_idx = np.argmax(perception_w["tpi"] >= perception_w["calibration_threshold"])
            signal_ts = int(perception_w[signal_idx]["ts_ns"])
            action_ts = int(latest_action["ts_ns"])
            diff_ns = abs(action_ts - signal_ts)
            if diff_ns > self.timing_threshold_ns:
                severity = min(1.0, diff_ns / 30_000_000_000.0)
                conflicts.append(ConflictRecord(tick_id=int(latest_action["trade_id"]), conflict_type=ConflictType.TIMING_MISALIGNMENT, severity=severity, description="Signal-action timestamp drift too large", layer="action", timestamp=latest_ts))

        result = ConflictResult(conflicts=conflicts, timestamp=latest_ts)
        result.recompute()
        return result
