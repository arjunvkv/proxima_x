"""
Metric Normalization — Wave 6: Metric Normalization Layer.

Fixes:
- M2: B+/B- imbalance normalization
- M5: rolling baseline drift correction
- M12: fixed-window ER bias
- M14: lock asymmetry edge cases under volatility spikes
- M20: EF convexity edge saturation
"""
from collections import deque
import math
from typing import List


class MetricNormalizer:
    def __init__(self, rolling_window: int = 64):
        self.rolling_window = rolling_window
        self.m5_buffer: deque = deque(maxlen=rolling_window)
        self.m12_buffer: deque = deque(maxlen=rolling_window)
        self.ef_buffer: deque = deque(maxlen=rolling_window)

    # M2: B+/B- imbalance normalization
    def normalize_m2(self, b_plus: float, b_minus: float) -> dict:
        denom = abs(b_plus) + abs(b_minus) + 1e-9
        raw = (b_plus - b_minus) / denom
        magnitude = math.tanh(denom)
        confidence = magnitude * (1.0 - abs(raw) * 0.25)
        normalized = math.tanh(raw) * confidence
        return {
            "raw_imbalance": raw,
            "normalized": normalized,
            "confidence_scale": confidence,
        }

    # M5: rolling baseline drift correction
    def update_m5(self, value: float) -> dict:
        self.m5_buffer.append(value)
        if len(self.m5_buffer) < 3:
            return {"drift_corrected_value": value, "baseline": value}
        baseline = sum(self.m5_buffer) / len(self.m5_buffer)
        variance = sum((x - baseline) ** 2 for x in self.m5_buffer) / len(self.m5_buffer)
        vol = math.sqrt(variance) + 1e-9
        drift_corrected = (value - baseline) / vol
        return {"drift_corrected_value": drift_corrected, "baseline": baseline}

    # M12: fixed-window ER bias correction
    def update_m12(self, er_value: float) -> dict:
        self.m12_buffer.append(er_value)
        if len(self.m12_buffer) < self.rolling_window:
            return {"er_raw": er_value, "er_corrected": er_value, "bias_factor": 1.0}
        baseline = sum(self.m12_buffer) / len(self.m12_buffer)
        deviation = er_value - baseline
        bias_factor = 1.0 / (1.0 + abs(deviation))
        corrected = er_value * bias_factor
        return {"er_raw": er_value, "er_corrected": corrected, "bias_factor": bias_factor}

    # M14: lock asymmetry under volatility spikes
    def evaluate_m14(self, lock_state: float, volatility: float) -> dict:
        asymmetry = lock_state * math.tanh(volatility * 2.0)
        lock_valid = asymmetry < 0.65
        penalty = asymmetry ** 2
        return {"lock_valid": lock_valid, "asymmetry_penalty": penalty}

    # M20: EF convexity edge saturation
    def update_m20(self, ef_value: float) -> dict:
        self.ef_buffer.append(ef_value)
        if len(self.ef_buffer) < 4:
            return {"ef_raw": ef_value, "ef_saturated": ef_value, "convexity_factor": 1.0}
        n = len(self.ef_buffer)
        x = list(range(n))
        mean_x = sum(x) / n
        mean_y = sum(self.ef_buffer) / n
        cov_xy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, self.ef_buffer))
        var_x = sum((xi - mean_x) ** 2 for xi in x) + 1e-9
        slope = cov_xy / var_x
        curvature = abs(slope)
        saturation = math.tanh(curvature)
        ef_saturated = ef_value * (1.0 - 0.5 * saturation)
        return {"ef_raw": ef_value, "ef_saturated": ef_saturated, "convexity_factor": saturation}
