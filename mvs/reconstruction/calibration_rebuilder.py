from __future__ import annotations

from typing import Dict

import numpy as np

from layer7.tpi_calibration import TPICalibrationLayer


class CalibrationRebuilder:
    __slots__ = (
        "calibration", "persistence_streak", "last_sign",
        "prev_v", "tpi_window", "max_window",
    )

    def __init__(self) -> None:
        self.calibration = TPICalibrationLayer()
        self.persistence_streak: int = 0
        self.last_sign: int = 0
        self.prev_v: float = 0.0
        self.tpi_window: list = []
        self.max_window: int = 100

    def _update_persistence(self, tpi_sign: int) -> None:
        if tpi_sign == self.last_sign and tpi_sign != 0:
            self.persistence_streak += 1
        else:
            self.persistence_streak = 1
        self.last_sign = tpi_sign

    def _derive_curvature(self, tpi: float) -> str:
        self.tpi_window.append(tpi)
        if len(self.tpi_window) > self.max_window:
            self.tpi_window.pop(0)

        if len(self.tpi_window) < 5:
            return "FLAT"

        window = np.array(self.tpi_window[-5:])
        x = np.arange(len(window))
        slope = np.polyfit(x, window, 1)[0]
        v = float(slope)
        a = v - self.prev_v
        self.prev_v = v

        eps = max(float(np.std(window)) * 0.2, 1e-6)

        sign_v = 1 if v > eps else -1 if v < -eps else 0
        sign_a = 1 if a > eps else -1 if a < -eps else 0

        if abs(v) < eps:
            return "FLAT"
        if sign_v == sign_a and sign_v != 0:
            return "ACCELERATION"
        if sign_v != sign_a and sign_a != 0:
            return "DECELERATION"
        if abs(a) > eps * 3:
            return "INFLECTION"
        return "FLAT"

    def _update_regime(self, tpi: float) -> None:
        self.tpi_window.append(abs(tpi))
        if len(self.tpi_window) > self.max_window:
            self.tpi_window.pop(0)
        if len(self.tpi_window) < 10:
            return
        ecdf = np.mean(np.array(self.tpi_window) <= abs(tpi))
        self.calibration.update_regime("__default__", ecdf)

    def on_tick(self, tick_id: int, symbol: str, ts_ns: int, tpi: float, entropy: float, regime: str) -> Dict:
        tpi_sign = 1 if tpi > 0 else -1 if tpi < 0 else 0

        self._update_persistence(tpi_sign)
        curvature = self._derive_curvature(tpi)
        self._update_regime(tpi)

        position_dir = tpi_sign if tpi_sign != 0 else 1
        result = self.calibration.evaluate(
            symbol, tpi_sign, self.persistence_streak, curvature, position_dir
        )

        threshold = 1.0 if result.get("blocked", False) else 0.3
        bucket = result.get("regime", "MODERATE_EDGE_BUCKET")

        return {
            "calibration_threshold": float(threshold),
            "calibration_bucket": str(bucket),
            "calibration_persistence": self.persistence_streak,
            "calibration_curvature": curvature,
            "calibration_ok": not result.get("blocked", False),
        }
