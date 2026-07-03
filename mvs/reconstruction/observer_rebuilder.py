from __future__ import annotations

from typing import Dict

import numpy as np

from mvs.observer.observer_features import (
    MAX_ENTROPY,
    normalize_tpi,
    compute_entropy_alignment,
    curvature_strength_from_std,
    persistence_ratio_from_window,
    compute_confidence,
    state_from_confidence,
)


class ObserverRebuilder:
    __slots__ = (
        "state", "max_entropy", "tpi_window", "max_window", "calibration_passed",
    )

    def __init__(self) -> None:
        self.state = "SUPPRESS"
        self.max_entropy = MAX_ENTROPY
        self.tpi_window: list = []
        self.max_window: int = 100
        self.calibration_passed: bool = False

    def set_calibration_passed(self, passed: bool) -> None:
        self.calibration_passed = passed

    def on_tick(self, tick_id: int, symbol: str, ts_ns: int,
                tpi: float, tpi_sign: int, entropy: float,
                regime: str, rf_prob: float) -> Dict:
        if not self.calibration_passed:
            return {"observer_state": "SUPPRESS", "observer_confidence": 0.0}

        self.tpi_window.append(abs(tpi))
        if len(self.tpi_window) > self.max_window:
            self.tpi_window.pop(0)

        tpi_arr = np.array(self.tpi_window)
        max_tpi = float(np.max(tpi_arr)) if len(tpi_arr) > 0 else 1.0
        std_tpi = float(np.std(tpi_arr)) if len(tpi_arr) > 5 else 0.01

        normalized_tpi = normalize_tpi(tpi, max_tpi)
        persistence = persistence_ratio_from_window(self.tpi_window)
        curvature = curvature_strength_from_std(std_tpi, max_tpi)
        entropy_alignment = compute_entropy_alignment(entropy, self.max_entropy)

        confidence = compute_confidence(normalized_tpi, persistence,
                                        curvature, entropy_alignment)
        self.state = state_from_confidence(confidence)

        return {"observer_state": self.state, "observer_confidence": float(confidence)}
