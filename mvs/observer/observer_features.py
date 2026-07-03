from __future__ import annotations

from typing import Dict, Optional

MAX_ENTROPY = 1.0986122886681098
CURVATURE_MAP = {"ACCELERATION": 0.8, "DECELERATION": 0.5,
                 "INFLECTION": 0.2, "NEUTRAL": 0.1, "FLAT": 0.1}
DEFAULT_WEIGHTS = {"tpi": 0.40, "persistence": 0.25,
                   "curvature": 0.20, "entropy": 0.15}
STATE_THRESHOLDS = {"EXECUTE": 0.75, "HESITATE": 0.55,
                    "REVIEW": 0.35, "SUPPRESS": 0.0}


def normalize_tpi(tpi: float, max_tpi: Optional[float] = None) -> float:
    if max_tpi is not None and max_tpi > 1e-10:
        return min(1.0, abs(tpi) / max_tpi)
    return min(1.0, abs(tpi))


def compute_entropy_alignment(normalized_entropy: float,
                               max_entropy: float = MAX_ENTROPY) -> float:
    return max(0.0, 1.0 - (normalized_entropy / max_entropy))


def curvature_strength_from_state(curvature_state: str) -> float:
    return CURVATURE_MAP.get(curvature_state, 0.1)


def curvature_strength_from_std(std_tpi: float,
                                 max_tpi: float) -> float:
    return min(1.0, std_tpi / max(max_tpi, 1e-10))


def persistence_ratio_from_streak(streak: int,
                                   max_streak: int = 10) -> float:
    return min(1.0, max(streak, 0) / max(max_streak, 1))


def persistence_ratio_from_window(tpi_window) -> float:
    if len(tpi_window) < 5:
        return 0.0
    sign_streak = 0
    last_s = 0
    for val in tpi_window:
        s = 1 if val > 0 else (-1 if val < 0 else 0)
        if s == last_s and s != 0:
            sign_streak += 1
        else:
            sign_streak = 1
            last_s = s
    return min(1.0, sign_streak / max(len(tpi_window), 1))


def compute_confidence(normalized_tpi: float,
                       persistence: float,
                       curvature: float,
                       entropy_alignment: float,
                       weights: Optional[Dict[str, float]] = None) -> float:
    w = weights or DEFAULT_WEIGHTS
    return (w["tpi"] * normalized_tpi
            + w["persistence"] * persistence
            + w["curvature"] * curvature
            + w["entropy"] * entropy_alignment)


def state_from_confidence(confidence: float,
                           thresholds: Optional[Dict[str, float]] = None) -> str:
    t = thresholds or STATE_THRESHOLDS
    if confidence >= t["EXECUTE"]:
        return "EXECUTE"
    if confidence >= t["HESITATE"]:
        return "HESITATE"
    if confidence >= t["REVIEW"]:
        return "REVIEW"
    return "SUPPRESS"
