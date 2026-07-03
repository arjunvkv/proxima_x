from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass(slots=True)
class GeometryResult:
    opportunity_score: float
    path_type: str
    convexity: float
    optimal_exit: float
    optimal_hold_ticks: int
    topology: Dict


class OpportunityGeometryEngine:
    __slots__ = ()

    def build(self, mfe_mae: dict, path_sig: str, topology: dict) -> GeometryResult:
        mfe_mae_ratio = mfe_mae["mfe"] / max(mfe_mae["mae"], 1e-9)
        opportunity_score = (mfe_mae_ratio * 0.4) + ((1.0 - topology["oscillation_count"] / 100.0) * 0.3) + (np.sign(topology["convexity"]) * 0.3)
        opportunity_score = float(np.clip(opportunity_score, 0.0, 1.0))
        optimal_exit = float(topology["max_excursion"])
        return GeometryResult(opportunity_score=opportunity_score, path_type=path_sig, convexity=float(topology["convexity"]), optimal_exit=optimal_exit, optimal_hold_ticks=int(topology["time_to_max_ticks"]), topology=topology)
