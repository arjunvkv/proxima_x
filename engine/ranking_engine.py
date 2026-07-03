from typing import Dict, Any, List, Tuple, Optional
from signals.outcome_surface_signal import OutcomeSurfaceSignal


class RankingEngine:
    """
    OSS Expected-Value Ranking Engine.

    Scores symbols by absolute OSS expected value (edge magnitude).
    Replaces V3/V4 ECDF+entropy ranking with research-validated OSS scoring.
    """
    def __init__(self, oss: Optional[OutcomeSurfaceSignal] = None):
        self._oss = oss

    def set_oss(self, oss: OutcomeSurfaceSignal):
        self._oss = oss

    def score_symbol(self, ecdf: float) -> float:
        if self._oss is None:
            return 0.0
        bucket = min(int(ecdf * 10), 9)
        ev = self._oss.bucket_ev(bucket)
        return float(abs(ev))

    def rank_all(self, eval_data: Dict[str, Dict[str, Any]]) -> List[Tuple[str, float]]:
        ranked = []
        for sym, data in eval_data.items():
            ecdf = data.get("ecdf_rank", 0.5)
            score = self.score_symbol(ecdf)
            ranked.append((sym, score))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
