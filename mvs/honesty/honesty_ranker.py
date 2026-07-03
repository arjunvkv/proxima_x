from __future__ import annotations

from typing import List, Tuple
from mvs.models.honesty_model import HonestyScore


class HonestyRanker:
    __slots__ = ()

    def rank_layers(self, scores: List[HonestyScore]) -> List[Tuple[str, float, int]]:
        ordered = sorted(scores, key=lambda x: x.score, reverse=True)
        return [(s.layer_name, s.score, i) for i, s in enumerate(ordered, start=1)]
