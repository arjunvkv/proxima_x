from __future__ import annotations

from typing import Dict
from mvs.models.conflict_model import ConflictResult


class LayerAttribution:
    __slots__ = ()

    def from_conflicts(self, conflicts: ConflictResult) -> Dict[str, float]:
        if not conflicts.conflicts:
            return {}
        counts: Dict[str, int] = {}
        for c in conflicts.conflicts:
            counts[c.layer] = counts.get(c.layer, 0) + 1
        total = sum(counts.values())
        return {layer: count / total for layer, count in counts.items()}
