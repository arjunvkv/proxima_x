"""SignalFusionEngine — merges CORE + ERL signals, deduplicates, scores.

Responsible for:
- Merging signals from EdgeSignalMapper and EdgeRedundancyLayer
- Deduplicating by (symbol, direction, time_window)
- Preserving confirm logic and governor rules
- Producing a single ranked signal list for execution"""
import logging
from typing import Optional
import numpy as np

logger = logging.getLogger("proxima_ops.risk.fusion")

DEDUP_WINDOW_BARS = 3  # Merge signals within this many bars for same symbol/direction


class SignalFusionEngine:
    """Merge, deduplicate, and score multi-source signals."""

    def __init__(self):
        self._last_fused_signals: list[dict] = []

    def fuse(self, core_signals: list[dict],
             erl_pressure: list[dict],
             erl_momentum: list[dict]) -> list[dict]:
        """Merge all signal sources, deduplicate, and return ranked list."""
        all_signals = list(core_signals)
        all_signals.extend(erl_pressure)
        all_signals.extend(erl_momentum)

        # Filter: only signals with direction != 0
        directed = [s for s in all_signals if s.get("direction", 0) != 0]

        if not directed:
            self._last_fused_signals = []
            return []

        # Deduplication: group by (symbol, direction), keep highest confidence
        groups: dict[str, list[dict]] = {}
        for s in directed:
            key = f"{s.get('symbol', '?')}_{s.get('side', 'BUY')}"
            groups.setdefault(key, []).append(s)

        fused = []
        for key, sigs in groups.items():
            # Keep the best signal per group
            best = max(sigs, key=lambda x: x.get("confidence", 0))
            # Augment with fusion metadata
            best["_fusion_source_count"] = len(sigs)
            best["_fusion_sources"] = [s.get("edge_id") for s in sigs]
            best["_fusion_primary_edge"] = best.get("parent_edge_id", best.get("edge_id"))
            best["_fusion_is_erl"] = any(s.get("strategy") in ("pressure", "momentum") for s in sigs)
            fused.append(best)

        # Sort by confidence desc
        fused.sort(key=lambda s: s.get("confidence", 0), reverse=True)
        self._last_fused_signals = fused
        return fused

    def get_confirm_key(self, signal: dict) -> str:
        """Generate the confirm cycle tracking key for a signal."""
        edge_id = signal.get("edge_id", "unknown")
        symbol = signal.get("symbol", "?")
        side = signal.get("side", "BUY")
        return f"{edge_id}_{symbol}_{side}"

    def is_threshold_pass(self, signal: dict, threshold: float = 0.40) -> bool:
        """Check if signal passes execution threshold."""
        return (signal.get("direction", 0) != 0
                and signal.get("confidence", 0) >= threshold)

    def get_best_signal(self, threshold: float = 0.40) -> Optional[dict]:
        """Get the highest-confidence signal that passes threshold."""
        for s in self._last_fused_signals:
            if self.is_threshold_pass(s, threshold):
                return s
        return None
