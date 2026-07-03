import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("proxima_demo")

LOOKBACK_TICKS = 20


class CausalFingerprintEngine:
    def __init__(self):
        self._fingerprint_history: Dict[str, List[tuple]] = defaultdict(list)
        self._event_history: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self._occurrence_counts: Dict[tuple, int] = defaultdict(int)
        self._event_counts: Dict[Tuple[tuple, str], int] = defaultdict(int)
        self._latency_sums: Dict[Tuple[tuple, str], int] = defaultdict(int)
        self._latency_counts: Dict[Tuple[tuple, str], int] = defaultdict(int)
        self._event_types: set = set()

    def record(self, symbol: str, fingerprint: tuple):
        self._fingerprint_history[symbol].append(fingerprint)
        self._occurrence_counts[fingerprint] += 1
        logger.info(f"[CAUSAL_FINGERPRINT] {symbol} "
                    f"fingerprint={fingerprint} total={len(self._fingerprint_history[symbol])}")

    def resolve(self, symbol: str, event_type: str):
        self._event_types.add(event_type)
        history = self._fingerprint_history.get(symbol, [])
        if not history:
            return
        lookback = min(LOOKBACK_TICKS, len(history))
        tick_idx = len(history) - 1
        self._event_history[symbol].append((event_type, tick_idx))
        for offset in range(lookback):
            fp_idx = tick_idx - offset
            if fp_idx < 0:
                break
            fp = history[fp_idx]
            key = (fp, event_type)
            self._event_counts[key] += 1
            self._latency_sums[key] += offset
            self._latency_counts[key] += 1
        logger.info(f"[FINGERPRINT_EVENT] {symbol} "
                    f"event={event_type} lookback={lookback} "
                    f"history_len={len(history)}")

    def confidence(self, fingerprint: tuple, event_type: str) -> float:
        occ = self._occurrence_counts.get(fingerprint, 0)
        if occ == 0:
            return 0.0
        ev = self._event_counts.get((fingerprint, event_type), 0)
        return min(1.0, ev / occ)

    def latency(self, fingerprint: tuple, event_type: str) -> float:
        s = self._latency_sums.get((fingerprint, event_type), 0)
        c = self._latency_counts.get((fingerprint, event_type), 0)
        return s / c if c > 0 else 0.0

    def rarity(self, fingerprint: tuple) -> int:
        return self._occurrence_counts.get(fingerprint, 0)

    def stats(self) -> dict:
        all_fps = set()
        for v in self._fingerprint_history.values():
            all_fps.update(v)
        fp_confidences = {}
        for fp in all_fps:
            for ev in self._event_types:
                c = self.confidence(fp, ev)
                if c > 0:
                    fp_confidences[(fp, ev)] = c
        top_conf = max(fp_confidences.values()) if fp_confidences else 0.0
        latencies = []
        for fp in all_fps:
            for ev in self._event_types:
                l = self.latency(fp, ev)
                if l > 0:
                    latencies.append(l)
        mean_lat = sum(latencies) / len(latencies) if latencies else 0.0
        rare_fps = [fp for fp in all_fps if self.rarity(fp) < 10]
        repeated_fps = [fp for fp in all_fps if self.rarity(fp) >= 5]
        high_conf_fps = [
            {"fingerprint": str(fp), "event": ev, "confidence": c,
             "latency": self.latency(fp, ev)}
            for (fp, ev), c in fp_confidences.items()
            if c >= 0.70
        ]
        total_obs = sum(len(v) for v in self._fingerprint_history.values())
        return {
            "total_observations": total_obs,
            "unique_fingerprints": len(all_fps),
            "event_types": len(self._event_types),
            "event_types_list": sorted(self._event_types),
            "top_confidence": round(top_conf, 3),
            "mean_latency": round(mean_lat, 1),
            "rare_fingerprints": len(rare_fps),
            "repeated_fingerprints": len(repeated_fps),
            "high_confidence_patterns": len(high_conf_fps),
        }
