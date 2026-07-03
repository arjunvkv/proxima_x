import logging
from typing import Dict, List, Optional, Tuple
from signals.thesis_buffer import ThesisBuffer

logger = logging.getLogger("proxima_demo")

BASE_HORIZONS = [5, 20, 50]

SYMBOL_VOL_RANK = {
    "EURJPY": 1.0, "USDJPY": 1.0, "GBPJPY": 1.2,
    "AUDJPY": 1.0, "CHFJPY": 0.8, "NZDJPY": 1.0,
    "EURUSD": 0.8, "GBPUSD": 1.0, "AUDUSD": 0.9,
    "USDCAD": 0.7, "NZDUSD": 0.9, "EURGBP": 0.6,
    "EURAUD": 0.9, "GBPAUD": 1.1, "AUDCAD": 1.0,
    "USDCHF": 0.7, "GBPCHF": 1.0, "EURCHF": 0.6,
}


def _scaled_horizons(symbol: str) -> List[int]:
    scale = SYMBOL_VOL_RANK.get(symbol, 1.0)
    scale = max(0.5, min(2.0, scale))
    return [max(1, int(h * scale)) for h in BASE_HORIZONS]


class ThesisGraph:
    def __init__(self, buffer: ThesisBuffer):
        self._buffer = buffer
        self._registered = set()

    def register(self, thesis_id: int, symbol: str, entry_price: float = 0.0):
        horizons = _scaled_horizons(symbol)
        self._buffer.attach_horizons(thesis_id, horizons, entry_price)
        self._registered.add(thesis_id)
        logger.info(f"[THESIS_GRAPH] registered id={thesis_id} {symbol} "
                    f"horizons={horizons}")

    def tick(self, symbol: str, price: float):
        self._buffer.tick_horizons(symbol, price)

    def survival_vector(self, thesis_id: int) -> Optional[Tuple[int, int, int]]:
        rec = self._buffer._records.get(thesis_id)
        if rec is None:
            return None
        return rec.horizon_labels

    def fracture_score(self, thesis_id: int) -> Optional[float]:
        rec = self._buffer._records.get(thesis_id)
        if rec is None:
            return None
        return rec.fracture_score

    def graph_stats(self) -> dict:
        total = len(self._registered)
        resolved = sum(1 for tid in self._registered
                       if self._buffer._records.get(tid) and
                       self._buffer._records[tid].horizon_labels is not None)
        fracture_pos = sum(1 for tid in self._registered
                           if self._buffer._records.get(tid) and
                           self._buffer._records[tid].fracture_score is not None and
                           self._buffer._records[tid].fracture_score > 0)
        vectors = set()
        for tid in self._registered:
            rec = self._buffer._records.get(tid)
            if rec and rec.horizon_labels is not None:
                vectors.add(rec.horizon_labels)
        return {
            "registered": total,
            "probes_resolved": resolved,
            "fracture_positive": fracture_pos,
            "fracture_rate": round(fracture_pos / max(resolved, 1), 3),
            "distinct_vectors": len(vectors),
            "vectors": sorted(str(v) for v in vectors) if vectors else [],
        }
