import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import polars as pl

logger = logging.getLogger("proxima_demo")

EVIDENCE_SCHEMA = {
    "decision_id": str,
    "timestamp": int,
    "symbol": str,
    "production_action": str,
    "observer_recommendation": str,
    "observer_quality": float,
    "observer_confidence": float,
    "observer_consensus": float,
    "observer_override": bool,
    "disagreement": bool,
    "thesis_id": str,
    "resolved_label": int,
    "pnl": float,
    "rf_prob": float,
    "memory_weight": float,
    "counterfactual_score": float,
    "rupture_probability": float,
    "path_probability": float,
    "causal_confidence": float,
    "attractor_strength": float,
    "transition_entropy": float,
    "fracture": float,
    "cohort_instability": float,
    "pressure": float,
    "topology_state": str,
    "trust_band": str,
    "pressure_band": str,
    "rupture_flag": int,
}


class EvidenceStore:
    def __init__(self, path: Optional[str] = None):
        self._records: List[dict] = []
        self._path = path

    def record(self, entry: dict):
        for key in EVIDENCE_SCHEMA:
            if key not in entry:
                entry[key] = None
        self._records.append(entry)

    def update_thesis(self, decision_id: str, thesis_id: str,
                      resolved_label: int, pnl: float):
        for r in self._records:
            if r["decision_id"] == decision_id:
                r["thesis_id"] = thesis_id
                r["resolved_label"] = resolved_label
                r["pnl"] = pnl
                return

    def to_polars(self) -> pl.DataFrame:
        return pl.DataFrame(self._records, schema=EVIDENCE_SCHEMA)

    def save(self, path: Optional[str] = None):
        dst = path or self._path
        if dst is None:
            raise ValueError("no save path specified")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        self.to_polars().write_parquet(dst)
        logger.info(f"[EVIDENCE_STORE] saved {len(self._records)} records to {dst}")

    @classmethod
    def load(cls, path: str) -> "EvidenceStore":
        df = pl.read_parquet(path)
        store = cls(path=path)
        store._records = df.to_dicts()
        return store

    def query(self, symbol: Optional[str] = None,
              min_quality: Optional[float] = None,
              recommendation: Optional[str] = None,
              disagreement: Optional[bool] = None) -> List[dict]:
        results = list(self._records)
        if symbol is not None:
            results = [r for r in results if r["symbol"] == symbol]
        if min_quality is not None:
            results = [r for r in results
                       if r["observer_quality"] is not None
                       and r["observer_quality"] >= min_quality]
        if recommendation is not None:
            results = [r for r in results
                       if r["observer_recommendation"] == recommendation]
        if disagreement is not None:
            results = [r for r in results
                       if r["disagreement"] == disagreement]
        return results

    def stats(self) -> dict:
        total = len(self._records)
        if total == 0:
            return {"records": 0, "symbols": 0, "resolved": 0, "disagreements": 0}
        resolved = sum(1 for r in self._records if r["resolved_label"] is not None)
        disagreements = sum(1 for r in self._records if r["disagreement"])
        symbols = len(set(r["symbol"] for r in self._records if r["symbol"]))
        recs = defaultdict(int)
        for r in self._records:
            if r["observer_recommendation"]:
                recs[r["observer_recommendation"]] += 1
        return {
            "records": total,
            "symbols": symbols,
            "resolved": resolved,
            "disagreements": disagreements,
            "recommendations": dict(recs),
        }

    def clear(self):
        self._records.clear()
