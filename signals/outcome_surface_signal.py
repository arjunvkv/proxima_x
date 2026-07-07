"""OutcomeSurfaceSignal — frozen lookup table from bucket EV statistics.

Layer 0: Primary directional signal. Built from training window bucket statistics.
Supports drift-conditioned buckets: P(up | ECDF_bucket, drift_state).
"""
from typing import Dict, List, Any, Optional
from collections import defaultdict


class OutcomeSurfaceSignal:
    def __init__(self, bucket_stats: Dict[str, Dict] = None, ev_threshold: float = 0.05):
        if bucket_stats is None:
            bucket_stats = {}
        self._buckets = {
            k: {
                "ev": v["ev"],
                "count": v["count"],
                "up_pct": v["up_pct"],
                "dn_pct": v["dn_pct"],
                "confidence": v["edginess"] / 100.0,
                "persist_hits": v.get("persist_hits", 0),
                "persist_total": v.get("persist_total", 0),
            }
            for k, v in bucket_stats.items()
        }
        self._ev_threshold = ev_threshold
        self._bucket_keys = sorted(self._buckets.keys())

    def predict(self, ecdf: float, drift_state: int = 0) -> int:
        bucket_key = self._make_key(ecdf, drift_state)
        info = self._buckets.get(bucket_key)
        if info is None:
            bucket_key = f"{int(ecdf * 10) / 10:.1f}|0"
            info = self._buckets.get(bucket_key)
        if info is None:
            return 0
        # Persistence-based signal: confidence-weighted attenuation (no sign inversion)
        persist_hits = info.get("persist_hits", 0)
        persist_total = info.get("persist_total", 0)
        p_cont = persist_hits / persist_total if persist_total > 0 else 0.5
        if drift_state == 0:
            return 0
        if p_cont >= 0.60:
            return drift_state  # drift persists → follow drift
        elif p_cont <= 0.40:
            return 0  # drift fails → no signal (confidence too low, no reversal)
        return 0

    def predict_with_info(self, ecdf: float, drift_state: int = 0) -> dict:
        ecdf_key = f"{int(ecdf * 10) / 10:.1f}"
        bucket_key = f"{ecdf_key}|{drift_state}"
        info = self._buckets.get(bucket_key)
        diagnostics = {
            "requested_bucket": bucket_key,
            "found_bucket": bucket_key,
            "fallback_reason": "exact",
            "requested_drift": drift_state,
            "found_drift": drift_state,
            "available_drifts": [],
        }
        if info is None:
            diagnostics["fallback_reason"] = "drift_fallback"
            diagnostics["found_drift"] = 0
            bucket_key = f"{ecdf_key}|0"
            diagnostics["found_bucket"] = bucket_key
            available = []
            for d in [-1, 0, 1]:
                if f"{ecdf_key}|{d}" in self._buckets:
                    available.append(d)
            diagnostics["available_drifts"] = available
            info = self._buckets.get(bucket_key)
        if info is None:
            diagnostics["fallback_reason"] = "default"
            diagnostics["found_bucket"] = "N/A"
            diagnostics["found_drift"] = 0
            return {"signal": 0, "bucket": bucket_key, "ev": 0.0, "ev_signal": 0, "diagnostics": diagnostics}
        # EV-based signal (telemetry only)
        ev = info["ev"]
        if ev > self._ev_threshold:
            ev_sig = 1
        elif ev < -self._ev_threshold:
            ev_sig = -1
        else:
            ev_sig = 0
        # Persistence-based signal (active)
        persist_hits = info.get("persist_hits", 0)
        persist_total = info.get("persist_total", 0)
        p_cont = persist_hits / persist_total if persist_total > 0 else 0.5
        if drift_state == 0:
            sig = 0
        elif p_cont >= 0.60:
            sig = drift_state
        elif p_cont <= 0.40:
            sig = 0  # confidence too low, no signal (no reversal)
        else:
            sig = 0
        return {
            "signal": sig,
            "ev_signal": ev_sig,
            "bucket": bucket_key,
            "ev": ev,
            "confidence": info["confidence"],
            "up_pct": info["up_pct"],
            "dn_pct": info["dn_pct"],
            "mean_abs_move": info.get("mean_abs_move", 0.0),
            "p_cont": p_cont,
            "persist_hits": persist_hits,
            "persist_total": persist_total,
            "count": info["count"],
            "diagnostics": diagnostics,
        }

    def bucket_expected_move(self, bucket_key: str) -> float:
        info = self._buckets.get(bucket_key)
        if info is None:
            return 0.0
        return info.get("mean_abs_move", 0.0)

    @classmethod
    def from_pipeline_records(cls, records: List[Dict], ev_threshold: float = 0.05) -> "OutcomeSurfaceSignal":
        buckets = defaultdict(lambda: {"n": 0, "up": 0, "dn": 0, "sum_outcome": 0.0, "sum_abs_move": 0.0, "persist_hits": 0, "persist_total": 0})
        for r in records:
            ecdf = int(r.get("ecdf", 0.5) * 10)
            drift = r.get("drift", 0)
            k = f"{ecdf / 10:.1f}|{drift}"
            o = r.get("outcome", 0)
            buckets[k]["n"] += 1
            buckets[k]["sum_outcome"] += o
            buckets[k]["sum_abs_move"] += r.get("abs_move", 0.0)
            if o > 0:
                buckets[k]["up"] += 1
            elif o < 0:
                buckets[k]["dn"] += 1
            if drift != 0:
                buckets[k]["persist_total"] += 1
                if (drift > 0 and o > 0) or (drift < 0 and o < 0):
                    buckets[k]["persist_hits"] += 1
        stats = {}
        for k, b in buckets.items():
            n = b["n"]
            stats[k] = {
                "count": n,
                "up_pct": b["up"] / n * 100 if n else 0,
                "dn_pct": b["dn"] / n * 100 if n else 0,
                "ev": b["sum_outcome"] / n if n else 0,
                "edginess": abs(b["up"] - b["dn"]) / n * 100 if n else 0,
                "mean_abs_move": b["sum_abs_move"] / n if n else 0.0,
                "persist_hits": b["persist_hits"],
                "persist_total": b["persist_total"],
            }
        return cls(stats, ev_threshold)

    def evaluate(self, records: List[Dict]) -> dict:
        correct = total = 0
        pnl = 0.0
        for r in records:
            pred = self.predict(r.get("ecdf", 0.5), r.get("drift", 0))
            outcome = r.get("outcome", 0)
            if pred != 0:
                total += 1
                if (pred > 0 and outcome > 0) or (pred < 0 and outcome < 0):
                    correct += 1
                pnl += pred * outcome
        return {
            "accuracy": correct / total if total else 0.0,
            "pnl": pnl,
            "n_signals": total,
            "n_records": len(records),
            "ev_threshold": self._ev_threshold,
        }

    def bucket_ev(self, bucket_key: str) -> float:
        info = self._buckets.get(bucket_key)
        if info is None:
            return 0.0
        return info["ev"]

    def bucket_count(self) -> int:
        return len(self._buckets)

    def get_cache_occupancy(self) -> dict:
        occupancy = {}
        for k, v in self._buckets.items():
            occupancy[k] = {
                "count": v["count"],
                "persist_total": v["persist_total"],
                "persist_hits": v["persist_hits"],
                "ev": v["ev"],
                "up_pct": v["up_pct"],
                "dn_pct": v["dn_pct"],
            }
        return occupancy

    def signal_density(self) -> float:
        n = len(self._buckets)
        if n == 0:
            return 0.0
        active = sum(1 for b in self._buckets.values() if abs(b["ev"]) > self._ev_threshold)
        return active / n

    @staticmethod
    def _make_key(ecdf: float, drift_state: int) -> str:
        return f"{int(ecdf * 10) / 10:.1f}|{drift_state}"