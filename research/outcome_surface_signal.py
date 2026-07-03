"""OutcomeSurfaceSignal — frozen lookup table from bucket EV statistics.

Built from training window bucket statistics. Production-ready signal generation
without online adaptation. Signal = f(ECDF_bucket) based on empirical expected value.
"""
from typing import Dict, List, Any, Optional


class OutcomeSurfaceSignal:
    def __init__(self, bucket_stats: Dict[str, Dict], ev_threshold: float = 0.05):
        self._buckets = {
            k: {
                "ev": v["ev"],
                "count": v["count"],
                "up_pct": v["up_pct"],
                "dn_pct": v["dn_pct"],
                "confidence": v["edginess"] / 100.0,
            }
            for k, v in bucket_stats.items()
        }
        self._ev_threshold = ev_threshold
        self._bucket_keys = sorted(self._buckets.keys(), key=float)

    def predict(self, ecdf: float) -> int:
        bucket_key = f"{int(ecdf * 10) / 10:.1f}"
        info = self._buckets.get(bucket_key)
        if info is None:
            return 0
        ev = info["ev"]
        if ev > self._ev_threshold:
            return 1
        elif ev < -self._ev_threshold:
            return -1
        return 0

    def predict_with_info(self, ecdf: float) -> dict:
        bucket_key = f"{int(ecdf * 10) / 10:.1f}"
        info = self._buckets.get(bucket_key)
        if info is None:
            return {"signal": 0, "bucket": bucket_key, "ev": 0.0}
        ev = info["ev"]
        if ev > self._ev_threshold:
            sig = 1
        elif ev < -self._ev_threshold:
            sig = -1
        else:
            sig = 0
        return {
            "signal": sig,
            "bucket": bucket_key,
            "ev": ev,
            "confidence": info["confidence"],
            "up_pct": info["up_pct"],
            "dn_pct": info["dn_pct"],
        }

    @classmethod
    def from_pipeline_records(cls, records: List[Dict], ev_threshold: float = 0.05) -> "OutcomeSurfaceSignal":
        from collections import defaultdict
        buckets = defaultdict(lambda: {"n": 0, "up": 0, "dn": 0, "sum_outcome": 0.0})
        for r in records:
            ecdf = int(r.get("ecdf", 0.5) * 10)
            k = f"{ecdf / 10:.1f}"
            o = r.get("outcome", 0)
            buckets[k]["n"] += 1
            buckets[k]["sum_outcome"] += o
            if o > 0:
                buckets[k]["up"] += 1
            elif o < 0:
                buckets[k]["dn"] += 1
        stats = {}
        for k, b in buckets.items():
            n = b["n"]
            stats[k] = {
                "count": n,
                "up_pct": b["up"] / n * 100 if n else 0,
                "dn_pct": b["dn"] / n * 100 if n else 0,
                "ev": b["sum_outcome"] / n if n else 0,
                "edginess": abs(b["up"] - b["dn"]) / n * 100 if n else 0,
            }
        return cls(stats, ev_threshold)

    def evaluate(self, records: List[Dict]) -> dict:
        correct = 0
        total = 0
        pnl = 0.0
        for r in records:
            pred = self.predict(r.get("ecdf", 0.5))
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

    def summary(self) -> str:
        lines = [f"OutcomeSurfaceSignal (EV threshold={self._ev_threshold})"]
        lines.append(f"  {'Bucket':>8} {'EV':>8} {'Conf':>6} {'Up%':>6} {'Dn%':>6} {'Count':>7}")
        for k in self._bucket_keys:
            b = self._buckets[k]
            signal = "BUY" if b["ev"] > self._ev_threshold else ("SELL" if b["ev"] < -self._ev_threshold else "FLAT")
            lines.append(f"  {k:>8} {b['ev']:>+8.4f} {b['confidence']:>6.2f} {b['up_pct']:>5.1f}% {b['dn_pct']:>5.1f}% {b['count']:>7}  {signal}")
        return "\n".join(lines)

    def bucket_count(self) -> int:
        return len(self._buckets)

    def signal_density(self) -> float:
        """Fraction of buckets that produce non-flat signals."""
        n = len(self._buckets)
        if n == 0:
            return 0.0
        active = sum(1 for b in self._buckets.values() if abs(b["ev"]) > self._ev_threshold)
        return active / n
