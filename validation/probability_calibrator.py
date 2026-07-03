import logging
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import polars as pl

logger = logging.getLogger("proxima_demo")


class ProbabilityCalibrator:
    def __init__(self, bins: int = 10):
        self._bins = bins
        self._records: List[dict] = []

    def add(self, raw_prob: float, outcome: bool):
        self._records.append({"raw": raw_prob, "outcome": int(outcome)})

    def add_batch(self, probs: List[float], outcomes: List[bool]):
        for p, o in zip(probs, outcomes):
            self.add(p, o)

    def clear(self):
        self._records.clear()

    def brier_score(self) -> float:
        if not self._records:
            return 0.0
        return sum((r["raw"] - r["outcome"]) ** 2 for r in self._records) / len(self._records)

    def reliability_curve(self) -> dict:
        if not self._records:
            return {"bins": [], "fractions": [], "counts": []}
        bin_edges = [i / self._bins for i in range(self._bins + 1)]
        bin_data = defaultdict(lambda: {"count": 0, "pos": 0, "sum_prob": 0.0})
        for r in self._records:
            p = r["raw"]
            idx = min(int(p * self._bins), self._bins - 1)
            bin_data[idx]["count"] += 1
            bin_data[idx]["pos"] += r["outcome"]
            bin_data[idx]["sum_prob"] += p
        bins = []
        fractions = []
        counts = []
        for i in range(self._bins):
            d = bin_data[i]
            if d["count"] > 0:
                bins.append(round((i + 0.5) / self._bins, 3))
                fractions.append(round(d["pos"] / d["count"], 4))
                counts.append(d["count"])
            else:
                bins.append(round((i + 0.5) / self._bins, 3))
                fractions.append(0.0)
                counts.append(0)
        return {"bins": bins, "fractions": fractions, "counts": counts}

    def calibration_error(self) -> float:
        rc = self.reliability_curve()
        total = sum(rc["counts"])
        if total == 0:
            return 0.0
        err = 0.0
        for b, f, c in zip(rc["bins"], rc["fractions"], rc["counts"]):
            err += abs(f - b) * (c / total)
        return round(err, 4)

    def _platt_transform(self, prob: float, a: float, b: float) -> float:
        logit = math.log(max(prob, 1e-15) / max(1 - prob, 1e-15))
        return 1.0 / (1.0 + math.exp(-(a * logit + b)))

    def calibrate_scores(self, method: str = "platt") -> Dict[str, float]:
        if len(self._records) < 10:
            return {"status": "insufficient_data", "records": len(self._records)}
        outcomes = [r["outcome"] for r in self._records]
        raw_probs = [r["raw"] for r in self._records]
        mean_outcome = sum(outcomes) / len(outcomes)
        if method == "platt":
            logits = [math.log(max(p, 1e-15) / max(1 - p, 1e-15)) for p in raw_probs]
            mean_logit = sum(logits) / len(logits)
            var_logit = sum((l - mean_logit) ** 2 for l in logits) / len(logits)
            if var_logit < 1e-10:
                return {
                    "status": "degenerate",
                    "records": len(self._records),
                    "mean_outcome": round(mean_outcome, 4),
                    "brier": round(self.brier_score(), 6),
                    "ece": self.calibration_error(),
                }
            n_pos = sum(outcomes)
            n_neg = len(outcomes) - n_pos
            prior = n_pos / max(n_pos + n_neg, 1)
            a = 1.0
            b = math.log(max(prior / max(1 - prior, 1e-10), 1e-10))
            calibrated = [self._platt_transform(p, a, b) for p in raw_probs]
            cal_brier = sum((c - o) ** 2 for c, o in zip(calibrated, outcomes)) / len(outcomes)
            return {
                "status": "platt",
                "records": len(self._records),
                "mean_outcome": round(mean_outcome, 4),
                "brier_raw": round(self.brier_score(), 6),
                "brier_calibrated": round(cal_brier, 6),
                "ece_raw": self.calibration_error(),
                "a": round(a, 4),
                "b": round(b, 4),
            }
        return {"status": "unknown_method"}

    def from_evidence(self, df: pl.DataFrame, prob_col: str = "rupture_probability",
                      outcome_col: str = "resolved_label") -> "ProbabilityCalibrator":
        cal = ProbabilityCalibrator(bins=self._bins)
        for row in df.iter_rows(named=True):
            p = row.get(prob_col)
            o = row.get(outcome_col)
            if p is not None and o is not None:
                cal.add(float(p), bool(o))
        return cal

    def stats(self) -> dict:
        n = len(self._records)
        if n == 0:
            return {"records": 0}
        outcomes = [r["outcome"] for r in self._records]
        raw_probs = [r["raw"] for r in self._records]
        return {
            "records": n,
            "mean_prob": round(sum(raw_probs) / n, 4),
            "mean_outcome": round(sum(outcomes) / n, 4),
            "brier": round(self.brier_score(), 6),
            "ece": self.calibration_error(),
        }
