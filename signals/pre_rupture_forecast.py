import logging
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
import math

logger = logging.getLogger("proxima_demo")

WARNING_LEVELS = ["GREEN", "YELLOW", "ORANGE", "RED"]
WARNING_THRESHOLDS = [0.30, 0.50, 0.70]


def _warning_level(prob: float) -> str:
    if prob < 0.30:
        return "GREEN"
    elif prob < 0.50:
        return "YELLOW"
    elif prob < 0.70:
        return "ORANGE"
    return "RED"


class PreRuptureForecastEngine:
    def __init__(self):
        self._history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self._forecasts: Dict[str, List[dict]] = defaultdict(list)
        self._resolved: Dict[str, List[dict]] = defaultdict(list)
        self._calibration: Dict[str, Dict[str, list]] = defaultdict(
            lambda: defaultdict(list)
        )

    def update(self, symbol: str, organism_state: dict):
        self._history[symbol].append(organism_state)

    def forecast(self, symbol: str) -> Optional[dict]:
        hist = list(self._history.get(symbol, []))
        if not hist:
            return None
        state = hist[-1]
        p = state.get("pressure", 0.0)
        f = state.get("fracture", 0.0)
        c = state.get("cohort_instability", 0.0)
        t = state.get("trust", 0.5)
        a = state.get("attractor_strength", 0.0)
        cc = state.get("causal_confidence", 0.0)

        rp = min(p * 0.30 + f * 0.20 + c * 0.15 + (1.0 - t) * 0.15 + a * 0.10 + cc * 0.10, 1.0)
        warning = _warning_level(rp)
        et = state.get("escape_time", 10.0)
        rt = state.get("return_time", 5.0)
        pressure_factor = max(p, 0.1)
        expected_ticks = int(max(round((et + rt) / pressure_factor), 1))
        drivers_raw = [
            ("pressure", round(p * 0.30, 4)),
            ("fracture", round(f * 0.20, 4)),
            ("cohort_instability", round(c * 0.15, 4)),
            ("low_trust", round((1.0 - t) * 0.15, 4)),
            ("attractor_strength", round(a * 0.10, 4)),
            ("causal_confidence", round(cc * 0.10, 4)),
        ]
        drivers_raw.sort(key=lambda x: x[1], reverse=True)
        result = {
            "rupture_probability": round(rp, 4),
            "confidence": round(1.0 - abs(rp - 0.5) * 2, 4),
            "warning_level": warning,
            "expected_ticks": expected_ticks,
            "drivers": drivers_raw,
        }
        self._forecasts[symbol].append(result)
        logger.info(f"[PRE_RUPTURE] {symbol} prob={rp:.2f} "
                    f"{warning} ticks={expected_ticks}")
        return result

    def resolve(self, symbol: str, ruptured: bool):
        fcasts = self._forecasts.get(symbol, [])
        if not fcasts:
            return
        f = fcasts[-1]
        entry = {
            "rupture_probability": f["rupture_probability"],
            "warning_level": f["warning_level"],
            "expected_ticks": f["expected_ticks"],
            "ruptured": ruptured,
            "correct": (f["rupture_probability"] > 0.5) == ruptured,
        }
        self._resolved[symbol].append(entry)
        band = f["warning_level"]
        self._calibration[symbol][band].append(entry)
        logger.info(f"[FORECAST_RESOLVE] {symbol} forecast={f['warning_level']} "
                    f"actual={'RUPTURE' if ruptured else 'NO_RUPTURE'} "
                    f"{'correct' if entry['correct'] else 'incorrect'}")

    def forecast_accuracy(self, symbol: str) -> dict:
        res = self._resolved.get(symbol, [])
        if not res:
            return {}
        bands = defaultdict(lambda: {"correct": 0, "total": 0})
        for r in res:
            b = r["warning_level"]
            bands[b]["total"] += 1
            if r["correct"]:
                bands[b]["correct"] += 1
        result = {}
        for b in WARNING_LEVELS:
            if bands[b]["total"] > 0:
                result[b] = round(bands[b]["correct"] / bands[b]["total"], 4)
        return result

    def calibration(self, symbol: str) -> dict:
        res = self._resolved.get(symbol, [])
        if not res:
            return {}
        bands = defaultdict(lambda: {"fp": 0, "fn": 0, "tp": 0, "tn": 0})
        for r in res:
            b = r["warning_level"]
            pred_pos = r["rupture_probability"] > 0.5
            actual_pos = r["ruptured"]
            if pred_pos and actual_pos:
                bands[b]["tp"] += 1
            elif pred_pos and not actual_pos:
                bands[b]["fp"] += 1
            elif not pred_pos and actual_pos:
                bands[b]["fn"] += 1
            else:
                bands[b]["tn"] += 1
        result = {}
        all_brier = 0.0
        all_count = 0
        for b in WARNING_LEVELS:
            d = bands[b]
            total_pos = d["tp"] + d["fn"]
            total_neg = d["fp"] + d["tn"]
            precision = d["tp"] / max(d["tp"] + d["fp"], 1)
            recall = d["tp"] / max(total_pos, 1)
            fpr = d["fp"] / max(total_neg, 1)
            npv = d["tn"] / max(d["tn"] + d["fn"], 1)
            for r in res:
                if r["warning_level"] == b:
                    all_brier += (r["rupture_probability"] - int(r["ruptured"])) ** 2
                    all_count += 1
            result[b] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "fpr": round(fpr, 4),
                "npv": round(npv, 4),
                "tp": d["tp"],
                "fp": d["fp"],
                "fn": d["fn"],
                "tn": d["tn"],
            }
        brier = round(all_brier / max(all_count, 1), 6)
        result["brier_score"] = brier
        return result

    def stats(self) -> dict:
        all_symbols = set(list(self._history.keys()) + list(self._forecasts.keys()))
        total_forecasts = sum(len(v) for v in self._forecasts.values())
        total_resolved = sum(len(v) for v in self._resolved.values())
        total_ruptures = sum(
            sum(1 for r in res if r["ruptured"])
            for res in self._resolved.values()
        )
        probs = []
        horizons = []
        warning_counts = defaultdict(int)
        for sym, fcasts in self._forecasts.items():
            for f in fcasts:
                probs.append(f["rupture_probability"])
                horizons.append(f["expected_ticks"])
                warning_counts[f["warning_level"]] += 1
        mean_prob = round(sum(probs) / max(len(probs), 1), 4)
        mean_horizon = round(sum(horizons) / max(len(horizons), 1), 2)
        return {
            "symbols": len(all_symbols),
            "forecasts": total_forecasts,
            "resolved": total_resolved,
            "ruptures": total_ruptures,
            "mean_probability": mean_prob,
            "mean_expected_horizon": mean_horizon,
            "warning_counts": dict(warning_counts),
        }
