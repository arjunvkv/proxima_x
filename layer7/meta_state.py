"""
DPL-21: Meta-State Fusion Engine.

Fuses all surviving informational layers into a single
composite directional model (MetaScore).

Inputs (all normalized to [0,1]):
1. TPI sign magnitude   — directional conviction
2. Persistence rank     — consecutive same-direction memory
3. Curvature support    — second-derivative confirmation
4. Decay confidence     — rolling EMA hit rate
5. Propagation score    — cross-asset influence
6. Tick pressure score  — microstructural pre-release activity

Output: MetaScore per symbol + ranking + accuracy lift over TPI alone.
"""
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# Default fusion weights (can be tuned)
DEFAULT_WEIGHTS = {
    "tpi": 0.25,
    "persistence": 0.20,
    "curvature": 0.15,
    "decay": 0.15,
    "propagation": 0.10,
    "pressure": 0.15,
}


class MetaStateFusionEngine:
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self._weights = weights or DEFAULT_WEIGHTS.copy()
        self._history: Dict[str, list] = defaultdict(list)
        self._outcomes: List[dict] = []
        self._max_history = 500

    def compute(self, symbol: str, tpi: float, persistence: dict,
                curvature: dict, decay: dict, propagation: float,
                pressure: dict) -> dict:
        """Compute MetaScore for a symbol from all sub-signals.
        
        All inputs are normalized to [0,1] internally.
        Returns dict with MetaScore, component scores, and direction.
        """
        # Normalize TPI sign magnitude to [0,1]
        tpi_norm = min(abs(tpi), 1.0)

        # Normalize persistence rank (already 0-100 scale)
        pers_norm = persistence.get("persistence_rank", 0) / 100.0 if persistence else 0.0

        # Curvature support: 1 if supportive, 0.5 if neutral, 0 if opposing
        curv_state = curvature.get("state", "NEUTRAL") if curvature else "NEUTRAL"
        if curv_state in ("ACCELERATION", "DECAY"):
            curv_norm = 1.0
        elif curv_state in ("EXHAUSTION", "REVERSAL_TENSION"):
            curv_norm = 0.6
        else:
            curv_norm = 0.4

        # Decay confidence (EMA hit rate, already 0-100)
        decay_norm = (decay.get("ema_confidence", 0) or 0) / 100.0 if decay else 0.0

        # Propagation score (already 0-1 scale from the engine)
        prop_norm = min(propagation, 1.0) if propagation else 0.0

        # Pressure score (already 0-1 scale)
        pressure_norm = pressure.get("pressure", 0) if pressure else 0.0

        w = self._weights
        meta = (
            w["tpi"] * tpi_norm
            + w["persistence"] * pers_norm
            + w["curvature"] * curv_norm
            + w["decay"] * decay_norm
            + w["propagation"] * prop_norm
            + w["pressure"] * pressure_norm
        )

        # Direction: from TPI sign (the primary directional signal)
        meta_direction = 1 if tpi > 0 else (-1 if tpi < 0 else 0)

        result = {
            "meta_score": round(meta, 4),
            "meta_direction": meta_direction,
            "components": {
                "tpi": round(tpi_norm, 4),
                "persistence": round(pers_norm, 4),
                "curvature": round(curv_norm, 4),
                "decay": round(decay_norm, 4),
                "propagation": round(prop_norm, 4),
                "pressure": round(pressure_norm, 4),
            },
            "attribution": self._compute_attribution(
                tpi_norm, pers_norm, curv_norm, decay_norm, prop_norm, pressure_norm, meta
            ),
        }

        self._history[symbol].append(meta)
        if len(self._history[symbol]) > self._max_history:
            self._history[symbol].pop(0)

        return result

    def _compute_attribution(self, tpi, pers, curv, decay, prop, pressure, meta) -> Dict[str, float]:
        """Compute relative contribution of each layer to the final MetaScore."""
        w = self._weights
        raw = {
            "tpi": w["tpi"] * tpi,
            "persistence": w["persistence"] * pers,
            "curvature": w["curvature"] * curv,
            "decay": w["decay"] * decay,
            "propagation": w["propagation"] * prop,
            "pressure": w["pressure"] * pressure,
        }
        total = sum(raw.values())
        if total > 0:
            return {k: round(v / total, 3) for k, v in raw.items()}
        return {k: 0.0 for k in raw}

    def record_outcome(self, symbol: str, meta_score: float, meta_direction: int,
                       tpi_direction: int, actual_direction: int) -> None:
        self._outcomes.append({
            "symbol": symbol,
            "meta_score": meta_score,
            "meta_dir": meta_direction,
            "tpi_dir": tpi_direction,
            "actual_dir": actual_direction,
            "meta_correct": meta_direction == actual_direction if meta_direction != 0 else False,
            "tpi_correct": tpi_direction == actual_direction if tpi_direction != 0 else False,
        })
        if len(self._outcomes) > self._max_history:
            self._outcomes.pop(0)

    def accuracy_report(self, threshold: float = 0.0) -> dict:
        """Compare MetaScore accuracy vs raw TPI accuracy."""
        if len(self._outcomes) < 10:
            return {"n_samples": len(self._outcomes), "status": "INSUFFICIENT"}
        meta_correct = sum(1 for o in self._outcomes if o["meta_correct"])
        tpi_correct = sum(1 for o in self._outcomes if o["tpi_correct"])
        total = len(self._outcomes)
        meta_acc = meta_correct / total
        tpi_acc = tpi_correct / total

        # High-confidence subset (meta_score > threshold)
        high_conf = [o for o in self._outcomes if o["meta_score"] > threshold]
        if len(high_conf) >= 10:
            hc_correct = sum(1 for o in high_conf if o["meta_correct"])
            hc_acc = hc_correct / len(high_conf)
        else:
            hc_acc = None

        return {
            "n_samples": total,
            "meta_accuracy": round(meta_acc, 4),
            "tpi_accuracy": round(tpi_acc, 4),
            "lift_vs_tpi": round((meta_acc - tpi_acc) * 100, 2),
            "high_conf_accuracy": round(hc_acc, 4) if hc_acc is not None else None,
            "n_high_conf": len(high_conf) if hc_acc is not None else 0,
            "status": "ACTIVE",
        }

    def summary(self, symbols: List[str], meta_scores: Dict[str, dict]) -> str:
        lines = []
        lines.append("  DPL-21: META-STATE FUSION")
        lines.append("-" * 52)
        acc = self.accuracy_report(threshold=0.5)
        if acc["status"] == "INSUFFICIENT":
            lines.append(f"  Collecting samples ({acc['n_samples']}/10 minimum)...")
        else:
            lines.append(f"  MetaAcc: {acc['meta_accuracy']:.1%}  TPIAcc: {acc['tpi_accuracy']:.1%}  Lift: {acc['lift_vs_tpi']:+.2f}%")
            if acc["high_conf_accuracy"] is not None:
                lines.append(f"  HighConf(>0.5): {acc['high_conf_accuracy']:.1%} (n={acc['n_high_conf']})")
        lines.append("")
        lines.append(f"  {'Symbol':<8s} {'Meta':<7s} {'Dir':<5s} {'TPI':<6s} {'Pers':<6s} {'Curv':<6s} {'Decay':<6s} {'Prop':<6s} {'Pres':<6s}")
        for sym in symbols:
            m = meta_scores.get(sym, {})
            if not m:
                lines.append(f"  {sym:<8s} {'PENDING':<20s}")
                continue
            ms = f"{m.get('meta_score', 0):.3f}"
            d = f"{'LONG' if m.get('meta_direction')==1 else 'SHORT' if m.get('meta_direction')==-1 else 'FLAT':>4s}"
            c = m.get("components", {})
            tpi_s = f"{c.get('tpi', 0):.3f}"
            pers_s = f"{c.get('persistence', 0):.3f}"
            curv_s = f"{c.get('curvature', 0):.3f}"
            dec_s = f"{c.get('decay', 0):.3f}"
            prop_s = f"{c.get('propagation', 0):.3f}"
            pres_s = f"{c.get('pressure', 0):.3f}"
            lines.append(f"  {sym:<8s} {ms:<7s} {d:<5s} {tpi_s:<6s} {pers_s:<6s} {curv_s:<6s} {dec_s:<6s} {prop_s:<6s} {pres_s:<6s}")
        lines.append("")
        lines.append("  Feature Attribution (global avg):")
        if self._outcomes:
            total_contrib = {"tpi": 0, "persistence": 0, "curvature": 0, "decay": 0, "propagation": 0, "pressure": 0}
            attrib_keys = list(total_contrib.keys())
            for sym, m in meta_scores.items():
                att = m.get("attribution", {})
                for k in attrib_keys:
                    total_contrib[k] += att.get(k, 0)
            n = max(len(meta_scores), 1)
            for k in attrib_keys:
                lines.append(f"    {k:<15s} {total_contrib[k]/n:.1%}")
        return "\n".join(lines)
