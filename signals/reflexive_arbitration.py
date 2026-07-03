import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import math

logger = logging.getLogger("proxima_demo")

RECOMMENDATIONS = ["EXECUTE", "HESITATE", "AVOID", "REVIEW"]


def _recommendation(quality: float) -> str:
    if quality >= 0.70:
        return "EXECUTE"
    elif quality >= 0.50:
        return "HESITATE"
    elif quality >= 0.30:
        return "REVIEW"
    return "AVOID"


class ReflexiveArbitrationEngine:
    def __init__(self):
        self._decisions: List[dict] = []
        self._outcomes: Dict[str, dict] = {}
        self._decision_id_counter: int = 0

    def evaluate(self, decision_state: dict) -> dict:
        rf = decision_state.get("thesis_rf_probability", 0.5)
        mem = decision_state.get("memory_weight", 0.5)
        cs = decision_state.get("counterfactual_score", 0.0)
        rp = decision_state.get("rupture_probability", 0.0)
        pp = decision_state.get("path_probability", 0.5)
        cc = decision_state.get("causal_confidence", 0.5)

        dq = min(rf * 0.25 + mem * 0.20 + max(cs, 0) * 0.20 +
                 (1.0 - rp) * 0.15 + pp * 0.10 + cc * 0.10, 1.0)
        rec = _recommendation(dq)

        reason_layers = [
            ("rf", round(rf, 4)),
            ("memory", round(mem, 4)),
            ("counterfactual", round(cs, 4)),
            ("forecast", round(1.0 - rp, 4)),
            ("path", round(pp, 4)),
            ("fingerprint", round(cc, 4)),
        ]
        reason_layers.sort(key=lambda x: x[1], reverse=True)

        override = dq < 0.40

        layer_signals = [rf, mem, max(cs, 0), 1.0 - rp, pp, cc]
        mean_v = sum(layer_signals) / len(layer_signals)
        var_v = sum((v - mean_v) ** 2 for v in layer_signals) / len(layer_signals)
        std_v = math.sqrt(var_v)
        consensus = round(max(1.0 - std_v * 3, 0.0), 4)

        conf = round(1.0 - abs(dq - 0.5) * 2, 4)

        decision_id = f"DEC_{self._decision_id_counter}"
        self._decision_id_counter += 1

        result = {
            "decision_id": decision_id,
            "decision_quality": round(dq, 4),
            "recommendation": rec,
            "confidence": conf,
            "consensus": consensus,
            "override": override,
            "reason_vector": reason_layers,
        }
        self._decisions.append(result)
        logger.info(f"[REFLEXIVE_DECISION] {decision_id} "
                    f"quality={dq:.2f} {rec} conf={conf:.2f}")
        return result

    def observe_outcome(self, decision_id: str, success: bool):
        for d in self._decisions:
            if d["decision_id"] == decision_id:
                self._outcomes[decision_id] = {
                    "decision_id": decision_id,
                    "recommendation": d["recommendation"],
                    "decision_quality": d["decision_quality"],
                    "success": success,
                }
                logger.info(f"[REFLEXIVE_OUTCOME] {decision_id} "
                            f"rec={d['recommendation']} success={success}")
                return

    def disagreement(self, actual_arbitration: str) -> dict:
        last = self._decisions[-1] if self._decisions else None
        if not last:
            return {"disagreement": False, "reflexive": None, "actual": actual_arbitration}
        rec = last["recommendation"]
        disc = rec != actual_arbitration
        logger.info(f"[REFLEXIVE_DISAGREE] reflexive={rec} "
                    f"actual={actual_arbitration} disagree={disc}")
        return {
            "disagreement": disc,
            "reflexive": rec,
            "actual": actual_arbitration,
        }

    def calibration(self) -> dict:
        result = {}
        for rec in RECOMMENDATIONS:
            outcomes = [o for o in self._outcomes.values()
                        if o["recommendation"] == rec]
            if not outcomes:
                continue
            correct = sum(1 for o in outcomes if o["success"])
            result[rec] = round(correct / len(outcomes), 4)
        return result

    def stats(self) -> dict:
        total = len(self._decisions)
        if total == 0:
            return {
                "decisions": 0, "recommendations": {}, "override_rate": 0.0,
                "mean_quality": 0.0, "mean_confidence": 0.0,
                "mean_consensus": 0.0, "outcomes": 0,
            }
        rec_counts = defaultdict(int)
        override_count = 0
        qual_sum = 0.0
        conf_sum = 0.0
        cons_sum = 0.0
        for d in self._decisions:
            rec_counts[d["recommendation"]] += 1
            if d["override"]:
                override_count += 1
            qual_sum += d["decision_quality"]
            conf_sum += d["confidence"]
            cons_sum += d["consensus"]
        return {
            "decisions": total,
            "recommendations": dict(rec_counts),
            "override_rate": round(override_count / total, 4),
            "mean_quality": round(qual_sum / total, 4),
            "mean_confidence": round(conf_sum / total, 4),
            "mean_consensus": round(cons_sum / total, 4),
            "outcomes": len(self._outcomes),
        }
