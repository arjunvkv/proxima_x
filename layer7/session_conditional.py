"""
DPL-20: Session Conditional TPI Engine.

Conditions all directional layers on market session structure:
  - Asia (00:00-08:00 UTC)
  - London (08:00-17:00 UTC)
  - NY (13:00-22:00 UTC)
  - Overlap (13:00-17:00 UTC)
  - Dead (22:00-24:00 UTC)

Measures accuracy, lift, and topology shifts per session.
"""
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime


def get_session(hour: int) -> str:
    if 0 <= hour < 8:
        return "ASIA"
    elif 8 <= hour < 13:
        return "LONDON"
    elif 13 <= hour < 17:
        return "OVERLAP"
    elif 17 <= hour < 22:
        return "NY"
    else:
        return "DEAD"


SESSIONS = ["ASIA", "LONDON", "OVERLAP", "NY", "DEAD"]


class SessionConditionalEngine:
    def __init__(self):
        self._records: Dict[str, list] = defaultdict(list)

    def record(self, session: str, symbol: str, layer: str,
               predicted_direction: int, actual_direction: int,
               confidence: float = 0.0, score: float = 0.0) -> None:
        correct = (predicted_direction != 0 and predicted_direction == actual_direction)
        self._records[session].append({
            "symbol": symbol, "layer": layer,
            "correct": correct, "confidence": confidence,
            "score": score, "predicted": predicted_direction,
            "actual": actual_direction,
        })

    def accuracy_by_session(self, layer: Optional[str] = None) -> Dict[str, dict]:
        result = {}
        for sess in SESSIONS:
            recs = self._records.get(sess, [])
            if not recs:
                result[sess] = {"n": 0, "accuracy": None}
                continue
            if layer:
                layer_recs = [r for r in recs if r["layer"] == layer]
            else:
                layer_recs = recs
            if not layer_recs:
                result[sess] = {"n": 0, "accuracy": None}
                continue
            correct = sum(1 for r in layer_recs if r["correct"])
            result[sess] = {
                "n": len(layer_recs),
                "accuracy": round(correct / len(layer_recs), 4),
            }
        return result

    def layer_lift(self, base_layer: str, target_layer: str) -> dict:
        """Measure accuracy lift of target_layer over base_layer by session."""
        result = {}
        for sess in SESSIONS:
            base_recs = [r for r in self._records.get(sess, []) if r["layer"] == base_layer]
            target_recs = [r for r in self._records.get(sess, []) if r["layer"] == target_layer]
            if len(base_recs) < 5 or len(target_recs) < 5:
                result[sess] = {"lift": None, "n_base": len(base_recs), "n_target": len(target_recs)}
                continue
            base_acc = sum(1 for r in base_recs if r["correct"]) / len(base_recs)
            target_acc = sum(1 for r in target_recs if r["correct"]) / len(target_recs)
            result[sess] = {
                "base_accuracy": round(base_acc, 4),
                "target_accuracy": round(target_acc, 4),
                "lift": round((target_acc - base_acc) * 100, 2),
            }
        return result

    def best_session(self, layer: str) -> Tuple[str, float]:
        best_sess, best_acc = "NONE", 0.0
        accs = self.accuracy_by_session(layer)
        for sess, a in accs.items():
            if a["accuracy"] is not None and a["accuracy"] > best_acc:
                best_acc = a["accuracy"]
                best_sess = sess
        return best_sess, round(best_acc, 4)

    def summary(self, all_meta_scores: dict = None) -> str:
        lines = []
        lines.append("  DPL-20: SESSION CONDITIONAL")
        lines.append("-" * 52)
        layers = ["tpi", "meta"]
        for layer in layers:
            accs = self.accuracy_by_session(layer)
            best_s, best_a = self.best_session(layer)
            line = f"  {layer.upper():5s}: "
            for sess in SESSIONS:
                a = accs.get(sess, {})
                if a.get("accuracy") is not None:
                    line += f"{sess[:4]}={a['accuracy']:.0%} "
                else:
                    line += f"{sess[:4]}=? "
            line += f"  best={best_s}({best_a:.0%})"
            lines.append(line)

        # Meta accuracy with enough samples per session
        lines.append("")
        lines.append("  Per-session accuracy (meta > 0 samples):")
        lines.append(f"  {'Session':<10s} {'n':<5s} {'Acc':<8s} {'BestLayer':<12s}")
        for sess in SESSIONS:
            recs = self._records.get(sess, [])
            n = len(recs)
            if n == 0:
                lines.append(f"  {sess:<10s} {'0':<5s} {'?':<8s} {'?':<12s}")
                continue
            meta_recs = [r for r in recs if r["layer"] == "meta"]
            if meta_recs:
                acc = sum(1 for r in meta_recs if r["correct"]) / len(meta_recs)
                lines.append(f"  {sess:<10s} {len(meta_recs):<5d} {acc:.1%}    {'META':<12s}")
            tpi_recs = [r for r in recs if r["layer"] == "tpi"]
            if not meta_recs and tpi_recs:
                acc = sum(1 for r in tpi_recs if r["correct"]) / len(tpi_recs)
                lines.append(f"  {sess:<10s} {len(tpi_recs):<5d} {acc:.1%}    {'TPI':<12s}")
        lines.append("")
        lines.append("  Best sessions per layer:")
        for layer in ["tpi", "persistence", "meta", "pressure"]:
            best_s, best_a = self.best_session(layer)
            if best_a > 0:
                lines.append(f"    {layer:<15s} {best_s:<8s} {best_a:.1%}")
        return "\n".join(lines)
