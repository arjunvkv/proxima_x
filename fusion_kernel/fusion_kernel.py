from typing import Dict, Any


class SignalFusionKernel:
    """
    V4 Directional Intelligence Layer

    Produces regime-aware directional signal per symbol.
    """

    def __init__(self,
                 entropy_flip_threshold: float = 0.65,
                 coherence_penalty: float = 0.15):
        self.entropy_flip_threshold = entropy_flip_threshold
        self.coherence_penalty = coherence_penalty

        self._last_signals: Dict[str, int] = {}
        self._exhaustion_hist: Dict[str, dict] = {}

    def _regime(self, eval_data: Dict[str, Dict[str, Any]]) -> str:
        entropies = [
            float(v.get("entropy", 0.5)) for v in eval_data.values()
        ]

        if not entropies:
            return "NEUTRAL"

        avg_entropy = sum(entropies) / len(entropies)

        if avg_entropy > 0.65:
            return "CHAOTIC"
        elif avg_entropy < 0.4:
            return "STRUCTURED"
        else:
            return "TRANSITION"

    def _detect_exhaustion(self, sym: str, ecdf: float, topo: dict) -> dict:
        result = {
            "status": "ACTIVE",
            "exhausted": False,
            "direction": 0,
            "score": 0.0,
            "reason": "NONE",
            "near_miss": {"ecdf_fail": False, "entropy_fail": False, "dH_fail": False, "dp_fail": False}
        }
        _entropy = topo.get("entropy", 0.5)
        _d_entropy = topo.get("d_entropy", 0)
        _d_pmax = topo.get("d_pmax", 0)
        if ecdf >= 0.80 and _entropy >= 0.88 and _d_entropy >= 0.0 and _d_pmax <= -0.010:
            result["exhausted"] = True
            result["direction"] = -1
            result["score"] = min(1.0, max(0.0, (ecdf - 0.80) / 0.20))
            result["reason"] = "SELL_EXHAUST"
        elif ecdf <= 0.20 and _entropy >= 0.88 and _d_entropy >= 0.0 and _d_pmax <= -0.010:
            result["exhausted"] = True
            result["direction"] = 1
            result["score"] = min(1.0, max(0.0, (0.20 - ecdf) / 0.20))
            result["reason"] = "BUY_EXHAUST"
        if not result["exhausted"]:
            nm = result["near_miss"]
            if not (ecdf >= 0.80 or ecdf <= 0.20):
                nm["ecdf_fail"] = True
            if not (_entropy >= 0.88):
                nm["entropy_fail"] = True
            if not (_d_entropy >= 0.0):
                nm["dH_fail"] = True
            if not (_d_pmax <= -0.010):
                nm["dp_fail"] = True
        self._exhaustion_hist[sym] = result
        return result

    def _base_signal(self, sym_data: Dict[str, Any]) -> int:
        ecdf = float(sym_data.get("ecdf_rank", 0.5))
        entropy = float(sym_data.get("entropy", 0.5))

        score = ecdf - entropy

        if score > 0.05:
            return 1
        elif score < -0.05:
            return -1
        else:
            return 0

    def _apply_flip_suppression(self,
                                sym: str,
                                signal: int,
                                entropy: float) -> int:
        prev = self._last_signals.get(sym, 0)

        # Suppress flips only: when a signal exists AND would flip from prev
        # Allow births (prev=0, signal!=0) to pass through
        if entropy > self.entropy_flip_threshold and prev != 0 and signal != 0 and signal != prev:
            return prev

        return signal

    def _apply_coherence_filter(self,
                               signals: Dict[str, int],
                               eval_data: Dict[str, Dict]) -> Dict[str, int]:
        # PHASE A: coherence metadata only — no signal overwrite
        # Original signals pass through unchanged
        return signals

    def generate(self,
                 eval_data: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
        regime = self._regime(eval_data)

        raw_signals = {}

        for sym, data in eval_data.items():
            sig = self._base_signal(data)
            sig = self._apply_flip_suppression(sym, sig, float(data.get("entropy", 0.5)))

            raw_signals[sym] = sig
            self._last_signals[sym] = sig

        if regime == "CHAOTIC":
            self._chaotic_strength = 0.7
            # Do NOT mutate raw signal polarity — shadow is a discrete state engine

        final = self._apply_coherence_filter(raw_signals, eval_data)

        return final
