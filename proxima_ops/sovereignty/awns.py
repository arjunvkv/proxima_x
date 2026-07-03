"""AWNS — Authority Weight Normalization System.

Normalize all module outputs into a unified probability distribution.
Sum of all normalized weights = 1.0.
"""

import math
from typing import Dict

_MODULE_KEYS = ["dce", "erf", "loef", "gmci", "aeem", "rfg", "eprg", "tamk"]
_NUM_MODULES = len(_MODULE_KEYS)  # 8
_ENTROPY_THRESHOLD = 0.8


class AuthorityWeightNormalization:
    """Normalize multi-module authority signals into a probability distribution.

    Each input signal is mapped to a raw weight in [0, 1] (some are inverted so
    that higher always means *more authoritative*).  Raw weights are then
    normalized so they sum to 1.0.
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize(
        self,
        dce_confidence: float,
        erf: float,
        loef_density: float,
        gmci_score: float,
        aeem_escape: float,
        rfg: float,
        eprg_reachability: float,
        tamk_authorized: bool,
    ) -> Dict[str, object]:
        """Produce a normalized authority distribution from module signals.

        Parameters
        ----------
        dce_confidence : float
            DCE confidence score in [0, 1].
        erf : float
            ERF score in [0, 1].
        loef_density : float
            LOEF density in [0, 1].
        gmci_score : float
            GMCI score in [0, 1].  **Inverted** internally (1 - score).
        aeem_escape : float
            AEEM escape energy in [0, 1].  **Inverted** internally
            (1 - escape).
        rfg : float
            RFG score in [0, 1].
        eprg_reachability : float
            EPRG reachability in [0, 1].
        tamk_authorized : bool
            Whether TAMK authority is granted.

        Returns
        -------
        dict
            ``normalized_weights`` — all values sum to 1.0.
            ``max_weight_authority`` — the key with the highest weight.
            ``authority_entropy`` — Shannon entropy of the distribution.
            ``dominance_margin`` — highest weight minus second-highest.
            ``conflicting_signals`` — ``True`` when entropy > 0.8.
        """
        try:
            raw = self._build_raw_vector(
                dce_confidence=dce_confidence,
                erf=erf,
                loef_density=loef_density,
                gmci_score=gmci_score,
                aeem_escape=aeem_escape,
                rfg=rfg,
                eprg_reachability=eprg_reachability,
                tamk_authorized=tamk_authorized,
            )

            normalized = self._normalize(raw)
            max_key, max_weight = self._find_max(normalized)
            second_max = self._find_second_max(normalized, max_key)
            entropy = self._shannon_entropy(normalized)
            margin = max_weight - second_max

            return {
                "normalized_weights": normalized,
                "max_weight_authority": max_key,
                "authority_entropy": entropy,
                "dominance_margin": margin,
                "conflicting_signals": entropy > _ENTROPY_THRESHOLD,
            }

        except Exception:
            # Fallback: uniform distribution
            uniform = {k: 1.0 / _NUM_MODULES for k in _MODULE_KEYS}
            return {
                "normalized_weights": uniform,
                "max_weight_authority": _MODULE_KEYS[0],
                "authority_entropy": math.log2(_NUM_MODULES),
                "dominance_margin": 0.0,
                "conflicting_signals": True,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_raw_vector(
        dce_confidence: float,
        erf: float,
        loef_density: float,
        gmci_score: float,
        aeem_escape: float,
        rfg: float,
        eprg_reachability: float,
        tamk_authorized: bool,
    ) -> Dict[str, float]:
        return {
            "dce": dce_confidence,
            "erf": erf,
            "loef": loef_density,
            "gmci": 1.0 - gmci_score,  # inverted: high = good
            "aeem": 1.0 - aeem_escape,  # inverted: low escape = good
            "rfg": rfg,
            "eprg": eprg_reachability,
            "tamk": 1.0 if tamk_authorized else 0.0,
        }

    @staticmethod
    def _normalize(raw: Dict[str, float]) -> Dict[str, float]:
        total = sum(raw.values())
        if total == 0.0:
            return {k: 1.0 / _NUM_MODULES for k in _MODULE_KEYS}
        return {k: v / total for k, v in raw.items()}

    @staticmethod
    def _find_max(
        weights: Dict[str, float],
    ) -> tuple:  # (key, value)
        key = max(weights, key=weights.__getitem__)
        return key, weights[key]

    @staticmethod
    def _find_second_max(
        weights: Dict[str, float], exclude_key: str
    ) -> float:
        best = -1.0
        for k, v in weights.items():
            if k == exclude_key:
                continue
            if v > best:
                best = v
        return best

    @staticmethod
    def _shannon_entropy(weights: Dict[str, float]) -> float:
        """Shannon entropy of a probability distribution (bits)."""
        h = 0.0
        for v in weights.values():
            if v > 0.0:
                h -= v * math.log2(v)
        return h
