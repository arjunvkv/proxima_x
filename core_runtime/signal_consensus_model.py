"""
Signal Consensus Model — resolves conflicts between OSS and ALT trading signals.

Supports five configurable strategies:
  MAJORITY, OSS_PRIORITY, ALT_PRIORITY, CONFIDENCE_WEIGHTED, SPLIT
"""

import logging
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------
class _SignalConsensusModel:
    """Resolve OSS / ALT signal conflicts."""

    VALID_STRATEGIES = frozenset({
        "MAJORITY",
        "OSS_PRIORITY",
        "ALT_PRIORITY",
        "CONFIDENCE_WEIGHTED",
        "SPLIT",
    })

    def __init__(self, instance_id: str = "default") -> None:
        self._instance_id = instance_id
        self._strategy: str = "MAJORITY"
        self._confidence_threshold: float = 0.3

        # statistics
        self._total_resolutions: int = 0
        self._agreements: int = 0
        self._conflicts: int = 0
        self._no_trades: int = 0
        self._split_decisions: int = 0
        self._oss_wins: int = 0
        self._alt_wins: int = 0

        logger.info("SignalConsensusModel '%s' initialised (strategy=%s, threshold=%.2f)",
                     instance_id, self._strategy, self._confidence_threshold)

    # -- Public config -------------------------------------------------------

    def set_strategy(self, strategy: str) -> None:
        """Set the conflict resolution strategy."""
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Valid: {sorted(self.VALID_STRATEGIES)}"
            )
        self._strategy = strategy
        logger.info("Strategy set to '%s' on instance '%s'", strategy, self._instance_id)

    def set_confidence_threshold(self, t: float) -> None:
        """Set the minimum absolute net confidence for CONFIDENCE_WEIGHTED mode (default 0.3)."""
        if not 0.0 <= t <= 1.0:
            raise ValueError("Confidence threshold must be in [0.0, 1.0]")
        self._confidence_threshold = t
        logger.info("Confidence threshold set to %.2f on instance '%s'", t, self._instance_id)

    # -- Core resolution -----------------------------------------------------

    def resolve(
        self,
        oss_signal: int,
        oss_confidence: float,
        alt_signal: int,
        alt_confidence: float,
    ) -> Dict[str, Any]:
        """Resolve a single OSS / ALT signal pair.

        Parameters
        ----------
        oss_signal : int
            OSS signal direction: -1, 0, or +1.
        oss_confidence : float
            Confidence of the OSS signal in [0.0, 1.0].
        alt_signal : int
            ALT signal direction: -1, 0, or +1.
        alt_confidence : float
            Confidence of the ALT signal in [0.0, 1.0].

        Returns
        -------
        dict with keys: consensus_signal, oss_input, alt_input, strategy_used,
                        split_position, confidence, explanation.
        """
        self._total_resolutions += 1
        _validate_signal(oss_signal, oss_confidence, "OSS")
        _validate_signal(alt_signal, alt_confidence, "ALT")

        result = self._resolve_internal(oss_signal, oss_confidence, alt_signal, alt_confidence)

        # --- Update statistics ---
        if oss_signal == alt_signal:
            self._agreements += 1
        else:
            self._conflicts += 1

        if result["split_position"]:
            self._split_decisions += 1
        elif result["consensus_signal"] == 0:
            self._no_trades += 1

        # Track who won in conflicts
        if oss_signal != alt_signal:
            if result["consensus_signal"] == oss_signal:
                self._oss_wins += 1
            elif result["consensus_signal"] == alt_signal:
                self._alt_wins += 1

        logger.debug("Resolve[%s] OSS=%d(%.2f) ALT=%d(%.2f) -> %s",
                     self._strategy, oss_signal, oss_confidence,
                     alt_signal, alt_confidence, result)

        return result

    def _resolve_internal(
        self,
        oss_signal: int,
        oss_confidence: float,
        alt_signal: int,
        alt_confidence: float,
    ) -> Dict[str, Any]:
        """Apply the current strategy (no stat recording)."""
        strategy = self._strategy

        if strategy == "MAJORITY":
            return self._resolve_majority(oss_signal, oss_confidence, alt_signal, alt_confidence)
        elif strategy == "OSS_PRIORITY":
            return self._resolve_oss_priority(oss_signal, oss_confidence, alt_signal, alt_confidence)
        elif strategy == "ALT_PRIORITY":
            return self._resolve_alt_priority(oss_signal, oss_confidence, alt_signal, alt_confidence)
        elif strategy == "CONFIDENCE_WEIGHTED":
            return self._resolve_confidence_weighted(oss_signal, oss_confidence, alt_signal, alt_confidence)
        elif strategy == "SPLIT":
            return self._resolve_split(oss_signal, oss_confidence, alt_signal, alt_confidence)
        else:
            # Should never reach here due to set_strategy validation
            raise RuntimeError(f"Unhandled strategy: {strategy}")

    @staticmethod
    def _base_dict(
        oss_signal: int,
        oss_confidence: float,
        alt_signal: int,
        alt_confidence: float,
        strategy: str,
    ) -> Dict[str, Any]:
        return {
            "oss_input": oss_signal,
            "oss_confidence": oss_confidence,
            "alt_input": alt_signal,
            "alt_confidence": alt_confidence,
            "strategy_used": strategy,
        }

    # --- Strategy implementations -------------------------------------------

    def _resolve_majority(
        self, oss: int, oc: float, alt: int, ac: float
    ) -> Dict[str, Any]:
        d = self._base_dict(oss, oc, alt, ac, "MAJORITY")
        if oss == alt:
            d["consensus_signal"] = oss
            d["confidence"] = (oc + ac) / 2.0
            d["split_position"] = False
            d["explanation"] = (
                f"OSS ({oss}) and ALT ({alt}) agree → "
                f"consensus {oss}"
            )
        else:
            d["consensus_signal"] = 0
            d["confidence"] = 0.0
            d["split_position"] = False
            d["explanation"] = (
                f"OSS ({oss}) and ALT ({alt}) disagree → NO TRADE"
            )
        return d

    def _resolve_oss_priority(
        self, oss: int, oc: float, alt: int, ac: float
    ) -> Dict[str, Any]:
        d = self._base_dict(oss, oc, alt, ac, "OSS_PRIORITY")
        d["consensus_signal"] = oss
        d["confidence"] = oc
        d["split_position"] = False
        if oss == alt:
            d["explanation"] = (
                f"OSS ({oss}) and ALT ({alt}) agree → consensus {oss}"
            )
        else:
            d["explanation"] = (
                f"OSS ({oss}) and ALT ({alt}) conflict → OSS priority: {oss}"
            )
        return d

    def _resolve_alt_priority(
        self, oss: int, oc: float, alt: int, ac: float
    ) -> Dict[str, Any]:
        d = self._base_dict(oss, oc, alt, ac, "ALT_PRIORITY")
        d["consensus_signal"] = alt
        d["confidence"] = ac
        d["split_position"] = False
        if oss == alt:
            d["explanation"] = (
                f"OSS ({oss}) and ALT ({alt}) agree → consensus {alt}"
            )
        else:
            d["explanation"] = (
                f"OSS ({oss}) and ALT ({alt}) conflict → ALT priority: {alt}"
            )
        return d

    def _resolve_confidence_weighted(
        self, oss: int, oc: float, alt: int, ac: float
    ) -> Dict[str, Any]:
        d = self._base_dict(oss, oc, alt, ac, "CONFIDENCE_WEIGHTED")
        net = (oss * oc) + (alt * ac)
        abs_net = abs(net)
        threshold = self._confidence_threshold

        if abs_net > threshold:
            consensus = 1 if net > 0 else -1
            d["consensus_signal"] = consensus
            d["confidence"] = abs_net
            d["split_position"] = False
            d["explanation"] = (
                f"OSS ({oss} @ {oc:.2f}) + ALT ({alt} @ {ac:.2f}) = "
                f"net {net:.2f} exceeds threshold {threshold} → consensus {consensus}"
            )
        else:
            d["consensus_signal"] = 0
            d["confidence"] = abs_net
            d["split_position"] = False
            d["explanation"] = (
                f"OSS ({oss} @ {oc:.2f}) + ALT ({alt} @ {ac:.2f}) = "
                f"net {net:.2f} does NOT exceed threshold {threshold} → NO TRADE"
            )
        return d

    def _resolve_split(
        self, oss: int, oc: float, alt: int, ac: float
    ) -> Dict[str, Any]:
        d = self._base_dict(oss, oc, alt, ac, "SPLIT")
        threshold = self._confidence_threshold

        if oss == alt:
            d["consensus_signal"] = oss
            d["confidence"] = (oc + ac) / 2.0
            d["split_position"] = False
            d["explanation"] = (
                f"OSS ({oss}) and ALT ({alt}) agree → consensus {oss}"
            )
            return d

        # They disagree.
        both_high = oc >= threshold and ac >= threshold

        if both_high:
            d["consensus_signal"] = 0
            d["confidence"] = min(oc, ac)
            d["split_position"] = True
            d["explanation"] = (
                f"OSS ({oss} @ {oc:.2f}) and ALT ({alt} @ {ac:.2f}) disagree "
                f"but both >= threshold {threshold} → SPLIT POSITION"
            )
        else:
            # Follow the higher-confidence signal
            if oc >= ac:
                d["consensus_signal"] = oss
                d["confidence"] = oc
            else:
                d["consensus_signal"] = alt
                d["confidence"] = ac
            d["split_position"] = False
            d["explanation"] = (
                f"OSS ({oss} @ {oc:.2f}) and ALT ({alt} @ {ac:.2f}) disagree, "
                f"one below threshold → follow higher-confidence signal {d['consensus_signal']}"
            )
        return d

    # -- Batch ---------------------------------------------------------------

    def resolve_batch(
        self, pairs: List[Tuple[int, float, int, float]]
    ) -> List[Dict[str, Any]]:
        """Resolve a list of (oss_sig, oss_conf, alt_sig, alt_conf) tuples."""
        return [self.resolve(oss_s, oss_c, alt_s, alt_c) for oss_s, oss_c, alt_s, alt_c in pairs]

    # -- Statistics ----------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return accumulated resolution statistics."""
        return {
            "total_resolutions": self._total_resolutions,
            "agreements": self._agreements,
            "conflicts": self._conflicts,
            "no_trades": self._no_trades,
            "split_decisions": self._split_decisions,
            "oss_wins": self._oss_wins,
            "alt_wins": self._alt_wins,
        }

    def reset(self) -> None:
        """Clear all statistics (keeps strategy and threshold)."""
        self._total_resolutions = 0
        self._agreements = 0
        self._conflicts = 0
        self._no_trades = 0
        self._split_decisions = 0
        self._oss_wins = 0
        self._alt_wins = 0
        logger.info("Statistics reset on instance '%s'", self._instance_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _validate_signal(signal: int, confidence: float, label: str) -> None:
    if signal not in (-1, 0, 1):
        raise ValueError(f"{label} signal must be -1, 0, or +1; got {signal}")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"{label} confidence must be in [0.0, 1.0]; got {confidence}")


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------
_instances: Dict[str, _SignalConsensusModel] = {}


def SignalConsensusModel(instance_id: str = "default") -> _SignalConsensusModel:
    """Factory that returns a shared _SignalConsensusModel singleton per instance_id."""
    if instance_id not in _instances:
        _instances[instance_id] = _SignalConsensusModel(instance_id)
    return _instances[instance_id]


# ===========================================================================
# Self-test
# ===========================================================================
def _self_test() -> None:
    """Exercise all five strategies across multiple conflict scenarios."""
    print("=" * 60)
    print("SignalConsensusModel :: Self-Test")
    print("=" * 60)

    scenarios = [
        # (oss_sig, oss_conf, alt_sig, alt_conf, scenario_name)
        (+1, 0.80, +1, 0.70, "Agree BUY"),
        (-1, 0.90, -1, 0.85, "Agree SELL"),
        (+1, 0.80, -1, 0.70, "Disagree BUY(high) vs SELL(high)"),
        (+1, 0.40, -1, 0.85, "Disagree BUY(low) vs SELL(high)"),
        (+1, 0.90, -1, 0.30, "Disagree BUY(high) vs SELL(low)"),
        (+1, 0.20, +1, 0.15, "Agree BUY (low confidence)"),
        (-1, 0.10, -1, 0.05, "Agree SELL (very low)"),
        (0,  0.00, +1, 0.80, "Neutral OSS vs BUY ALT"),
        (+1, 0.80, 0,  0.00, "BUY OSS vs Neutral ALT"),
    ]

    strategies = [
        "MAJORITY",
        "OSS_PRIORITY",
        "ALT_PRIORITY",
        "CONFIDENCE_WEIGHTED",
        "SPLIT",
    ]

    for strategy in strategies:
        print(f"\n--- Strategy: {strategy} ---")
        model = SignalConsensusModel(f"test_{strategy}")
        model.set_strategy(strategy)
        if strategy in ("CONFIDENCE_WEIGHTED", "SPLIT"):
            model.set_confidence_threshold(0.3)

        for oss_s, oss_c, alt_s, alt_c, name in scenarios:
            result = model.resolve(oss_s, oss_c, alt_s, alt_c)
            print(
                f"  {name:40s} -> sig={result['consensus_signal']:2d}  "
                f"conf={result['confidence']:.2f}  "
                f"split={result['split_position']}  |  {result['explanation']}"
            )

        stats = model.get_statistics()
        print(f"  --- Stats: {stats}")

    # -- Batch test ----------------------------------------------------------
    print("\n--- Batch resolve test ---")
    batch_model = SignalConsensusModel("batch_test")
    batch_model.set_strategy("CONFIDENCE_WEIGHTED")
    batch_model.set_confidence_threshold(0.3)
    pairs = [
        (+1, 0.80, -1, 0.70),
        (-1, 0.90, -1, 0.95),
        (+1, 0.60, +1, 0.65),
        (-1, 0.50, +1, 0.45),
    ]
    results = batch_model.resolve_batch(pairs)
    for i, r in enumerate(results):
        print(f"  Pair {i}: oss={r['oss_input']}, alt={r['alt_input']} -> sig={r['consensus_signal']}")

    # -- Reset test ----------------------------------------------------------
    print("\n--- Reset test ---")
    m = SignalConsensusModel("reset_test")
    m.resolve(+1, 0.9, +1, 0.8)
    assert m.get_statistics()["total_resolutions"] == 1
    m.reset()
    assert m.get_statistics()["total_resolutions"] == 0
    print("  reset() OK")

    # -- Singleton test ------------------------------------------------------
    print("\n--- Singleton test ---")
    a = SignalConsensusModel("singleton_test")
    b = SignalConsensusModel("singleton_test")
    assert a is b
    c = SignalConsensusModel("other")
    assert c is not a
    print("  Singleton pattern verified")

    print("\n" + "=" * 60)
    print("All self-tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    _self_test()
