"""
Layer Weight Dynamics Controller — dynamically adjusts the influence weights
of all four cognitive layers (SDIL, CSRF, SAAL, MRSRL) based on real-time
market regime and layer state.

Weights are NOT fixed between layers:
  - SAAL weight increases in stable regimes
  - SDIL weight increases in volatility / collapse spikes
  - MRSRL weight increases in noise regimes
  - CSRF weight increases when OSS/ALT truth sources are reliable

This module is part of the Unified Execution Synthesis Layer (UESL) — the
system that resolves disagreement between all cognitive layers in real time.

Usage
-----
    from core_runtime.layer_weight_dynamics_controller import (
        LayerWeightDynamicsController,
    )

    controller = LayerWeightDynamicsController()
    new_weights = controller.update_weights(
        sdil_state={"collapse_detected": True, "entropy": "HIGH"},
        csfr_signal={"truth_source": "OSS", "accuracy": 0.85},
        saal_authority={"authority_stable": True},
        mrsrl_resolution={"regime": "NOISE"},
    )
    print(controller.get_weights())
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton registry
# ---------------------------------------------------------------------------

_instances: Dict[str, "_LayerWeightDynamicsController"] = {}


def LayerWeightDynamicsController(instance_id: str = "default"):
    """Singleton accessor — returns the same ``_LayerWeightDynamicsController``
    for a given *instance_id*.

    Parameters
    ----------
    instance_id : str
        Logical identifier (default ``"default"``).

    Returns
    -------
    _LayerWeightDynamicsController
    """
    if instance_id not in _instances:
        logger.info(
            "Creating new LayerWeightDynamicsController instance '%s'",
            instance_id,
        )
        _instances[instance_id] = _LayerWeightDynamicsController(instance_id)
    return _instances[instance_id]


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------


class _LayerWeightDynamicsController:
    """Dynamically adjusts layer influence weights based on layer states.

    Parameters
    ----------
    instance_id : str
        Logical instance identifier (used for logging and singleton lookup).
    """

    # Layer names in canonical order
    LAYER_NAMES = ["sdil", "csfr", "saal", "mrsrl"]

    # Weight bounds per layer
    WEIGHT_BOUNDS: Dict[str, tuple] = {
        "sdil": (0.1, 0.5),
        "csfr": (0.1, 0.4),
        "saal": (0.15, 0.5),
        "mrsrl": (0.05, 0.35),
    }

    # Default weights
    DEFAULT_WEIGHTS: Dict[str, float] = {
        "sdil": 0.25,
        "csfr": 0.25,
        "saal": 0.30,
        "mrsrl": 0.20,
    }

    # Adjustment deltas
    SDIL_COLLAPSE_DELTA = 0.15
    SDIL_LOW_ENTROPY_DELTA = -0.05
    CSRF_TRUTH_DELTA = 0.10
    CSRF_NEITHER_DELTA = -0.10
    SAAL_STABLE_DELTA = 0.10
    SAAL_UNSTABLE_DELTA = -0.10
    MRSRL_NOISE_DELTA = 0.15
    MRSRL_TREND_DELTA = -0.05

    # Anti-oscillation window
    ANTI_OSCILLATION_WINDOW = 3

    def __init__(self, instance_id: str = "default"):
        self._instance_id = instance_id

        # ── Current weights (deep copy of defaults) ────────────────────
        self._weights: Dict[str, float] = copy.deepcopy(self.DEFAULT_WEIGHTS)
        self._previous_weights: Dict[str, float] = copy.deepcopy(
            self.DEFAULT_WEIGHTS
        )

        # ── Parameters ──────────────────────────────────────────────────
        self._stability_factor: float = 0.95
        self._adaptation_rate: float = 0.1

        # ── Weight history (last 100 states) ────────────────────────────
        self._weight_history: List[Dict[str, float]] = [
            copy.deepcopy(self.DEFAULT_WEIGHTS)
        ]

        # ── Direction tracking for anti-oscillation ─────────────────────
        # Each entry: {layer_name: +1 (up), -1 (down), 0 (no change)}
        self._direction_history: List[Dict[str, int]] = []

        logger.info(
            "LayerWeightDynamicsController(%r) initialised: weights=%s "
            "stability=%.2f adaptation=%.2f",
            instance_id,
            self._weights,
            self._stability_factor,
            self._adaptation_rate,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_weights(
        self,
        sdil_state: dict,
        csfr_signal: dict,
        saal_authority: dict,
        mrsrl_resolution: dict,
    ) -> Dict[str, float]:
        """Adjust layer weights based on current layer states.

        Parameters
        ----------
        sdil_state : dict
            Expected keys: ``collapse_detected`` (bool), ``entropy`` (str).
        csfr_signal : dict
            Expected keys: ``truth_source`` (str), ``accuracy`` (float, optional).
        saal_authority : dict
            Expected keys: ``authority_stable`` (bool).
        mrsrl_resolution : dict
            Expected keys: ``regime`` (str).

        Returns
        -------
        dict
            New weights after adjustment and normalisation.
            Keys: ``"sdil"``, ``"csfr"``, ``"saal"``, ``"mrsrl"``.
        """
        # ── Record previous weights for anti-oscillation ────────────────
        self._previous_weights = copy.deepcopy(self._weights)

        raw_weights = copy.deepcopy(self._weights)

        # ── 1. SDIL weight adjustment ───────────────────────────────────
        collapse = sdil_state.get("collapse_detected", False)
        entropy = sdil_state.get("entropy", "MODERATE")

        if collapse:
            raw_weights["sdil"] += self.SDIL_COLLAPSE_DELTA
        if entropy == "LOW":
            raw_weights["sdil"] += self.SDIL_LOW_ENTROPY_DELTA

        # ── 2. CSRF weight adjustment ───────────────────────────────────
        truth_source = csfr_signal.get("truth_source", "NEITHER")
        csfr_accuracy = csfr_signal.get("accuracy", 0.5)

        if truth_source in ("OSS", "ALT") and csfr_accuracy > 0.6:
            raw_weights["csfr"] += self.CSRF_TRUTH_DELTA
        elif truth_source == "NEITHER":
            raw_weights["csfr"] += self.CSRF_NEITHER_DELTA

        # ── 3. SAAL weight adjustment ───────────────────────────────────
        authority_stable = saal_authority.get("authority_stable", False)

        if authority_stable:
            raw_weights["saal"] += self.SAAL_STABLE_DELTA
        else:
            raw_weights["saal"] += self.SAAL_UNSTABLE_DELTA

        # ── 4. MRSRL weight adjustment ──────────────────────────────────
        regime = mrsrl_resolution.get("regime", "UNKNOWN")

        if regime == "NOISE":
            raw_weights["mrsrl"] += self.MRSRL_NOISE_DELTA
        elif regime == "MACRO_TREND":
            raw_weights["mrsrl"] += self.MRSRL_TREND_DELTA

        # ── 5. Clamp each weight to its bounds ──────────────────────────
        for layer in self.LAYER_NAMES:
            lo, hi = self.WEIGHT_BOUNDS[layer]
            raw_weights[layer] = max(lo, min(hi, raw_weights[layer]))

        # ── 6. Normalise to sum to 1.0 ──────────────────────────────────
        total = sum(raw_weights.values())
        if total > 0:
            for layer in self.LAYER_NAMES:
                raw_weights[layer] /= total

        # ── 7. Apply anti-oscillation ───────────────────────────────────
        # Determine direction of change for each layer
        directions: Dict[str, int] = {}
        for layer in self.LAYER_NAMES:
            delta = raw_weights[layer] - self._previous_weights[layer]
            if delta > 1e-6:
                directions[layer] = 1   # up
            elif delta < -1e-6:
                directions[layer] = -1  # down
            else:
                directions[layer] = 0   # unchanged

        self._direction_history.append(directions)

        # Check if direction changed in last N ticks
        if len(self._direction_history) >= self.ANTI_OSCILLATION_WINDOW:
            recent = self._direction_history[
                -self.ANTI_OSCILLATION_WINDOW:
            ]
            for layer in self.LAYER_NAMES:
                # Extract direction sequence for this layer
                seq = [d.get(layer, 0) for d in recent]
                # Check for alternation: +1, -1, +1 or -1, +1, -1
                if self._is_oscillating(seq):
                    # Halve the change magnitude
                    adj = (raw_weights[layer] - self._previous_weights[layer]) / 2.0
                    raw_weights[layer] = self._previous_weights[layer] + adj
                    logger.debug(
                        "Anti-oscillation applied to %s: new weight=%.4f",
                        layer, raw_weights[layer],
                    )

        # ── 8. Apply stability factor (smooth changes) ──────────────────
        stable_weights: Dict[str, float] = {}
        for layer in self.LAYER_NAMES:
            sf = self._stability_factor
            stable_weights[layer] = (
                sf * raw_weights[layer]
                + (1.0 - sf) * self._previous_weights[layer]
            )

        # ── 9. Renormalise after anti-oscillation / stabilisation ──────
        total_s = sum(stable_weights.values())
        if total_s > 0:
            for layer in self.LAYER_NAMES:
                stable_weights[layer] /= total_s

        # ── Store ────────────────────────────────────────────────────────
        self._weights = stable_weights

        # ── Record history (keep last 100) ───────────────────────────────
        self._weight_history.append(copy.deepcopy(self._weights))
        if len(self._weight_history) > 100:
            self._weight_history = self._weight_history[-100:]

        # Trim direction history (keep last 10)
        if len(self._direction_history) > 10:
            self._direction_history = self._direction_history[-10:]

        logger.debug(
            "Weights updated: %s", self._weights,
        )
        return dict(self._weights)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_weights(self) -> Dict[str, float]:
        """Return current layer weights.

        Returns
        -------
        dict
            Keys: ``"sdil"``, ``"csfr"``, ``"saal"``, ``"mrsrl"``.
        """
        return dict(self._weights)

    def get_weight_history(self) -> List[Dict[str, float]]:
        """Return the last 100 weight states.

        Returns
        -------
        list of dict
            Each entry is a weight dict as returned by ``get_weights()``.
        """
        return list(self._weight_history)

    def reset(self) -> None:
        """Reset weights to defaults and clear all history."""
        self._weights = copy.deepcopy(self.DEFAULT_WEIGHTS)
        self._previous_weights = copy.deepcopy(self.DEFAULT_WEIGHTS)
        self._weight_history = [copy.deepcopy(self.DEFAULT_WEIGHTS)]
        self._direction_history.clear()
        logger.info(
            "LayerWeightDynamicsController(%r) reset to defaults: %s",
            self._instance_id, self._weights,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_oscillating(seq: List[int]) -> bool:
        """Detect if a direction sequence is oscillating (alternating).

        Examples of oscillating:
            [1, -1, 1]   → yes
            [-1, 1, -1]  → yes
            [1, 1, -1]   → no (two same in a row)
            [0, 1, -1]   → no (zero in middle)
            [-1, -1, 1]  → no

        Parameters
        ----------
        seq : list of int
            Direction values: +1 (up), -1 (down), 0 (unchanged).

        Returns
        -------
        bool
            True if the sequence shows an alternation pattern.
        """
        if len(seq) < 3:
            return False
        # Filter out zeros (no-change steps)
        filtered = [s for s in seq if s != 0]
        if len(filtered) < 3:
            return False
        # Check the last 3 non-zero values
        last_three = filtered[-3:]
        # Oscillation if values strictly alternate
        return (
            last_three[0] != 0
            and last_three[1] != 0
            and last_three[2] != 0
            and last_three[0] == -last_three[1]
            and last_three[1] == -last_three[2]
        )

    def __repr__(self) -> str:
        return (
            f"LayerWeightDynamicsController('{self._instance_id}', "
            f"weights={self._weights})"
        )


# ===========================================================================
# Self-test
# ===========================================================================

def _run_self_test() -> None:
    """Run multiple scenarios to verify weight adjustment logic."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("LayerWeightDynamicsController — Self-Test")
    print("=" * 60)

    _state = {"passed": True}

    def _check(cond: bool, msg: str) -> None:
        if not cond:
            _state["passed"] = False
            print(f"  FAIL: {msg}")
        else:
            print(f"  PASS: {msg}")

    def _round_weights(w: dict) -> dict:
        return {k: round(v, 4) for k, v in w.items()}

    # Helper to verify weights sum to 1.0 (within tolerance)
    def _sum_is_one(w: dict, tolerance: float = 1e-4) -> bool:
        return abs(sum(w.values()) - 1.0) < tolerance

    # ------------------------------------------------------------------
    # Scenario 1: Stable regime — SAAL should increase
    # ------------------------------------------------------------------
    print("\n--- SCE 1: Stable regime → SAAL increases ---")
    ctrl1 = LayerWeightDynamicsController("selftest_1")
    ctrl1.reset()

    w1 = ctrl1.update_weights(
        sdil_state={"collapse_detected": False, "entropy": "MODERATE"},
        csfr_signal={"truth_source": "OSS", "accuracy": 0.80},
        saal_authority={"authority_stable": True},
        mrsrl_resolution={"regime": "MESO_STRUCTURE"},
    )
    print(f"  weights = {_round_weights(w1)}")
    _check(
        w1["saal"] > ctrl1.DEFAULT_WEIGHTS["saal"],
        f"SAAL weight increased ({w1['saal']:.4f} > "
        f"{ctrl1.DEFAULT_WEIGHTS['saal']})",
    )
    _check(
        _sum_is_one(w1),
        f"Weights sum to 1.0 (sum={sum(w1.values()):.6f})",
    )

    # ------------------------------------------------------------------
    # Scenario 2: Collapse detected — SDIL should increase
    # ------------------------------------------------------------------
    print("\n--- SCE 2: Collapse detected → SDIL increases ---")
    ctrl2 = LayerWeightDynamicsController("selftest_2")
    ctrl2.reset()

    w2 = ctrl2.update_weights(
        sdil_state={"collapse_detected": True, "entropy": "HIGH"},
        csfr_signal={"truth_source": "NEITHER", "accuracy": 0.3},
        saal_authority={"authority_stable": False},
        mrsrl_resolution={"regime": "MACRO_TREND"},
    )
    print(f"  weights = {_round_weights(w2)}")
    _check(
        w2["sdil"] > ctrl2.DEFAULT_WEIGHTS["sdil"],
        f"SDIL weight increased ({w2['sdil']:.4f} > "
        f"{ctrl2.DEFAULT_WEIGHTS['sdil']})",
    )
    _check(
        _sum_is_one(w2),
        f"Weights sum to 1.0 (sum={sum(w2.values()):.6f})",
    )

    # ------------------------------------------------------------------
    # Scenario 3: Noise regime — MRSRL should increase
    # ------------------------------------------------------------------
    print("\n--- SCE 3: Noise regime → MRSRL increases ---")
    ctrl3 = LayerWeightDynamicsController("selftest_3")
    ctrl3.reset()

    w3 = ctrl3.update_weights(
        sdil_state={"collapse_detected": False, "entropy": "HIGH"},
        csfr_signal={"truth_source": "NEITHER", "accuracy": 0.3},
        saal_authority={"authority_stable": False},
        mrsrl_resolution={"regime": "NOISE"},
    )
    print(f"  weights = {_round_weights(w3)}")
    _check(
        w3["mrsrl"] > ctrl3.DEFAULT_WEIGHTS["mrsrl"],
        f"MRSRL weight increased ({w3['mrsrl']:.4f} > "
        f"{ctrl3.DEFAULT_WEIGHTS['mrsrl']})",
    )
    _check(
        _sum_is_one(w3),
        f"Weights sum to 1.0 (sum={sum(w3.values()):.6f})",
    )

    # ------------------------------------------------------------------
    # Scenario 4: Full regime sequence — verify dynamic adaptation
    # ------------------------------------------------------------------
    print("\n--- SCE 4: Regime sequence (stable → collapse → noise) ---")
    ctrl4 = LayerWeightDynamicsController("selftest_4")
    ctrl4.reset()

    # Phase A: Stable regime
    print("  Phase A: Stable")
    wa = ctrl4.update_weights(
        sdil_state={"collapse_detected": False, "entropy": "LOW"},
        csfr_signal={"truth_source": "OSS", "accuracy": 0.90},
        saal_authority={"authority_stable": True},
        mrsrl_resolution={"regime": "MESO_STRUCTURE"},
    )
    print(f"    weights = {_round_weights(wa)}")
    _check(
        wa["saal"] > ctrl4.DEFAULT_WEIGHTS["saal"],
        f"Phase A: SAAL increased ({wa['saal']:.4f})",
    )
    _check(
        wa["sdil"] < ctrl4.DEFAULT_WEIGHTS["sdil"],
        f"Phase A: SDIL decreased due to LOW entropy "
        f"({wa['sdil']:.4f} < {ctrl4.DEFAULT_WEIGHTS['sdil']})",
    )

    # Phase B: Collapse
    print("  Phase B: Collapse spike")
    wb = ctrl4.update_weights(
        sdil_state={"collapse_detected": True, "entropy": "HIGH"},
        csfr_signal={"truth_source": "NEITHER", "accuracy": 0.2},
        saal_authority={"authority_stable": False},
        mrsrl_resolution={"regime": "MICRO_STRUCTURE"},
    )
    print(f"    weights = {_round_weights(wb)}")
    _check(
        wb["sdil"] > wa["sdil"],
        f"Phase B: SDIL weight increased ({wb['sdil']:.4f} > {wa['sdil']:.4f})",
    )
    _check(
        _sum_is_one(wb),
        f"Phase B: Weights sum to 1.0 (sum={sum(wb.values()):.6f})",
    )

    # Phase C: Noise regime
    print("  Phase C: Noise regime")
    wc = ctrl4.update_weights(
        sdil_state={"collapse_detected": False, "entropy": "HIGH"},
        csfr_signal={"truth_source": "NEITHER", "accuracy": 0.2},
        saal_authority={"authority_stable": False},
        mrsrl_resolution={"regime": "NOISE"},
    )
    print(f"    weights = {_round_weights(wc)}")
    _check(
        wc["mrsrl"] > wb["mrsrl"],
        f"Phase C: MRSRL weight increased ({wc['mrsrl']:.4f} > {wb['mrsrl']:.4f})",
    )
    _check(
        _sum_is_one(wc),
        f"Phase C: Weights sum to 1.0 (sum={sum(wc.values()):.6f})",
    )

    # Verify history
    history4 = ctrl4.get_weight_history()
    _check(
        len(history4) == 4,  # init + 3 updates
        f"History has 4 entries, got {len(history4)}",
    )

    # ------------------------------------------------------------------
    # Scenario 5: Bounds clamping
    # ------------------------------------------------------------------
    print("\n--- SCE 5: Weight bounds clamping ---")
    ctrl5 = LayerWeightDynamicsController("selftest_5")
    ctrl5.reset()

    # Aggressively push weights to extremes
    for _ in range(10):
        ctrl5.update_weights(
            sdil_state={"collapse_detected": True, "entropy": "LOW"},
            csfr_signal={"truth_source": "OSS", "accuracy": 0.95},
            saal_authority={"authority_stable": True},
            mrsrl_resolution={"regime": "NOISE"},
        )

    w5 = ctrl5.get_weights()
    print(f"  weights after 10 aggressive updates = {_round_weights(w5)}")
    _check(
        ctrl5.WEIGHT_BOUNDS["sdil"][0]
        <= w5["sdil"]
        <= ctrl5.WEIGHT_BOUNDS["sdil"][1],
        f"SDIL weight {w5['sdil']:.4f} within [{ctrl5.WEIGHT_BOUNDS['sdil'][0]}, "
        f"{ctrl5.WEIGHT_BOUNDS['sdil'][1]}]",
    )
    _check(
        ctrl5.WEIGHT_BOUNDS["csfr"][0]
        <= w5["csfr"]
        <= ctrl5.WEIGHT_BOUNDS["csfr"][1],
        f"CSFR weight {w5['csfr']:.4f} within bounds",
    )
    _check(
        ctrl5.WEIGHT_BOUNDS["saal"][0]
        <= w5["saal"]
        <= ctrl5.WEIGHT_BOUNDS["saal"][1],
        f"SAAL weight {w5['saal']:.4f} within bounds",
    )
    _check(
        ctrl5.WEIGHT_BOUNDS["mrsrl"][0]
        <= w5["mrsrl"]
        <= ctrl5.WEIGHT_BOUNDS["mrsrl"][1],
        f"MRSRL weight {w5['mrsrl']:.4f} within bounds",
    )
    _check(
        _sum_is_one(w5),
        f"Weights sum to 1.0 after clamping (sum={sum(w5.values()):.6f})",
    )

    # ------------------------------------------------------------------
    # Scenario 6: Anti-oscillation detection
    # ------------------------------------------------------------------
    print("\n--- SCE 6: Anti-oscillation ---")
    ctrl6 = LayerWeightDynamicsController("selftest_6")
    ctrl6.reset()

    # Force oscillation by alternating between two states
    oscillating_state_a = {
        "sdil": {"collapse_detected": True, "entropy": "HIGH"},
        "csfr": {"truth_source": "OSS", "accuracy": 0.85},
        "saal": {"authority_stable": True},
        "mrsrl": {"regime": "NOISE"},
    }
    oscillating_state_b = {
        "sdil": {"collapse_detected": False, "entropy": "LOW"},
        "csfr": {"truth_source": "NEITHER", "accuracy": 0.3},
        "saal": {"authority_stable": False},
        "mrsrl": {"regime": "MACRO_TREND"},
    }

    # Record weights before oscillation
    w_before = ctrl6.get_weights()

    # Run alternating updates to trigger anti-oscillation
    for i in range(5):
        state_a = (i % 2 == 0)
        s = oscillating_state_a if state_a else oscillating_state_b
        ctrl6.update_weights(
            sdil_state=s["sdil"],
            csfr_signal=s["csfr"],
            saal_authority=s["saal"],
            mrsrl_resolution=s["mrsrl"],
        )

    w_after = ctrl6.get_weights()
    print(f"  weights before oscillation = {_round_weights(w_before)}")
    print(f"  weights after oscillation  = {_round_weights(w_after)}")
    _check(
        _sum_is_one(w_after),
        f"Weights sum to 1.0 (sum={sum(w_after.values()):.6f})",
    )
    # The weights should still be valid and not stuck at extremes
    _check(
        all(
            ctrl6.WEIGHT_BOUNDS[layer][0]
            <= w_after[layer]
            <= ctrl6.WEIGHT_BOUNDS[layer][1]
            for layer in ctrl6.LAYER_NAMES
        ),
        "All weights within bounds after oscillation",
    )

    # ------------------------------------------------------------------
    # Scenario 7: get_weights() and get_weight_history()
    # ------------------------------------------------------------------
    print("\n--- SCE 7: Introspection ---")
    ctrl7 = LayerWeightDynamicsController("selftest_7")
    ctrl7.reset()

    weights7 = ctrl7.get_weights()
    _check(
        weights7 == ctrl7.DEFAULT_WEIGHTS,
        "get_weights returns defaults after reset",
    )

    ctrl7.update_weights(
        sdil_state={"collapse_detected": False, "entropy": "MODERATE"},
        csfr_signal={"truth_source": "OSS", "accuracy": 0.80},
        saal_authority={"authority_stable": True},
        mrsrl_resolution={"regime": "MESO_STRUCTURE"},
    )

    hist7 = ctrl7.get_weight_history()
    _check(
        len(hist7) == 2,  # init + 1 update
        f"History has 2 entries, got {len(hist7)}",
    )
    _check(
        hist7[0] == ctrl7.DEFAULT_WEIGHTS,
        "First history entry is default weights",
    )
    _check(
        hist7[1] != ctrl7.DEFAULT_WEIGHTS,
        "Second history entry differs from defaults",
    )

    # ------------------------------------------------------------------
    # Scenario 8: History capped at 100
    # ------------------------------------------------------------------
    print("\n--- SCE 8: History cap at 100 ---")
    ctrl8 = LayerWeightDynamicsController("selftest_8")
    ctrl8.reset()

    for _ in range(120):
        ctrl8.update_weights(
            sdil_state={"collapse_detected": True, "entropy": "HIGH"},
            csfr_signal={"truth_source": "OSS", "accuracy": 0.85},
            saal_authority={"authority_stable": False},
            mrsrl_resolution={"regime": "NOISE"},
        )

    hist8 = ctrl8.get_weight_history()
    _check(
        len(hist8) <= 100,
        f"History length {len(hist8)} <= 100",
    )
    _check(
        len(hist8) == 100,
        f"History length exactly 100, got {len(hist8)}",
    )

    # ------------------------------------------------------------------
    # Scenario 9: reset()
    # ------------------------------------------------------------------
    print("\n--- SCE 9: reset() ---")
    ctrl9 = LayerWeightDynamicsController("selftest_9")
    ctrl9.reset()

    ctrl9.update_weights(
        sdil_state={"collapse_detected": True, "entropy": "LOW"},
        csfr_signal={"truth_source": "ALT", "accuracy": 0.90},
        saal_authority={"authority_stable": True},
        mrsrl_resolution={"regime": "NOISE"},
    )
    ctrl9.reset()

    w9 = ctrl9.get_weights()
    _check(
        w9 == ctrl9.DEFAULT_WEIGHTS,
        "Weights return to defaults after reset",
    )
    hist9 = ctrl9.get_weight_history()
    _check(
        len(hist9) == 1,
        f"History has 1 entry after reset, got {len(hist9)}",
    )
    _check(
        hist9[0] == ctrl9.DEFAULT_WEIGHTS,
        "History entry is defaults after reset",
    )

    # ------------------------------------------------------------------
    # Scenario 10: Singleton identity
    # ------------------------------------------------------------------
    print("\n--- SCE 10: Singleton identity ---")
    ctrl_default_1 = LayerWeightDynamicsController()
    ctrl_default_2 = LayerWeightDynamicsController("default")
    ctrl_other = LayerWeightDynamicsController("other")

    _check(
        ctrl_default_1 is ctrl_default_2,
        "Default singleton identity",
    )
    _check(
        ctrl_other is not ctrl_default_1,
        "Different instance_id returns different object",
    )
    _check(
        ctrl1 is LayerWeightDynamicsController("selftest_1"),
        "Same instance_id returns same object",
    )

    # ------------------------------------------------------------------
    # Scenario 11: Stability factor smoothing
    # ------------------------------------------------------------------
    print("\n--- SCE 11: Stability factor smoothing ---")
    ctrl11 = LayerWeightDynamicsController("selftest_11")
    ctrl11.reset()

    # First update pushes weights in one direction
    w11a = ctrl11.update_weights(
        sdil_state={"collapse_detected": True, "entropy": "HIGH"},
        csfr_signal={"truth_source": "OSS", "accuracy": 0.85},
        saal_authority={"authority_stable": True},
        mrsrl_resolution={"regime": "NOISE"},
    )
    # The stability factor should mean the new weights are a blend
    # of the raw adjustment and the previous weights
    print(f"  After update 1: {_round_weights(w11a)}")
    _check(
        _sum_is_one(w11a),
        f"Weights sum to 1.0 (sum={sum(w11a.values()):.6f})",
    )

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    if _state["passed"]:
        print("ALL SELF-TESTS PASSED")
        sys.exit(0)
    else:
        print("SOME SELF-TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    _run_self_test()
