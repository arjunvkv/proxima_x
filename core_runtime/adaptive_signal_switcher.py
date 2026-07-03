"""
Adaptive Signal Switcher — automatically switches between signal generation
modes based on the current market regime.

For ALT: switches between ZSCORE_BREAKOUT, MOMENTUM, and ADAPTIVE_EMA modes.
For OSS: switches between RAW and SMOOTHED surface modes.
"""

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton registry
# ---------------------------------------------------------------------------

_instances = {}


def AdaptiveSignalSwitcher(instance_id="default"):
    """Singleton accessor — returns the same ``_AdaptiveSignalSwitcher``
    for a given *instance_id*.

    Parameters
    ----------
    instance_id : str
        Logical identifier (default ``"default"``).

    Returns
    -------
    _AdaptiveSignalSwitcher
    """
    if instance_id not in _instances:
        _instances[instance_id] = _AdaptiveSignalSwitcher(instance_id)
    return _instances[instance_id]


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------

# Regime-to-mode mapping
_REGIME_ALT_MAP = {
    "NOISE": "ZSCORE_BREAKOUT",
    "MICRO_STRUCTURE": "MOMENTUM",
    "MESO_STRUCTURE": "ADAPTIVE_EMA",
    "MACRO_TREND": "ADAPTIVE_EMA",
}

_REGIME_OSS_MAP = {
    "NOISE": "SMOOTHED",
    "MICRO_STRUCTURE": "RAW",
    "MESO_STRUCTURE": "RAW",
    "MACRO_TREND": "RAW",
}

# Low-viability override (only applies to NOISE)
_LOW_VIABILITY_ALT = "NO_SIGNAL"
_LOW_VIABILITY_OSS = "FLAT"

# Anti-flap threshold
_ANTI_FLAP_THRESHOLD = 3


class _AdaptiveSignalSwitcher:
    """Automatically selects ALT and OSS modes per market regime.

    Parameters
    ----------
    instance_id : str
        Logical instance identifier.
    """

    def __init__(self, instance_id="default"):
        self._instance_id = instance_id

        # Registered mode sets
        self._alt_modes = ["ZSCORE_BREAKOUT", "MOMENTUM", "ADAPTIVE_EMA", "NO_SIGNAL"]
        self._oss_modes = ["RAW", "SMOOTHED", "FLAT"]

        # Current active modes (defaults)
        self._current_alt_mode = "ZSCORE_BREAKOUT"
        self._current_oss_mode = "RAW"

        # Stability tracking — start high to allow first switch
        self._mode_stability = 99

        # Tick counter for anti-flapping
        self._tick_since_last_change = 99

        logger.debug(
            "AdaptiveSignalSwitcher(%r) initialised alt=%s oss=%s",
            instance_id, self._current_alt_mode, self._current_oss_mode,
        )

    # ------------------------------------------------------------------
    # Public API — manual mode setting
    # ------------------------------------------------------------------

    def set_alt_mode(self, mode):
        """Manually set the active ALT mode.

        Parameters
        ----------
        mode : str
            Must be one of the registered alt_modes.
        """
        if mode not in self._alt_modes:
            logger.warning(
                "set_alt_mode: unknown mode '%s' — valid: %s",
                mode, self._alt_modes,
            )
            return
        if mode != self._current_alt_mode:
            self._current_alt_mode = mode
            self._mode_stability = 0
            self._tick_since_last_change = 0
            logger.info("ALT mode set to %s (stability reset)", mode)

    def set_oss_mode(self, mode):
        """Manually set the active OSS mode.

        Parameters
        ----------
        mode : str
            Must be one of the registered oss_modes.
        """
        if mode not in self._oss_modes:
            logger.warning(
                "set_oss_mode: unknown mode '%s' — valid: %s",
                mode, self._oss_modes,
            )
            return
        if mode != self._current_oss_mode:
            self._current_oss_mode = mode
            self._mode_stability = 0
            self._tick_since_last_change = 0
            logger.info("OSS mode set to %s (stability reset)", mode)

    # ------------------------------------------------------------------
    # Public API — automatic regime-based switching
    # ------------------------------------------------------------------

    def switch_for_regime(self, resolution_classification, signal_viability="NORMAL"):
        """Automatically select modes based on the current market regime.

        Parameters
        ----------
        resolution_classification : str
            One of ``NOISE``, ``MICRO_STRUCTURE``, ``MESO_STRUCTURE``,
            ``MACRO_TREND``.
        signal_viability : str
            ``"NORMAL"`` or ``"LOW"``.

        Returns
        -------
        dict
            ``selected_alt_mode``  — the ALT mode in effect.

            ``selected_oss_mode``  — the OSS mode in effect.

            ``reason``             — human-readable explanation.

            ``mode_stability``     — consecutive ticks in same mode.
        """
        regime = resolution_classification.upper()

        # ---- Determine target modes ----
        target_alt = _REGIME_ALT_MAP.get(regime, self._current_alt_mode)
        target_oss = _REGIME_OSS_MAP.get(regime, self._current_oss_mode)

        # Low-viability override: only for NOISE regime
        if signal_viability.upper() == "LOW" and regime == "NOISE":
            target_alt = _LOW_VIABILITY_ALT
            target_oss = _LOW_VIABILITY_OSS

        alt_changed = target_alt != self._current_alt_mode
        oss_changed = target_oss != self._current_oss_mode
        any_change = alt_changed or oss_changed

        # ---- Anti-flapping logic ----
        if any_change and self._mode_stability < _ANTI_FLAP_THRESHOLD:
            # Prefer previous mode — too soon to switch again
            self._tick_since_last_change += 1
            self._mode_stability += 1
            reason = (
                f"Anti-flap triggered (stability={self._mode_stability} < "
                f"{_ANTI_FLAP_THRESHOLD}). Keeping alt={self._current_alt_mode} "
                f"oss={self._current_oss_mode} (would switch to "
                f"alt={target_alt} oss={target_oss} for regime={regime})"
            )
            logger.debug("switch_for_regime: %s", reason)
            return {
                "selected_alt_mode": self._current_alt_mode,
                "selected_oss_mode": self._current_oss_mode,
                "reason": reason,
                "mode_stability": self._mode_stability,
            }

        # ---- Apply switch (or confirm same mode) ----
        if any_change:
            self._current_alt_mode = target_alt
            self._current_oss_mode = target_oss
            self._mode_stability = 0
            self._tick_since_last_change = 0
            reason = (
                f"Regime={regime} viab={signal_viability}: switched to "
                f"alt={target_alt} oss={target_oss}"
            )
            logger.info("switch_for_regime: %s", reason)
        else:
            self._mode_stability += 1
            self._tick_since_last_change += 1
            reason = (
                f"Regime={regime} viab={signal_viability}: unchanged "
                f"alt={target_alt} oss={target_oss} "
                f"(stability={self._mode_stability})"
            )

        return {
            "selected_alt_mode": self._current_alt_mode,
            "selected_oss_mode": self._current_oss_mode,
            "reason": reason,
            "mode_stability": self._mode_stability,
        }

    def get_current_modes(self):
        """Return the current ALT and OSS modes.

        Returns
        -------
        dict
            ``alt_mode``  — current ALT mode.

            ``oss_mode``  — current OSS mode.

            ``stability`` — current mode-stability counter.
        """
        return {
            "alt_mode": self._current_alt_mode,
            "oss_mode": self._current_oss_mode,
            "stability": self._mode_stability,
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("Adaptive Signal Switcher — Self Test")
    print("=" * 60)

    _state = {"passed": True}

    def _check(cond, msg):
        if not cond:
            _state["passed"] = False
            print(f"  FAIL: {msg}")
        else:
            print(f"  PASS: {msg}")

    # --------------------------------------------------------------
    # Scenario 1: NOISE regime -> ZSCORE_BREAKOUT / SMOOTHED
    # --------------------------------------------------------------
    print("\n--- SCE1: NOISE regime ---")
    sw1 = AdaptiveSignalSwitcher("sce1")
    result = sw1.switch_for_regime("NOISE", "NORMAL")
    print(f"  alt={result['selected_alt_mode']} oss={result['selected_oss_mode']} "
          f"stability={result['mode_stability']}")
    _check(result["selected_alt_mode"] == "ZSCORE_BREAKOUT",
           f"Expected ZSCORE_BREAKOUT, got {result['selected_alt_mode']}")
    _check(result["selected_oss_mode"] == "SMOOTHED",
           f"Expected SMOOTHED, got {result['selected_oss_mode']}")
    _check(result["mode_stability"] == 0,
           "Stability reset to 0 after first switch")

    # --------------------------------------------------------------
    # Scenario 2: MICRO_STRUCTURE regime -> MOMENTUM / RAW
    # --------------------------------------------------------------
    print("\n--- SCE2: MICRO_STRUCTURE regime ---")
    sw2 = AdaptiveSignalSwitcher("sce2")
    result = sw2.switch_for_regime("MICRO_STRUCTURE", "NORMAL")
    print(f"  alt={result['selected_alt_mode']} oss={result['selected_oss_mode']}")
    _check(result["selected_alt_mode"] == "MOMENTUM",
           f"Expected MOMENTUM, got {result['selected_alt_mode']}")
    _check(result["selected_oss_mode"] == "RAW",
           f"Expected RAW, got {result['selected_oss_mode']}")

    # --------------------------------------------------------------
    # Scenario 3: MESO_STRUCTURE regime -> ADAPTIVE_EMA / RAW
    # --------------------------------------------------------------
    print("\n--- SCE3: MESO_STRUCTURE regime ---")
    sw3 = AdaptiveSignalSwitcher("sce3")
    result = sw3.switch_for_regime("MESO_STRUCTURE", "NORMAL")
    print(f"  alt={result['selected_alt_mode']} oss={result['selected_oss_mode']}")
    _check(result["selected_alt_mode"] == "ADAPTIVE_EMA",
           f"Expected ADAPTIVE_EMA, got {result['selected_alt_mode']}")
    _check(result["selected_oss_mode"] == "RAW",
           f"Expected RAW, got {result['selected_oss_mode']}")

    # --------------------------------------------------------------
    # Scenario 4: MACRO_TREND regime -> ADAPTIVE_EMA / RAW
    # --------------------------------------------------------------
    print("\n--- SCE4: MACRO_TREND regime ---")
    sw4 = AdaptiveSignalSwitcher("sce4")
    result = sw4.switch_for_regime("MACRO_TREND", "NORMAL")
    print(f"  alt={result['selected_alt_mode']} oss={result['selected_oss_mode']}")
    _check(result["selected_alt_mode"] == "ADAPTIVE_EMA",
           f"Expected ADAPTIVE_EMA, got {result['selected_alt_mode']}")
    _check(result["selected_oss_mode"] == "RAW",
           f"Expected RAW, got {result['selected_oss_mode']}")

    # --------------------------------------------------------------
    # Scenario 5: NOISE + LOW viability -> NO_SIGNAL / FLAT
    # --------------------------------------------------------------
    print("\n--- SCE5: NOISE + LOW viability ---")
    sw5 = AdaptiveSignalSwitcher("sce5")
    result = sw5.switch_for_regime("NOISE", "LOW")
    print(f"  alt={result['selected_alt_mode']} oss={result['selected_oss_mode']}")
    _check(result["selected_alt_mode"] == "NO_SIGNAL",
           f"Expected NO_SIGNAL, got {result['selected_alt_mode']}")
    _check(result["selected_oss_mode"] == "FLAT",
           f"Expected FLAT, got {result['selected_oss_mode']}")

    # --------------------------------------------------------------
    # Scenario 6: LOW viability on non-NOISE regime -> no override
    # --------------------------------------------------------------
    print("\n--- SCE6: LOW viability on MICRO_STRUCTURE (no override) ---")
    sw6 = AdaptiveSignalSwitcher("sce6")
    result = sw6.switch_for_regime("MICRO_STRUCTURE", "LOW")
    print(f"  alt={result['selected_alt_mode']} oss={result['selected_oss_mode']}")
    _check(result["selected_alt_mode"] == "MOMENTUM",
           f"Expected MOMENTUM, got {result['selected_alt_mode']}")
    _check(result["selected_oss_mode"] == "RAW",
           f"Expected RAW, got {result['selected_oss_mode']}")

    # --------------------------------------------------------------
    # Scenario 7: Anti-flapping — rapid regime changes
    # --------------------------------------------------------------
    print("\n--- SCE7: Anti-flapping ---")
    sw7 = AdaptiveSignalSwitcher("sce7")
    # Call 1: NOISE -> switch to NOISE modes, stability=0
    r1 = sw7.switch_for_regime("NOISE", "NORMAL")
    print(f"  1: NOISE  -> alt={r1['selected_alt_mode']} oss={r1['selected_oss_mode']} "
          f"stab={r1['mode_stability']}")
    _check(r1["mode_stability"] == 0, "Stability 0 after first switch")
    # Call 2: MICRO_STRUCTURE -> anti-flap (stability=0 < 3), keep NOISE modes
    r2 = sw7.switch_for_regime("MICRO_STRUCTURE", "NORMAL")
    print(f"  2: MICRO  -> alt={r2['selected_alt_mode']} oss={r2['selected_oss_mode']} "
          f"stab={r2['mode_stability']}")
    _check(r2["selected_alt_mode"] == "ZSCORE_BREAKOUT",
           f"Anti-flap should keep ZSCORE_BREAKOUT, got {r2['selected_alt_mode']}")
    _check(r2["mode_stability"] == 1, "Stability 1 after anti-flap")
    # Call 3: still MICRO (stability=1 < 3 still blocked)
    r3 = sw7.switch_for_regime("MICRO_STRUCTURE", "NORMAL")
    print(f"  3: MICRO  -> alt={r3['selected_alt_mode']} oss={r3['selected_oss_mode']} "
          f"stab={r3['mode_stability']}")
    _check(r3["selected_alt_mode"] == "ZSCORE_BREAKOUT",
           f"Anti-flap still active, got {r3['selected_alt_mode']}")
    _check(r3["mode_stability"] == 2, "Stability 2 after 2nd anti-flap")
    # Call 4: MICRO (stability=2 < 3 still blocked, becomes 3)
    r4 = sw7.switch_for_regime("MICRO_STRUCTURE", "NORMAL")
    print(f"  4: MICRO  -> alt={r4['selected_alt_mode']} oss={r4['selected_oss_mode']} "
          f"stab={r4['mode_stability']}")
    _check(r4["selected_alt_mode"] == "ZSCORE_BREAKOUT",
           f"Anti-flap still on call 4, got {r4['selected_alt_mode']}")
    _check(r4["mode_stability"] == 3, "Stability 3 after 3rd anti-flap")
    # Call 5: MICRO (stability=3 not < 3 -> switch allowed)
    r5 = sw7.switch_for_regime("MICRO_STRUCTURE", "NORMAL")
    print(f"  5: MICRO  -> alt={r5['selected_alt_mode']} oss={r5['selected_oss_mode']} "
          f"stab={r5['mode_stability']}")
    _check(r5["selected_alt_mode"] == "MOMENTUM",
           f"Expected MOMENTUM after anti-flap, got {r5['selected_alt_mode']}")
    _check(r5["mode_stability"] == 0,
           "Stability reset to 0 after switch")

    # --------------------------------------------------------------
    # Scenario 8: get_current_modes
    # --------------------------------------------------------------
    print("\n--- SCE8: get_current_modes ---")
    modes = sw1.get_current_modes()
    print(f"  alt_mode={modes['alt_mode']} oss_mode={modes['oss_mode']} "
          f"stability={modes['stability']}")
    _check(modes["alt_mode"] == "ZSCORE_BREAKOUT",
           f"Expected ZSCORE_BREAKOUT, got {modes['alt_mode']}")
    _check(modes["oss_mode"] == "SMOOTHED",
           f"Expected SMOOTHED, got {modes['oss_mode']}")

    # --------------------------------------------------------------
    # Scenario 9: Manual set_alt_mode / set_oss_mode
    # --------------------------------------------------------------
    print("\n--- SCE9: Manual mode setting ---")
    sw9 = AdaptiveSignalSwitcher("sce9")
    sw9.set_alt_mode("MOMENTUM")
    _check(sw9.get_current_modes()["alt_mode"] == "MOMENTUM",
           "set_alt_mode to MOMENTUM")
    sw9.set_oss_mode("SMOOTHED")
    _check(sw9.get_current_modes()["oss_mode"] == "SMOOTHED",
           "set_oss_mode to SMOOTHED")
    # Invalid modes should be ignored
    sw9.set_alt_mode("INVALID_MODE")
    _check(sw9.get_current_modes()["alt_mode"] == "MOMENTUM",
           "Invalid alt mode ignored")
    sw9.set_oss_mode("INVALID_MODE")
    _check(sw9.get_current_modes()["oss_mode"] == "SMOOTHED",
           "Invalid oss mode ignored")

    # --------------------------------------------------------------
    # Scenario 10: Singleton identity
    # --------------------------------------------------------------
    print("\n--- SCE10: Singleton identity ---")
    sw1_again = AdaptiveSignalSwitcher("sce1")
    _check(sw1_again is sw1, "Same instance_id returns same object")
    default_a = AdaptiveSignalSwitcher()
    default_b = AdaptiveSignalSwitcher("default")
    _check(default_a is default_b, "Default singleton identity")
    other = AdaptiveSignalSwitcher("other")
    _check(other is not sw1, "Different instance_id returns different object")

    # --------------------------------------------------------------
    # Scenario 11: Stability tracking across same-mode calls
    # --------------------------------------------------------------
    print("\n--- SCE11: Stability accumulation ---")
    sw11 = AdaptiveSignalSwitcher("sce11")
    # Call 1: switch to NOISE modes, stability=0
    r11a = sw11.switch_for_regime("NOISE", "NORMAL")
    _check(r11a["mode_stability"] == 0, "Stability 0 after first switch")
    # Call 2: same regime -> stability increments to 1
    r11b = sw11.switch_for_regime("NOISE", "NORMAL")
    _check(r11b["mode_stability"] == 1, f"Expected stability 1, got {r11b['mode_stability']}")
    # Call 3: same regime -> stability increments to 2
    r11c = sw11.switch_for_regime("NOISE", "NORMAL")
    _check(r11c["mode_stability"] == 2, f"Expected stability 2, got {r11c['mode_stability']}")
    print(f"  stability steps: 0 -> {r11b['mode_stability']} -> {r11c['mode_stability']}")

    # --------------------------------------------------------------
    # Scenario 12: Anti-flap transition after stability threshold
    # --------------------------------------------------------------
    print("\n--- SCE12: Transition after anti-flap threshold ---")
    sw12 = AdaptiveSignalSwitcher("sce12")
    # Stay in NOISE for 5 calls to build stability past 3
    sw12.switch_for_regime("NOISE", "NORMAL")   # switch -> stab=0
    sw12.switch_for_regime("NOISE", "NORMAL")   # stab=1
    sw12.switch_for_regime("NOISE", "NORMAL")   # stab=2
    sw12.switch_for_regime("NOISE", "NORMAL")   # stab=3
    sw12.switch_for_regime("NOISE", "NORMAL")   # stab=4
    # Now switch should be allowed (stab=4 >= 3)
    r12 = sw12.switch_for_regime("MESO_STRUCTURE", "NORMAL")
    print(f"  NOISE x5 then MESO: alt={r12['selected_alt_mode']} stab={r12['mode_stability']}")
    _check(r12["selected_alt_mode"] == "ADAPTIVE_EMA",
           f"Expected ADAPTIVE_EMA, got {r12['selected_alt_mode']}")
    _check(r12["mode_stability"] == 0, "Stability reset after switch")
    # Subsequent same-mode call increments stability
    r12b = sw12.switch_for_regime("MESO_STRUCTURE", "NORMAL")
    _check(r12b["mode_stability"] == 1,
           f"Stability builds to 1, got {r12b['mode_stability']}")

    # --------------------------------------------------------------
    # Final result
    # --------------------------------------------------------------
    print()
    print("=" * 60)
    if _state["passed"]:
        print("ALL SELF-TESTS PASSED ✓")
    else:
        print("SOME SELF-TESTS FAILED ✗")
    print("=" * 60)

    import sys
    sys.exit(0 if _state["passed"] else 1)
