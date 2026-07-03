"""
Validation signal injector module.

Injects controlled directional bias signals for validation mode ONLY.
Not production logic — purely for testing execution pipeline independence.
Proves that the execution layer can execute trades when given ANY valid signal.
"""

import os
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class _Mode(Enum):
    ALTERNATING = "ALTERNATING"
    PERSISTENT_BUY = "PERSISTENT_BUY"
    PERSISTENT_SELL = "PERSISTENT_SELL"
    PERIODIC = "PERIODIC"
    OFF = "OFF"


class _ValidationSignalInjector:
    """Internal implementation of the validation signal injector.

    All modes are symbol-agnostic — the same signal is returned for every
    symbol on a given tick.
    """

    def __init__(self, instance_id="default"):
        self._instance_id = instance_id
        self._mode = _Mode.OFF

        # General tick counter
        self._tick_counter = 0

        # ALTERNATING mode
        self._cycle_length = 5

        # PERIODIC mode state — uses a purely positional counter so that
        # tick() advances by one position and get_signal() reads the current
        # position without ambiguity about phase boundaries.
        self._periodic_on_ticks = 3
        self._periodic_off_ticks = 3
        self._periodic_position = 0  # 0 = before any tick; incremented by tick()

        # Statistics
        self._total_ticks = 0
        self._buy_count = 0
        self._sell_count = 0
        self._flat_count = 0

        logger.info(
            "ValidationSignalInjector '%s': initialised, mode=OFF",
            instance_id,
        )

    # ------------------------------------------------------------------
    # Public configuration
    # ------------------------------------------------------------------

    def set_mode(self, mode):
        """Set injection mode.

        Args:
            mode: One of ``"ALTERNATING"``, ``"PERSISTENT_BUY"``,
                  ``"PERSISTENT_SELL"``, ``"PERIODIC"``, ``"OFF"``.
        """
        if isinstance(mode, str):
            try:
                mode = _Mode[mode.upper()]
            except KeyError:
                valid = ", ".join(m.name for m in _Mode)
                raise ValueError(
                    f"Unknown mode '{mode}'. Valid modes: {valid}"
                )
        self._mode = mode
        self._tick_counter = 0
        self._periodic_position = 0
        logger.info(
            "ValidationSignalInjector '%s': mode set to %s",
            self._instance_id,
            self._mode.value,
        )

    def set_cycle_length(self, n):
        """Set *N* for ALTERNATING mode (default: 5).

        The signal toggles every *N* ticks.
        """
        if n < 1:
            raise ValueError("Cycle length must be >= 1")
        self._cycle_length = n
        logger.info(
            "ValidationSignalInjector '%s': cycle_length set to %d",
            self._instance_id,
            n,
        )

    def set_periodic_params(self, on_ticks, off_ticks):
        """Set parameters for PERIODIC mode.

        Signal is injected for *on_ticks* consecutive ticks, then zero for
        *off_ticks* ticks, repeating indefinitely.

        Args:
            on_ticks:  Number of ticks to inject signal (>= 1).
            off_ticks: Number of ticks to inject 0 (>= 1).
        """
        if on_ticks < 1 or off_ticks < 1:
            raise ValueError("on_ticks and off_ticks must be >= 1")
        self._periodic_on_ticks = on_ticks
        self._periodic_off_ticks = off_ticks
        self._periodic_position = 0
        logger.info(
            "ValidationSignalInjector '%s': periodic_params set to on=%d, off=%d",
            self._instance_id,
            on_ticks,
            off_ticks,
        )

    # ------------------------------------------------------------------
    # Tick / signal
    # ------------------------------------------------------------------

    def tick(self):
        """Increment internal tick counter.  Call once per cycle."""
        self._tick_counter += 1
        self._total_ticks += 1
        if self._mode == _Mode.PERIODIC:
            self._periodic_position += 1

    def get_signal(self, symbol):
        """Return the injected signal for *symbol*.

        The signal is symbol-agnostic — every symbol sees the same value.

        Returns
            ``int``: ``-1`` (sell), ``0`` (flat), or ``+1`` (buy).

        Safety guard
            Returns ``0`` unless the environment variable
            ``VALIDATION_MODE`` is set.
        """
        if not os.environ.get("VALIDATION_MODE"):
            return 0
        signal = self._compute_signal(symbol)
        self._record_signal(signal)
        return signal

    def is_active(self):
        """Whether injection is currently producing a non-zero signal."""
        if not os.environ.get("VALIDATION_MODE"):
            return False
        if self._mode == _Mode.OFF:
            return False
        if self._mode in (_Mode.PERSISTENT_BUY, _Mode.PERSISTENT_SELL):
            return True
        if self._mode == _Mode.ALTERNATING:
            # Alternating is always +1 or -1, never 0.
            return True
        if self._mode == _Mode.PERIODIC:
            return self._is_in_periodic_on_phase()
        return False

    def get_injection_stats(self):
        """Return injection statistics as a dictionary."""
        return {
            "total_ticks": self._total_ticks,
            "buy_count": self._buy_count,
            "sell_count": self._sell_count,
            "flat_count": self._flat_count,
            "mode": self._mode.value,
        }

    def reset_stats(self):
        """Reset all statistics counters (does *not* reset tick counter)."""
        self._total_ticks = 0
        self._buy_count = 0
        self._sell_count = 0
        self._flat_count = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_signal(self, symbol):
        """Compute the signal based on current mode and tick."""
        _ = symbol  # unused — all modes are symbol-agnostic

        if self._mode == _Mode.OFF:
            return 0

        if self._mode == _Mode.PERSISTENT_BUY:
            return 1

        if self._mode == _Mode.PERSISTENT_SELL:
            return -1

        if self._mode == _Mode.ALTERNATING:
            # Toggle every N ticks: period 0 → +1, period 1 → -1, ...
            # Use (tick_counter - 1) so that tick 1 starts in period 0
            # and the first N ticks all fall in the same period.
            period = (self._tick_counter - 1) // self._cycle_length
            return 1 if period % 2 == 0 else -1

        if self._mode == _Mode.PERIODIC:
            if self._periodic_position == 0:
                return 0
            cycle_len = self._periodic_on_ticks + self._periodic_off_ticks
            pos = (self._periodic_position - 1) % cycle_len
            return 1 if pos < self._periodic_on_ticks else 0

        return 0

    def _is_in_periodic_on_phase(self):
        """Return ``True`` if PERIODIC mode is currently in the on-phase."""
        if self._periodic_position == 0:
            return False
        cycle_len = self._periodic_on_ticks + self._periodic_off_ticks
        pos = (self._periodic_position - 1) % cycle_len
        return pos < self._periodic_on_ticks

    def _record_signal(self, signal):
        """Record one signal observation for statistics."""
        if signal > 0:
            self._buy_count += 1
        elif signal < 0:
            self._sell_count += 1
        else:
            self._flat_count += 1


# ------------------------------------------------------------------
# Singleton accessor
# ------------------------------------------------------------------

_instances = {}


def ValidationSignalInjector(instance_id="default"):
    """Get or create a :class:`_ValidationSignalInjector` instance.

    Singleton accessor pattern — returns the same instance for a given
    *instance_id* on every call.

    Args:
        instance_id: Unique identifier for the instance (default ``"default"``).

    Returns:
        _ValidationSignalInjector instance.
    """
    if instance_id not in _instances:
        _instances[instance_id] = _ValidationSignalInjector(instance_id)
    return _instances[instance_id]


# ------------------------------------------------------------------
# Self-test
# ------------------------------------------------------------------

def _run_self_test():
    """Run through all modes and verify signal distributions.

    Returns ``True`` if all checks pass, ``False`` otherwise.
    """
    logger.info("=" * 60)
    logger.info("ValidationSignalInjector self-test")
    logger.info("=" * 60)

    # Remember and force VALIDATION_MODE for the duration of the test.
    old_val = os.environ.get("VALIDATION_MODE")
    os.environ["VALIDATION_MODE"] = "1"

    inj = ValidationSignalInjector("_selftest")
    test_passed = True

    def _check(cond, msg):
        nonlocal test_passed
        if cond:
            logger.info("  PASS: %s", msg)
        else:
            test_passed = False
            logger.error("  FAIL: %s", msg)

    # ----- safety guard: no VALIDATION_MODE ---------------------------
    logger.info("")
    logger.info("--- Safety guard ---")
    del os.environ["VALIDATION_MODE"]
    inj.set_mode("PERSISTENT_BUY")
    _check(inj.get_signal("X") == 0, "returns 0 when VALIDATION_MODE is missing")
    _check(not inj.is_active(), "is_active is False when VALIDATION_MODE is missing")
    os.environ["VALIDATION_MODE"] = "1"

    # ----- OFF mode ---------------------------------------------------
    logger.info("")
    logger.info("--- OFF mode ---")
    inj.reset_stats()
    inj.set_mode("OFF")
    signals = [inj.get_signal("X") for _ in range(10)]
    _check(all(s == 0 for s in signals), "all signals are 0 in OFF mode")
    _check(not inj.is_active(), "is_active is False in OFF mode")
    stats = inj.get_injection_stats()
    _check(stats["flat_count"] == 10, "OFF mode flat_count == 10")
    _check(stats["buy_count"] == 0, "OFF mode buy_count == 0")
    _check(stats["sell_count"] == 0, "OFF mode sell_count == 0")

    # ----- PERSISTENT_BUY mode ----------------------------------------
    logger.info("")
    logger.info("--- PERSISTENT_BUY mode ---")
    inj.reset_stats()
    inj.set_mode("PERSISTENT_BUY")
    signals = [inj.get_signal("X") for _ in range(10)]
    _check(all(s == 1 for s in signals), "all signals are +1 in PERSISTENT_BUY")
    _check(inj.is_active(), "is_active is True in PERSISTENT_BUY")
    stats = inj.get_injection_stats()
    _check(stats["buy_count"] == 10, "PERSISTENT_BUY buy_count == 10")
    _check(stats["sell_count"] == 0, "PERSISTENT_BUY sell_count == 0")
    _check(stats["flat_count"] == 0, "PERSISTENT_BUY flat_count == 0")
    _check(stats["mode"] == "PERSISTENT_BUY", "mode string is PERSISTENT_BUY")

    # ----- PERSISTENT_SELL mode ---------------------------------------
    logger.info("")
    logger.info("--- PERSISTENT_SELL mode ---")
    inj.reset_stats()
    inj.set_mode("PERSISTENT_SELL")
    signals = [inj.get_signal("X") for _ in range(10)]
    _check(all(s == -1 for s in signals), "all signals are -1 in PERSISTENT_SELL")
    stats = inj.get_injection_stats()
    _check(stats["sell_count"] == 10, "PERSISTENT_SELL sell_count == 10")
    _check(stats["buy_count"] == 0, "PERSISTENT_SELL buy_count == 0")
    _check(stats["flat_count"] == 0, "PERSISTENT_SELL flat_count == 0")

    # ----- ALTERNATING mode -------------------------------------------
    logger.info("")
    logger.info("--- ALTERNATING mode (N=3) ---")
    inj.reset_stats()
    inj.set_mode("ALTERNATING")
    inj.set_cycle_length(3)
    inj._tick_counter = 0  # reset internal counter after set_mode
    signals_alt = []
    for _ in range(9):
        inj.tick()
        signals_alt.append(inj.get_signal("X"))
    # tick 1: _tick_counter=1 → 1//3=0 (even) → +1
    # tick 2: _tick_counter=2 → 2//3=0 (even) → +1
    # tick 3: _tick_counter=3 → 3//3=1 (odd)  → -1
    # tick 4: _tick_counter=4 → 4//3=1 (odd)  → -1
    # tick 5: _tick_counter=5 → 5//3=1 (odd)  → -1
    # tick 6: _tick_counter=6 → 6//3=2 (even) → +1
    # tick 7: _tick_counter=7 → 7//3=2 (even) → +1
    # tick 8: _tick_counter=8 → 8//3=2 (even) → +1
    # tick 9: _tick_counter=9 → 9//3=3 (odd)  → -1
    expected_alt = [1, 1, 1, -1, -1, -1, 1, 1, 1]
    _check(
        signals_alt == expected_alt,
        f"ALTERNATING pattern: expected {expected_alt}, got {signals_alt}",
    )
    stats = inj.get_injection_stats()
    _check(stats["buy_count"] == 6, f"ALTERNATING buy_count == 6 (got {stats['buy_count']})")
    _check(stats["sell_count"] == 3, f"ALTERNATING sell_count == 3 (got {stats['sell_count']})")
    _check(stats["flat_count"] == 0, f"ALTERNATING flat_count == 0 (got {stats['flat_count']})")

    # ----- PERIODIC mode ----------------------------------------------
    logger.info("")
    logger.info("--- PERIODIC mode (on=2, off=3) ---")
    inj.reset_stats()
    inj.set_mode("PERIODIC")
    inj.set_periodic_params(2, 3)
    inj._periodic_position = 0
    signals_per = []
    for _ in range(10):
        inj.tick()
        signals_per.append(inj.get_signal("X"))
    # on for 2, off for 3:
    # tick  1: periodic_position=1 → (1-1)%5=0 (0<2)  → +1
    # tick  2: periodic_position=2 → (2-1)%5=1 (1<2)  → +1
    # tick  3: periodic_position=3 → (3-1)%5=2 (2>=2) → 0
    # tick  4: periodic_position=4 → (4-1)%5=3 (3>=2) → 0
    # tick  5: periodic_position=5 → (5-1)%5=4 (4>=2) → 0
    # tick  6: periodic_position=6 → (6-1)%5=0 (0<2)  → +1
    # tick  7: periodic_position=7 → (7-1)%5=1 (1<2)  → +1
    # tick  8: periodic_position=8 → (8-1)%5=2 (2>=2) → 0
    # tick  9: periodic_position=9 → (9-1)%5=3 (3>=2) → 0
    # tick 10: periodic_position=10→(10-1)%5=4 (4>=2) → 0
    expected_per = [1, 1, 0, 0, 0, 1, 1, 0, 0, 0]
    _check(
        signals_per == expected_per,
        f"PERIODIC pattern: expected {expected_per}, got {signals_per}",
    )
    stats = inj.get_injection_stats()
    _check(stats["buy_count"] == 4, f"PERIODIC buy_count == 4 (got {stats['buy_count']})")
    _check(stats["flat_count"] == 6, f"PERIODIC flat_count == 6 (got {stats['flat_count']})")
    _check(stats["sell_count"] == 0, f"PERIODIC sell_count == 0 (got {stats['sell_count']})")

    # ----- is_active during PERIODIC on/off phases --------------------
    logger.info("")
    logger.info("--- is_active in PERIODIC mode ---")
    inj.reset_stats()
    inj.set_mode("PERIODIC")
    inj.set_periodic_params(2, 3)
    inj._periodic_position = 0
    active_states = []
    for _ in range(6):
        inj.tick()
        active_states.append(inj.is_active())
    # tick 1: on  → True
    # tick 2: on  → True
    # tick 3: off → False
    # tick 4: off → False
    # tick 5: off → False
    # tick 6: on  → True
    expected_active = [True, True, False, False, False, True]
    _check(
        active_states == expected_active,
        f"is_active PERIODIC: expected {expected_active}, got {active_states}",
    )

    # ----- singleton behaviour ----------------------------------------
    logger.info("")
    logger.info("--- Singleton accessor ---")
    inst_a = ValidationSignalInjector("_selftest")
    inst_b = ValidationSignalInjector("_selftest")
    inst_c = ValidationSignalInjector("other")
    _check(inst_a is inst_b, "same instance_id returns same object")
    _check(inst_a is not inst_c, "different instance_id returns different object")

    # ----- stats dict keys --------------------------------------------
    logger.info("")
    logger.info("--- Stats dictionary ---")
    stats = inj.get_injection_stats()
    expected_keys = {"total_ticks", "buy_count", "sell_count", "flat_count", "mode"}
    _check(
        set(stats.keys()) == expected_keys,
        f"stats keys match: expected {expected_keys}, got {set(stats.keys())}",
    )

    # ------------------------------------------------------------------
    # Restore original env
    # ------------------------------------------------------------------
    if old_val is not None:
        os.environ["VALIDATION_MODE"] = old_val
    else:
        os.environ.pop("VALIDATION_MODE", None)

    logger.info("")
    logger.info("=" * 60)
    if test_passed:
        logger.info("RESULT: ALL PASSED")
    else:
        logger.error("RESULT: SOME FAILED")
    logger.info("=" * 60)

    return test_passed


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    success = _run_self_test()
    import sys
    sys.exit(0 if success else 1)
