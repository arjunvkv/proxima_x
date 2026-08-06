"""C5 regression: RiskGovernor day-boundary reset uses tape-day, not wall-clock.

Gap: risk_governor.check() used date.today() for the day key. Under replay
the daily-loss counter reset on the MACHINE's midnight, not the tape's trading
day — mis-mapping FirmRisk daily-loss limits vs MT5 server timezone.

Fix: __init__(clock=None) injects a clock whose now().date() is the day key.
None = legacy wall clock (fully backward-compatible). The day key is lazily
adopted on the first check (so an epoch-start ReplayClock cannot wipe the
first day's accrued loss).

Verifies:
 1. epoch-start ReplayClock does not wipe day-1 accrued loss on first check.
 2. crossing a tape-timeday key resets the daily counter, then new losses
    accumulate on the fresh key.
 3. default (wall clock) path still works.
"""
import sys
sys.path.insert(0, r"C:\Trading\Proxima_X")

from datetime import datetime  # noqa: E402

from core.adapters.clock import ReplayClock  # noqa: E402
from proxima_ops.risk.risk_governor import RiskGovernor  # noqa: E402


def test_epoch_clock_no_first_check_wipe():
    clk = ReplayClock()
    clk.advance_to(datetime(2026, 8, 3, 12, 0).timestamp())
    g = RiskGovernor(clock=clk)
    g.set_start_equity(100000)
    g.record_result(-900)
    assert g.check()["state"] == "HEALTHY"
    assert g._daily_loss == -900.0, f"day-1 loss wiped: {g._daily_loss}"


def test_tape_day_boundary_resets_then_accumulates():
    clk = ReplayClock()
    g = RiskGovernor(clock=clk)
    g.set_start_equity(100000)
    clk.advance_to(datetime(2026, 8, 3, 10, 0).timestamp())
    g.record_result(-900)
    assert g.check()["state"] == "HEALTHY"
    # cross into the next tape-timeline day
    clk.advance_to(datetime(2026, 8, 4, 1, 0).timestamp())
    g.record_result(0)
    g.check()
    assert g._daily_loss == 0.0, f"boundary didn't clear day-1: {g._daily_loss}"
    assert g._today == "2026-08-04"
    # fresh day-2 trades accumulate on the new key
    g.record_result(-700)
    g.check()
    assert g._daily_loss == -700.0, f"day-2 loss wrong: {g._daily_loss}"
    assert g._today == "2026-08-04"


def test_wall_clock_default():
    g = RiskGovernor()  # backward-compat no-arg construction
    g.set_start_equity(100000)
    g.record_result(-500)
    assert g.check()["state"] == "HEALTHY"


def main():
    tests = [test_epoch_clock_no_first_check_wipe,
             test_tape_day_boundary_resets_then_accumulates,
             test_wall_clock_default]
    fails = 0
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
        except AssertionError as e:
            fails += 1
            print(f"  FAIL {t.__name__}: {e}")
    print(f"C5 RESULT: {len(tests) - fails}/{len(tests)} passed" + ("" if fails else " (all green)"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())