"""verify_execution_parity.py — Phase 7 acceptance harness.

GPT-recommended next step (2026-08-06): prove the missing link in the
backtest<->live alignment story:

    canonical tick -> execution -> portfolio state  (fills / friction / PnL / risk)

Run from repo root:
    unset PYTHONPATH && ./.venv/Scripts/python.exe scripts/verify_execution_parity.py

Offline (no MT5 terminal): drives PaperBroker + ExecutionCost with a
broker-realistic EURJPY tick tape (point=0.001, 16-point spread) and asserts the
exact execution-contract that live MT5 fills follow:

  BUY  entry = ask  (adverse slip makes fill >= ask)
  SELL entry = bid  (adverse slip makes fill <= bid)
  BUY  exit  = bid
  SELL exit  = ask
  commission charged on BOTH legs
  slippage adverse-only, deterministic per symbol/side
  JPY pip value 1000/entry (== ExecutionCost.pip_value_per_lot, MT5 semantics)
  open positions leave balance untouched; PnL realized only on close
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from data.execution_cost import ExecutionCost, pip_value_per_lot
from core.adapters.broker import PaperBroker

PASS = 0
FAIL = []


def check(name: str, cond: bool, detail: str = ""):
    global PASS
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL.append(name)
        print(f"  [FAIL] {name}  " + (detail or ""))


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def time(self):
        return self.now

    def sleep(self, s):
        self.now += s

    def advance(self, s):
        self.now += s


class TickSource:
    """Iterates a per-symbol tape; records the last tick served (so the harness
    can assert the EXACT fill/close price the broker saw)."""

    def __init__(self, ticks: dict):
        self._q = {k: list(v) for k, v in ticks.items()}
        self._idx = {}
        self.last = {}  # symbol -> last tick returned

    def get_tick(self, symbol):
        q = self._q.get(symbol, [])
        if not q:
            self.last[symbol] = None
            return None
        i = self._idx.get(symbol, 0)
        if i < len(q):
            t = q[i]
            self._idx[symbol] = i + 1
        else:
            t = q[-1]  # static tail (replay-like steady state)
        self.last[symbol] = t
        return t


def build_eurjpy_tape(n=40, base=182.000, step=0.001, spread=0.016, spread_pts=16):
    ticks = []
    for i in range(n):
        bid = base + i * step
        ticks.append({
            "symbol": "EURJPY", "bid": bid, "ask": bid + spread,
            "spread": spread, "spread_pts": spread_pts,
            "point": 0.001, "digits": 3, "time_sec": int(1_000 + i),
        })
    return ticks


def run():
    tape = build_eurjpy_tape()
    src = TickSource({"EURJPY": tape})
    clock = Clock()

    ec = ExecutionCost(commission_per_lot=3.5, min_commission=0.0,
                       slippage_bps_range=(0.0, 3.0), enabled=True)
    pb = PaperBroker(tick_source=src, clock=clock, execution_cost=ec,
                     initial_balance=100000.0)

    # ---- BUY entry (fill = ask + adverse slip) ----
    rb = pb.place_order("EURJPY", "BUY", 1.0, 0.0)
    tick_b = src.last.get("EURJPY")
    pos_b = pb._positions[rb["ticket"]]
    check("BUY entry >= ask (fill side + adverse slip)",
          pos_b["entry"] >= tick_b["ask"],
          f"entry={pos_b['entry']:.6f} ask={tick_b['ask']:.6f}")
    check("BUY open commission = one leg (per-lot)",
          abs(pos_b["commission"] - ec.commission(1.0)) < 1e-9 and ec.commission(1.0) > 0,
          f"comm={pos_b['commission']:.2f}")
    check("open position leaves balance untouched", pb._balance == 100000.0)

    # ---- SELL entry ----
    rs = pb.place_order("EURJPY", "SELL", 1.0, 0.0)
    tick_s = src.last.get("EURJPY")
    fill_s = pb._positions[rs["ticket"]]
    check("SELL entry <= bid (fill side + adverse slip)",
          fill_s["entry"] <= tick_s["bid"],
          f"entry={fill_s['entry']:.6f} bid={tick_s['bid']:.6f}")

    # advance then close both
    clock.advance(2.0)
    pb.close_order(rb["ticket"])   # BUY exits at bid
    last_after = src.last.get("EURJPY")
    hb = pb._positions[rb["ticket"]]
    check("BUY exit == bid at close tick", abs(hb["close_price"] - last_after["bid"]) < 1e-12,
          f"close={hb['close_price']:.6f} bid={last_after['bid']:.6f}")

    pb.close_order(rs["ticket"])   # SELL exits at ask
    last2 = src.last.get("EURJPY")
    hs = pb._positions[rs["ticket"]]
    check("SELL exit == ask at close tick", abs(hs["close_price"] - last2["ask"]) < 1e-12,
          f"close={hs['close_price']:.6f} ask={last2['ask']:.6f}")

    # ---- PnL accounting ----
    pv = pip_value_per_lot("EURJPY", price=hb["entry"])
    check("JPY pip value == 1000/entry", abs(pv - 1000.0 / hb["entry"]) < 1e-9,
          f"pip_val={pv:.6f}")
    # both legs commission borne by the trade (charged at open, charged at close)
    check("round-trip commission both legs",
          abs(hb["commission"] - 2 * ec.commission(1.0)) < 1e-6 and
          abs(hs["commission"] - 2 * ec.commission(1.0)) < 1e-6,
          f"BUY_comm={hb['commission']:.2f} SELL_comm={hs['commission']:.2f}")
    # favorable long move -> net PnL counts commission + (spread recoverable)
    check("realized PnL updates balance/equity",
          pb._balance != 100000.0 and abs(pb.total_pnl - (pb._balance - 100000.0)) < 1e-6,
          f"balance={pb._balance:.2f} pnl={pb.total_pnl:.2f}")

    # legacy zero-friction default is fully untouched (backward compat)
    pb_legacy = PaperBroker(tick_source=TickSource({"EURJPY": build_eurjpy_tape()}),
                            clock=Clock())
    tl_src = pb_legacy._tick_source
    rl = pb_legacy.place_order("EURJPY", "BUY", 1.0, 0.0)
    tleg = tl_src.last.get("EURJPY")  # tick the broker actually filled on
    check("no-arg PaperBroker == zero friction (legacy parity)",
          abs(pb_legacy._positions[rl["ticket"]]["entry"] - tleg["ask"]) < 1e-12 and
          pb_legacy._positions[rl["ticket"]]["commission"] == 0.0,
          f"entry={pb_legacy._positions[rl['ticket']]['entry']:.6f} ask={tleg['ask']:.6f} comm={pb_legacy._positions[rl['ticket']]['commission']}")

    print(f"\nPASS_COUNT: {PASS}  FAILURES: {len(FAIL)}")
    if FAIL:
        print("FAILED:", "; ".join(FAIL))
        sys.exit(1)
    print("EXECUTION PARITY OK")


if __name__ == "__main__":
    run()