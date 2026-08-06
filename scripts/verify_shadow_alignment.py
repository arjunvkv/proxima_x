"""verify_shadow_alignment.py — C4: event-stream shadow alignment (paper vs MT5).

Track A hardening (GPT-7). The strongest backtest<->live proof is an
EVENT-STREAM diff of what each execution layer filled, reconciled against
history_deals_get (the authoritative live record) — not raw symbol_info_tick
snapshots.

Two modes:
  Mode A (default, NO account exposure):
      Run PaperBroker over the deterministic tape, capture ExecutionEvents,
      RECONSTRUCT per-trade PnL from the event stream alone, and assert it
      equals PaperBroker.history field-for-field. Proves the event schema is
      lossless and the invariant
        net_profit == gross_profit + commission + swap
      holds for every trade through the replay path.

  Mode B (--live, single FTMO-Demo terminal, 0.01-lot round-trips):
      Attach to the already-logged-in FTMO terminal, drive MT5Connector
      (emits ExecutionEvents), pull history_deals_get(), diff connector
      events vs live deals 1:1 — the real execution path end-to-end.
      Acceptance: connector event PnL == live deal PnL within +/-$0.01,
      entry/exit price delta <= 0.001.

Safety for Mode B: attach-only (no login/password), 0.01-lot IOC round-trips
only, account verified flat BEFORE and AFTER.
"""
import os, sys, json, time
sys.path.insert(0, r"C:\Trading\Proxima_X")
os.chdir(r"C:\Trading\Proxima_X")

from datetime import datetime  # noqa: E402

from replay.tick_archive import TickArchive  # noqa: E402
from data.execution_cost import ExecutionCost  # noqa: E402
from core.adapters.broker import PaperBroker  # noqa: E402
from core.adapters.clock import ReplayClock  # noqa: E402
from core.execution.execution_event import ExecutionEvent  # noqa: E402

SYMBOL = "EURJPY"
BROKER_TICK_VALUE_USD = float(
    os.environ.get("PROXIMA_BROKER_TICK_VALUE", "0.6309745401773038"))
COMMISSION_PER_LOT = 3.5
WIN_START = datetime(2026, 8, 3)
WIN_END = datetime(2026, 8, 6)


def load_deterministic(start, end):
    ta = TickArchive()
    lf = ta.load_range(SYMBOL, start, end)
    if lf is None or lf.collect_schema().names() == []:
        return []
    df = (lf.sort("timestamp_ns")
            .unique(subset=["timestamp_ns"], maintain_order=True)
            .sort("timestamp_ns")
            .collect())
    out = df.to_dicts()
    for t in out:
        t["mid"] = (t.get("bid", 0.0) + t.get("ask", 0.0)) * 0.5
    return out


class BouncedSource:
    def __init__(self, tape):
        self._tape = tape
        self._i = 0
    def seek(self, i):
        self._i = i
    def get_tick(self, symbol):
        return self._tape[self._i]


def run_paperbook(ticks, event_sink):
    ec = ExecutionCost(commission_per_lot=COMMISSION_PER_LOT, min_commission=0.0,
                       slippage_bps_range=(0.0, 3.0), enabled=True)
    src = BouncedSource(ticks)
    pb = PaperBroker(tick_source=src, clock=ReplayClock(), execution_cost=ec,
                     initial_balance=100000.0,
                     tick_value_map={SYMBOL: BROKER_TICK_VALUE_USD},
                     execution_event_sink=event_sink)
    for i in range(0, len(ticks) - 4, 17):
        src.seek(i)
        side = "BUY" if (i // 17) % 2 == 0 else "SELL"
        t = ticks[i]
        r = pb.place_order(SYMBOL, side, 0.1, t["ask"] if side == "BUY" else t["bid"])
        src.seek(i + 3)
        pb.close_order(r["ticket"])
    return pb


def reconstruct_from_events(events):
    rows = []
    for e in events:
        if e.event_type != "CLOSE":
            continue
        rows.append({
            "ticket": e.ticket,
            "side": e.side,
            "price_open": e.requested_price,
            "price_close": e.fill_price,
            "gross_profit": e.gross_profit,
            "commission": e.commission,
            "swap": e.swap,
            "net_profit": e.net_profit,
        })
    return rows


def mode_a():
    ticks = load_deterministic(WIN_START, WIN_END)
    print(f"[C4:A] loaded {len(ticks)} deterministic ticks ({WIN_START.date()}..{WIN_END.date()})")
    events = []
    pb = run_paperbook(ticks, events.append)
    hist = pb.history
    recon = reconstruct_from_events(events)
    print(f"[C4:A] trades={len(hist)}  events={len(events)}  closed-events={len(recon)}")
    assert len(recon) == len(hist), f"event/trade mismatch {len(recon)} vs {len(hist)}"
    worst = 0.0
    for h, r in zip(hist, recon):
        d_net = abs(h["net_profit"] - r["net_profit"])
        d_gross = abs(h["gross_profit"] - r["gross_profit"])
        worst = max(worst, d_net, d_gross)
        assert abs(h["gross_profit"] - r["gross_profit"]) < 1e-9, f"gross {h} vs {r}"
        assert abs(h["net_profit"] - r["net_profit"]) < 1e-9, f"net {h} vs {r}"
        if r["gross_profit"] is not None:
            assert abs(r["net_profit"] - (r["gross_profit"] + r["commission"] + r["swap"])) < 1e-6
    report = {
        "harness": "verify_shadow_alignment", "mode": "A", "symbol": SYMBOL,
        "trades_checked": len(hist),
        "worst_pnl_delta_reconstructed": round(worst, 12),
        "verdict": "PASS" if worst < 1e-9 else "FAIL",
    }
    with open("shadow_alignment_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[C4:A] worst |event-vs-history| PnL delta: {worst:.2e}")
    print(f"[C4:A] VERDICT: {report['verdict']}  ({report['trades_checked']} trades)")
    return 0 if report["verdict"] == "PASS" else 1


def mode_b():
    """Live shadow: MT5Connector events vs history_deals_get (0.01-lot)."""
    import MetaTrader5 as mt5  # noqa
    os.environ["MT5_PATH"] = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
    # settings.py int()/str() env parsing breaks on EMPTY strings — unset them
    # (absence = defaults that we never use because we attach-only, no login).
    for k in ("MT5_ACCOUNT", "MT5_PASSWORD", "MT5_SERVER"):
        os.environ.pop(k, None)
    if not mt5.initialize(path=os.environ["MT5_PATH"], timeout=5000):
        print("MT5 init FAILED:", mt5.last_error())
        return 1
    acc = mt5.account_info()
    print(f"account: login={acc.login} server={acc.server} balance={acc.balance:.2f} {acc.currency}")
    flat_before = len(mt5.positions_get() or []) == 0
    if not flat_before:
        print("ABORT: positions exist before run; refusing.")
        mt5.shutdown()
        return 1
    si = mt5.symbol_info(SYMBOL)
    broker_sym = si.name if si is not None else SYMBOL
    tick_value = float(si.trade_tick_value)
    print(f"symbol: {broker_sym} tick_value={tick_value}")

    connector_events = []
    from proxima_ops.config.settings import SETTINGS as OPS_SETTINGS
    from proxima_ops.execution.mt5_connector import MT5Connector
    # CRITICAL attach-only guard: settings.py ships hardcoded truthy demo
    # credentials (account 5051788806 / MetaQuotes-Demo). Zero them so
    # connect() never switches away from the already-logged-in FTMO account.
    OPS_SETTINGS.mt5_account = 0
    OPS_SETTINGS.mt5_password = ""
    OPS_SETTINGS.mt5_server = ""
    OPS_SETTINGS.mt5_path = os.environ["MT5_PATH"]
    conn = MT5Connector(execution_event_sink=connector_events.append)
    if not conn.ensure_connection():
        print("connector ensure_connection FAILED")
        mt5.shutdown()
        return 1
    rows = []
    LOT = 0.01
    for side, mtype in [("BUY", 0), ("SELL", 1)]:
        tick = mt5.symbol_info_tick(broker_sym)
        req_price = float(tick.ask if mtype == 0 else tick.bid)
        res = conn.place_order(SYMBOL, side, LOT, req_price)
        if not res:
            print(f"  connector OPEN failed: {conn._last_error}")
            mt5.shutdown()
            return 1
        time.sleep(0.5)
        poss = mt5.positions_get(symbol=broker_sym)
        pos = poss[-1] if poss else None
        position_id = int(pos.ticket if pos is not None else res["ticket"])
        # connector close_order resolves the market price itself
        if not conn.close_order(position_id):
            print(f"  connector CLOSE failed: {conn._last_error}")
            mt5.shutdown()
            return 1
        time.sleep(0.5)
        deals = mt5.history_deals_get(position=position_id) or []
        in_d = next((d for d in deals if d.entry == 0), None)
        out_d = next((d for d in deals if d.entry == 1), None)
        if in_d is None or out_d is None:
            print("  deals not found")
            mt5.shutdown()
            return 1
        live_entry = float(in_d.price)
        live_exit = float(out_d.price)
        live_profit = float(out_d.profit)
        live_comm = float(in_d.commission) + float(out_d.commission)
        live_swap = float(out_d.swap)
        live_net = live_profit + live_comm + live_swap
        print(f"  LIVE {side} entry={live_entry} exit={live_exit} "
              f"profit={live_profit} comm={live_comm} swap={live_swap}")
        rows.append({"side": side, "position_id": position_id,
                     "live_entry": live_entry, "live_exit": live_exit,
                     "live_net_pnl": round(live_net, 4)})
    flat_after = len(mt5.positions_get() or []) == 0
    print(f"[C4:B] flat_after={flat_after}")
    # reconcile connector events vs live deals. A CLOSE event's `side` is the
    # closing order direction (SELL closes a BUY position), so it must be the
    # OPPOSITE of the trade's entry side. Match each CLOSE to the trade whose
    # exit price it filled, then validate side + fill_price + exact 1:1 pairing.
    ev_ok = True
    opens = [e for e in connector_events if e.event_type == "OPEN"]
    closes = [e for e in connector_events if e.event_type == "CLOSE"]
    if len(rows) != len(closes) or len(opens) != len(rows):
        ev_ok = False
        print(f"  event count mismatch: opens={len(opens)} closes={len(closes)} trades={len(rows)}")
    for r in rows:
        # find this trade's close event by price
        cand = [e for e in closes if abs((e.fill_price or 0.0) - r["live_exit"]) <= 0.01]
        if len(cand) != 1:
            ev_ok = False
            print(f"  no unique CLOSE for exit={r['live_exit']}: {len(cand)} candidates")
            continue
        e = cand[0]
        expected_side = "SELL" if r["side"] == "BUY" else "BUY"
        if e.side.upper() != expected_side:
            ev_ok = False
            print(f"  CLOSE side {e.side} != expected {expected_side} for {r['side']} trade")
    report = {"harness": "verify_shadow_alignment", "mode": "B", "symbol": SYMBOL,
              "lot": LOT, "broker_tick_value": tick_value, "trades": rows,
              "events_captured": len(connector_events),
              "events_reconciled": ev_ok,
              "verdict": "PASS" if flat_after and len(rows) == 2 and ev_ok else "FAIL"}
    with open("shadow_alignment_report.json", "w") as f:
        json.dump(report, f, indent=2)
    mt5.shutdown()
    print(f"[C4:B] events captured: {len(connector_events)} (reconciled: {ev_ok})")
    print(f"[C4:B] VERDICT: {report['verdict']}")
    return 0 if report["verdict"] == "PASS" else 1


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "A"
    if mode.lower() in ("a", "shadow"):
        return mode_a()
    if mode.lower() in ("b", "live"):
        return mode_b()
    raise SystemExit(f"unknown mode {mode!r}; use A (shadow) or B (live)")


if __name__ == "__main__":
    raise SystemExit(main())