"""verify_live_deal_alignment.py — FTMO-Demo MICRO-RUN: per-trade backtest == live.

Places 2 tiny EURIPY round-trips (0.01 lot) on the live FTMO-Demo terminal,
captures the live entry/exit deal prices, then feeds the SAME prices through
PaperBroker (with the broker's authoritative tick_value) and compares each
trade 1:1: entry, exit, commission, net PnL.

Acceptance: backtest per-trade net PnL == live per-trade net PnL within
+/-$0.01 (MT5 reports deal profit rounded to cents; PaperBroker keeps full
precision), entry/exit price delta <= 0.001. This is the user's explicit
criterion, exercised against the real terminal with minimal, reversible,
closed-in-seconds exposure.
"""
import os, sys, time, json
sys.path.insert(0, r"C:\Trading\Proxima_X")
os.chdir(r"C:\Trading\Proxima_X")

os.environ["MT5_PATH"] = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
os.environ["MT5_ACCOUNT"] = ""
os.environ["MT5_PASSWORD"] = ""
os.environ["MT5_SERVER"] = ""

import MetaTrader5 as mt5
from data.execution_cost import ExecutionCost
from core.adapters.broker import PaperBroker

SYMBOL = "EURJPY"
LOT = 0.01
COMMISSION_PER_LOT = 3.0


class Feed:
    def __init__(self, pb):
        self.pb = pb

    def get_tick(self, sym):
        return self.pb._tick


def run_one(pb, side, live_entry, live_exit):
    """Replay one live trade through PaperBroker; return its history row."""
    is_buy = side == "BUY"
    # PaperBroker fills BUY @ask, SELL @bid; closes BUY @bid, SELL @ask.
    if is_buy:
        pb._tick = {"bid": live_entry - 0.03, "ask": live_entry,
                    "point": 0.001, "digits": 3, "time_sec": 0, "symbol": SYMBOL}
    else:
        pb._tick = {"bid": live_entry, "ask": live_entry + 0.03,
                    "point": 0.001, "digits": 3, "time_sec": 0, "symbol": SYMBOL}
    r = pb.place_order(SYMBOL, side, LOT, 0.0)
    if not r:
        return None
    if is_buy:
        pb._tick = {"bid": live_exit, "ask": live_exit + 0.03,
                    "point": 0.001, "digits": 3, "time_sec": 0, "symbol": SYMBOL}
    else:
        pb._tick = {"bid": live_exit - 0.03, "ask": live_exit,
                    "point": 0.001, "digits": 3, "time_sec": 0, "symbol": SYMBOL}
    pb.close_order(r["ticket"])
    return pb._history[-1]


def main():
    if not mt5.initialize(path=os.environ["MT5_PATH"], timeout=5000):
        print("MT5 init FAILED:", mt5.last_error())
        return 1
    acc = mt5.account_info()
    print(f"account: login={acc.login} server={acc.server} "
          f"balance={acc.balance:.2f} {acc.currency}")
    si = mt5.symbol_info(SYMBOL)
    print(f"symbol: {SYMBOL} point={si.point} digits={si.digits} "
          f"tick_size={si.trade_tick_size} tick_value={si.trade_tick_value}")
    broker_sym = si.name
    tick_value = float(si.trade_tick_value)

    ec = ExecutionCost(commission_per_lot=COMMISSION_PER_LOT, min_commission=0.0,
                       slippage_bps_range=(0.0, 0.0))
    pb = PaperBroker(execution_cost=ec, tick_value_map={SYMBOL: tick_value})
    pb._clock = type("C", (), {"time": lambda s: 0.0,
                               "sleep": lambda s, x: time.sleep(x)})()
    pb._tick_source = Feed(pb)

    rows = []
    for side, mtype in [("BUY", 0), ("SELL", 1)]:
        print(f"\n=== {side} round-trip (0.01 lot) ===")
        mt5.symbol_info_tick(broker_sym)
        req = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": broker_sym,
            "volume": LOT, "type": mtype, "price": 0.0, "deviation": 5,
            "magic": 777000 + mtype, "comment": "HERMES_ALIGN", "type_time": 0,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        res = mt5.order_send(req)
        if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"  OPEN failed retcode={getattr(res, 'retcode', None)} "
                  f"comment={getattr(res, 'comment', None)}")
            mt5.shutdown()
            return 1
        time.sleep(0.5)
        poss = mt5.positions_get(symbol=broker_sym)
        pos = poss[-1] if poss else None
        position_id = int(pos.ticket) if pos is not None else int(res.order)
        t_fill = mt5.symbol_info_tick(broker_sym)

        out_type = 1 - mtype
        exit_price = float(t_fill.bid if mtype == 0 else t_fill.ask)
        creq = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": broker_sym,
            "volume": LOT, "type": out_type, "position": position_id,
            "price": exit_price, "deviation": 5,
            "magic": 777100 + mtype, "comment": "HERMES_ALIGN_EXIT", "type_time": 0,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        cres = mt5.order_send(creq)
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
        print(f"  LIVE entry={live_entry} exit={live_exit} "
              f"profit={live_profit} comm={live_comm} swap={float(out_d.swap)}")

        h = run_one(pb, side, live_entry, live_exit)
        if h is None:
            print("  paper replay failed")
            continue
        # net-to-net: live net = deal profit + both-leg commission (both MT5
        # signed); paper net = PaperBroker realized PnL (already net of comm).
        live_net = live_profit + live_comm
        paper_net = float(h["profit"])
        d_entry = live_entry - h["entry"]
        d_exit = live_exit - h["close"]
        d_net = live_net - paper_net
        ok = abs(d_net) <= 0.01 and abs(d_entry) <= 0.001 and abs(d_exit) <= 0.001
        print(f"  PAPER entry={h['entry']} exit={h['close']} profit={paper_net:.4f}")
        rows.append({
            "side": side, "position_id": position_id, "pass": ok,
            "live_entry": live_entry, "paper_entry": round(h["entry"], 4),
            "delta_entry": round(d_entry, 4),
            "live_exit": live_exit, "paper_exit": round(h["close"], 4),
            "delta_exit": round(d_exit, 4),
            "live_net_pnl": round(live_net, 4), "paper_net_pnl": round(paper_net, 4),
            "delta_net_pnl": round(d_net, 4),
            "commission_live": round(live_comm, 2),
        })

    npass = sum(1 for r in rows if r["pass"])
    report = {
        "harness": "verify_live_deal_alignment", "symbol": SYMBOL, "lot": LOT,
        "broker_tick_value": tick_value, "trades": rows,
        "aligned": npass, "total": len(rows),
        "verdict": ("PASS" if rows and npass == len(rows) else "MISALIGNED"),
        "note": ("per-trade net PnL == live within +/-$0.01 (MT5 profit-field "
                 "cent-rounding); entry/exit price delta <= 0.001"),
    }
    try:
        with open("deal_alignment_report.json", "w") as f:
            json.dump(report, f, indent=2)
    except Exception as exc:
        print("  could not write report:", exc)

    print("\n=== SUMMARY ===")
    for r in rows:
        print(json.dumps(r))
    print(f"VERDICT: {report['verdict']}  ({report['aligned']}/{report['total']} trades aligned)")
    mt5.shutdown()
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
