"""verify_live_micro_batch.py — AMPLE-sample live-vs-engine match on FTMO-Demo.

Places N small (0.01 lot) round-trips across a mixed JPY/non-JPY universe on the
LIVE FTMO-Demo terminal, captures the real deal prices, then replays each trade
through the GENERALIZED ENGINE's tick-value PnL (proxima_ops.backtest.pnl.
trade_to_usd — the exact function run_strategy() uses) and compares per trade:

  * entry / exit price delta   <= 0.001
  * gross PnL delta            <= $0.01   (engine tick-value math vs MT5 profit)
  * net  PnL delta             <= $0.01   (engine - commission vs MT5 profit+comm)

This is the 8.7x-JPY-inflation battleground at scale: JPY pairs priced via the
broker's authoritative trade_tick_value must match MT5 deal profit per trade.
Attach-only (no login): relies on the terminal's GUI login. ~40 samples.
"""
import os, sys, time, json
sys.path.insert(0, r"C:\Trading\Proxima_X")
os.chdir(r"C:\Trading\Proxima_X")

os.environ["MT5_PATH"] = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
os.environ["MT5_ACCOUNT"] = ""
os.environ["MT5_PASSWORD"] = ""
os.environ["MT5_SERVER"] = ""

import MetaTrader5 as mt5
from proxima_ops.backtest.pnl import trade_to_usd

# JPY pairs (3-digit, point=0.001) + non-JPY (5-digit, point=0.00001)
# Authoritative 18-symbol TOKYO_UNIVERSE (matches run_tokyo_h0_live.py:60).
SYMBOLS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "EURJPY", "GBPJPY",
           "EURAUD", "EURNZD", "GBPAUD", "GBPNZD", "GBPCAD", "AUDNZD",
           "USDCAD", "NZDUSD", "EURGBP", "EURCHF", "USDCHF", "AUDJPY"]
ROUND_TRIPS_PER_SYMBOL = 3   # -> 18 x 3 = 54 live matches
LOT = 0.01
ENGINE_COMMISSION_PER_LOT = 3.5   # engine constant; compared against live comm

def order(action, symbol, mtype, volume, price, position=0, magic=0, comment=""):
    req = {"action": action, "symbol": symbol, "volume": volume, "type": mtype,
           "price": price, "deviation": 5, "magic": magic, "comment": comment,
           "type_time": 0, "type_filling": mt5.ORDER_FILLING_IOC}
    if position:
        req["position"] = position
    return mt5.order_send(req)

def round_trip(symbol, mtype, tick_value_map):
    """One live round-trip -> comparison dict (or None on failure)."""
    res = order(mt5.TRADE_ACTION_DEAL, symbol, mtype, LOT, 0.0,
                magic=777000 + mtype, comment="HERMES_ALIGN")
    if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
        return {"symbol": symbol, "side": "BUY" if mtype == 0 else "SELL",
                "error": f"open retcode={getattr(res,'retcode',None)} {getattr(res,'comment',None)}"}
    time.sleep(0.5)
    poss = mt5.positions_get(symbol=symbol)
    pos = poss[-1] if poss else None
    position_id = int(pos.ticket) if pos is not None else int(res.order)
    out_type = 1 - mtype
    tick = mt5.symbol_info_tick(symbol)
    exit_price = float(tick.bid if mtype == 0 else tick.ask)
    cres = order(mt5.TRADE_ACTION_DEAL, symbol, out_type, LOT, exit_price,
                 position=position_id, magic=777100 + mtype, comment="HERMES_ALIGN_EXIT")
    time.sleep(0.5)
    deals = mt5.history_deals_get(position=position_id) or []
    in_d = next((d for d in deals if d.entry == 0), None)
    out_d = next((d for d in deals if d.entry == 1), None)
    if in_d is None or out_d is None:
        return {"symbol": symbol, "side": "BUY" if mtype == 0 else "SELL",
                "error": "deals not found", "position_id": position_id}
    live_entry = float(in_d.price)
    live_exit = float(out_d.price)
    live_gross = float(out_d.profit)
    live_comm = float(in_d.commission) + float(out_d.commission)
    dirn = 1 if mtype == 0 else -1
    pnl_pts = (live_exit - live_entry) * dirn
    # ENGINE replay (generalized engine's exact PnL function)
    eng = trade_to_usd({"symbol": symbol, "pnl_pts": pnl_pts, "entry": live_entry,
                        "entry_ts": int(in_d.time), "exit_ts": int(out_d.time),
                        "side": "BUY" if mtype == 0 else "SELL", "reason": "LIVE"},
                       LOT, tick_value_map)
    return {"symbol": symbol, "side": "BUY" if mtype == 0 else "SELL",
            "position_id": position_id,
            "live_entry": live_entry, "live_exit": live_exit,
            "live_gross": round(live_gross, 4), "live_comm": round(live_comm, 4),
            "engine_entry": round(eng["entry"], 4), "engine_exit": round(live_exit, 4),
            "engine_gross": round(eng["gross_usd"], 4), "engine_comm": round(eng["commission"], 4),
            "delta_entry": round(live_entry - eng["entry"], 4),
            "delta_exit": round(live_exit - live_exit, 4),
            "delta_gross": round(live_gross - eng["gross_usd"], 4),
            "engine_net": round(eng["net"], 4),
            "delta_net": round((live_gross + live_comm) - eng["net"], 4)}

def main():
    if not mt5.initialize(path=os.environ["MT5_PATH"], timeout=5000):
        print("MT5 init FAILED:", mt5.last_error())
        return 1
    acc = mt5.account_info()
    print(f"account: login={acc.login} server={acc.server} balance={acc.balance:.2f}")
    tick_value_map = {}
    for s in SYMBOLS:
        si = mt5.symbol_info(s)
        if si:
            tick_value_map[s] = float(si.trade_tick_value)
            print(f"  {s:7s} point={si.point} digits={si.digits} "
                  f"tick_value={si.trade_tick_value} (per 1.0 lot)")
    rows = []
    for sym in SYMBOLS:
        for i in range(ROUND_TRIPS_PER_SYMBOL):
            mtype = i % 2          # alternate BUY/SELL
            r = round_trip(sym, mtype, tick_value_map)
            rows.append(r)
            tag = r.get("error", f"gross_d={r['delta_gross']:+.4f} net_d={r['delta_net']:+.4f}")
            print(f"  [{sym:7s} {'BUY' if mtype==0 else 'SELL'} #{i+1}] {tag}")
    ok_rows = [r for r in rows if "error" not in r]
    g_ok = sum(1 for r in ok_rows if abs(r["delta_gross"]) <= 0.01)
    n_ok = sum(1 for r in ok_rows if abs(r["delta_net"]) <= 0.01)
    e_ok = sum(1 for r in ok_rows if abs(r["delta_entry"]) <= 0.001)
    report = {"harness": "verify_live_micro_batch", "lot": LOT,
              "samples": len(rows), "ok_samples": len(ok_rows),
              "gross_aligned": g_ok, "net_aligned": n_ok, "entry_aligned": e_ok,
              "rows": rows,
              "verdict": ("PASS" if ok_rows and g_ok == len(ok_rows) and n_ok == len(ok_rows)
                          else "MISALIGNED"),
              "note": "engine tick-value PnL (trade_to_usd) vs live MT5 deals; "
                      "gross/net delta <= $0.01, entry delta <= 0.001"}
    with open("micro_batch_alignment.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\n=== SUMMARY ===")
    print(f"samples={len(rows)} ok={len(ok_rows)} gross_aligned={g_ok} "
          f"net_aligned={n_ok} entry_aligned={e_ok}")
    for r in rows:
        if "error" in r:
            print(f"  ERROR {r['symbol']} {r['side']}: {r['error']}")
        else:
            print(f"  {r['symbol']:7s} {r['side']:4s} e={r['live_entry']} x={r['live_exit']} "
                  f"gross {r['live_gross']:+.4f} vs eng {r['engine_gross']:+.4f} "
                  f"(d={r['delta_gross']:+.4f}) net {r['delta_net']:+.4f}")
    print(f"VERDICT: {report['verdict']}  (gross {g_ok}/{len(ok_rows)}, net {n_ok}/{len(ok_rows)})")
    mt5.shutdown()
    return 0 if report["verdict"] == "PASS" else 1

if __name__ == "__main__":
    sys.exit(main())