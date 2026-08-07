"""reconcile_live_history_deals.py — Read MT5 broker deal ledger vs OFFLINE engine.

Reads the AUTHORITATIVE MT5 history_deals ledger directly (every deal the broker
persisted for HERMES_ALIGN positions), re-groups each closed position into its
entry+exit deal pair, replays each through the ENGINE's trade_to_usd (tick-value
+ commission), and compares per pair + sum-of-nets.

This is the "get mt5 trade history and match it to the offline run" pass: it
reads the broker-persisted ledger, NOT the in-memory capture of the batch that
placed the trades. If they agree 1:1, engine PnL == broker PnL for every deal.
Attach-only (no daemon).
"""
import os, sys, time, json, collections
sys.path.insert(0, r"C:\Trading\Proxima_X")
os.chdir(r"C:\Trading\Proxima_X")
os.environ["MT5_PATH"] = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
os.environ["MT5_PASSWORD"] = ""
os.environ["MT5_SERVER"] = ""

import MetaTrader5 as mt5
from proxima_ops.backtest.pnl import trade_to_usd

TAG = "HERMES_ALIGN"

def main() -> int:
    if not mt5.initialize(path=os.environ["MT5_PATH"], timeout=5000):
        print("MT5 init FAILED:", mt5.last_error()); return 1
    acc = mt5.account_info()
    print(f"account: login={acc.login} server={acc.server} balance={acc.balance:.2f}")

    # Authoritative keys: the position_ids captured by verify_live_micro_batch.
    # (Time-windowed history_deals_get is unreliable on this terminal; per-
    #  position reads are exact.)  Full 18-symbol run = 54 round trips.
    batch = json.load(open("micro_batch_alignment.json"))
    rows_batch = [r for r in batch["rows"] if "error" not in r and r.get("position_id")]
    pids = [r["position_id"] for r in rows_batch]
    print(f"reconciling {len(pids)} captured positions from micro-batch JSON")

    tick_value_map = {}
    for r in rows_batch:
        s = r["symbol"]
        if s not in tick_value_map:
            si = mt5.symbol_info(s)
            if si:
                tick_value_map[s] = float(si.trade_tick_value)

    rows, errs = [], 0
    for pid in pids:
        ds = mt5.history_deals_get(position=pid) or []
        in_d = next((x for x in ds if x.entry == 0), None)
        out_d = next((x for x in ds if x.entry == 1), None)
        if in_d is None or out_d is None:
            errs += 1
            rows.append({"pos": pid, "error": f"deals not found ({len(ds)})"})
            continue
        sym, lot = in_d.symbol, float(in_d.volume)
        live_entry, live_exit = float(in_d.price), float(out_d.price)
        live_profit = float(out_d.profit)
        live_comm = float(in_d.commission) + float(out_d.commission)
        dirn = 1 if in_d.type == 0 else -1
        eng = trade_to_usd({"symbol": sym,
                            "pnl_pts": (live_exit - live_entry) * dirn,
                            "entry": live_entry, "side": "BUY" if in_d.type == 0 else "SELL"},
                           lot, tick_value_map)
        rows.append({
            "pos": pid, "symbol": sym, "side": "BUY" if in_d.type == 0 else "SELL",
            "entry": live_entry, "exit": live_exit, "volume": lot,
            "ledger_gross": round(live_profit, 4), "ledger_comm": round(live_comm, 4),
            "engine_gross": round(eng["gross_usd"], 4), "engine_comm": round(eng["commission"], 4),
            "ledger_net": round(live_profit + live_comm, 4), "engine_net": round(eng["net"], 4),
            "delta_gross": round(live_profit - eng["gross_usd"], 6),
            "delta_net": round((live_profit + live_comm) - eng["net"], 6),
        })

    n = len(rows)
    g_ok = sum(1 for r in rows if abs(r["delta_gross"]) <= 0.01)
    n_ok = sum(1 for r in rows if abs(r["delta_net"]) <= 0.01)
    sum_ledger = round(sum(r["ledger_net"] for r in rows), 2)
    sum_engine = round(sum(r["engine_net"] for r in rows), 2)
    report = {"harness": "reconcile_live_history_deals", "positions": n,
              "gross_aligned": g_ok, "net_aligned": n_ok, "unpaired_positions": errs,
              "sum_ledger_net": sum_ledger, "sum_engine_net": sum_engine,
              "sum_diff": round(sum_ledger - sum_engine, 2), "rows": rows,
              "verdict": "PASS" if n and g_ok == n and n_ok == n and abs(sum_ledger - sum_engine) <= 1.0
                         else "MISALIGNED"}
    with open("reconcile_live_history_deals.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== LIVE LEDGER vs OFFLINE ENGINE ===")
    print(f"positions={n} gross_aligned={g_ok} net_aligned={n_ok} unpaired={errs}")
    for r in rows:
        print(f"  {r['symbol']:7s} {r['side']:4s} e={r['entry']} x={r['exit']} "
              f"ledger_net={r['ledger_net']:+.2f} engine_net={r['engine_net']:+.2f} "
              f"(d={r['delta_net']:+.4f})")
    print(f"SUM ledger_net={sum_ledger:.2f}  engine_net={sum_engine:.2f}  diff={sum_ledger - sum_engine:+.2f}")
    print(f"VERDICT: {report['verdict']}")
    mt5.shutdown()
    return 0 if report["verdict"] == "PASS" else 1

if __name__ == "__main__":
    sys.exit(main())