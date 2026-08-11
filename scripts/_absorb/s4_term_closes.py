"""s4_term_closes.py — dump terminal daily closes (DXY.cash + EURUSD) to JSON.

Runs on the VPS (needs MT5 terminal). The cached side is dumped locally with
s4_live_feed_trace.py --source cached --dump; alignment happens locally.
"""
import sys, os, json
import numpy as np
import MetaTrader5 as mt5

def m5_to_daily(sym, n=15000):
    r = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 0, n)
    if r is None:
        print(f"no M5 for {sym}: {mt5.last_error()}"); sys.exit(1)
    days, opens, closes, last_ts = [], [], [], None
    for x in r:
        d = int(x["time"]) // 86400
        if last_ts is not None and d != last_ts:
            days.append(last_ts); opens.append(o); closes.append(c)
        last_ts, o, c = d, float(x["open"]), float(x["close"])
    days.append(last_ts); opens.append(o); closes.append(c)
    return np.array(days), np.array(opens), np.array(closes)

if __name__ == "__main__":
    if not mt5.initialize(path=os.environ["MT5_PATH"], timeout=4000):
        print("MT5 init FAILED"); sys.exit(1)
    for s in ("DXY.cash", "EURUSD"):
        mt5.symbol_select(s, True)
    out = {}
    for sym in ("DXY.cash", "EURUSD"):
        d, o, c = m5_to_daily(sym)
        out[sym] = {"days": [int(x) for x in d], "open": [float(x) for x in o],
                    "close": [float(x) for x in c]}
        print(f"[{sym}] n={len(d)} range={d[0]}..{d[-1]}")
    with open("/tmp/s4_term_closes.json", "w") as f:
        json.dump(out, f)
    print("dumped /tmp/s4_term_closes.json")
    mt5.shutdown()
