"""eur_align_check.py — dump cached-vs-terminal EURUSD daily closes on common days."""
import sys, json, os
import numpy as np
import MetaTrader5 as mt5

sys.path.insert(0, "scripts"); sys.path.insert(0, "scripts/_absorb"); sys.path.insert(0, ".")
from proxima_ops.backtest.feed import load_bars_cached
from proxima_ops.nova import feed as nfeed
from proxima_ops.nova import factors as F

def lastc(s):
    b = nfeed.bars_list_to_arrays(load_bars_cached(s))
    d = F.bar_day(b["ts"])
    dd = np.unique(d)
    return dd, b["open"][np.searchsorted(d, dd, side="left")], \
        b["close"][np.searchsorted(d, dd, side="right") - 1]

def m5_to_daily_terminal(sym, n=15000):
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
    cd, co, cc = lastc("EURUSD")
    td, to, tc = m5_to_daily_terminal("EURUSD")
    print(f"[cached EUR] n={len(cd)} {cd[0]}..{cd[-1]}")
    print(f"[term   EUR] n={len(td)} {td[0]}..{td[-1]}")
    common = np.intersect1d(cd, td)
    i1 = np.searchsorted(cd, common); i2 = np.searchsorted(td, common)
    print(f"common={len(common)}")
    print("day     cached_close  term_close   diff")
    for k in range(max(0, len(common) - 12), len(common)):
        d = common[k]
        print(f"{d}  {cc[i1[k]]:.5f}  {tc[i2[k]]:.5f}  {cc[i1[k]] - tc[i2[k]]:+.5f}")
    # also check DXY closes diff
    cd2, co2, cc2 = lastc("DXY.cash")
    td2, to2, tc2 = m5_to_daily_terminal("DXY.cash")
    common2 = np.intersect1d(cd2, td2)
    j1 = np.searchsorted(cd2, common2); j2 = np.searchsorted(td2, common2)
    print("DXY day     cached_close  term_close   diff")
    for k in range(max(0, len(common2) - 12), len(common2)):
        d = common2[k]
        print(f"{d}  {cc2[j1[k]]:.4f}  {tc2[j2[k]]:.4f}  {cc2[j1[k]] - tc2[j2[k]]:+.4f}")
    mt5.shutdown()
