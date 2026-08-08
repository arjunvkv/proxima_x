"""Pull REAL FTMO-Demo spreads (bid/ask) for all 18 book pairs from the live
terminal on the VPS. Spread in pips + points + USD/lot round-trip."""
import sys, time, datetime
import MetaTrader5 as mt5

MT5_PATH = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
UNIVERSE = ["EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD","EURNZD",
            "GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF",
            "USDCHF","AUDJPY"]

ok = mt5.initialize(path=MT5_PATH)
if not ok:
    print("FATAL init:", mt5.last_error()); sys.exit(2)
acct = mt5.account_info()
print(f"attached login={acct.login} server={acct.server} "
      f"time={datetime.datetime.utcnow():%H:%M:%S}Z", flush=True)

def pip(sym): return 0.01 if "JPY" in sym else 0.0001
def point(sym): return 0.001 if "JPY" in sym else 0.00001

rows = []
for s in UNIVERSE:
    t = mt5.symbol_info_tick(s)
    si = mt5.symbol_info(s)
    if t is None or si is None:
        rows.append((s, None)); continue
    sp = t.ask - t.bid
    rows.append((s, sp, sp/pip(s), sp/point(s), si.trade_tick_value, si.trade_tick_size))

print(f"{'sym':<8}{'spread':>10}{'pips':>7}{'pts':>6}{'tickval':>9}{'tick_sz':>9}")
tot = 0.0
for r in rows:
    if r[1] is None:
        print(f"{r[0]:<8}  NO-TICK"); continue
    s, sp, pips, pts, tv, tsz = r
    usd_lot = sp / tsz * tv * 1.0  # USD per 1.0 lot round trip (spread cost)
    tot += usd_lot
    print(f"{s:<8}{sp:>10.5f}{pips:>7.2f}{pts:>8.0f}{tv:>9.4f}{tsz:>10.5f}   ${usd_lot:.2f}/lot")
print(f"\nspread sum over 18 pairs: ${tot:.2f}/lot if all traded once")
mt5.shutdown()