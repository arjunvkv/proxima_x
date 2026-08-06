"""Read-only probe #2: M5/M1 history depth available per symbol on FTMO."""
import os, sys
sys.path.insert(0, r"C:\Trading\Proxima_X")
import proxima_ops.config.settings as S
for attr in ("mt5_account", "mt5_password", "mt5_login"):
    if hasattr(S, attr):
        try: setattr(S, attr, None)
        except Exception: pass
if hasattr(S, "mt5_path"):
    try: S.mt5_path = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
    except Exception: pass

import MetaTrader5 as mt5
FTMO = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
if not mt5.initialize(path=FTMO, timeout=4000):
    print("INIT FAILED", mt5.last_error()); sys.exit(1)

from datetime import datetime, timedelta
syms = ["EURJPY","EURUSD","GBPJPY","EURNZD","AUDNZD","GBPAUD","USDJPY","EURGBP"]
# how far back can we get M5 rates?
for sym in syms:
    rates = mt5.copy_rates_range(sym, mt5.TIMEFRAME_M5,
        datetime.now() - timedelta(days=200), datetime.now())
    if rates is None:
        print(f"{sym}: FAIL {mt5.last_error()}"); continue
    n = len(rates)
    # estimate first bar date
    first = datetime.fromtimestamp(rates[0]['time'])
    last  = datetime.fromtimestamp(rates[-1]['time'])
    print(f"{sym}: M5 bars={n}  span {first.date()} -> {last.date()}  (~{(last-first).days}d)")
mt5.shutdown(); print("DEPTH PROBE OK")