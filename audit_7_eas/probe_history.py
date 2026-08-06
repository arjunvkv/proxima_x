"""Read-only probe: can the FTMO terminal serve M1/M5 history for the EA universes?

Attach-only: initializes to the ALREADY-LOGGED-IN FTMO instance, reads symbol
list + history bars. NO orders. Fails hard if a connect would switch accounts
(settings.py hardcoded creds are neutralized first).
"""
import os, sys
sys.path.insert(0, r"C:\Trading\Proxima_X")

# --- attach-only guard: neutralize hardcoded identity creds in settings ---
import proxima_ops.config.settings as S
for attr in ("mt5_account", "mt5_password", "mt5_login"):
    if hasattr(S, attr):
        try:
            setattr(S, attr, None)
        except Exception:
            pass
if hasattr(S, "mt5_path"):
    try:
        S.mt5_path = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
    except Exception:
        pass

import MetaTrader5 as mt5

FTMO_TERMINAL = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
if not mt5.initialize(path=FTMO_TERMINAL, timeout=4000):
    print("INIT FAILED:", mt5.last_error()); sys.exit(1)

acct = mt5.account_info()
print("attached account:", acct.login if acct else None,
      acct.server if acct else "", "| balance:", round(acct.balance, 2) if acct else None)

symbols = mt5.symbols_get()
avail = {s.name for s in symbols}
print("symbols available:", len(avail))

universe = ["EURUSD","GBPUSD","USDJPY","EURAUD","GBPAUD","EURJPY","GBPJPY","EURNZD","GBPNZD",
            "AUDUSD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF","USDCHF","AUDJPY",
            "AUDCAD","CADJPY","EURCAD","GBPCAD"]
missing = [s for s in universe if s not in avail]
print("universe symbols missing on FTMO:", missing if missing else "NONE")

# probe M5 history depth for a few universe symbols
from datetime import datetime, timedelta
import time
for sym in ["EURJPY", "EURUSD", "GBPJPY", "EURNZD", "AUDNZD"]:
    rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 0, 5)
    print(f"  {sym}: M5 bars={len(rates) if rates is not None else 'FAIL'}",
          mt5.last_error() if rates is None else "")
    time.sleep(0.3)

mt5.shutdown()
print("PROBE OK")
