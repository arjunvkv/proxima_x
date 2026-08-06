import sys
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timezone, timedelta

ULTRA_MONSTER_UNIVERSE = [
    "EURAUD", "GBPAUD", "EURNZD", "GBPNZD", "GBPUSD",
    "EURUSD", "EURJPY", "USDJPY", "GBPJPY", "AUDCAD",
    "GBPCAD", "AUDNZD", "EURCAD", "NZDUSD", "AUDCHF"
]

account_info = {"login": 1514168544, "password": "$!4fwBIc", "server": "FTMO-Demo"}
mt5.initialize()
mt5.login(login=int(account_info["login"]), password=account_info["password"], server=account_info["server"])

from_dt = datetime(2026, 8, 3, 0, 0, 0)
to_dt = datetime(2026, 8, 4, 4, 0, 0)

loaded = []
for pair in ULTRA_MONSTER_UNIVERSE:
    mt5.symbol_select(pair, True)
    rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M5, from_dt, to_dt)
    if rates is not None and len(rates) > 0:
        loaded.append(pair)

print(f"Loaded {len(loaded)} / {len(ULTRA_MONSTER_UNIVERSE)} pairs:", loaded)
missing = set(ULTRA_MONSTER_UNIVERSE) - set(loaded)
print("Missing pairs:", missing)

mt5.shutdown()
