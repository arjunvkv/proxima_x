import sys
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timezone, timedelta

ULTRA_MONSTER_UNIVERSE = [
    "EURAUD", "GBPAUD", "EURNZD", "GBPNZD", "GBPUSD",
    "EURUSD", "EURJPY", "USDJPY", "GBPJPY", "AUDCAD",
    "GBPCAD", "AUDNZD", "EURCAD", "NZDUSD", "AUDCHF"
]

CROSS_PIP_MULT = {
    "EURAUD": 6.70, "GBPAUD": 6.70, "AUDNZD": 5.80,
    "EURNZD": 6.10, "GBPNZD": 6.10, "GBPCAD": 7.80,
    "EURCAD": 7.80, "AUDCAD": 7.80, "AUDCHF": 10.50,
}

def pip_val_usd(pair: str) -> float:
    return CROSS_PIP_MULT.get(pair, 10.0)

def pip_size(pair: str) -> float:
    return 0.01 if "JPY" in pair else 0.0001

def load_all_m5_data(target_date_str="2026-08-03"):
    account_info = {"login": 1514168544, "password": "$!4fwBIc", "server": "FTMO-Demo"}
    if not mt5.initialize():
        return {}
    mt5.login(login=int(account_info["login"]), password=account_info["password"], server=account_info["server"])

    dt = datetime.strptime(target_date_str, "%Y-%m-%d")
    # Fetch from 1 day before target date (for rolling lookbacks) to end of target date
    from_dt = dt - timedelta(days=1)
    to_dt = dt + timedelta(days=1, hours=4)

    pair_dfs = {}
    for pair in ULTRA_MONSTER_UNIVERSE:
        rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M5, from_dt, to_dt)
        if rates is not None and len(rates) > 0:
            df = pd.DataFrame(rates)
            # MT5 server is EET (UTC+3), subtract 10800s for UTC
            df['utc_time'] = pd.to_datetime(df['time'] - 10800, unit='s')
            df.set_index('utc_time', inplace=True)
            pair_dfs[pair] = df

    mt5.shutdown()
    return pair_dfs

print("Testing loading M5 rates for 15 universe pairs...")
pair_dfs = load_all_m5_data("2026-08-03")
print(f"Successfully loaded data for {len(pair_dfs)} pairs!")
for pair, df in pair_dfs.items():
    print(f"  {pair}: {len(df)} M5 candles from {df.index[0]} to {df.index[-1]}")
