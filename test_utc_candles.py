import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone, timedelta

account_info = {"login": 1514168544, "password": "$!4fwBIc", "server": "FTMO-Demo"}
mt5.initialize()
mt5.login(login=int(account_info["login"]), password=account_info["password"], server=account_info["server"])

# Query rates for 2026-08-03 00:00 UTC
# 00:00 UTC corresponds to 03:00 Server Time
from_dt = datetime(2026, 8, 3, 0, 0, 0)
to_dt = datetime(2026, 8, 3, 23, 59, 59)

rates = mt5.copy_rates_range("EURUSD", mt5.TIMEFRAME_M5, from_dt, to_dt)
if rates is not None and len(rates) > 0:
    df = pd.DataFrame(rates)
    # Convert broker epoch timestamp (EET) to UTC by subtracting 3 hours (10800s)
    df['utc_time'] = pd.to_datetime(df['time'] - 10800, unit='s')
    print("First 5 bars (converted to UTC):")
    print(df[['time', 'utc_time', 'open', 'high', 'low', 'close']].head(5))
    print("\nBar around 13:00 UTC:")
    bar13 = df[df['utc_time'] == '2026-08-03 13:00:00']
    print(bar13[['utc_time', 'open', 'high', 'low', 'close']])

mt5.shutdown()
