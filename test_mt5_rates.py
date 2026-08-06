import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone, timedelta

account_info = {"login": 1514168544, "password": "$!4fwBIc", "server": "FTMO-Demo"}

if not mt5.initialize():
    print("MT5 Init Failed:", mt5.last_error())
    sys.exit(1)

if not mt5.login(login=int(account_info["login"]), password=account_info["password"], server=account_info["server"]):
    print("MT5 Login Failed:", mt5.last_error())
    mt5.shutdown()
    sys.exit(1)

print("MT5 Connected Successfully!")
# Fetch M5 rates for EURUSD today
from_dt = datetime(2026, 8, 3, 0, 0, 0, tzinfo=timezone.utc)
to_dt = datetime.now(timezone.utc)

rates = mt5.copy_rates_range("EURUSD", mt5.TIMEFRAME_M5, from_dt, to_dt)
if rates is not None and len(rates) > 0:
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    # Adjust for MT5 EET (UTC+3) server offset if needed, or check timestamps
    print(f"Fetched {len(df)} M5 bars for EURUSD")
    print(df.head(5))
else:
    print("No rates fetched:", mt5.last_error())

mt5.shutdown()
