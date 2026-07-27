"""Download recent M1 data from MT5 to verify live trades."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import pandas as pd
import datetime
import MetaTrader5 as mt5
import time as _time

PAIRS = ["GBPNZD", "EURNZD", "GBPAUD", "EURAUD", "GBPCAD", "AUDNZD"]
N_BARS = 500

if not mt5.initialize():
    print("MT5 initialize failed")
    sys.exit(1)

# Get data using current time (MT5 server time) — fetch last N_BARS bars
now = int(_time.time())
rates = mt5.copy_rates_from(PAIRS[0], mt5.TIMEFRAME_M1, now, 1)
if rates is not None and len(rates) > 0:
    server_now = rates[0][0]  # time of most recent bar
    sdt = datetime.datetime.fromtimestamp(server_now)
    udt = datetime.datetime.fromtimestamp(now)
    offset = (server_now - now) / 3600
    print(f"Current time (local): {udt} UTC epoch={now}")
    print(f"Last M1 bar (server): {sdt} epoch={server_now}")
    print(f"Server offset: {offset:+.1f}h from UTC")

# Download for each pair
for pair in PAIRS:
    rates = mt5.copy_rates_from(pair, mt5.TIMEFRAME_M1, now, N_BARS)
    if rates is None or len(rates) == 0:
        print(f"  {pair}: NO DATA")
        continue
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")
    out_path = f"research/cppf/_live_data/{pair.lower()}.parquet"
    os.makedirs("research/cppf/_live_data", exist_ok=True)
    df.to_parquet(out_path)
    print(f"  {pair}: {len(df)} bars, {df.index[0]} to {df.index[-1]}")

mt5.shutdown()
print("Done")
