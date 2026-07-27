"""Check MT5 M1 data availability for each pair."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import datetime
import MetaTrader5 as mt5

PAIRS = ["GBPNZD", "EURNZD", "GBPAUD", "EURAUD", "GBPCAD", "AUDNZD"]

if not mt5.initialize():
    print("MT5 init failed")
    sys.exit(1)

print(f"Timestamp now: {int(datetime.datetime.now().timestamp())}")
print(f"  UTC: {datetime.datetime.utcnow()}")

for pair in PAIRS:
    # Get oldest available bar by requesting a huge count from epoch 0
    rates_chunk = mt5.copy_rates_from(pair, mt5.TIMEFRAME_M1, 86400, 1)
    if rates_chunk is None or len(rates_chunk) == 0:
        print(f"{pair}: No data")
        continue
    
    # Get count by requesting from year 2000
    from_ts = int(datetime.datetime(2015, 1, 1).timestamp())
    rates_from_start = mt5.copy_rates_from(pair, mt5.TIMEFRAME_M1, from_ts, 100000)
    if rates_from_start is not None and len(rates_from_start) > 0:
        first = datetime.datetime.fromtimestamp(rates_from_start[0][0])
        last = datetime.datetime.fromtimestamp(rates_from_start[-1][0])
        n = len(rates_from_start)
        print(f"{pair}: {n} M1 bars, {first} to {last} (UTC)")
    else:
        # Try getting bars from now backwards with large count
        now = int(datetime.datetime.now().timestamp())
        rates_now = mt5.copy_rates_from(pair, mt5.TIMEFRAME_M1, now, 50000)
        if rates_now is not None and len(rates_now) > 0:
            first = datetime.datetime.fromtimestamp(rates_now[0][0])
            last = datetime.datetime.fromtimestamp(rates_now[-1][0])
            n = len(rates_now)
            print(f"{pair}: {n} M1 bars, {first} to {last} (UTC)")
        else:
            print(f"{pair}: Could not get data")

mt5.shutdown()
