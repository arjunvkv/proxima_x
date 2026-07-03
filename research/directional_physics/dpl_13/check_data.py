import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from proxima_ops.execution.mt5_connector import MT5Connector
import MetaTrader5 as mt5
from datetime import datetime, timedelta

conn = MT5Connector()
if conn.connect():
    print("MT5 connected")
    for sym in ["XAUUSD", "EURJPY", "USDJPY", "GBPJPY"]:
        broker = conn._get_broker_symbol(sym)
        print(f"{sym} -> broker={broker}")
        mt5.symbol_select(broker, True)
        rates = mt5.copy_rates_from_pos(broker, mt5.TIMEFRAME_M5, 0, 10)
        if rates is not None:
            t0 = datetime.fromtimestamp(rates[0]["time"])
            t1 = datetime.fromtimestamp(rates[-1]["time"])
            print(f"  Got {len(rates)} M5 bars: {t0} to {t1}")
            start = datetime.now() - timedelta(days=365*3)
            rates_all = mt5.copy_rates_from(broker, mt5.TIMEFRAME_M5, start, 100000)
            if rates_all is not None:
                dt0 = datetime.fromtimestamp(rates_all[0]["time"])
                dt1 = datetime.fromtimestamp(rates_all[-1]["time"])
                print(f"  Available from {start.date()}: {len(rates_all)} bars ({dt0.date()} to {dt1.date()})")
            else:
                print(f"  copy_rates_from failed: {mt5.last_error()}")
        else:
            print(f"  No data: {mt5.last_error()}")
    conn.disconnect()
else:
    print("MT5 connection failed")
