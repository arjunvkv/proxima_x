"""Download EURUSD M1 data from FundedNext Server 3."""
import MetaTrader5 as mt5
from datetime import datetime
import numpy as np
import pandas as pd

# Try known FundedNext terminal paths
paths = [
    r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe",
    r"C:\Program Files\MetaTrader 5\terminal64.exe",
]
path = None
for p in paths:
    if mt5.initialize(path=p):
        path = p
        break
    mt5.shutdown()

if path is None:
    # Auto-detect
    import os
    for root, dirs, files in os.walk(r"C:\Program Files"):
        if "terminal64.exe" in files and "mt5" in root.lower():
            p = os.path.join(root, "terminal64.exe")
            if mt5.initialize(path=p):
                path = p
                break
            mt5.shutdown()

if path is None:
    print("Could not initialize MT5. Trying default...")
    if not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}")
        exit(1)

print(f"MT5 initialized via: {path or 'default'}")
print(f"Terminal info: {mt5.terminal_info()._asdict() if mt5.terminal_info() else 'N/A'}")

# Check EURUSD availability
info = mt5.symbol_info("EURUSD")
if info:
    print(f"EURUSD: spread={info.spread} ({info.spread/10:.1f} pips), "
          f"ask={info.ask:.5f}, bid={info.bid:.5f}")
else:
    print(f"EURUSD NOT on this server ({mt5.last_error()})")
    # List available symbols
    symbols = mt5.symbols_get()
    print(f"Server has {len(symbols)} symbols. Sample: {[s.name for s in symbols[:10]]}")
    mt5.shutdown()
    exit(1)

# Check what data is available
from datetime import timedelta
today = datetime.utcnow()
print(f"\nToday: {today}")

# Try to get EURUSD data range
# First check from what date data exists
for months_back in [3, 6, 12, 24]:
    from_dt = today - timedelta(days=months_back * 30)
    rates = mt5.copy_rates_range("EURUSD", mt5.TIMEFRAME_M1, from_dt, today)
    if rates is not None and len(rates) > 0:
        first = datetime.utcfromtimestamp(rates[0][0])
        last = datetime.utcfromtimestamp(rates[-1][0])
        print(f"EURUSD data available: {len(rates)} bars, {first} to {last}")
        # Check spreads
        spreads = [r[6] for r in rates]
        print(f"  Spread stats: min={min(spreads)} max={max(spreads)} "
              f"median={np.median(spreads):.0f} (points)")
        break
    else:
        print(f"No data for {months_back}mo range")

# Try downloading Apr 21 - Jul 1 2026 to match cross pairs
print("\nDownloading Apr 21 - Jul 1 2026...")
from_dt = datetime(2026, 4, 21)
to_dt = datetime(2026, 7, 1)
rates = mt5.copy_rates_range("EURUSD", mt5.TIMEFRAME_M1, from_dt, to_dt)
if rates is None or len(rates) == 0:
    # Try from pos instead
    rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M1, 0, 100000)

if rates is not None and len(rates) > 0:
    first = datetime.utcfromtimestamp(rates[0][0])
    last = datetime.utcfromtimestamp(rates[-1][0])
    print(f"Downloaded: {len(rates)} bars, {first} to {last}")
    spreads = [r[6] for r in rates]
    print(f"  Spread stats: min={min(spreads)} max={max(spreads)} "
          f"median={np.median(spreads):.0f} p75={np.percentile(spreads,75):.0f} "
          f"p90={np.percentile(spreads,90):.0f}")
    
    # Save
    fpath = r"C:\Trading\Agentic_Trading\proxima_x\research\fundednext_data\EURUSD.npy"
    np.save(fpath, np.array(rates))
    print(f"Saved: {fpath}")
else:
    print(f"Download failed: {mt5.last_error()}")

mt5.shutdown()
print("Done.")
