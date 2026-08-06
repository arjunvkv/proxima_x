"""Check what bars exist around midnight 2026-08-03 in the proven M5 dataset."""
import pandas as pd
import numpy as np
from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align

PAIRS_ALL = ["EURUSD", "GBPUSD", "USDJPY", "EURAUD", "GBPAUD", "EURJPY", "GBPJPY", "EURNZD", "GBPNZD"]
raw, _ = load_and_align()
pieces = [df.set_index("time")[["close","open","high","low"]] for df in raw.values()]
for i, p in enumerate(raw.keys()):
    pieces[i].columns = [p, f"{p}_open", f"{p}_high", f"{p}_low"]
df_all = pd.concat(pieces, axis=1, sort=True).ffill().bfill()
times = pd.to_datetime(df_all.index)

# Show what bars exist around midnight Aug 3
mask = (times >= pd.Timestamp("2026-08-02 23:45:00")) & (times <= pd.Timestamp("2026-08-03 01:00:00"))
print("Bars around midnight 2026-08-03 (proven backtest M5 data):")
print(df_all[mask][["GBPJPY", "GBPJPY_open", "GBPJPY_high", "GBPJPY_low"]].to_string())
print()

# Check most recent dates available
print("Last 5 timestamps in M5 dataset:")
print(times[-5:])
print()
print("First date:", times[0])
print("Last date:", times[-1])
