"""Download Dark Consensus pairs from FundedNext."""
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import os

FN_TERMINAL = r"C:\Program Files\FundedNext MT5 Terminal\terminal64.exe"
PAIRS = ["EURJPY", "EURUSD", "GBPJPY"]
FROM = datetime(2026, 6, 8)
TO = datetime(2026, 7, 26)
OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'research', 'dark_research')

if not os.path.exists(OUT):
    os.makedirs(OUT)

print(f"Connecting to FundedNext terminal...")
if not mt5.initialize(path=FN_TERMINAL):
    print(f"FAILED: {mt5.last_error()}")
    sys.exit(1)

print(f"Connected: {mt5.terminal_info().name}")

for pair in PAIRS:
    print(f"\nDownloading {pair}...")
    rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M1, FROM, TO)
    if rates is None or len(rates) == 0:
        print(f"  {pair}: NO DATA")
        continue

    df = pd.DataFrame(rates)
    s = df['spread']
    print(f"  {len(df)} bars, spread: med={s.median()}, p90={s.quantile(0.90)}, max={s.max()}")

    fname = f"fundednext_{pair.lower()}_m1.npy"
    fpath = os.path.join(OUT, fname)
    np.save(fpath, rates)
    print(f"  Saved to {fname}")

mt5.shutdown()
print("\nDone.")
