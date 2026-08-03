"""Download all 6 cross pairs M1 data from FTMO for Jun 8-Jul 25."""
import MetaTrader5 as mt5
from datetime import datetime
import numpy as np

pairs = ['AUDNZD', 'EURAUD', 'EURNZD', 'GBPAUD', 'GBPCAD', 'GBPNZD']
path = r'C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe'
mt5.initialize(path=path)

for pair in pairs:
    rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M1,
                                 datetime(2026, 5, 1), datetime(2026, 7, 26))
    if rates is not None and len(rates) > 0:
        from datetime import datetime as dt
        fname = fr'C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\ftmo_{pair.lower()}_m1.npy'
        np.save(fname, rates)
        print(f"{pair}: {len(rates)} bars, {dt.utcfromtimestamp(rates[0][0])} to {dt.utcfromtimestamp(rates[-1][0])}")
    else:
        print(f"{pair}: FAILED ({mt5.last_error()})")

mt5.shutdown()
