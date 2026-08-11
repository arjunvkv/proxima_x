"""scripts/_absorb/pull_new_assets.py — acquire 200d M5 + live spreads for new
asset classes into the audit cache (audit_7_eas/market/<SYM>.pqt).

FTMO-Demo terminal ONLY. ONE attach, all symbols sequentially (history-service
degradation lesson). Schema matches cache: time, open, high, low, close.
Also samples live bid/ask for a measured-spread table (worst/typical).
"""
import sys, os, time, json
import numpy as np
import polars as pl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

MT5_PATH = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MARKET = os.path.join(ROOT, "audit_7_eas", "market")
TARGETS = ["US30.cash","US500.cash","GER40.cash","UK100.cash","JP225.cash",
           "HK50.cash","USOIL.cash","UKOIL.cash","DXY.cash"]

import MetaTrader5 as mt5
ok = mt5.initialize(MT5_PATH)
print("initialize:", ok, mt5.last_error() if not ok else "")
if not ok:
    sys.exit(1)

def pull(sym):
    from datetime import datetime, timedelta
    now = datetime.now()
    rates = mt5.copy_rates_range(sym, mt5.TIMEFRAME_M5, now - timedelta(days=210), now)
    if rates is None or len(rates) == 0:
        return None
    df = pl.DataFrame({
        "time": rates["time"].astype(np.int64),
        "open": rates["open"].astype(float),
        "high": rates["high"].astype(float),
        "low": rates["low"].astype(float),
        "close": rates["close"].astype(float),
    }).sort("time").unique(subset="time", keep="last")
    df.write_parquet(os.path.join(MARKET, f"{sym}.pqt"))
    return df

pulled = {}
for sym in TARGETS:
    df = pull(sym)
    if df is None:
        print(f"{sym}: MISSING")
        pulled[sym] = None
        continue
    first = int(df["time"][0]); last = int(df["time"][-1])
    print(f"{sym}: {len(df)} bars {time.strftime('%Y-%m-%d', time.gmtime(first))}..{time.strftime('%Y-%m-%d', time.gmtime(last))}")
    pulled[sym] = len(df)

# ---- live spread sampling (only if market open) ----
spreads = {}
open_syms = [s for s in pulled if pulled[s]]
for sym in open_syms:
    try:
        info = mt5.symbol_info(sym)
        if info is None or not info.visible:
            mt5.symbol_select(sym, True); info = mt5.symbol_info(sym)
        bid, ask = info.bid, info.ask
        if bid > 0 and ask > 0:
            pt = info.point
            spreads[sym] = round((ask - bid) / pt, 1)
            print(f"spread {sym}: {spreads[sym]} pts (bid {bid} ask {ask})")
    except Exception as e:
        print(f"spread {sym}: ERR {e}")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "new_asset_spreads.json"), "w") as f:
    json.dump({"pulled": {k: v for k, v in pulled.items() if v}, "spread_pts": spreads}, f, indent=1)
mt5.shutdown()
print("done")