"""Phase 0 probe: how deep does the FTMO terminal serve tick history, and do trade ticks exist?

Attaches to the RUNNING local FTMO terminal (blank creds, explicit MT5_PATH), pulls
copy_ticks_range for the 7 primary research symbols over the last ~60 days, and reports
the actually-served window + tick counts + LAST-flag share. Also samples spread stats.

Run: unset PYTHONPATH && ./.venv/Scripts/python.exe scripts/_probe_tick_depth.py
"""
import os, time
from datetime import datetime, timedelta

os.environ.setdefault("MT5_PATH", r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe")
import MetaTrader5 as mt5

SYMS = ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "GBPJPY", "AUDUSD", "USDCAD"]
LOOKBACK_DAYS = 60

if not mt5.initialize(path=os.environ["MT5_PATH"], timeout=8000):
    raise SystemExit(f"MT5 init failed: {mt5.last_error()}")

acct = mt5.account_info()
print(f"attached account: {acct.login} @ {acct.server} (name={acct.name})" if acct else "no account")

now = int(time.time())
t0 = now - LOOKBACK_DAYS * 86400
print(f"probing {t0}..{now} ({LOOKBACK_DAYS}d)")

for sym in SYMS:
    ticks = mt5.copy_ticks_range(sym, t0, now, mt5.COPY_TICKS_ALL)
    if ticks is None or len(ticks) == 0:
        print(f"{sym}: EMPTY (err={mt5.last_error()})")
        continue
    n = len(ticks)
    first = datetime.utcfromtimestamp(int(ticks[0]["time_msc"]) // 1000).strftime("%Y-%m-%d %H:%M")
    last = datetime.utcfromtimestamp(int(ticks[-1]["time_msc"]) // 1000).strftime("%Y-%m-%d %H:%M")
    days_served = (int(ticks[-1]["time_msc"]) - int(ticks[0]["time_msc"])) / 86400e3
    # flags: 1=bid 2=ask 4=last 8=volume ; trade ticks carry LAST flag
    flags = set(int(t["flags"]) for t in ticks[:20000])
    n_last = sum(1 for t in ticks[:20000] if int(t["flags"]) & 4)
    n_vol = sum(1 for t in ticks[:20000] if float(t["volume"]) > 0)
    sample_spread = []
    for t in ticks[:20000]:
        b, a = float(t["bid"]), float(t["ask"])
        if b > 0 and a > 0:
            sample_spread.append(a - b)
    ss = sorted(sample_spread)
    med = ss[len(ss) // 2] if ss else float("nan")
    print(f"{sym}: {n:,} ticks served, window {first} -> {last} ({days_served:.1f}d), "
          f"LAST-share {n_last/20000:.2%}, vol>0 {n_vol/20000:.2%}, flags={sorted(flags)}, "
          f"spread med {med:.6f}")

mt5.shutdown()
print("done")