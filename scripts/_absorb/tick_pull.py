"""scripts/_absorb/tick_pull.py — pull 60d of genuine ticks per symbol and
aggregate to 1-min bars with price-based signed flow. Saves a small research
cache under results/ticks/<SYM>.pqt — NOT a tick archive, just minute aggregates
(the absorption->impact study at tick resolution).

Read-only vs the engine: uses the same FTMO terminal the live worker attaches to
(blank creds + explicit MT5_PATH), no engine files touched.

FEED HONESTY (verified 2026-08-10): the FTMO-Demo feed sets the LAST flag and
movements on both sides, but `last` is ALWAYS 0.0 and there is no tick volume,
no BUY/SELL flag bit. True aggressor volume CANNOT be measured. The signed-flow
column is therefore the standard ONE-SIDED QUOTE RULE (Conti-style) applied to
real bid/ask updates:
    +1 when the ask lifts alone (d_ask>0 & d_bid==0)   -> buy pressure
    -1 when the bid drops alone (d_bid<0 & d_ask==0)   -> sell pressure
     0 otherwise (ambiguous / two-sided re-marking)
It is a pressure proxy, explicitly NOT claimed to be transaction flow.

Aggregates per server-minute (server epoch = same basis as the M5 tape):
  ts         : minute start, server epoch seconds
  open/high/low/close : mid-price (bid+ask)/2 OHLC of the minute
  n_last     : count of trade-flag ticks (flags & TICK_FLAG_LAST) — activity
  imb        : one-sided quote pressure sum (signed flow proxy)
  spread_med : median(ask - bid) in price units
  n_quotes   : count of quotes with valid bid & ask

Run: unset PYTHONPATH && ./.venv/Scripts/python.exe scripts/_absorb/tick_pull.py [SYM]
"""
from __future__ import annotations
import json, os, sys, time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import polars as pl

MT5_PATH = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
UNIVERSE = ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "GBPJPY", "AUDUSD", "USDCAD"]
DAYS = 60
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "ticks")
FLAG_LAST = 4

os.makedirs(OUT, exist_ok=True)


def pull_symbol(mt5, sym: str, from_ms: int, to_ms: int) -> pl.DataFrame:
    ticks = mt5.copy_ticks_range(sym, from_ms, to_ms, mt5.COPY_TICKS_ALL)
    if ticks is None or len(ticks) == 0:
        print(f"  {sym}: no ticks in range")
        return pl.DataFrame()
    n = len(ticks)
    t_ms = ticks["time_msc"].astype("int64")
    bid = ticks["bid"].astype("float64")
    ask = ticks["ask"].astype("float64")
    flags = ticks["flags"].astype("int64")
    is_last = (flags & FLAG_LAST) > 0
    valid_q = (bid > 0) & (ask > 0)
    # one-sided quote rule on real bid/ask updates (see module docstring)
    def _prev(col):
        p = np.empty(n)
        p[0] = np.nan
        p[1:] = col[:-1]
        return p
    p_bid, p_ask = _prev(bid), _prev(ask)
    d_bid = np.where(valid_q & ~np.isnan(p_bid), bid - p_bid, np.nan)
    d_ask = np.where(valid_q & ~np.isnan(p_ask), ask - p_ask, np.nan)
    signed = np.zeros(n, dtype=np.float64)
    signed[(d_ask > 0) & (d_bid == 0)] = 1.0     # ask lifted alone = buy press
    signed[(d_bid < 0) & (d_ask == 0)] = -1.0    # bid dropped alone = sell press
    mid = np.where(valid_q, (bid + ask) / 2.0, np.nan)
    spread = np.where(valid_q, ask - bid, np.nan)

    df = pl.DataFrame({
        "minute": (t_ms // 60000),
        "mid": mid,
        "is_last": is_last,
        "signed": signed,
        "spread": spread,
    })
    agg = df.group_by("minute", maintain_order=True).agg([
        pl.col("mid").first().alias("open"),
        pl.col("mid").max().alias("high"),
        pl.col("mid").min().alias("low"),
        pl.col("mid").last().alias("close"),
        pl.col("is_last").sum().alias("n_last"),
        pl.col("signed").sum().alias("imb"),
        pl.col("spread").median().alias("spread_med"),
        pl.col("mid").count().alias("n_quotes"),
    ]).filter(pl.col("n_quotes") >= 1)
    agg = agg.with_columns((pl.col("minute") * 60).alias("ts"))
    print(f"  {sym}: {n:,} ticks -> {len(agg):,} minutes "
          f"({agg['n_last'].sum():,} trade prints)", flush=True)
    return agg


def main(sym: Optional[str] = None) -> None:
    import MetaTrader5 as mt5  # noqa: F401  (module import has side-effects)
    if not mt5.initialize(path=MT5_PATH, timeout=8000):
        raise SystemExit(f"MT5 init failed: {mt5.last_error()}")
    info = mt5.account_info()
    print(f"attached {info.login if info else '?'} @ "
          f"{info.server if info else '?'} "
          f"({info.name if info else '?'})")
    to_s = int(time.time())
    from_s = to_s - DAYS * 86400
    print(f"pulling {DAYS}d window "
          f"{datetime.fromtimestamp(from_s, tz=timezone.utc):%Y-%m-%d %H:%M} -> "
          f"{datetime.fromtimestamp(to_s, tz=timezone.utc):%Y-%m-%d %H:%M}")
    window = {"from_s": from_s, "to_s": to_s}
    with open(os.path.join(OUT, "_window.json"), "w") as f:
        json.dump(window, f)
    syms = [sym] if sym else UNIVERSE
    for s in syms:
        try:
            df = pull_symbol(mt5, s, from_s, to_s)
        except SystemError:
            print(f"  {s}: SystemError (unrecoverable this run)", flush=True)
            df = pl.DataFrame()
        if df.height > 0:
            df.write_parquet(os.path.join(OUT, f"{s}.pqt"))
            print(f"  {s}: saved {len(df):,} minutes "
                  f"({df['n_last'].sum():,} trade prints)", flush=True)
        else:
            print(f"  {s}: no data this run", flush=True)
    mt5.shutdown()
    print("done")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)