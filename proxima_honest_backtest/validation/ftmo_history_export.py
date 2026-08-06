#!/usr/bin/env python3
"""FTMO history export — pull M5 bars from the real broker into parquet.

Why (apples-to-apples "SAME SOURCE" contract): the backtest must run on the SAME
broker candles the live feed will see, else the first live decision diverges.
Export last N M5 bars/pair from FTMO, then run the backtest on that export and
compute the expected bootstrap history hash to pass to run_proxima_live --hash.

RUN: python validation/ftmo_history_export.py [--bars 200] [--out m5_live]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "proxima_honest_backtest"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from proxima_honest_backtest.strategies.tokyo_h0.strategy import ALL_PAIRS

OUT_COLS = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", type=int, default=200)
    ap.add_argument("--out", default="m5_live")
    ap.add_argument("--pairs", nargs="*", default=None)
    args = ap.parse_args()

    import MetaTrader5 as mt5
    if not mt5.initialize():
        print("MT5 initialize FAILED")
        return
    try:
        pairs = args.pairs or ALL_PAIRS
        out_dir = ROOT / "data" / args.out
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = {}
        for sym in pairs:
            rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 0, args.bars)
            if rates is None or len(rates) == 0:
                print(f"[skip] {sym} no data")
                continue
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df["time"] = df["time"].dt.tz_localize(None)
            for col in ("tick_volume", "spread", "real_volume"):
                if col not in df.columns:
                    df[col] = 0
            df = df[OUT_COLS]
            path = out_dir / f"{sym}.parquet"
            df.to_parquet(path, index=False)
            rows[sym] = len(df)
            print(f"[ok] {sym}: {len(df)} bars -> {path}")
        print("exported:", rows)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()