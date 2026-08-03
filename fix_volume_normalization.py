#!/usr/bin/env python3
"""Fix Volume Normalization across all EAs to prevent [Invalid volume] close errors."""

import glob

def fix_ea(path):
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()

    old_vol = "(float)PositionGetDouble(POSITION_VOLUME)"
    new_vol = "NormalizeDouble(PositionGetDouble(POSITION_VOLUME), 2)"

    if old_vol in code:
        code = code.replace(old_vol, new_vol)
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"🟢 Fixed volume normalization in {path}")
    else:
        print(f"ℹ️ {path} already uses clean volume normalization.")

def main():
    eas = glob.glob(r"paper_trade\mt5_backtest\*.mq5")
    for ea in eas:
        fix_ea(ea)

if __name__ == "__main__":
    main()
