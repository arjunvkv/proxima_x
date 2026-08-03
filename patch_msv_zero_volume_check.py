#!/usr/bin/env python3
"""Patch MSV_Asian_Exhaustion_MT5 and all EAs with zero-volume protection."""

def patch_msv():
    path = r"paper_trade\mt5_backtest\MSV_Asian_Exhaustion_MT5.mq5"
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()

    old_close = "double vol = NormalizeDouble(PositionGetDouble(POSITION_VOLUME), 2);"
    new_close = """double vol = NormalizeDouble(PositionGetDouble(POSITION_VOLUME), 2);
   if(vol <= 0.0) { g_t[p].active = false; g_t[p].ticket = 0; return false; }"""

    if old_close in code:
        code = code.replace(old_close, new_close)
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        print("🟢 MSV_Asian_Exhaustion_MT5.mq5 zero-volume protection applied!")

def main():
    patch_msv()

if __name__ == "__main__":
    main()
