#!/usr/bin/env python3
"""Patch ORB_Ride_MT5.mq5 to allow London Breakout Checks throughout 12:30 PM - 03:30 PM IST."""
import re

def main():
    path = r"c:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\ORB_Ride_MT5.mq5"
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()

    # Old restricted 5-minute window
    old_line = "if(dt.hour == 10 && dt.min >= 0 && dt.min <= 5) {"

    # New expanded 3-hour breakout window (Server Hour 10 to 12 -> 12:30 PM to 03:30 PM IST)
    new_line = "if(dt.hour >= 10 && dt.hour <= 12) {"

    if old_line in code:
        code = code.replace(old_line, new_line)
        code = code.replace("ORB Breakout Ride (#4) EA Init", "ORB Breakout Ride (#4) EA Init v1.02")

        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        print("🟢 ORB_Ride_MT5.mq5 successfully patched to v1.02 (Expanded 3-Hour Breakout Window)!")
    else:
        print("⚠️ Target line not found or already patched.")

if __name__ == "__main__":
    main()
