#!/usr/bin/env python3
"""Patch TokyoH0_MT5.mq5 to fix g_last_entry initialization bug."""
import re

def main():
    path = r"c:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\TokyoH0_MT5.mq5"
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()

    # Look for CheckEntry() logic
    old_block = """   if(today == g_last_entry) return;
   g_last_entry = today;"""

    new_block = """   if(today == g_last_entry) return;"""

    if old_block in code:
        code = code.replace(old_block, new_block)
        # Update version string to v1.06
        code = code.replace("Tokyo H0 v1.04", "Tokyo H0 v1.06")
        code = code.replace("Tokyo H0 v1.05", "Tokyo H0 v1.06")
        
        # Add g_last_entry = today at the bottom of CheckEntry() after execution loop
        old_loop_end = 'Print("  Entered ", en, "/", te);\n}'
        new_loop_end = 'Print("  Entered ", en, "/", te);\n   g_last_entry = today;\n}'
        code = code.replace(old_loop_end, new_loop_end)

        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        print("🟢 TokyoH0_MT5.mq5 successfully patched to v1.06!")
    else:
        print("⚠️ Target block not found or already patched.")

if __name__ == "__main__":
    main()
