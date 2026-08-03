#!/usr/bin/env python3
"""Update CPMC_Z_MT5.mq5 default HOLD_BARS to 18 (90 Minutes Hold Time)."""

def main():
    path = r"paper_trade\mt5_backtest\CPMC_Z_MT5.mq5"
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()

    old_line = "input int      HOLD_BARS           = 9;       // 45m hold time"
    new_line = "input int      HOLD_BARS           = 18;      // 90m hold time (Optimized for Peak Profit)"

    if old_line in code:
        code = code.replace(old_line, new_line)
        code = code.replace("=== CPMC Momentum Continuation v1.00", "=== CPMC Engine MT5 v1.05 (90m Hold Optimization)")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        print("🟢 CPMC_Z_MT5.mq5 default HOLD_BARS successfully updated to 18 (90 Minutes)!")
    else:
        print("⚠️ Target HOLD_BARS line not found or already updated.")

if __name__ == "__main__":
    main()
