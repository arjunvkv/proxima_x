#!/usr/bin/env python3
"""Patch Ultra_Monster_MT5.mq5 to v1.05 with 12.0 Pip Volatility Gate + 1.0 Pip Breakout Buffer."""

def main():
    path = r"c:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\Ultra_Monster_MT5.mq5"
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()

    # 1. Update MIN_RANGE_PIPS default to 12.0 pips
    code = code.replace("input double   MIN_RANGE_PIPS      = 6.0;", "input double   MIN_RANGE_PIPS      = 12.0;")

    # 2. Update Breakout conditions with 1.0 Pip Buffer
    old_buy = "if(c_now > orb_high) {"
    old_sell = "} else if(c_now < orb_low) {"

    new_buy = "double buf = (StringFind(s, \"JPY\") >= 0) ? 0.010 : 0.00010;\n         if(c_now > (orb_high + buf)) {"
    new_sell = "} else if(c_now < (orb_low - buf)) {"

    code = code.replace(old_buy, new_buy)
    code = code.replace(old_sell, new_sell)
    code = code.replace("ULTRA MONSTER Engine MT5 Init", "ULTRA MONSTER Engine MT5 Init v1.05 (12p Gate + 1p Buffer)")

    with open(path, "w", encoding="utf-8") as f:
        f.write(code)

    print("🟢 Ultra_Monster_MT5.mq5 successfully upgraded to v1.05 (12.0 Pip Gate + 1.0 Pip Buffer)!")

if __name__ == "__main__":
    main()
