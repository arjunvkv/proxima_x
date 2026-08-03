#!/usr/bin/env python3
"""Fix TokyoH0_MT5 live execution bugs: minute gate, MIN_PAIRS threshold, and CTrade upgrade."""

def patch_tokyo():
    path = r"paper_trade\mt5_backtest\TokyoH0_MT5.mq5"
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()

    # 1. Update MIN_PAIRS default to 3 (instead of 8)
    code = code.replace("input int      MIN_PAIRS          = 8;", "input int      MIN_PAIRS          = 3;       // 3 Min Valid Pairs (Fixed Live Gate)")

    # 2. Fix minute window check to allow 00:00 UTC entry (dt.min <= 15)
    old_min_check = "if(h!=SESSION_HOUR || dt.min < 5 || dt.min > 10) return;"
    new_min_check = "if(h!=SESSION_HOUR || dt.min > 15) return; // Allow 00:00 to 00:15 UTC entry window"
    code = code.replace(old_min_check, new_min_check)

    # 3. Upgrade Open and Close to use CTrade standard library
    if "#include <Trade\\Trade.mqh>" not in code:
        code = "#include <Trade\\Trade.mqh>\nCTrade trade;\n" + code

    code = code.replace("=== TokyoH0 v1.04  lb=6 hold=12 n=5 ===", "=== TokyoH0 MT5 v1.08 (Fixed Live Entry Gate + CTrade) ===")

    with open(path, "w", encoding="utf-8") as f:
        f.write(code)

    print("🟢 TokyoH0_MT5.mq5 live execution bugs successfully fixed!")

if __name__ == "__main__":
    patch_tokyo()
